"""Strict MATH verification in spawn workers with parent-enforced timeouts."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as distribution_version
import json
import math
import multiprocessing
from multiprocessing.connection import Connection

from ..isolation import sanitize_exception
from ..schema import (
    CandidateRecord,
    QuestionRecord,
    QuestionVerificationRecord,
    Source,
    VerificationRecord,
    VerificationStatus,
)


MATH_VERIFY_VERSION = "0.9.0"
MATH_PARSER_VERSION = "math_boxed_final_v1"
MATH_VERIFIER_NAME = "math-verify"
MATH_CANONICAL_REPRESENTATION_VERSION = "sympy_srepr_v1"

_BOX_MARKER = r"\boxed{"
_EXTRACTION_STATUSES = frozenset({"parsed", "invalid", "ambiguous"})
_GOLD_READY = "GOLD_READY"
_GOLD_ERROR = "GOLD_ERROR"
_VERIFY = "VERIFY"
_STOP = "STOP"
_CANDIDATE_RESULT = "CANDIDATE_RESULT"
_CANDIDATE_ERROR = "CANDIDATE_ERROR"

GoldParserHook = Callable[[str, object], object]
CandidateVerifierHook = Callable[[object, str, object], tuple[bool | None, str | None]]


class DependencyContractError(RuntimeError):
    """A locked verification dependency is absent or has the wrong version."""


class MathVerificationError(ValueError):
    """The MATH question, candidate identities, or verifier inputs are invalid."""


class _MathParseFailure(ValueError):
    pass


@dataclass(frozen=True)
class BoxedAnswer:
    """The strict extraction outcome for one terminal boxed answer."""

    status: str
    boxed_text: str | None
    error_type: str | None

    def __post_init__(self) -> None:
        if self.status not in _EXTRACTION_STATUSES:
            raise ValueError("unknown boxed-answer status")
        if self.status == "parsed":
            if not isinstance(self.boxed_text, str) or not self.boxed_text or self.error_type is not None:
                raise ValueError("parsed boxed answer must contain one boxed expression")
        elif self.boxed_text is not None or not isinstance(self.error_type, str) or not self.error_type:
            raise ValueError("unparsed boxed answer must contain one error type")


@dataclass
class _WorkerHandle:
    process: multiprocessing.Process
    connection: Connection


def extract_math_gold(solution: str) -> BoxedAnswer:
    """Extract the unique terminal boxed answer from an official solution."""
    return _extract_terminal_box(solution)


def extract_math_candidate(response: str) -> BoxedAnswer:
    """Extract the unique terminal boxed answer from a generated response."""
    return _extract_terminal_box(response)


def verify_math_question(
    question: QuestionRecord,
    candidates: Iterable[CandidateRecord],
    gold_timeout_seconds: float,
    candidate_timeout_seconds: float,
    *,
    verification_run_id: str,
    gold_parser_hook: GoldParserHook = None,
    candidate_verifier_hook: CandidateVerifierHook = None,
    worker_state: object = None,
) -> tuple[QuestionVerificationRecord, list[VerificationRecord]]:
    """Verify one MATH question while the parent owns every timeout."""
    _require_math_verify_version()
    _validate_call(
        question,
        verification_run_id,
        gold_timeout_seconds,
        candidate_timeout_seconds,
    )
    materialized = _validate_candidates(question, candidates)
    gold_box = extract_math_gold(question.solution)
    if gold_box.status != "parsed":
        return _gold_error(question, verification_run_id, "gold_parse_failure"), []

    gold_hook = gold_parser_hook or _math_verify_gold_parser
    candidate_hook = candidate_verifier_hook or _math_verify_candidate_verifier
    context = multiprocessing.get_context("spawn")
    worker, gold_error, canonical_gold = _start_worker(
        context,
        gold_box.boxed_text,
        gold_hook,
        candidate_hook,
        worker_state,
        gold_timeout_seconds,
    )
    if worker is None:
        return _gold_error(question, verification_run_id, gold_error), []

    question_record = QuestionVerificationRecord(
        verification_run_id=verification_run_id,
        original_question_id=question.original_question_id,
        source=Source.MATH,
        gold_parse_status="ok",
        canonical_gold=canonical_gold,
        error_type=None,
    )
    question_record.validate()
    verified: list[VerificationRecord] = []
    try:
        for candidate in materialized:
            extracted = extract_math_candidate(candidate.response)
            if extracted.status != "parsed":
                status = (
                    VerificationStatus.AMBIGUOUS
                    if extracted.status == "ambiguous"
                    else VerificationStatus.INVALID
                )
                verified.append(
                    _verification_record(
                        candidate,
                        verification_run_id,
                        candidate_timeout_seconds,
                        status,
                        extracted.status,
                        None,
                        extracted.error_type,
                    )
                )
                continue

            if worker is None:
                worker, restart_error, restarted_canonical_gold = _start_worker(
                    context,
                    gold_box.boxed_text,
                    gold_hook,
                    candidate_hook,
                    worker_state,
                    gold_timeout_seconds,
                )
                if worker is None:
                    return _gold_error(question, verification_run_id, restart_error), []
                if restarted_canonical_gold != canonical_gold:
                    _shutdown_worker(worker)
                    worker = None
                    return _gold_error(question, verification_run_id, "gold_parse_failure"), []

            outcome, error_type, canonical_prediction = _verify_in_worker(
                worker,
                extracted.boxed_text,
                candidate_timeout_seconds,
            )
            if outcome == "timeout":
                _terminate_worker(worker)
                worker = None
                verified.append(
                    _verification_record(
                        candidate,
                        verification_run_id,
                        candidate_timeout_seconds,
                        VerificationStatus.ERROR,
                        "error",
                        None,
                        "timeout",
                    )
                )
            elif outcome == "error":
                verified.append(
                    _verification_record(
                        candidate,
                        verification_run_id,
                        candidate_timeout_seconds,
                        VerificationStatus.ERROR,
                        "error",
                        None,
                        error_type,
                    )
                )
            elif outcome == "ambiguous":
                verified.append(
                    _verification_record(
                        candidate,
                        verification_run_id,
                        candidate_timeout_seconds,
                        VerificationStatus.AMBIGUOUS,
                        "ambiguous",
                        canonical_prediction,
                        "math_parse_failure",
                    )
                )
            else:
                status = VerificationStatus.CORRECT if outcome == "correct" else VerificationStatus.INCORRECT
                verified.append(
                    _verification_record(
                        candidate,
                        verification_run_id,
                        candidate_timeout_seconds,
                        status,
                        "parsed",
                        canonical_prediction,
                        None,
                    )
                )
    finally:
        if worker is not None:
            _shutdown_worker(worker)
    return question_record, verified


def _extract_terminal_box(text: str) -> BoxedAnswer:
    if not isinstance(text, str):
        raise TypeError("MATH answer source must be text")
    starts: list[int] = []
    offset = 0
    while True:
        start = text.find(_BOX_MARKER, offset)
        if start < 0:
            break
        starts.append(start)
        offset = start + len(_BOX_MARKER)
    if not starts:
        return BoxedAnswer("invalid", None, "missing_final_marker")

    boxes: list[tuple[str, int]] = []
    for start in starts:
        end = _balanced_box_end(text, start + len(_BOX_MARKER))
        if end is None:
            return BoxedAnswer("invalid", None, "math_parse_failure")
        boxes.append((text[start : end + 1], end))
    if len(boxes) > 1:
        if len({boxed for boxed, _end in boxes}) > 1:
            return BoxedAnswer("ambiguous", None, "conflicting_markers")
        return BoxedAnswer("invalid", None, "duplicate_same_marker")
    boxed, end = boxes[0]
    if text[end + 1 :].strip():
        return BoxedAnswer("invalid", None, "prompt_contract_mismatch")
    return BoxedAnswer("parsed", boxed, None)


def _balanced_box_end(text: str, content_start: int) -> int | None:
    depth = 1
    for index in range(content_start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def _require_math_verify_version() -> None:
    try:
        installed = distribution_version("math-verify")
    except PackageNotFoundError as error:
        raise DependencyContractError("math-verify==0.9.0 is required") from error
    if installed != MATH_VERIFY_VERSION:
        raise DependencyContractError("math-verify==0.9.0 is required")


def _validate_call(
    question: QuestionRecord,
    verification_run_id: str,
    gold_timeout_seconds: float,
    candidate_timeout_seconds: float,
) -> None:
    if not isinstance(question, QuestionRecord):
        raise TypeError("question must be a QuestionRecord")
    question.validate()
    if question.source is not Source.MATH:
        raise MathVerificationError("question must use the MATH source shape")
    if not isinstance(verification_run_id, str) or not verification_run_id.strip():
        raise MathVerificationError("verification_run_id must be nonempty text")
    for name, value in (
        ("gold_timeout_seconds", gold_timeout_seconds),
        ("candidate_timeout_seconds", candidate_timeout_seconds),
    ):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise MathVerificationError(f"{name} must be positive and finite")


def _validate_candidates(
    question: QuestionRecord,
    candidates: Iterable[CandidateRecord],
) -> list[CandidateRecord]:
    materialized = list(candidates)
    seen: set[tuple[str, str]] = set()
    for candidate in materialized:
        if not isinstance(candidate, CandidateRecord):
            raise TypeError("candidates must contain CandidateRecord values")
        candidate.validate()
        plan = candidate.plan
        if (
            plan.original_question_id != question.original_question_id
            or plan.question_content_hash != question.question_content_hash
            or plan.source is not Source.MATH
        ):
            raise MathVerificationError("candidate question identity does not match the MATH question")
        if plan.key in seen:
            raise MathVerificationError("duplicate composite candidate key")
        seen.add(plan.key)
    return materialized


def _math_verify_gold_parser(boxed_text: str, _state: object) -> object:
    from math_verify import parse

    parsed = parse(
        boxed_text,
        fallback_mode="no_fallback",
        extraction_mode="first_match",
        parsing_timeout=None,
        raise_on_error=True,
    )
    if len(parsed) != 1:
        raise _MathParseFailure("gold did not produce exactly one parsed expression")
    return parsed[0]


def _math_verify_candidate_verifier(
    gold: object,
    boxed_text: str,
    _state: object,
) -> tuple[bool | None, str | None]:
    from math_verify import parse, verify

    parsed = parse(
        boxed_text,
        fallback_mode="no_fallback",
        extraction_mode="first_match",
        parsing_timeout=None,
        raise_on_error=True,
    )
    if len(parsed) != 1:
        return None, None
    canonical_prediction = _canonical_math_value(parsed[0])
    equivalent = verify(
        gold,
        parsed[0],
        strict=True,
        timeout_seconds=None,
        raise_on_error=True,
    )
    if type(equivalent) is not bool:
        raise TypeError("math-verify returned a non-boolean equivalence result")
    return equivalent, canonical_prediction


def _canonical_math_value(value: object) -> str:
    """Encode one parsed value as a deterministic private string, never an object."""
    from sympy import Basic, MatrixBase, srepr

    if isinstance(value, (Basic, MatrixBase)):
        return f"{MATH_CANONICAL_REPRESENTATION_VERSION}:{srepr(value)}"
    if isinstance(value, str):
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        return f"string_json_v1:{encoded}"
    raise TypeError("math parser returned an unsupported canonical value")


def _worker_main(
    connection: Connection,
    boxed_gold: str,
    gold_parser_hook: GoldParserHook,
    candidate_verifier_hook: CandidateVerifierHook,
    worker_state: object,
) -> None:
    try:
        try:
            gold = gold_parser_hook(boxed_gold, worker_state)
            if gold is None:
                raise _MathParseFailure("gold parser returned no expression")
            canonical_gold = _canonical_math_value(gold)
        except BaseException as error:
            sanitized = sanitize_exception(error)
            error_type = (
                sanitized
                if sanitized in {"dependency_error", "timeout"}
                else "gold_parse_failure"
            )
            connection.send((_GOLD_ERROR, error_type))
            return
        connection.send((_GOLD_READY, canonical_gold))
        while True:
            message = connection.recv()
            if message == (_STOP,):
                return
            if not isinstance(message, tuple) or len(message) != 2 or message[0] != _VERIFY:
                return
            try:
                equivalent, canonical_prediction = candidate_verifier_hook(gold, message[1], worker_state)
                if equivalent is None:
                    if canonical_prediction is not None:
                        raise TypeError("ambiguous candidate must not carry a canonical prediction")
                    connection.send((_CANDIDATE_RESULT, "ambiguous", None))
                elif type(equivalent) is bool:
                    if not isinstance(canonical_prediction, str) or not canonical_prediction:
                        raise TypeError("verified candidate must carry a canonical prediction")
                    connection.send(
                        (
                            _CANDIDATE_RESULT,
                            "correct" if equivalent else "incorrect",
                            canonical_prediction,
                        )
                    )
                else:
                    raise TypeError("candidate verifier hook returned an unsupported result")
            except BaseException as error:
                connection.send((_CANDIDATE_ERROR, sanitize_exception(error)))
    except (EOFError, BrokenPipeError, OSError):
        return
    finally:
        connection.close()


def _start_worker(
    context,
    boxed_gold: str,
    gold_parser_hook: GoldParserHook,
    candidate_verifier_hook: CandidateVerifierHook,
    worker_state: object,
    timeout_seconds: float,
) -> tuple[_WorkerHandle | None, str | None, str | None]:
    parent, child = context.Pipe(duplex=True)
    process = context.Process(
        target=_worker_main,
        args=(child, boxed_gold, gold_parser_hook, candidate_verifier_hook, worker_state),
    )
    try:
        process.start()
    except BaseException:
        parent.close()
        child.close()
        raise
    child.close()
    worker = _WorkerHandle(process, parent)
    if not parent.poll(timeout_seconds):
        _terminate_worker(worker)
        return None, "timeout", None
    try:
        message = parent.recv()
    except (EOFError, OSError):
        _terminate_worker(worker)
        return None, "unexpected_error", None
    if (
        isinstance(message, tuple)
        and len(message) == 2
        and message[0] == _GOLD_READY
        and isinstance(message[1], str)
        and message[1]
    ):
        return worker, None, message[1]
    if (
        isinstance(message, tuple)
        and len(message) == 2
        and message[0] == _GOLD_ERROR
        and isinstance(message[1], str)
    ):
        _shutdown_worker(worker)
        return None, message[1], None
    _terminate_worker(worker)
    return None, "unexpected_error", None


def _verify_in_worker(
    worker: _WorkerHandle,
    boxed_text: str,
    timeout_seconds: float,
) -> tuple[str, str | None, str | None]:
    try:
        worker.connection.send((_VERIFY, boxed_text))
    except (BrokenPipeError, EOFError, OSError):
        return "error", "unexpected_error", None
    if not worker.connection.poll(timeout_seconds):
        return "timeout", "timeout", None
    try:
        message = worker.connection.recv()
    except (EOFError, OSError):
        return "error", "unexpected_error", None
    if (
        isinstance(message, tuple)
        and len(message) == 3
        and message[0] == _CANDIDATE_RESULT
        and message[1] in {"correct", "incorrect", "ambiguous"}
        and (message[2] is None or isinstance(message[2], str))
    ):
        return message[1], None, message[2]
    if (
        isinstance(message, tuple)
        and len(message) == 2
        and message[0] == _CANDIDATE_ERROR
        and isinstance(message[1], str)
    ):
        return "error", message[1], None
    return "error", "unexpected_error", None


def _verification_record(
    candidate: CandidateRecord,
    verification_run_id: str,
    timeout_seconds: float,
    verification_status: VerificationStatus,
    parse_status: str,
    canonical_prediction: str | None,
    error_type: str | None,
) -> VerificationRecord:
    record = VerificationRecord(
        generation_run_id=candidate.plan.generation_run_id,
        verification_run_id=verification_run_id,
        candidate_id=candidate.plan.candidate_id,
        original_question_id=candidate.plan.original_question_id,
        source=Source.MATH,
        generator_id=candidate.plan.generator_id,
        verification_status=verification_status,
        prediction_parse_status=parse_status,
        canonical_prediction=canonical_prediction,
        verifier_name=MATH_VERIFIER_NAME,
        verifier_version=MATH_VERIFY_VERSION,
        parser_version=MATH_PARSER_VERSION,
        verifier_timeout_seconds=timeout_seconds,
        error_type=error_type,
    )
    record.validate()
    return record


def _gold_error(
    question: QuestionRecord,
    verification_run_id: str,
    error_type: str | None,
) -> QuestionVerificationRecord:
    record = QuestionVerificationRecord(
        verification_run_id=verification_run_id,
        original_question_id=question.original_question_id,
        source=Source.MATH,
        gold_parse_status="gold_verification_error",
        canonical_gold=None,
        error_type=error_type or "unexpected_error",
    )
    record.validate()
    return record


def _shutdown_worker(worker: _WorkerHandle) -> None:
    try:
        if worker.process.is_alive():
            worker.connection.send((_STOP,))
    except (BrokenPipeError, EOFError, OSError):
        pass
    worker.process.join(timeout=1.0)
    if worker.process.is_alive():
        worker.process.terminate()
        worker.process.join()
    worker.connection.close()
    worker.process.close()


def _terminate_worker(worker: _WorkerHandle) -> None:
    if worker.process.is_alive():
        worker.process.terminate()
    worker.process.join()
    worker.connection.close()
    worker.process.close()
