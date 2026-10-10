"""Official-label ReClor verification with a strict terminal answer parser."""

from collections.abc import Iterable
from dataclasses import dataclass
import math
import re

from ..schema import (
    CandidateRecord,
    QuestionRecord,
    QuestionVerificationRecord,
    Source,
    VerificationRecord,
    VerificationStatus,
)


RECLOR_PARSER_VERSION = "reclor_final_answer_v1"
RECLOR_VERIFIER_NAME = "reclor_exact_match"
RECLOR_VERIFIER_VERSION = "reclor_exact_match_v1"

_MARKER = "Final Answer:"
_VALID_AFTER_MARKER = re.compile(r"Final Answer:[ \t]*([A-D])(?=\s|\Z)")
_VALID_TERMINAL = re.compile(r"Final Answer:[ \t]*([A-D])\s*\Z")
_PARSE_STATUSES = frozenset({"parsed", "invalid", "ambiguous"})


class ReClorVerificationError(ValueError):
    """The question, candidate identities, or verifier inputs are inconsistent."""


@dataclass(frozen=True)
class ParsedAnswer:
    """The strict parser outcome for one candidate response."""

    status: str
    answer: str | None
    error_type: str | None

    def __post_init__(self) -> None:
        if self.status not in _PARSE_STATUSES:
            raise ValueError("unknown parsed-answer status")
        if self.status == "parsed":
            if self.answer not in ("A", "B", "C", "D") or self.error_type is not None:
                raise ValueError("parsed answer must contain one canonical option")
        elif self.answer is not None or not isinstance(self.error_type, str) or not self.error_type:
            raise ValueError("unparsed answer must contain one error type")


def parse_reclor_candidate(response: str) -> ParsedAnswer:
    """Parse exactly one case-sensitive terminal ``Final Answer: X`` marker."""
    if not isinstance(response, str):
        raise TypeError("response must be text")
    marker_count = response.count(_MARKER)
    if marker_count == 0:
        return ParsedAnswer("invalid", None, "missing_final_marker")

    marked_answers = _VALID_AFTER_MARKER.findall(response)
    if marker_count > 1:
        if len(marked_answers) != marker_count:
            return ParsedAnswer("invalid", None, "prompt_contract_mismatch")
        if len(set(marked_answers)) > 1:
            return ParsedAnswer("ambiguous", None, "conflicting_markers")
        return ParsedAnswer("invalid", None, "duplicate_same_marker")

    terminal = _VALID_TERMINAL.search(response)
    if terminal is None:
        return ParsedAnswer("invalid", None, "prompt_contract_mismatch")
    return ParsedAnswer("parsed", terminal.group(1), None)


def verify_reclor_question(
    question: QuestionRecord,
    candidates: Iterable[CandidateRecord],
    *,
    verification_run_id: str,
    verifier_timeout_seconds: float,
) -> tuple[QuestionVerificationRecord, list[VerificationRecord]]:
    """Verify ReClor candidates against only the question's official gold label."""
    _validate_call(question, verification_run_id, verifier_timeout_seconds)
    materialized = list(candidates)
    seen: set[tuple[str, str]] = set()
    verified: list[VerificationRecord] = []

    for candidate in materialized:
        if not isinstance(candidate, CandidateRecord):
            raise TypeError("candidates must contain CandidateRecord values")
        candidate.validate()
        plan = candidate.plan
        if (
            plan.original_question_id != question.original_question_id
            or plan.question_content_hash != question.question_content_hash
            or plan.source is not Source.RECLOR
        ):
            raise ReClorVerificationError("candidate question identity does not match the ReClor question")
        if plan.key in seen:
            raise ReClorVerificationError("duplicate composite candidate key")
        seen.add(plan.key)

        parsed = parse_reclor_candidate(candidate.response)
        if parsed.status == "parsed":
            status = (
                VerificationStatus.CORRECT
                if parsed.answer == question.gold_label
                else VerificationStatus.INCORRECT
            )
        elif parsed.status == "ambiguous":
            status = VerificationStatus.AMBIGUOUS
        else:
            status = VerificationStatus.INVALID
        record = VerificationRecord(
            generation_run_id=plan.generation_run_id,
            verification_run_id=verification_run_id,
            candidate_id=plan.candidate_id,
            original_question_id=question.original_question_id,
            source=Source.RECLOR,
            generator_id=plan.generator_id,
            verification_status=status,
            prediction_parse_status=parsed.status,
            canonical_prediction=parsed.answer,
            verifier_name=RECLOR_VERIFIER_NAME,
            verifier_version=RECLOR_VERIFIER_VERSION,
            parser_version=RECLOR_PARSER_VERSION,
            verifier_timeout_seconds=verifier_timeout_seconds,
            error_type=parsed.error_type,
        )
        record.validate()
        verified.append(record)

    gold = QuestionVerificationRecord(
        verification_run_id=verification_run_id,
        original_question_id=question.original_question_id,
        source=Source.RECLOR,
        gold_parse_status="ok",
        canonical_gold=question.gold_label,
        error_type=None,
    )
    gold.validate()
    return gold, verified


def _validate_call(
    question: QuestionRecord,
    verification_run_id: str,
    verifier_timeout_seconds: float,
) -> None:
    if not isinstance(question, QuestionRecord):
        raise TypeError("question must be a QuestionRecord")
    question.validate()
    if question.source is not Source.RECLOR:
        raise ReClorVerificationError("question must use the ReClor source shape")
    if not isinstance(verification_run_id, str) or not verification_run_id.strip():
        raise ReClorVerificationError("verification_run_id must be nonempty text")
    if (
        type(verifier_timeout_seconds) not in (int, float)
        or not math.isfinite(verifier_timeout_seconds)
        or verifier_timeout_seconds <= 0
    ):
        raise ReClorVerificationError("verifier_timeout_seconds must be positive and finite")
