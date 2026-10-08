from dataclasses import replace
import multiprocessing

import pytest

from permstudy.data_pipeline.schema import (
    CandidatePlan,
    CandidateRecord,
    QuestionRecord,
    Source,
    Split,
    VerificationStatus,
)


GENERATION_RUN_ID = "a" * 64
VERIFICATION_RUN_ID = "b" * 64
QUESTION_HASH = "c" * 64
SPLIT_HASH = "d" * 64
PROMPT_HASH = "e" * 64
MODEL_REVISION = "f" * 40


def math_question(*, solution=r"Therefore the result is \boxed{\frac{1}{2}}"):
    return QuestionRecord(
        schema_version="question_record_v1",
        source=Source.MATH,
        original_question_id="math:train:" + "1" * 40,
        question_content_hash=QUESTION_HASH,
        source_snapshot_id="math_snapshot_1",
        source_revision="2" * 40,
        source_row_id="algebra:synthetic_1",
        context=None,
        question=None,
        answers=(),
        gold_label=None,
        problem="Compute the synthetic result.",
        solution=solution,
        category="algebra",
        level="Level 1",
        synthetic=True,
    )


def candidate(candidate_id, response, *, sampling_index=0, generation_run_id=GENERATION_RUN_ID):
    plan = CandidatePlan(
        schema_version="candidate_plan_v1",
        generation_run_id=generation_run_id,
        candidate_id=candidate_id,
        original_question_id="math:train:" + "1" * 40,
        question_content_hash=QUESTION_HASH,
        source=Source.MATH,
        split=Split.TRAIN,
        split_manifest_hash=SPLIT_HASH,
        generator_id="qwen2.5-7b-instruct",
        sampling_index=sampling_index,
        shard_id="00000",
        prompt_hash=PROMPT_HASH,
    )
    return CandidateRecord(
        plan=plan,
        response=response,
        finish_reason="stop",
        generated_token_count=8,
        model_revision=MODEL_REVISION,
    )


def counting_gold_parser(boxed_text, counter):
    with counter.get_lock():
        counter.value += 1
    return boxed_text


def exact_text_verifier(gold, boxed_text, _state):
    return gold == boxed_text


def blocking_gold_parser(boxed_text, state):
    started, release = state
    started.set()
    release.wait(30)
    return boxed_text


def fail_if_candidate_called(_gold, _boxed_text, _state):
    raise AssertionError("candidate hook must not run before GOLD_READY")


def counting_gold_with_blocking_candidate(boxed_text, state):
    counter, _release = state
    with counter.get_lock():
        counter.value += 1
    return boxed_text


def block_one_candidate_then_compare(gold, boxed_text, state):
    _counter, release = state
    if boxed_text == r"\boxed{999}":
        release.wait(30)
    return gold == boxed_text


def raising_gold_parser(_boxed_text, _state):
    raise RuntimeError("private gold payload")


@pytest.mark.parametrize(
    ("text", "status", "boxed_text", "error_type"),
    [
        (
            "work\n" + r"\boxed{\frac{1}{2}}",
            "parsed",
            r"\boxed{\frac{1}{2}}",
            None,
        ),
        (r"\boxed{2}" + " \r\n\t", "parsed", r"\boxed{2}", None),
        ("no boxed answer", "invalid", None, "missing_final_marker"),
        (r"\boxed{\frac{1}{2}", "invalid", None, "math_parse_failure"),
        (r"\boxed{1}\n\boxed{1}", "invalid", None, "duplicate_same_marker"),
        (r"\boxed{1}\n\boxed{2}", "ambiguous", None, "conflicting_markers"),
        (r"\boxed{1} trailing", "invalid", None, "prompt_contract_mismatch"),
    ],
)
def test_strict_boxed_extraction(text, status, boxed_text, error_type):
    from permstudy.data_pipeline.verification import extract_math_candidate

    extracted = extract_math_candidate(text)

    assert (extracted.status, extracted.boxed_text, extracted.error_type) == (
        status,
        boxed_text,
        error_type,
    )


def test_math_verify_explicitly_establishes_equivalent_and_non_equivalent_candidates():
    from permstudy.data_pipeline.verification import verify_math_question

    records = [
        candidate("equivalent", r"Reasoning\n\boxed{0.5}"),
        candidate("not_equivalent", r"Reasoning\n\boxed{2}", sampling_index=1),
        candidate("invalid", "No final box", sampling_index=2),
        candidate("unparseable", r"Reasoning\n\boxed{???}", sampling_index=3),
    ]

    gold, verified = verify_math_question(
        math_question(),
        records,
        gold_timeout_seconds=5.0,
        candidate_timeout_seconds=5.0,
        verification_run_id=VERIFICATION_RUN_ID,
    )

    assert gold.gold_parse_status == "ok"
    assert gold.canonical_gold == r"\boxed{\frac{1}{2}}"
    assert [record.verification_status for record in verified] == [
        VerificationStatus.CORRECT,
        VerificationStatus.INCORRECT,
        VerificationStatus.INVALID,
        VerificationStatus.AMBIGUOUS,
    ]
    assert [record.prediction_parse_status for record in verified] == [
        "parsed",
        "parsed",
        "invalid",
        "ambiguous",
    ]
    assert [record.canonical_prediction for record in verified] == [
        r"\boxed{0.5}",
        r"\boxed{2}",
        None,
        r"\boxed{???}",
    ]
    assert [record.error_type for record in verified] == [
        None,
        None,
        "missing_final_marker",
        "math_parse_failure",
    ]
    assert [(record.generation_run_id, record.candidate_id) for record in verified] == [
        item.plan.key for item in records
    ]
    assert all(record.verifier_name == "math-verify" for record in verified)
    assert all(record.verifier_version == "0.9.0" for record in verified)
    assert all(record.parser_version == "math_boxed_final_v1" for record in verified)
    assert all(record.verifier_timeout_seconds == 5.0 for record in verified)


@pytest.mark.parametrize(
    "solution",
    [
        "No official boxed result.",
        r"Malformed \boxed{\frac{1}{2}",
        r"\boxed{1}\n\boxed{2}",
        r"\boxed{???}",
    ],
    ids=["missing", "malformed", "conflicting", "unparseable"],
)
def test_gold_extraction_failure_emits_one_question_error_and_no_candidate_records(solution):
    from permstudy.data_pipeline.verification import verify_math_question

    gold, verified = verify_math_question(
        math_question(solution=solution),
        [candidate("candidate_1", r"\boxed{1}")],
        gold_timeout_seconds=1.0,
        candidate_timeout_seconds=1.0,
        verification_run_id=VERIFICATION_RUN_ID,
    )

    assert gold.gold_parse_status == "gold_verification_error"
    assert gold.canonical_gold is None
    assert gold.error_type == "gold_parse_failure"
    assert verified == []


def test_gold_worker_exception_is_one_gold_parse_failure_with_no_candidates():
    from permstudy.data_pipeline.verification import verify_math_question

    gold, verified = verify_math_question(
        math_question(),
        [candidate("candidate_1", r"\boxed{1}")],
        gold_timeout_seconds=3.0,
        candidate_timeout_seconds=3.0,
        verification_run_id=VERIFICATION_RUN_ID,
        gold_parser_hook=raising_gold_parser,
        candidate_verifier_hook=fail_if_candidate_called,
    )

    assert gold.gold_parse_status == "gold_verification_error"
    assert gold.error_type == "gold_parse_failure"
    assert verified == []


def test_normal_six_candidate_question_parses_gold_once_under_spawn():
    from permstudy.data_pipeline.verification import verify_math_question

    context = multiprocessing.get_context("spawn")
    gold_parse_count = context.Value("i", 0)
    records = [
        candidate(f"candidate_{index}", r"\boxed{\frac{1}{2}}", sampling_index=index)
        for index in range(6)
    ]

    gold, verified = verify_math_question(
        math_question(),
        records,
        gold_timeout_seconds=3.0,
        candidate_timeout_seconds=3.0,
        verification_run_id=VERIFICATION_RUN_ID,
        gold_parser_hook=counting_gold_parser,
        candidate_verifier_hook=exact_text_verifier,
        worker_state=gold_parse_count,
    )

    assert gold.gold_parse_status == "ok"
    assert len(verified) == 6
    assert all(record.verification_status is VerificationStatus.CORRECT for record in verified)
    assert gold_parse_count.value == 1


def test_gold_timeout_terminates_worker_and_emits_no_candidate_records():
    from permstudy.data_pipeline.verification import verify_math_question

    context = multiprocessing.get_context("spawn")
    started = context.Event()
    release = context.Event()

    gold, verified = verify_math_question(
        math_question(),
        [candidate("candidate_1", r"\boxed{\frac{1}{2}}")],
        gold_timeout_seconds=2.0,
        candidate_timeout_seconds=1.0,
        verification_run_id=VERIFICATION_RUN_ID,
        gold_parser_hook=blocking_gold_parser,
        candidate_verifier_hook=fail_if_candidate_called,
        worker_state=(started, release),
    )

    assert started.is_set()
    assert gold.gold_parse_status == "gold_verification_error"
    assert gold.error_type == "timeout"
    assert verified == []
    assert multiprocessing.active_children() == []


def test_candidate_timeout_restarts_worker_and_repeats_gold_handshake():
    from permstudy.data_pipeline.verification import verify_math_question

    context = multiprocessing.get_context("spawn")
    gold_parse_count = context.Value("i", 0)
    release = context.Event()
    records = [
        candidate("timed_out", r"\boxed{999}"),
        candidate("after_restart", r"\boxed{\frac{1}{2}}", sampling_index=1),
    ]

    gold, verified = verify_math_question(
        math_question(),
        records,
        gold_timeout_seconds=3.0,
        candidate_timeout_seconds=0.1,
        verification_run_id=VERIFICATION_RUN_ID,
        gold_parser_hook=counting_gold_with_blocking_candidate,
        candidate_verifier_hook=block_one_candidate_then_compare,
        worker_state=(gold_parse_count, release),
    )

    assert gold.gold_parse_status == "ok"
    assert [record.verification_status for record in verified] == [
        VerificationStatus.ERROR,
        VerificationStatus.CORRECT,
    ]
    assert [record.error_type for record in verified] == ["timeout", None]
    assert gold_parse_count.value == 2
    assert multiprocessing.active_children() == []


def raising_candidate_verifier(_gold, _boxed_text, _state):
    raise RuntimeError("private candidate payload")


def test_picklable_worker_candidate_exception_becomes_sanitized_error():
    from permstudy.data_pipeline.verification import verify_math_question

    context = multiprocessing.get_context("spawn")
    counter = context.Value("i", 0)
    _, verified = verify_math_question(
        math_question(),
        [candidate("candidate_1", r"\boxed{1}")],
        gold_timeout_seconds=3.0,
        candidate_timeout_seconds=3.0,
        verification_run_id=VERIFICATION_RUN_ID,
        gold_parser_hook=counting_gold_parser,
        candidate_verifier_hook=raising_candidate_verifier,
        worker_state=counter,
    )

    assert verified[0].verification_status is VerificationStatus.ERROR
    assert verified[0].prediction_parse_status == "error"
    assert verified[0].error_type == "unexpected_error"


def test_math_verify_version_mismatch_fails_before_worker_start(monkeypatch):
    import permstudy.data_pipeline.verification.math as math_verification
    from permstudy.data_pipeline.verification import DependencyContractError, verify_math_question

    monkeypatch.setattr(math_verification, "distribution_version", lambda _name: "0.8.0")

    with pytest.raises(DependencyContractError, match="0.9.0"):
        verify_math_question(
            math_question(),
            [candidate("candidate_1", r"\boxed{1}")],
            gold_timeout_seconds=1.0,
            candidate_timeout_seconds=1.0,
            verification_run_id=VERIFICATION_RUN_ID,
        )


def test_math_verification_rejects_candidate_question_identity_mismatch():
    from permstudy.data_pipeline.verification import MathVerificationError, verify_math_question

    record = candidate("candidate_1", r"\boxed{1}")
    mismatched = replace(record, plan=replace(record.plan, question_content_hash="9" * 64))

    with pytest.raises(MathVerificationError, match="question identity"):
        verify_math_question(
            math_question(),
            [mismatched],
            gold_timeout_seconds=1.0,
            candidate_timeout_seconds=1.0,
            verification_run_id=VERIFICATION_RUN_ID,
        )
