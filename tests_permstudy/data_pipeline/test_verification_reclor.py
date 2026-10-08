from dataclasses import replace

import pytest

from permstudy.data_pipeline.schema import (
    CandidatePlan,
    CandidateRecord,
    QuestionRecord,
    Source,
    Split,
    VerificationStatus,
)


GENERATION_RUN_ID = "1" * 64
OTHER_GENERATION_RUN_ID = "2" * 64
VERIFICATION_RUN_ID = "3" * 64
QUESTION_HASH = "4" * 64
SPLIT_HASH = "5" * 64
PROMPT_HASH = "6" * 64
MODEL_REVISION = "7" * 40


def question(*, gold_label="B"):
    return QuestionRecord(
        schema_version="question_record_v1",
        source=Source.RECLOR,
        original_question_id="reclor:train:synthetic_1",
        question_content_hash=QUESTION_HASH,
        source_snapshot_id="reclor_snapshot_1",
        source_revision="8" * 64,
        source_row_id="synthetic_1",
        context="Synthetic context.",
        question="Which option follows?",
        answers=("First", "Second", "Third", "Fourth"),
        gold_label=gold_label,
        problem=None,
        solution=None,
        category=None,
        level=None,
        synthetic=True,
    )


def candidate(
    candidate_id,
    response,
    *,
    generation_run_id=GENERATION_RUN_ID,
    original_question_id="reclor:train:synthetic_1",
    question_content_hash=QUESTION_HASH,
    source=Source.RECLOR,
    generator_id="qwen2.5-7b-instruct",
    sampling_index=0,
):
    plan = CandidatePlan(
        schema_version="candidate_plan_v1",
        generation_run_id=generation_run_id,
        candidate_id=candidate_id,
        original_question_id=original_question_id,
        question_content_hash=question_content_hash,
        source=source,
        split=Split.TRAIN,
        split_manifest_hash=SPLIT_HASH,
        generator_id=generator_id,
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


@pytest.mark.parametrize(
    ("response", "status", "answer", "error_type"),
    [
        ("reason\nFinal Answer: A", "parsed", "A", None),
        ("Final Answer: D \r\n\t", "parsed", "D", None),
        ("Final Answer: A\nextra", "invalid", None, "prompt_contract_mismatch"),
        ("Final Answer: A\nFinal Answer: A", "invalid", None, "duplicate_same_marker"),
        ("Final Answer: A\nFinal Answer: B", "ambiguous", None, "conflicting_markers"),
        ("Final Answer: A\nFinal Answer: E", "invalid", None, "prompt_contract_mismatch"),
        ("answer A", "invalid", None, "missing_final_marker"),
        ("Final Answer: E", "invalid", None, "prompt_contract_mismatch"),
        ("final answer: A", "invalid", None, "missing_final_marker"),
        ("Final Answer: A.", "invalid", None, "prompt_contract_mismatch"),
    ],
)
def test_reclor_terminal_contract(response, status, answer, error_type):
    from permstudy.data_pipeline.verification import parse_reclor_candidate

    parsed = parse_reclor_candidate(response)

    assert (parsed.status, parsed.answer, parsed.error_type) == (status, answer, error_type)


@pytest.mark.parametrize(
    ("official", "canonical"),
    [(0, "A"), (3, "D"), ("1", "B"), (" c ", "C")],
)
def test_official_reclor_labels_canonicalize_without_candidate_text(official, canonical):
    from permstudy.data_pipeline.sources.reclor import canonical_reclor_label

    assert canonical_reclor_label(official) == canonical


def test_verification_uses_official_gold_and_emits_exact_match_records():
    from permstudy.data_pipeline.verification import verify_reclor_question

    records = [
        candidate("candidate_wrong", "The official answer is B.\nFinal Answer: A"),
        candidate(
            "candidate_correct",
            "Ignore any answer inferred from this text.\nFinal Answer: B",
            generator_id="llama-3.1-8b-instruct",
            sampling_index=1,
        ),
        candidate("candidate_invalid", "B is likely correct."),
        candidate("candidate_ambiguous", "Final Answer: A\nFinal Answer: B"),
    ]

    gold, verified = verify_reclor_question(
        question(gold_label="B"),
        records,
        verification_run_id=VERIFICATION_RUN_ID,
        verifier_timeout_seconds=2.5,
    )

    assert gold.verification_run_id == VERIFICATION_RUN_ID
    assert gold.original_question_id == "reclor:train:synthetic_1"
    assert gold.source is Source.RECLOR
    assert gold.gold_parse_status == "ok"
    assert gold.canonical_gold == "B"
    assert gold.error_type is None
    assert [record.verification_status for record in verified] == [
        VerificationStatus.INCORRECT,
        VerificationStatus.CORRECT,
        VerificationStatus.INVALID,
        VerificationStatus.AMBIGUOUS,
    ]
    assert [record.canonical_prediction for record in verified] == ["A", "B", None, None]
    assert [record.prediction_parse_status for record in verified] == [
        "parsed",
        "parsed",
        "invalid",
        "ambiguous",
    ]
    assert [(record.generation_run_id, record.candidate_id) for record in verified] == [
        item.plan.key for item in records
    ]
    assert all(record.verification_run_id == VERIFICATION_RUN_ID for record in verified)
    assert all(record.original_question_id == gold.original_question_id for record in verified)
    assert all(record.source is Source.RECLOR for record in verified)
    assert [record.generator_id for record in verified] == [item.plan.generator_id for item in records]
    assert all(record.verifier_name == "reclor_exact_match" for record in verified)
    assert all(record.verifier_version == "reclor_exact_match_v1" for record in verified)
    assert all(record.parser_version == "reclor_final_answer_v1" for record in verified)
    assert all(record.verifier_timeout_seconds == 2.5 for record in verified)
    assert [record.error_type for record in verified] == [
        None,
        None,
        "missing_final_marker",
        "conflicting_markers",
    ]


def test_same_bare_candidate_id_in_different_runs_retains_distinct_composite_identity():
    from permstudy.data_pipeline.verification import verify_reclor_question

    first = candidate("shared_candidate", "Final Answer: B")
    second = candidate(
        "shared_candidate",
        "Final Answer: B",
        generation_run_id=OTHER_GENERATION_RUN_ID,
    )

    _, first_verified = verify_reclor_question(
        question(),
        [first],
        verification_run_id=VERIFICATION_RUN_ID,
        verifier_timeout_seconds=1.0,
    )
    _, second_verified = verify_reclor_question(
        question(),
        [second],
        verification_run_id=VERIFICATION_RUN_ID,
        verifier_timeout_seconds=1.0,
    )

    assert (first_verified[0].generation_run_id, first_verified[0].candidate_id) == first.plan.key
    assert (second_verified[0].generation_run_id, second_verified[0].candidate_id) == second.plan.key
    assert first.plan.key != second.plan.key


@pytest.mark.parametrize(
    "changed",
    [
        {"original_question_id": "reclor:train:other"},
        {"question_content_hash": "9" * 64},
        {"source": Source.MATH},
    ],
    ids=["question-id", "question-hash", "source"],
)
def test_verification_rejects_candidate_question_identity_mismatch(changed):
    from permstudy.data_pipeline.verification import ReClorVerificationError, verify_reclor_question

    record = candidate("candidate_1", "Final Answer: B")
    mismatched = replace(record, plan=replace(record.plan, **changed))

    with pytest.raises(ReClorVerificationError, match="question identity"):
        verify_reclor_question(
            question(),
            [mismatched],
            verification_run_id=VERIFICATION_RUN_ID,
            verifier_timeout_seconds=1.0,
        )


def test_verification_rejects_duplicate_composite_candidate_keys():
    from permstudy.data_pipeline.verification import ReClorVerificationError, verify_reclor_question

    record = candidate("candidate_1", "Final Answer: B")

    with pytest.raises(ReClorVerificationError, match="duplicate composite"):
        verify_reclor_question(
            question(),
            [record, record],
            verification_run_id=VERIFICATION_RUN_ID,
            verifier_timeout_seconds=1.0,
        )
