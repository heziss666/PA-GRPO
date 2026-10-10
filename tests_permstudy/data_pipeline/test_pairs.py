"""Deterministic verified-response deduplication and pair selection."""

from dataclasses import replace

import pytest

from permstudy.data_pipeline.canonical import canonical_json_bytes, sha256_hex
from permstudy.data_pipeline.schema import (
    CandidatePlan,
    CandidateRecord,
    PairRecord,
    QuestionRecord,
    Source,
    Split,
    VerificationRecord,
    VerificationStatus,
)


GENERATION_RUN_ID = "1" * 64
OTHER_GENERATION_RUN_ID = "2" * 64
VERIFICATION_RUN_ID = "3" * 64
QUESTION_HASH = "4" * 64
SPLIT_HASH = "5" * 64
PROMPT_HASH = "6" * 64
MODEL_REVISION = "7" * 40
TOKENIZER_REVISION = "8" * 40
GENERATION_MANIFEST_HASH = "9" * 64
VERIFICATION_MANIFEST_HASH = "a" * 64
UPSTREAM_BINDINGS = {
    "generation": GENERATION_MANIFEST_HASH,
    "verification": VERIFICATION_MANIFEST_HASH,
}


def question(question_id="reclor:train:synthetic_1", question_hash=QUESTION_HASH):
    return QuestionRecord(
        schema_version="question_record_v1",
        source=Source.RECLOR,
        original_question_id=question_id,
        question_content_hash=question_hash,
        source_snapshot_id="reclor_snapshot_1",
        source_revision="b" * 64,
        source_row_id=question_id.rsplit(":", 1)[-1],
        context="Synthetic context.",
        question="Which option follows?",
        answers=("First", "Second", "Third", "Fourth"),
        gold_label="A",
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
    status=VerificationStatus.CORRECT,
    generation_run_id=GENERATION_RUN_ID,
    verification_run_id=VERIFICATION_RUN_ID,
    generator_id="qwen2.5-7b-instruct",
    sampling_index=0,
    finish_reason="stop",
    prediction_parse_status="parsed",
    canonical_prediction="A",
    question_record=None,
):
    question_record = question_record or question()
    plan = CandidatePlan(
        schema_version="candidate_plan_v1",
        generation_run_id=generation_run_id,
        candidate_id=candidate_id,
        original_question_id=question_record.original_question_id,
        question_content_hash=question_record.question_content_hash,
        source=question_record.source,
        split=Split.TRAIN,
        split_manifest_hash=SPLIT_HASH,
        generator_id=generator_id,
        sampling_index=sampling_index,
        shard_id="00000",
        prompt_hash=PROMPT_HASH,
    )
    generated = CandidateRecord(
        plan=plan,
        response=response,
        finish_reason=finish_reason,
        generated_token_count=8,
        model_revision=MODEL_REVISION,
    )
    verified = VerificationRecord(
        generation_run_id=generation_run_id,
        verification_run_id=verification_run_id,
        candidate_id=candidate_id,
        original_question_id=question_record.original_question_id,
        source=question_record.source,
        generator_id=generator_id,
        verification_status=status,
        prediction_parse_status=prediction_parse_status,
        canonical_prediction=canonical_prediction,
        verifier_name="reclor_exact_match",
        verifier_version="reclor_exact_match_v1",
        parser_version="reclor_terminal_final_v1",
        verifier_timeout_seconds=1.0,
        error_type=None,
    )
    return generated, verified


class RecordingTokenizer:
    def __init__(self, lengths):
        self.lengths = lengths
        self.calls = []

    def encode(self, text, *, add_special_tokens):
        self.calls.append((text, add_special_tokens))
        return list(range(self.lengths[text]))


def test_dedup_normalizes_nfc_and_line_endings_and_retains_provenance():
    from permstudy.data_pipeline.pairs import build_dedup_groups

    first = candidate(
        "candidate_b",
        " Cafe\u0301\r\nFinal Answer: A \r\n",
        generator_id="qwen2.5-7b-instruct",
    )
    second = candidate(
        "candidate_a",
        "Caf\u00e9\nFinal Answer: A",
        generator_id="llama-3.1-8b-instruct",
        sampling_index=1,
    )

    groups = build_dedup_groups((first, second))

    assert len(groups) == 1
    assert groups[0].response_hash == sha256_hex("Caf\u00e9\nFinal Answer: A".encode("utf-8"))
    assert groups[0].representative_key == (GENERATION_RUN_ID, "candidate_a")
    assert groups[0].member_keys == (
        (GENERATION_RUN_ID, "candidate_a"),
        (GENERATION_RUN_ID, "candidate_b"),
    )
    assert groups[0].member_generators == (
        "llama-3.1-8b-instruct",
        "qwen2.5-7b-instruct",
    )
    assert groups[0].verification_status is VerificationStatus.CORRECT


def test_dedup_excludes_ineligible_generation_and_verification_states():
    from permstudy.data_pipeline.pairs import build_dedup_groups

    eligible = candidate("eligible", "reason\nFinal Answer: A")
    records = (
        eligible,
        candidate("length", "reason\nFinal Answer: A", finish_reason="length", sampling_index=1),
        candidate("empty", "  \r\n", sampling_index=2),
        candidate(
            "missing_final",
            "The answer may be A.",
            prediction_parse_status="invalid",
            canonical_prediction=None,
            sampling_index=3,
        ),
        candidate(
            "ambiguous",
            "Final Answer: A\nFinal Answer: B",
            status=VerificationStatus.AMBIGUOUS,
            prediction_parse_status="ambiguous",
            canonical_prediction=None,
            sampling_index=4,
        ),
        candidate(
            "error",
            "Final Answer: A",
            status=VerificationStatus.ERROR,
            prediction_parse_status="error",
            canonical_prediction=None,
            sampling_index=5,
        ),
    )

    groups = build_dedup_groups(records)

    assert [group.member_keys for group in groups] == [((GENERATION_RUN_ID, "eligible"),)]


def test_dedup_fails_closed_when_same_response_hash_has_conflicting_statuses():
    from permstudy.data_pipeline.pairs import PairIntegrityError, build_dedup_groups

    records = (
        candidate("candidate_correct", "same\nFinal Answer: A"),
        candidate(
            "candidate_incorrect",
            "same\r\nFinal Answer: A",
            status=VerificationStatus.INCORRECT,
            canonical_prediction="B",
            sampling_index=1,
        ),
    )

    with pytest.raises(PairIntegrityError, match="conflicting verification statuses"):
        build_dedup_groups(records)


def test_dedup_never_merges_identical_responses_across_questions():
    from permstudy.data_pipeline.pairs import build_dedup_groups

    other_question = question("reclor:train:synthetic_2", "e" * 64)
    records = (
        candidate("candidate_q1", "same\nFinal Answer: A"),
        candidate(
            "candidate_q2",
            "same\nFinal Answer: A",
            question_record=other_question,
            sampling_index=1,
        ),
    )

    groups = build_dedup_groups(records)

    assert len(groups) == 2
    assert {group.representative_key for group in groups} == {
        (GENERATION_RUN_ID, "candidate_q1"),
        (GENERATION_RUN_ID, "candidate_q2"),
    }


def test_dedup_rejects_candidate_verification_composite_identity_mismatch():
    from permstudy.data_pipeline.pairs import PairIntegrityError, build_dedup_groups

    generated, verified = candidate("candidate_1", "Final Answer: A")
    mismatched = replace(verified, generation_run_id=OTHER_GENERATION_RUN_ID)

    with pytest.raises(PairIntegrityError, match="composite identity"):
        build_dedup_groups(((generated, mismatched),))


def test_pair_stage_config_role_binds_generation_and_verification_manifests():
    from permstudy.data_pipeline.ids import run_id
    from permstudy.data_pipeline.pairs import pair_stage_config

    baseline = pair_stage_config(TOKENIZER_REVISION, UPSTREAM_BINDINGS)
    changed_generation = pair_stage_config(
        TOKENIZER_REVISION,
        {**UPSTREAM_BINDINGS, "generation": "c" * 64},
    )
    changed_verification = pair_stage_config(
        TOKENIZER_REVISION,
        {**UPSTREAM_BINDINGS, "verification": "d" * 64},
    )

    assert baseline["upstream_bindings"] == {
        "generation": GENERATION_MANIFEST_HASH,
        "verification": VERIFICATION_MANIFEST_HASH,
    }
    assert run_id("pairs", baseline) != run_id("pairs", changed_generation)
    assert run_id("pairs", baseline) != run_id("pairs", changed_verification)


@pytest.mark.parametrize("revision", ["main", "A" * 40, "1" * 39, "g" * 40])
def test_pair_stage_config_requires_lowercase_immutable_tokenizer_revision(revision):
    from permstudy.data_pipeline.pairs import pair_stage_config

    with pytest.raises(ValueError, match="tokenizer revision"):
        pair_stage_config(revision, UPSTREAM_BINDINGS)


def test_selection_minimizes_token_gap_and_uses_canonical_response_contract():
    from permstudy.data_pipeline.pairs import TOKENIZER_REPOSITORY, select_pair

    records = (
        candidate("pos_a", " positive-short\r\nFinal Answer: A ", sampling_index=0),
        candidate("pos_b", "positive-long\nFinal Answer: A", sampling_index=1),
        candidate(
            "neg_a",
            "negative-mid\nFinal Answer: B",
            status=VerificationStatus.INCORRECT,
            canonical_prediction="B",
            sampling_index=2,
        ),
        candidate(
            "neg_b",
            "negative-long\nFinal Answer: B",
            status=VerificationStatus.INCORRECT,
            canonical_prediction="B",
            sampling_index=3,
        ),
    )
    lengths = {
        "positive-short\nFinal Answer: A": 2,
        "positive-long\nFinal Answer: A": 7,
        "negative-mid\nFinal Answer: B": 4,
        "negative-long\nFinal Answer: B": 8,
    }
    tokenizer = RecordingTokenizer(lengths)

    selected = select_pair(
        question(),
        tuple(reversed(records)),
        tokenizer,
        TOKENIZER_REVISION,
        upstream_bindings=UPSTREAM_BINDINGS,
    )

    assert selected is not None
    assert selected.positive_candidate_id == "pos_b"
    assert selected.negative_candidate_id == "neg_b"
    assert (selected.positive_token_count, selected.negative_token_count) == (7, 8)
    assert selected.tokenizer_repo == TOKENIZER_REPOSITORY
    assert selected.tokenizer_revision == TOKENIZER_REVISION
    assert sorted(tokenizer.calls) == sorted((text, False) for text in lengths)


def test_selection_breaks_equal_gap_ties_by_candidate_ids():
    from permstudy.data_pipeline.pairs import select_pair

    records = (
        candidate("pos_b", "positive-b\nFinal Answer: A", sampling_index=0),
        candidate("pos_a", "positive-a\nFinal Answer: A", sampling_index=1),
        candidate(
            "neg_b",
            "negative-b\nFinal Answer: B",
            status=VerificationStatus.INCORRECT,
            canonical_prediction="B",
            sampling_index=2,
        ),
        candidate(
            "neg_a",
            "negative-a\nFinal Answer: B",
            status=VerificationStatus.INCORRECT,
            canonical_prediction="B",
            sampling_index=3,
        ),
    )
    tokenizer = RecordingTokenizer({record[0].response: 5 for record in records})

    selected = select_pair(
        question(),
        records,
        tokenizer,
        TOKENIZER_REVISION,
        upstream_bindings=UPSTREAM_BINDINGS,
    )

    assert selected is not None
    assert (selected.positive_candidate_id, selected.negative_candidate_id) == ("pos_a", "neg_a")


def test_selection_requires_both_labels_and_fails_closed_on_question_mismatch():
    from permstudy.data_pipeline.pairs import PairIntegrityError, select_pair

    correct_only = (candidate("correct", "Final Answer: A"),)
    assert select_pair(
        question(),
        correct_only,
        RecordingTokenizer({"Final Answer: A": 1}),
        TOKENIZER_REVISION,
        upstream_bindings=UPSTREAM_BINDINGS,
    ) is None

    other_question = question("reclor:train:synthetic_2", "e" * 64)
    mixed = correct_only + (
        candidate(
            "wrong_question",
            "Final Answer: B",
            status=VerificationStatus.INCORRECT,
            canonical_prediction="B",
            question_record=other_question,
        ),
    )
    with pytest.raises(PairIntegrityError, match="question identity"):
        select_pair(
            question(),
            mixed,
            RecordingTokenizer({"Final Answer: A": 1, "Final Answer: B": 1}),
            TOKENIZER_REVISION,
            upstream_bindings=UPSTREAM_BINDINGS,
        )


def test_pair_identity_is_stable_within_run_and_changes_across_generation_runs():
    from permstudy.data_pipeline.pairs import select_pair

    def records(run_id):
        return (
            candidate("positive", "Final Answer: A", generation_run_id=run_id),
            candidate(
                "negative",
                "Final Answer: B",
                status=VerificationStatus.INCORRECT,
                canonical_prediction="B",
                generation_run_id=run_id,
                sampling_index=1,
            ),
        )

    tokenizer = RecordingTokenizer({"Final Answer: A": 1, "Final Answer: B": 1})
    first = select_pair(
        question(), records(GENERATION_RUN_ID), tokenizer, TOKENIZER_REVISION,
        upstream_bindings=UPSTREAM_BINDINGS,
    )
    repeated = select_pair(
        question(), tuple(reversed(records(GENERATION_RUN_ID))), tokenizer, TOKENIZER_REVISION,
        upstream_bindings=UPSTREAM_BINDINGS,
    )
    changed = select_pair(
        question(), records(OTHER_GENERATION_RUN_ID), tokenizer, TOKENIZER_REVISION,
        upstream_bindings={**UPSTREAM_BINDINGS, "generation": "f" * 64},
    )

    assert first is not None and repeated is not None and changed is not None
    assert first == repeated
    assert first.pair_id != changed.pair_id


def test_pair_manifest_uses_canonical_sorted_jsonl_and_literal_hash():
    from permstudy.data_pipeline.pairs import pair_manifest_bytes, pair_manifest_hash

    common = dict(
        schema_version="pair_v1",
        pair_run_id="pair_run_1",
        generation_run_id=GENERATION_RUN_ID,
        verification_run_id=VERIFICATION_RUN_ID,
        question_content_hash=QUESTION_HASH,
        source=Source.RECLOR,
        split=Split.TRAIN,
        split_manifest_hash=SPLIT_HASH,
        positive_candidate_id="positive",
        negative_candidate_id="negative",
        positive_response_hash="b" * 64,
        negative_response_hash="c" * 64,
        positive_generator_id="qwen2.5-7b-instruct",
        negative_generator_id="llama-3.1-8b-instruct",
        positive_token_count=3,
        negative_token_count=4,
        tokenizer_repo="Qwen/Qwen2.5-7B-Instruct",
        tokenizer_revision=TOKENIZER_REVISION,
    )
    later = PairRecord(
        **common,
        pair_id="e" * 64,
        original_question_id="reclor:train:synthetic_2",
    )
    earlier = PairRecord(
        **common,
        pair_id="d" * 64,
        original_question_id="reclor:train:synthetic_1",
    )

    rendered = pair_manifest_bytes((later, earlier))

    expected = canonical_json_bytes(earlier.to_dict()) + b"\n" + canonical_json_bytes(later.to_dict()) + b"\n"
    assert rendered == expected
    assert pair_manifest_hash((later, earlier)) == "549f7c7d99f3bace634260c528b04a3ae18e19af78d80efd788f17b5ccf3e9a3"
