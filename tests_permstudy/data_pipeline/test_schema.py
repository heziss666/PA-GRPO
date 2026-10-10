"""Synthetic contract fixtures; no source dataset or generated payloads."""

from dataclasses import FrozenInstanceError, replace
from importlib.metadata import version
import json

import pytest

from permstudy.data_pipeline import schema as s


PLAN = dict(schema_version="candidate_plan_v1", generation_run_id="gen_abc",
            candidate_id="cand_def", original_question_id="reclor:train:synthetic_1",
            question_content_hash="a" * 64, source="reclor", split="train",
            split_manifest_hash="b" * 64, generator_id="qwen_7b",
            sampling_index=0, shard_id="00000", prompt_hash="c" * 64)
QUESTION = dict(schema_version="question_v1", source="reclor",
                original_question_id="reclor:train:synthetic_1",
                question_content_hash="a" * 64, source_snapshot_id="snapshot_1",
                source_revision="revision_1", source_row_id="synthetic_1",
                context="Synthetic context", question="Synthetic question?",
                answers=["Synthetic A", "Synthetic B", "Synthetic C", "Synthetic D"],
                gold_label="A", problem=None, solution=None, category=None,
                level=None, synthetic=True)


def record_payloads():
    yield s.ArtifactRef, dict(relative_path="sources/synthetic/questions.jsonl", sha256="a" * 64, record_count=1)
    yield s.RunManifest, dict(schema_version="manifest_v1", stage="sources", run_id="run_1", config_hash="a" * 64,
                             upstream_manifest_hashes=["b" * 64], artifacts=[dict(relative_path="questions.jsonl", sha256="c" * 64, record_count=1)],
                             counts={"questions": 1}, created_at_utc="2026-10-06T00:00:00Z", output_manifest_hash="d" * 64)
    yield s.QuestionRecord, QUESTION
    yield s.SplitAssignment, dict(original_question_id="synthetic_1", question_content_hash="a" * 64, source="reclor", split="train", stratum="source")
    yield s.CandidatePlan, PLAN
    yield s.CandidateRecord, dict(plan=PLAN, response="Synthetic response", finish_reason="stop", generated_token_count=3, model_revision="rev_1")
    yield s.FailureRecord, dict(plan=PLAN, error_type="synthetic_error", retry_count=0)
    yield s.VerificationRecord, dict(generation_run_id="gen_abc", verification_run_id="verify_1", candidate_id="cand_def", original_question_id="synthetic_1",
                                     source="reclor", generator_id="qwen_7b", verification_status="correct", prediction_parse_status="ok", canonical_prediction="A",
                                     verifier_name="reclor", verifier_version="1", parser_version="1", verifier_timeout_seconds=1.0, error_type=None)
    yield s.QuestionVerificationRecord, dict(verification_run_id="verify_1", original_question_id="synthetic_1", source="reclor", gold_parse_status="ok", canonical_gold="A", error_type=None)
    yield s.PairRecord, dict(schema_version="pair_v1", pair_run_id="pair_run_1", pair_id="pair_1", generation_run_id="gen_abc", verification_run_id="verify_1",
                            original_question_id="synthetic_1", question_content_hash="a" * 64, source="reclor", split="train", split_manifest_hash="b" * 64,
                            positive_candidate_id="cand_def", negative_candidate_id="cand_neg", positive_response_hash="c" * 64, negative_response_hash="d" * 64,
                            positive_generator_id="qwen_7b", negative_generator_id="qwen_7b", positive_token_count=3, negative_token_count=4,
                            tokenizer_repo="synthetic/tokenizer", tokenizer_revision="revision_1")
    yield s.PermutationRecord, dict(permutation_run_id="perm_1", pair_id="pair_1", generation_run_id="gen_abc", original_question_id="synthetic_1", split="train",
                                   split_manifest_hash="b" * 64, permutation_id=0, permutation_label="AB", surface_a_candidate_id="cand_def", surface_b_candidate_id="cand_neg", correct_surface="A")
    yield s.AuditSelectionRecord, dict(audit_run_id="audit_1", generation_run_id="gen_abc", verification_run_id="verify_1", record_kind="candidate", original_question_id="synthetic_1",
                                      candidate_id="cand_def", source="reclor", generator_id="qwen_7b", verification_status="correct", gold_parse_status=None, reason_code="other")
    yield s.AuditDecision, dict(audit_run_id="audit_1", generation_run_id="gen_abc", verification_run_id="verify_1", record_kind="candidate", original_question_id="synthetic_1",
                               candidate_id="cand_def", verdict="AGREE", reason_code="other", confirmed_disagree=False)


@pytest.mark.parametrize("cls,payload", list(record_payloads()))
def test_records_round_trip_json_and_are_frozen(cls, payload):
    record = cls.from_dict(payload)
    record.validate()
    assert record.to_dict() == payload
    assert cls.from_dict(json.loads(json.dumps(record.to_dict()))) == record
    with pytest.raises(FrozenInstanceError):
        setattr(record, next(iter(payload)), "changed")


def test_candidate_plan_is_frozen_and_uses_composite_key():
    plan = s.CandidatePlan.from_dict(PLAN)
    assert plan.key == ("gen_abc", "cand_def")
    with pytest.raises(FrozenInstanceError):
        plan.candidate_id = "changed"


def test_math_verify_is_exactly_pinned():
    assert version("math-verify") == "0.9.0"


@pytest.mark.parametrize("split", ["train", "internal_holdout"])
def test_both_internal_partitions_validate_and_round_trip(split):
    record = s.CandidatePlan.from_dict(dict(PLAN, split=split))
    record.validate()
    assert record.to_dict()["split"] == split
    if split == "internal_holdout":
        assert record.split is s.Split.INTERNAL_HOLDOUT


@pytest.mark.parametrize("field,value", [("candidate_id", ""), ("generation_run_id", " "), ("question_content_hash", "g" * 64),
                                        ("prompt_hash", "a" * 63), ("sampling_index", -1), ("sampling_index", True), ("source", "unknown"), ("split", "test")])
def test_candidate_plan_rejects_invalid_identity_and_lineage(field, value):
    with pytest.raises(ValueError):
        replace(s.CandidatePlan.from_dict(PLAN), **{field: value}).validate()


@pytest.mark.parametrize("changes", [dict(answers=["A", "B", "C"]), dict(gold_label="E"), dict(problem="Synthetic problem"), dict(context=None)])
def test_reclor_requires_exact_source_shape(changes):
    with pytest.raises(ValueError):
        replace(s.QuestionRecord.from_dict(QUESTION), **changes).validate()


def test_math_requires_its_fields_and_excludes_reclor_fields():
    payload = dict(QUESTION, source="math", context=None, question=None, answers=[], gold_label=None,
                   problem="Synthetic problem", solution="Synthetic solution", category="Algebra", level="Level 1")
    record = s.QuestionRecord.from_dict(payload)
    record.validate()
    for changes in (dict(solution=None), dict(category=""), dict(answers=("A",)), dict(gold_label="A")):
        with pytest.raises(ValueError):
            replace(record, **changes).validate()


@pytest.mark.parametrize("path", ["/absolute.json", "C:/absolute.json", "../escape.json", "a/../b.json", "a\\b.json", "", "a//b.json", "./a.json"])
def test_artifact_paths_must_be_relative_posix(path):
    with pytest.raises(ValueError):
        s.ArtifactRef(path, "a" * 64, 1).validate()


def test_manifest_and_question_collections_are_immutable_snapshots():
    payload = next(payload for cls, payload in record_payloads() if cls is s.RunManifest)
    manifest = s.RunManifest.from_dict(payload)
    assert isinstance(manifest.artifacts, tuple)
    assert isinstance(manifest.artifacts[0], s.ArtifactRef)
    payload["artifacts"].clear()
    assert len(manifest.artifacts) == 1
    with pytest.raises(TypeError):
        manifest.counts["questions"] = 2
    question = s.QuestionRecord.from_dict(QUESTION)
    assert isinstance(question.answers, tuple)


def test_gold_error_is_one_question_level_state():
    record = s.QuestionVerificationRecord("verify_abc", "math:train:" + "a" * 40, s.Source.MATH,
                                         "gold_verification_error", None, "math_parse_failure")
    record.validate()
    for status in ("error", "correct", ""):
        with pytest.raises(ValueError):
            replace(record, gold_parse_status=status).validate()
    with pytest.raises(ValueError):
        replace(record, canonical_gold="1").validate()


def test_candidate_verification_cannot_carry_gold_status():
    payload = next(payload for cls, payload in record_payloads() if cls is s.VerificationRecord)
    with pytest.raises(ValueError):
        s.VerificationRecord.from_dict(dict(payload, gold_parse_status="gold_verification_error"))


def test_question_gold_audit_has_no_candidate_identity():
    payload = next(payload for cls, payload in record_payloads() if cls is s.AuditSelectionRecord)
    record = s.AuditSelectionRecord.from_dict(dict(payload, record_kind="question_gold", candidate_id=None, generator_id=None,
                                                  verification_status=None, gold_parse_status="gold_verification_error", reason_code="gold_parse_failure"))
    record.validate()
    for changes in (dict(candidate_id="cand_def"), dict(generator_id="qwen_7b"), dict(verification_status=s.VerificationStatus.ERROR)):
        with pytest.raises(ValueError):
            replace(record, **changes).validate()


def test_audit_confirmation_requires_disagreement():
    payload = next(payload for cls, payload in record_payloads() if cls is s.AuditDecision)
    with pytest.raises(ValueError):
        replace(s.AuditDecision.from_dict(payload), confirmed_disagree=True).validate()


@pytest.mark.parametrize("changes", [dict(permutation_id=2), dict(permutation_label="BA"), dict(correct_surface="B"), dict(surface_b_candidate_id="cand_def")])
def test_permutations_reject_wrong_mapping_or_repeated_candidates(changes):
    payload = next(payload for cls, payload in record_payloads() if cls is s.PermutationRecord)
    with pytest.raises(ValueError):
        replace(s.PermutationRecord.from_dict(payload), **changes).validate()


def test_ba_permutation_preserves_the_positive_b_surface():
    payload = next(payload for cls, payload in record_payloads() if cls is s.PermutationRecord)
    record = s.PermutationRecord.from_dict(dict(payload, permutation_id=1, permutation_label="BA", correct_surface="B",
                                               surface_a_candidate_id="cand_neg", surface_b_candidate_id="cand_def"))
    assert record.to_dict()["surface_b_candidate_id"] == "cand_def"


@pytest.mark.parametrize("cls,payload", list(record_payloads()))
def test_records_reject_unknown_fields(cls, payload):
    with pytest.raises(ValueError):
        cls.from_dict(dict(payload, unrelated="unexpected"))
