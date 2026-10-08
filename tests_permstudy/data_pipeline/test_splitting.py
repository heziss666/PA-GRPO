"""Deterministic question-level split and smoke-selection behavior."""

from collections import Counter
from dataclasses import replace
import hashlib
import importlib
import json
from types import MappingProxyType

import pytest

from permstudy.data_pipeline.schema import (
    ArtifactRef,
    CandidatePlan,
    PairRecord,
    PermutationRecord,
    QuestionRecord,
    Source,
    Split,
)
from permstudy.data_pipeline.io import append_record, scan_jsonl, write_atomic_manifest
from permstudy.data_pipeline.ids import run_id
from permstudy.data_pipeline.sources import SourceSnapshot


EXPECTED_BASELINE_SPLIT_HASH = "08aaa028fcb859df62a792be1b79c4c74bbbdb7a17a883202719f16ec2885bf1"


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def splitting_module():
    try:
        return importlib.import_module("permstudy.data_pipeline.splitting")
    except ModuleNotFoundError:
        pytest.fail("permstudy.data_pipeline.splitting is not implemented")


def reclor_question(index: int, label: str) -> QuestionRecord:
    qid = f"reclor:train:r{index:03d}"
    record = QuestionRecord(
        schema_version="question_v1",
        source=Source.RECLOR,
        original_question_id=qid,
        question_content_hash=digest(qid),
        source_snapshot_id="reclor_snapshot",
        source_revision="reclor_revision",
        source_row_id=f"r{index:03d}",
        context=f"Synthetic context {index}",
        question=f"Synthetic question {index}?",
        answers=("Option A", "Option B", "Option C", "Option D"),
        gold_label=label,
        problem=None,
        solution=None,
        category=None,
        level=None,
        synthetic=True,
    )
    record.validate()
    return record


def math_question(index: int, category: str, level: str) -> QuestionRecord:
    problem = f"math synthetic {index:03d}"
    content_hash = digest(problem)
    record = QuestionRecord(
        schema_version="question_v1",
        source=Source.MATH,
        original_question_id=f"math:train:{content_hash[:40]}",
        question_content_hash=content_hash,
        source_snapshot_id="math_snapshot",
        source_revision="a" * 40,
        source_row_id=f"{category}:{index:03d}",
        context=None,
        question=None,
        answers=(),
        gold_label=None,
        problem=problem,
        solution=f"synthetic solution {index}",
        category=category,
        level=level,
        synthetic=True,
    )
    record.validate()
    return record


def baseline_questions() -> list[QuestionRecord]:
    reclor = [reclor_question(index, "ABCD"[index % 4]) for index in range(40)]
    groups = (
        ("algebra", "Level 1"),
        ("algebra", "Level 2"),
        ("geometry", "Level 1"),
        ("geometry", "Level 2"),
    )
    math = [math_question(index, *groups[index % 4]) for index in range(40)]
    return reclor + math


def build(questions):
    return splitting_module().build_internal_split(questions, split_seed=42)


def source_snapshots(tmp_path):
    snapshots = {}
    bindings = {}
    for source in Source:
        questions = tuple(question for question in baseline_questions() if question.source is source)
        relative_path = f"sources/{source.value}/questions.jsonl"
        artifact_path = tmp_path / relative_path
        artifact_path.parent.mkdir(parents=True)
        for question in questions:
            append_record(artifact_path, question.to_dict())
        scan = scan_jsonl(artifact_path)
        artifact = ArtifactRef(relative_path, scan.file_sha256, len(scan.records))
        payload = {
            "schema_version": f"{source.value}_source_manifest_v1",
            "stage": "sources",
            "run_id": f"{source.value}_source_run",
            "config_hash": digest(f"{source.value}_config"),
            "upstream_manifest_hashes": [digest(f"{source.value}_upstream")],
            "artifacts": [artifact.to_dict()],
            "counts": {"questions": len(questions)},
            "source": source.value,
            "source_revision": questions[0].source_revision,
            "source_snapshot_id": questions[0].source_snapshot_id,
            "question_normalization_version": "question_normalization_v1",
        }
        manifest_path = tmp_path / f"sources/{source.value}/manifest.json"
        manifest_hash = write_atomic_manifest(manifest_path, payload)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        role = f"{source.value}_questions"
        snapshots[role] = SourceSnapshot(
            source=source,
            source_revision=questions[0].source_revision,
            source_snapshot_id=questions[0].source_snapshot_id,
            questions=questions,
            private_manifest=manifest,
        )
        bindings[role] = manifest_hash
    return snapshots, bindings


def test_split_is_input_order_invariant_and_has_exact_90_10_counts():
    questions = baseline_questions()

    first = build(questions)
    second = build(list(reversed(questions)))

    assert first == second
    assert first.stratification_level_by_source == {"math": "category+level", "reclor": "gold_label"}
    assert first.fallback_reasons == {}
    counts = Counter((assignment.source.value, assignment.split.value) for assignment in first.assignments)
    assert counts == {
        ("math", "train"): 36,
        ("math", "internal_holdout"): 4,
        ("reclor", "train"): 36,
        ("reclor", "internal_holdout"): 4,
    }
    assert len({assignment.original_question_id for assignment in first.assignments}) == 80


def test_split_stage_config_role_binds_both_source_manifests():
    module = splitting_module()
    config = module.split_stage_config(
        split_seed=42,
        upstream_bindings={"reclor_questions": "2" * 64, "math_questions": "1" * 64},
    )
    assert config == {
        "lineage_schema": "role_tagged_upstream_v1",
        "parameters": {
            "split_algorithm": "deterministic_stratified_question_split_v1",
            "split_schema_version": "split_manifest_v1",
            "split_seed": 42,
        },
        "upstream_bindings": {
            "math_questions": "1" * 64,
            "reclor_questions": "2" * 64,
        },
    }


def test_split_run_namespace_changes_with_algorithm_version(monkeypatch):
    module = splitting_module()
    bindings = {"math_questions": "1" * 64, "reclor_questions": "2" * 64}
    first = module.split_stage_config(42, bindings)
    repeated = module.split_stage_config(42, dict(reversed(tuple(bindings.items()))))

    assert run_id("split", first) == run_id("split", repeated)

    monkeypatch.setattr(module, "SPLIT_ALGORITHM", "deterministic_stratified_question_split_v2")
    changed = module.split_stage_config(42, bindings)

    assert run_id("split", first) != run_id("split", changed)


def test_bound_split_verifies_named_manifests_and_artifacts_before_building(tmp_path):
    module = splitting_module()
    snapshots, bindings = source_snapshots(tmp_path)

    bound = module.build_bound_internal_split(
        snapshots,
        upstream_bindings=bindings,
        data_root=tmp_path,
        split_seed=42,
    )

    assert bound == build(baseline_questions())


def test_bound_split_rejects_swapped_manifest_roles(tmp_path):
    module = splitting_module()
    snapshots, bindings = source_snapshots(tmp_path)
    swapped = {
        "math_questions": bindings["reclor_questions"],
        "reclor_questions": bindings["math_questions"],
    }

    with pytest.raises(ValueError, match="source manifest lineage"):
        module.build_bound_internal_split(snapshots, swapped, tmp_path)


def test_bound_split_rejects_manifest_payload_tampering(tmp_path):
    module = splitting_module()
    snapshots, bindings = source_snapshots(tmp_path)
    math = snapshots["math_questions"]
    tampered_manifest = dict(math.private_manifest)
    tampered_manifest["config_hash"] = "f" * 64
    snapshots["math_questions"] = replace(math, private_manifest=tampered_manifest)

    with pytest.raises(ValueError, match="source manifest lineage"):
        module.build_bound_internal_split(snapshots, bindings, tmp_path)


def test_bound_split_rejects_artifact_tampering(tmp_path):
    module = splitting_module()
    snapshots, bindings = source_snapshots(tmp_path)
    artifact_path = tmp_path / snapshots["math_questions"].private_manifest["artifacts"][0]["relative_path"]
    with artifact_path.open("ab") as stream:
        stream.write(b"tampered")

    with pytest.raises(ValueError, match="source artifact integrity"):
        module.build_bound_internal_split(snapshots, bindings, tmp_path)


def test_bound_split_rejects_a_different_question_normalization_contract(tmp_path):
    module = splitting_module()
    snapshots, bindings = source_snapshots(tmp_path)
    math = snapshots["math_questions"]
    payload = {
        key: value
        for key, value in math.private_manifest.items()
        if key not in {"created_at_utc", "output_manifest_hash"}
    }
    payload["question_normalization_version"] = "question_normalization_v2"
    manifest_path = tmp_path / "sources/math/manifest-v2.json"
    bindings["math_questions"] = write_atomic_manifest(manifest_path, payload)
    snapshots["math_questions"] = replace(
        math,
        private_manifest=json.loads(manifest_path.read_text(encoding="utf-8")),
    )

    with pytest.raises(ValueError, match="normalization"):
        module.build_bound_internal_split(snapshots, bindings, tmp_path)


def test_split_result_metadata_is_an_immutable_snapshot():
    result = build(baseline_questions())
    assert isinstance(result.stratification_level_by_source, MappingProxyType)
    assert isinstance(result.fallback_reasons, MappingProxyType)
    with pytest.raises(TypeError):
        result.stratification_level_by_source["math"] = "source"
    with pytest.raises(TypeError):
        result.fallback_reasons["math"] = ("changed",)


def test_reclor_holdout_is_stratified_by_official_gold_label():
    result = build(baseline_questions())
    holdout = [
        assignment
        for assignment in result.assignments
        if assignment.source is Source.RECLOR and assignment.split is Split.INTERNAL_HOLDOUT
    ]

    assert Counter(assignment.stratum for assignment in holdout) == {
        "gold_label=A": 1,
        "gold_label=B": 1,
        "gold_label=C": 1,
        "gold_label=D": 1,
    }


def test_math_falls_back_whole_source_from_category_level_to_category():
    specs = (
        [("algebra", "Level 1")] * 19
        + [("algebra", "Level 2")]
        + [("geometry", "Level 1")] * 10
        + [("geometry", "Level 2")] * 10
    )
    questions = [reclor_question(index, "ABCD"[index % 4]) for index in range(40)]
    questions += [math_question(index, *spec) for index, spec in enumerate(specs)]

    result = build(questions)

    assert result.stratification_level_by_source["math"] == "category"
    assert result.fallback_reasons == {"math": ("category+level_infeasible",)}
    assert {a.stratum for a in result.assignments if a.source is Source.MATH} == {
        "category=algebra",
        "category=geometry",
    }


def test_math_falls_back_whole_source_to_source_level():
    specs = [("algebra", "Level 1")] + [("geometry", "Level 1")] * 39
    questions = [reclor_question(index, "ABCD"[index % 4]) for index in range(40)]
    questions += [math_question(index, *spec) for index, spec in enumerate(specs)]

    result = build(questions)

    assert result.stratification_level_by_source["math"] == "source"
    assert result.fallback_reasons == {
        "math": ("category+level_infeasible", "category_infeasible")
    }
    assert {a.stratum for a in result.assignments if a.source is Source.MATH} == {"source=math"}


def test_holdout_quota_uses_exact_group_ratio_before_key_tie_breaking():
    specs = [("algebra", "Level 1")] * 10 + [("geometry", "Level 1")] * 40
    questions = [reclor_question(index, "ABCD"[index % 4]) for index in range(40)]
    questions += [math_question(index, *spec) for index, spec in enumerate(specs)]

    result = build(questions)
    math_holdout = [
        assignment
        for assignment in result.assignments
        if assignment.source is Source.MATH and assignment.split is Split.INTERNAL_HOLDOUT
    ]

    assert Counter(assignment.stratum for assignment in math_holdout) == {
        "category=algebra|level=Level 1": 1,
        "category=geometry|level=Level 1": 4,
    }


def test_smoke_quota_is_proportional_for_unequal_train_strata():
    specs = [("algebra", "Level 1")] * 10 + [("geometry", "Level 1")] * 40
    questions = [reclor_question(index, "ABCD"[index % 4]) for index in range(40)]
    questions += [math_question(index, *spec) for index, spec in enumerate(specs)]
    result = build(questions)

    smoke = splitting_module().select_smoke_questions(result.assignments, per_source=20, split_seed=42)
    math_smoke = [assignment for assignment in smoke if assignment.source is Source.MATH]

    assert Counter(assignment.stratum for assignment in math_smoke) == {
        "category=algebra|level=Level 1": 4,
        "category=geometry|level=Level 1": 16,
    }


def test_split_manifest_hash_is_literal_and_cross_platform_stable():
    result = build(baseline_questions())
    assert result.split_manifest_hash == EXPECTED_BASELINE_SPLIT_HASH


def test_split_manifest_hash_covers_exact_canonical_jsonl_bytes():
    module = splitting_module()
    questions = baseline_questions()
    result = build(questions)

    manifest_bytes = module.split_manifest_bytes(questions, result, split_seed=42)
    lines = manifest_bytes.splitlines(keepends=True)
    metadata = json.loads(lines[0])
    assignments = [json.loads(line) for line in lines[1:]]

    assert hashlib.sha256(manifest_bytes).hexdigest() == result.split_manifest_hash
    assert manifest_bytes.endswith(b"\n")
    assert b"\r" not in manifest_bytes
    assert all(line.endswith(b"\n") for line in lines)
    assert metadata == {
        "fallback_reasons": {},
        "question_normalization_versions": {
            "math": "question_normalization_v1",
            "reclor": "question_normalization_v1",
        },
        "record_type": "split_metadata",
        "schema_version": "split_manifest_v1",
        "source_provenance": {
            "math": {
                "source_revision": "a" * 40,
                "source_snapshot_id": "math_snapshot",
            },
            "reclor": {
                "source_revision": "reclor_revision",
                "source_snapshot_id": "reclor_snapshot",
            },
        },
        "split_algorithm": "deterministic_stratified_question_split_v1",
        "split_seed": 42,
        "stratification_level_by_source": {
            "math": "category+level",
            "reclor": "gold_label",
        },
    }
    assert len(assignments) == 80
    assert [row["original_question_id"] for row in assignments] == sorted(
        row["original_question_id"] for row in assignments
    )
    assert all(row["record_type"] == "split_assignment" for row in assignments)


def test_split_manifest_bytes_are_input_order_invariant():
    module = splitting_module()
    questions = baseline_questions()
    result = build(questions)

    assert module.split_manifest_bytes(questions, result, split_seed=42) == module.split_manifest_bytes(
        reversed(questions),
        result,
        split_seed=42,
    )


def test_smoke_selection_is_deterministic_train_only_and_proportional():
    module = splitting_module()
    result = build(baseline_questions())

    first = module.select_smoke_questions(result.assignments, per_source=20, split_seed=42)
    second = module.select_smoke_questions(tuple(reversed(result.assignments)), per_source=20, split_seed=42)

    assert first == second
    assert len(first) == 40
    assert all(assignment.split is Split.TRAIN for assignment in first)
    assert Counter(assignment.source.value for assignment in first) == {"math": 20, "reclor": 20}
    assert Counter((assignment.source.value, assignment.stratum) for assignment in first) == {
        ("math", "category=algebra|level=Level 1"): 5,
        ("math", "category=algebra|level=Level 2"): 5,
        ("math", "category=geometry|level=Level 1"): 5,
        ("math", "category=geometry|level=Level 2"): 5,
        ("reclor", "gold_label=A"): 5,
        ("reclor", "gold_label=B"): 5,
        ("reclor", "gold_label=C"): 5,
        ("reclor", "gold_label=D"): 5,
    }


def test_smoke_selection_fails_when_a_source_has_fewer_than_requested_train_questions():
    module = splitting_module()
    assignments = list(build(baseline_questions()).assignments)
    removed = 0
    reduced = []
    for assignment in assignments:
        if assignment.source is Source.RECLOR and assignment.split is Split.TRAIN and removed < 17:
            removed += 1
            continue
        reduced.append(assignment)

    with pytest.raises(ValueError, match="reclor.*20"):
        module.select_smoke_questions(reduced, per_source=20, split_seed=42)


def test_duplicate_question_identity_is_rejected_before_assignment():
    questions = baseline_questions()
    with pytest.raises(ValueError, match="duplicate"):
        build(questions + [questions[0]])


def test_split_rejects_mixed_source_snapshot_provenance():
    questions = baseline_questions()
    questions[0] = replace(questions[0], source_snapshot_id="different_reclor_snapshot")

    with pytest.raises(ValueError, match="source provenance"):
        build(questions)


def test_source_revision_participates_in_split_manifest_hash():
    baseline = baseline_questions()
    changed = [
        replace(question, source_revision="b" * 40)
        if question.source is Source.MATH
        else question
        for question in baseline
    ]

    assert build(baseline).split_manifest_hash != build(changed).split_manifest_hash


def test_reclor_gold_stratification_must_be_feasible():
    reclor = [reclor_question(index, "A" if index == 0 else "B") for index in range(40)]
    math = [question for question in baseline_questions() if question.source is Source.MATH]
    with pytest.raises(ValueError, match="ReClor.*gold_label"):
        build(reclor + math)


def matching_downstream_records(assignment, split_manifest_hash):
    plan = CandidatePlan(
        schema_version="candidate_plan_v1",
        generation_run_id="generation_1",
        candidate_id="candidate_1",
        original_question_id=assignment.original_question_id,
        question_content_hash=assignment.question_content_hash,
        source=assignment.source,
        split=assignment.split,
        split_manifest_hash=split_manifest_hash,
        generator_id="generator_1",
        sampling_index=0,
        shard_id="00000",
        prompt_hash="a" * 64,
    )
    pair = PairRecord(
        schema_version="pair_v1",
        pair_run_id="pair_run_1",
        pair_id="pair_1",
        generation_run_id="generation_1",
        verification_run_id="verification_1",
        original_question_id=assignment.original_question_id,
        question_content_hash=assignment.question_content_hash,
        source=assignment.source,
        split=assignment.split,
        split_manifest_hash=split_manifest_hash,
        positive_candidate_id="candidate_1",
        negative_candidate_id="candidate_2",
        positive_response_hash="b" * 64,
        negative_response_hash="c" * 64,
        positive_generator_id="generator_1",
        negative_generator_id="generator_2",
        positive_token_count=10,
        negative_token_count=11,
        tokenizer_repo="synthetic/tokenizer",
        tokenizer_revision="revision_1",
    )
    permutation = PermutationRecord(
        permutation_run_id="permutation_1",
        pair_id="pair_1",
        generation_run_id="generation_1",
        original_question_id=assignment.original_question_id,
        split=assignment.split,
        split_manifest_hash=split_manifest_hash,
        permutation_id=0,
        permutation_label="AB",
        surface_a_candidate_id="candidate_1",
        surface_b_candidate_id="candidate_2",
        correct_surface="A",
    )
    for record in (plan, pair, permutation):
        record.validate()
    return plan, pair, permutation


def test_downstream_candidate_pair_and_permutation_must_inherit_split_lineage():
    module = splitting_module()
    result = build(baseline_questions())
    assignment = next(a for a in result.assignments if a.split is Split.TRAIN)
    records = matching_downstream_records(assignment, result.split_manifest_hash)

    module.validate_split_inheritance(result.assignments, records, result.split_manifest_hash)

    wrong_split = replace(records[0], split=Split.INTERNAL_HOLDOUT)
    wrong_content = replace(records[1], question_content_hash="f" * 64)
    wrong_manifest = replace(records[2], split_manifest_hash="e" * 64)
    for wrong in (wrong_split, wrong_content, wrong_manifest):
        with pytest.raises(ValueError, match="split lineage"):
            module.validate_split_inheritance(result.assignments, [wrong], result.split_manifest_hash)
