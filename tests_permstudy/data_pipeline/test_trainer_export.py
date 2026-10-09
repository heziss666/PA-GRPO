"""Trainer-ready pairwise rows preserve canonical data-pipeline identity."""

from dataclasses import replace
import json

import pytest

from permstudy.data_pipeline.canonical import normalize_text_v1, sha256_hex
from permstudy.data_pipeline.ids import (
    math_question_identity,
    pair_id,
    reclor_content_hash,
)
from permstudy.data_pipeline.permutations import build_permutations
from permstudy.data_pipeline.schema import (
    CandidatePlan,
    CandidateRecord,
    PairRecord,
    QuestionRecord,
    Source,
    Split,
)


GENERATION_RUN_ID = "1" * 64
VERIFICATION_RUN_ID = "2" * 64
PAIR_RUN_ID = "3" * 64
SPLIT_MANIFEST_HASH = "4" * 64
PAIR_MANIFEST_HASH = "5" * 64
PROMPT_HASH = "6" * 64
MODEL_REVISION = "7" * 40
TOKENIZER_REVISION = "8" * 40


def _math_question() -> QuestionRecord:
    question_id, content_hash = math_question_identity("What is 1 + 1?")
    return QuestionRecord(
        schema_version="question_record_v1",
        source=Source.MATH,
        original_question_id=question_id,
        question_content_hash=content_hash,
        source_snapshot_id="math_snapshot_v1",
        source_revision="9" * 40,
        source_row_id="synthetic_math_1",
        context=None,
        question=None,
        answers=(),
        gold_label=None,
        problem="What is 1 + 1?",
        solution="The answer is 2.",
        category="algebra",
        level="Level 1",
        synthetic=True,
    )


def _reclor_question() -> QuestionRecord:
    context = "Every raven is a bird. Kai is a raven."
    question = "What follows?"
    answers = (
        "Kai is a bird",
        "Kai is a fish",
        "No raven is a bird",
        "Nothing follows",
    )
    return QuestionRecord(
        schema_version="question_record_v1",
        source=Source.RECLOR,
        original_question_id="reclor:train:synthetic_1",
        question_content_hash=reclor_content_hash(context, question, answers),
        source_snapshot_id="reclor_snapshot_v1",
        source_revision="a" * 40,
        source_row_id="synthetic_1",
        context=context,
        question=question,
        answers=answers,
        gold_label="A",
        problem=None,
        solution=None,
        category=None,
        level=None,
        synthetic=True,
    )


def _candidate(question, candidate_id, response, generator_id, sampling_index):
    return CandidateRecord(
        plan=CandidatePlan(
            schema_version="candidate_plan_v1",
            generation_run_id=GENERATION_RUN_ID,
            candidate_id=candidate_id,
            original_question_id=question.original_question_id,
            question_content_hash=question.question_content_hash,
            source=question.source,
            split=Split.TRAIN,
            split_manifest_hash=SPLIT_MANIFEST_HASH,
            generator_id=generator_id,
            sampling_index=sampling_index,
            shard_id="00000",
            prompt_hash=PROMPT_HASH,
        ),
        response=response,
        finish_reason="stop",
        generated_token_count=16,
        model_revision=MODEL_REVISION,
    )


def _pair_bundle(question, suffix):
    positive = _candidate(
        question,
        f"{suffix}_positive",
        f"Reasoning for {suffix}.\nFinal Answer: A",
        "qwen2.5-7b-instruct",
        0,
    )
    negative = _candidate(
        question,
        f"{suffix}_negative",
        f"Incorrect reasoning for {suffix}.\nFinal Answer: B",
        "llama-3.1-8b-instruct",
        1,
    )
    positive_response = normalize_text_v1(positive.response)
    negative_response = normalize_text_v1(negative.response)
    record = PairRecord(
        schema_version="pair_v1",
        pair_run_id=PAIR_RUN_ID,
        pair_id=pair_id(
            GENERATION_RUN_ID,
            question.original_question_id,
            positive.plan.candidate_id,
            negative.plan.candidate_id,
            "pair_v1",
        ),
        generation_run_id=GENERATION_RUN_ID,
        verification_run_id=VERIFICATION_RUN_ID,
        original_question_id=question.original_question_id,
        question_content_hash=question.question_content_hash,
        source=question.source,
        split=Split.TRAIN,
        split_manifest_hash=SPLIT_MANIFEST_HASH,
        positive_candidate_id=positive.plan.candidate_id,
        negative_candidate_id=negative.plan.candidate_id,
        positive_response_hash=sha256_hex(positive_response.encode("utf-8")),
        negative_response_hash=sha256_hex(negative_response.encode("utf-8")),
        positive_generator_id=positive.plan.generator_id,
        negative_generator_id=negative.plan.generator_id,
        positive_token_count=8,
        negative_token_count=9,
        tokenizer_repo="Qwen/Qwen2.5-7B-Instruct",
        tokenizer_revision=TOKENIZER_REVISION,
    )
    permutations = build_permutations(
        record, upstream_bindings={"pairs": PAIR_MANIFEST_HASH}
    )
    return (positive, negative), record, permutations


def _records():
    math = _math_question()
    reclor = _reclor_question()
    math_candidates, math_pair, math_permutations = _pair_bundle(math, "math")
    reclor_candidates, reclor_pair, reclor_permutations = _pair_bundle(reclor, "reclor")
    return (
        (math, reclor),
        math_candidates + reclor_candidates,
        (math_pair, reclor_pair),
        math_permutations + reclor_permutations,
    )


def test_builds_exact_sorted_rows_with_pair_and_permutation_identity():
    from permstudy.data_pipeline.trainer_export import build_trainer_rows

    questions, candidates, pairs, permutations = _records()

    rows = build_trainer_rows(
        questions, candidates, pairs, tuple(reversed(permutations))
    )

    assert [
        (row["extra_info"]["original_question_id"], row["extra_info"]["permutation_id"])
        for row in rows
    ] == [
        (questions[0].original_question_id, 0),
        (questions[0].original_question_id, 1),
        (questions[1].original_question_id, 0),
        (questions[1].original_question_id, 1),
    ]
    assert all(
        set(row) == {"data_source", "prompt", "ability", "reward_model", "extra_info"}
        for row in rows
    )
    assert [row["ability"] for row in rows] == [
        "math",
        "math",
        "logical_reasoning",
        "logical_reasoning",
    ]
    assert {row["data_source"] for row in rows} == {"permstudy_pairwise_judge_v1"}

    math_ab, math_ba = rows[:2]
    math_pair = pairs[0]
    assert math_ab["prompt"] == [
        {"role": "system", "content": "Reply with only A or B."},
        {
            "role": "user",
            "content": (
                "Question:\nWhat is 1 + 1?\n\n"
                "Response A:\nReasoning for math.\nFinal Answer: A\n\n"
                "Response B:\nIncorrect reasoning for math.\nFinal Answer: B\n\n"
                "Which response is more correct?\nAnswer with A or B only."
            ),
        },
    ]
    assert math_ab["reward_model"] == {"ground_truth": "A", "style": "rule"}
    assert math_ba["reward_model"] == {"ground_truth": "B", "style": "rule"}
    assert math_ab["extra_info"] == {
        "pair_id": math_pair.pair_id,
        "original_question_id": math_pair.original_question_id,
        "permutation_id": 0,
        "permutation_label": "AB",
        "source": "math",
        "split": "train",
        "split_manifest_hash": SPLIT_MANIFEST_HASH,
        "generation_run_id": GENERATION_RUN_ID,
        "pair_run_id": PAIR_RUN_ID,
        "permutation_run_id": permutations[0].permutation_run_id,
    }
    assert type(math_ab["extra_info"]["permutation_id"]) is int

    reclor_user = rows[2]["prompt"][1]["content"]
    assert "Context:\nEvery raven is a bird. Kai is a raven." in reclor_user
    assert "Question:\nWhat follows?" in reclor_user
    assert (
        "A. Kai is a bird\nB. Kai is a fish\nC. No raven is a bird\nD. Nothing follows"
        in reclor_user
    )


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("response_hash", "response hash"),
        ("question_hash", "question content hash"),
        ("split_hash", "split lineage"),
        ("missing_candidate", "selected candidate"),
        ("duplicate_permutation", "exactly AB and BA"),
        ("non_mirrored", "surface mapping"),
    ],
)
def test_row_builder_fails_closed_on_corrupted_join(mutation, message):
    from permstudy.data_pipeline.trainer_export import (
        TrainerExportIntegrityError,
        build_trainer_rows,
    )

    questions, candidates, pairs, permutations = _records()
    if mutation == "response_hash":
        pairs = (replace(pairs[0], positive_response_hash="f" * 64), pairs[1])
    elif mutation == "question_hash":
        questions = (
            replace(questions[0], question_content_hash="e" * 64),
            questions[1],
        )
    elif mutation == "split_hash":
        permutations = (
            replace(permutations[0], split_manifest_hash="d" * 64),
        ) + permutations[1:]
    elif mutation == "missing_candidate":
        candidates = tuple(
            candidate
            for candidate in candidates
            if candidate.plan.candidate_id != "math_positive"
        )
    elif mutation == "duplicate_permutation":
        permutations = (permutations[0], permutations[0]) + permutations[2:]
    else:
        permutations = (
            replace(permutations[0], surface_a_candidate_id="unexpected_candidate"),
        ) + permutations[1:]

    with pytest.raises(TrainerExportIntegrityError, match=message):
        build_trainer_rows(questions, candidates, pairs, permutations)


def test_export_stage_config_role_binds_all_upstreams_and_template_contract():
    from permstudy.data_pipeline.ids import run_id
    from permstudy.data_pipeline.trainer_export import export_stage_config

    bindings = {
        "split": "1" * 64,
        "generation": "2" * 64,
        "pairs": "3" * 64,
        "permutations": "4" * 64,
    }
    baseline = export_stage_config(bindings)
    repeated = export_stage_config(dict(reversed(tuple(bindings.items()))))
    changed = export_stage_config({**bindings, "pairs": "5" * 64})
    swapped = export_stage_config(
        {
            **bindings,
            "pairs": bindings["permutations"],
            "permutations": bindings["pairs"],
        }
    )

    assert baseline == repeated
    assert baseline["upstream_bindings"] == bindings
    assert (
        baseline["parameters"]["prompt_template_version"] == "pairwise_judge_direct_v1"
    )
    assert baseline["parameters"]["row_schema_version"] == "trainer_pairwise_parquet_v1"
    assert set(baseline["parameters"]) == {
        "export_config_schema",
        "prompt_template_hash",
        "prompt_template_version",
        "row_schema_version",
    }
    assert run_id("trainer_export", baseline) == run_id("trainer_export", repeated)
    assert run_id("trainer_export", baseline) != run_id("trainer_export", changed)
    assert run_id("trainer_export", baseline) != run_id("trainer_export", swapped)


def test_exact_prompt_template_change_changes_hash_and_export_run_id():
    from permstudy.data_pipeline import trainer_export
    from permstudy.data_pipeline.ids import run_id

    bindings = {
        "split": "1" * 64,
        "generation": "2" * 64,
        "pairs": "3" * 64,
        "permutations": "4" * 64,
    }
    changed_template = trainer_export.USER_PROMPT_TEMPLATE.replace(
        "Question:\n{rendered_question}",
        "Question:\n\n{rendered_question}",
    )
    changed_hash = trainer_export.prompt_template_hash(
        trainer_export.SYSTEM_PROMPT_TEMPLATE,
        changed_template,
    )
    baseline = trainer_export.export_stage_config(bindings)
    changed = trainer_export._export_stage_config(bindings, changed_hash)

    assert changed_hash != trainer_export.PROMPT_TEMPLATE_HASH
    assert run_id("trainer_export", baseline) != run_id("trainer_export", changed)


def _write_upstream_manifest(
    data_root, role, filename, records, *, upstream_manifest_hashes=()
):
    from permstudy.data_pipeline.io import (
        append_record,
        scan_jsonl,
        write_atomic_manifest,
    )
    from permstudy.data_pipeline.schema import RunManifest

    artifact_path = data_root / "upstreams" / role / filename
    artifact_path.parent.mkdir(parents=True)
    for record in records:
        append_record(artifact_path, record.to_dict())
    scan = scan_jsonl(artifact_path)
    if role == "generation":
        run_identity = GENERATION_RUN_ID
    elif role == "pairs":
        run_identity = PAIR_RUN_ID
    elif role == "permutations":
        run_identity = records[0].permutation_run_id
    else:
        run_identity = sha256_hex(f"{role}:run".encode())
    payload = {
        "schema_version": "run_manifest_v1",
        "stage": role,
        "run_id": run_identity,
        "config_hash": sha256_hex(f"{role}:config".encode()),
        "upstream_manifest_hashes": list(upstream_manifest_hashes),
        "artifacts": [
            {
                "relative_path": artifact_path.relative_to(data_root).as_posix(),
                "sha256": scan.file_sha256,
                "record_count": len(scan.records),
            }
        ],
        "counts": {"records": len(scan.records)},
        "created_at_utc": "2026-10-09T00:00:00.000000Z",
    }
    manifest_path = data_root / "upstreams" / role / "manifest.json"
    digest = write_atomic_manifest(manifest_path, payload)
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = RunManifest.from_dict(stored)
    assert manifest.output_manifest_hash == digest
    return manifest


def _write_upstreams(data_root):
    questions, candidates, pairs, permutations = _records()
    split = _write_upstream_manifest(data_root, "split", "questions.jsonl", questions)
    generation = _write_upstream_manifest(
        data_root,
        "generation",
        "candidates.jsonl",
        candidates,
        upstream_manifest_hashes=(split.output_manifest_hash,),
    )
    pairs_manifest = _write_upstream_manifest(
        data_root,
        "pairs",
        "pairs.jsonl",
        pairs,
        upstream_manifest_hashes=(generation.output_manifest_hash,),
    )
    permutation_manifest = _write_upstream_manifest(
        data_root,
        "permutations",
        "permutations.jsonl",
        permutations,
        upstream_manifest_hashes=(pairs_manifest.output_manifest_hash,),
    )
    return {
        "split": split,
        "generation": generation,
        "pairs": pairs_manifest,
        "permutations": permutation_manifest,
    }


def test_exports_hash_pinned_parquet_and_standard_manifest(tmp_path):
    from permstudy.data_pipeline.trainer_export import export_trainer_parquet

    manifests = _write_upstreams(tmp_path)

    summary = export_trainer_parquet(manifests, tmp_path)

    parquet_path = tmp_path / summary.dataset_artifact.relative_path
    manifest_path = parquet_path.with_name("manifest.json")
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert (
        parquet_path
        == tmp_path / "trainer_exports" / summary.export_run_id / "train.parquet"
    )
    assert summary.row_count == 4
    assert summary.dataset_artifact.record_count == 4
    assert summary.prompt_template_hash == stored["prompt_template_hash"]
    assert summary.dataset_artifact.sha256 == stored["dataset_sha256"]
    assert stored["counts"] == {"rows": 4}
    assert stored["artifacts"] == [summary.dataset_artifact.to_dict()]
    assert stored["upstream_bindings"] == {
        role: manifest.output_manifest_hash
        for role, manifest in sorted(manifests.items())
    }
    assert summary.output_manifest_hash == stored["output_manifest_hash"]


def test_export_refuses_mutated_or_unreferenced_upstream_artifact(tmp_path):
    from permstudy.data_pipeline.trainer_export import (
        TrainerExportIntegrityError,
        export_trainer_parquet,
    )

    manifests = _write_upstreams(tmp_path)
    candidate_ref = manifests["generation"].artifacts[0]
    candidate_path = tmp_path / candidate_ref.relative_path
    candidate_path.write_bytes(candidate_path.read_bytes() + b"{}\n")

    with pytest.raises(TrainerExportIntegrityError, match="generation artifact"):
        export_trainer_parquet(manifests, tmp_path)


def test_export_accepts_hash_valid_extended_manifest_envelopes(tmp_path):
    from permstudy.data_pipeline.io import write_atomic_manifest
    from permstudy.data_pipeline.trainer_export import export_trainer_parquet

    manifests = _write_upstreams(tmp_path)
    generation_path = tmp_path / "upstreams" / "generation" / "manifest.json"
    extended = json.loads(generation_path.read_text(encoding="utf-8"))
    extended["generation_run_id"] = GENERATION_RUN_ID
    extended["backend"] = "fake"
    extended["upstream_bindings"] = {"split": manifests["split"].output_manifest_hash}
    write_atomic_manifest(generation_path, extended)
    manifests["generation"] = json.loads(generation_path.read_text(encoding="utf-8"))
    pair_path = tmp_path / "upstreams" / "pairs" / "manifest.json"
    pair_manifest = json.loads(pair_path.read_text(encoding="utf-8"))
    pair_manifest["upstream_manifest_hashes"] = [
        manifests["generation"]["output_manifest_hash"]
    ]
    write_atomic_manifest(pair_path, pair_manifest)
    manifests["pairs"] = json.loads(pair_path.read_text(encoding="utf-8"))
    permutation_path = tmp_path / "upstreams" / "permutations" / "manifest.json"
    permutation_manifest = json.loads(permutation_path.read_text(encoding="utf-8"))
    permutation_manifest["upstream_manifest_hashes"] = [
        manifests["pairs"]["output_manifest_hash"]
    ]
    write_atomic_manifest(permutation_path, permutation_manifest)
    manifests["permutations"] = json.loads(permutation_path.read_text(encoding="utf-8"))

    summary = export_trainer_parquet(manifests, tmp_path)

    assert summary.row_count == 4
    assert (
        summary.upstream_bindings["generation"]
        == manifests["generation"]["output_manifest_hash"]
    )


@pytest.mark.parametrize(
    ("downstream_role", "upstream_role"),
    [
        ("generation", "split"),
        ("pairs", "generation"),
        ("permutations", "pairs"),
    ],
)
def test_export_rejects_rehashed_manifest_with_wrong_direct_upstream(
    tmp_path, downstream_role, upstream_role
):
    from permstudy.data_pipeline.io import write_atomic_manifest
    from permstudy.data_pipeline.trainer_export import (
        TrainerExportIntegrityError,
        export_trainer_parquet,
    )

    manifests = _write_upstreams(tmp_path)
    manifest_path = tmp_path / "upstreams" / downstream_role / "manifest.json"
    contradictory = json.loads(manifest_path.read_text(encoding="utf-8"))
    contradictory["upstream_manifest_hashes"] = ["f" * 64]
    write_atomic_manifest(manifest_path, contradictory)
    manifests[downstream_role] = json.loads(manifest_path.read_text(encoding="utf-8"))

    with pytest.raises(
        TrainerExportIntegrityError,
        match=f"{downstream_role}.*{upstream_role}",
    ):
        export_trainer_parquet(manifests, tmp_path)


def test_export_rejects_absent_or_role_contradictory_lineage_evidence(tmp_path):
    from permstudy.data_pipeline.io import write_atomic_manifest
    from permstudy.data_pipeline.trainer_export import (
        TrainerExportIntegrityError,
        export_trainer_parquet,
    )

    manifests = _write_upstreams(tmp_path)
    pair_path = tmp_path / "upstreams" / "pairs" / "manifest.json"
    absent = json.loads(pair_path.read_text(encoding="utf-8"))
    absent["upstream_manifest_hashes"] = []
    write_atomic_manifest(pair_path, absent)
    manifests["pairs"] = json.loads(pair_path.read_text(encoding="utf-8"))
    with pytest.raises(TrainerExportIntegrityError, match="pairs.*generation"):
        export_trainer_parquet(manifests, tmp_path)

    role_root = tmp_path / "role_tagged"
    manifests = _write_upstreams(role_root)
    permutation_path = role_root / "upstreams" / "permutations" / "manifest.json"
    contradictory = json.loads(permutation_path.read_text(encoding="utf-8"))
    actual_pair_hash = manifests["pairs"].output_manifest_hash
    contradictory["upstream_manifest_hashes"] = [actual_pair_hash, "f" * 64]
    contradictory["upstream_bindings"] = {"pairs": "f" * 64}
    write_atomic_manifest(permutation_path, contradictory)
    manifests["permutations"] = json.loads(permutation_path.read_text(encoding="utf-8"))
    with pytest.raises(TrainerExportIntegrityError, match="role binding.*pairs"):
        export_trainer_parquet(manifests, role_root)


def test_export_rejects_record_run_identity_not_bound_by_its_manifest(tmp_path):
    from permstudy.data_pipeline.io import write_atomic_manifest
    from permstudy.data_pipeline.trainer_export import (
        TrainerExportIntegrityError,
        export_trainer_parquet,
    )

    manifests = _write_upstreams(tmp_path)
    pair_path = tmp_path / "upstreams" / "pairs" / "manifest.json"
    mismatched = json.loads(pair_path.read_text(encoding="utf-8"))
    mismatched["run_id"] = "f" * 64
    write_atomic_manifest(pair_path, mismatched)
    manifests["pairs"] = json.loads(pair_path.read_text(encoding="utf-8"))
    permutation_path = tmp_path / "upstreams" / "permutations" / "manifest.json"
    permutation_manifest = json.loads(permutation_path.read_text(encoding="utf-8"))
    permutation_manifest["upstream_manifest_hashes"] = [
        manifests["pairs"]["output_manifest_hash"]
    ]
    write_atomic_manifest(permutation_path, permutation_manifest)
    manifests["permutations"] = json.loads(permutation_path.read_text(encoding="utf-8"))

    with pytest.raises(TrainerExportIntegrityError, match="pair run lineage"):
        export_trainer_parquet(manifests, tmp_path)


def test_export_rejects_repository_local_output_before_writing(tmp_path):
    from pathlib import Path

    from permstudy.data_pipeline.isolation import IsolationError
    from permstudy.data_pipeline.trainer_export import export_trainer_parquet

    manifests = _write_upstreams(tmp_path)
    repository = Path(__file__).resolve().parents[2]
    forbidden = repository / "artifacts" / "task13a-private-data-must-not-exist"

    with pytest.raises(IsolationError, match="external path"):
        export_trainer_parquet(manifests, forbidden)

    assert not forbidden.exists()


class FakeTrainerTokenizer:
    chat_template = "synthetic"
    pad_token_id = 0

    def apply_chat_template(
        self, messages, *, add_generation_prompt, tokenize, **_kwargs
    ):
        assert add_generation_prompt is True
        assert tokenize is False
        return (
            "\n".join(f"{message['role']}:{message['content']}" for message in messages)
            + "\nassistant:"
        )

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return [ord(character) % 251 + 1 for character in text]

    def __call__(self, text, *, return_tensors, add_special_tokens):
        import torch

        assert return_tensors == "pt"
        token_ids = self.encode(text, add_special_tokens=add_special_tokens)
        return {
            "input_ids": torch.tensor([token_ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(token_ids)), dtype=torch.long),
        }


def _load_export_through_trainer(export_path, cache_dir):
    from omegaconf import OmegaConf
    from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn

    dataset = RLHFDataset(
        data_files=str(export_path),
        tokenizer=FakeTrainerTokenizer(),
        config=OmegaConf.create(
            {
                "prompt_key": "prompt",
                "max_prompt_length": 4096,
                "filter_overlong_prompts": False,
                "truncation": "error",
                "return_raw_chat": True,
                "cache_dir": str(cache_dir),
            }
        ),
    )
    return dataset, collate_fn([dataset[index] for index in range(len(dataset))])


def test_parquet_round_trip_through_real_trainer_identity_path(tmp_path):
    from permstudy.data_pipeline.trainer_export import export_trainer_parquet
    from permstudy.rollout_identity import (
        attach_permutation_identity,
        merge_identity_into_extra_infos,
        repeat_for_rollout,
    )
    from verl import DataProto

    summary = export_trainer_parquet(_write_upstreams(tmp_path), tmp_path)
    export_path = tmp_path / summary.dataset_artifact.relative_path

    dataset, loaded = _load_export_through_trainer(export_path, tmp_path / "hf_cache")
    batch = DataProto.from_single_dict(loaded)
    attach_permutation_identity(batch, identity_mode="explicit")
    repeated = repeat_for_rollout(batch, repeat_times=2, identity_mode="explicit")
    merged = merge_identity_into_extra_infos(repeated)

    _, _, pairs, _ = _records()
    expected_pair_ids = [
        pairs[0].pair_id,
        pairs[0].pair_id,
        pairs[1].pair_id,
        pairs[1].pair_id,
    ]
    assert len(dataset) == 4
    assert loaded["raw_prompt"][0][0]["role"] == "system"
    assert loaded["reward_model"][0] == {"ground_truth": "A", "style": "rule"}
    assert loaded["extra_info"][0]["pair_id"] == pairs[0].pair_id
    assert batch.non_tensor_batch["pair_id"].tolist() == expected_pair_ids
    assert batch.non_tensor_batch["permutation_id"].tolist() == [0, 1, 0, 1]
    assert repeated.non_tensor_batch["pair_id"].tolist() == [
        value for value in expected_pair_ids for _ in range(2)
    ]
    assert repeated.non_tensor_batch["permutation_id"].tolist() == [
        0,
        0,
        1,
        1,
        0,
        0,
        1,
        1,
    ]
    assert repeated.non_tensor_batch["rollout_slot"].tolist() == [
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
    ]
    assert [entry["rollout_slot"] for entry in merged] == [0, 1, 0, 1, 0, 1, 0, 1]
    assert [entry["pair_id"] for entry in merged] == repeated.non_tensor_batch[
        "pair_id"
    ].tolist()


def test_explicit_trainer_identity_uses_pair_id_without_question_alias_and_export_requires_pair_id(
    tmp_path,
):
    import pandas as pd

    from permstudy.data_pipeline.trainer_export import (
        TrainerExportIntegrityError,
        build_trainer_rows,
        export_trainer_parquet,
        validate_trainer_rows,
    )
    from permstudy.rollout_identity import attach_permutation_identity
    from verl import DataProto

    summary = export_trainer_parquet(_write_upstreams(tmp_path), tmp_path)
    export_path = tmp_path / summary.dataset_artifact.relative_path
    frame = pd.read_parquet(export_path)
    frame["extra_info"] = frame["extra_info"].map(
        lambda extra: {
            key: value for key, value in extra.items() if key != "original_question_id"
        }
    )
    no_question_path = tmp_path / "no_question_alias.parquet"
    frame.to_parquet(no_question_path, index=False)

    _, loaded = _load_export_through_trainer(
        no_question_path, tmp_path / "hf_cache_alias"
    )
    batch = DataProto.from_single_dict(loaded)
    attach_permutation_identity(batch, identity_mode="explicit")
    assert all(len(value) == 64 for value in batch.non_tensor_batch["pair_id"])

    rows = build_trainer_rows(*_records())
    rows[0]["extra_info"].pop("pair_id")
    with pytest.raises(TrainerExportIntegrityError, match="pair_id"):
        validate_trainer_rows(rows)
