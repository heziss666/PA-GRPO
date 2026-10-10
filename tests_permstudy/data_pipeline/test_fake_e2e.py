"""One synthetic research path, including its real trainer identity consumer."""

from collections import Counter
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf
import pyarrow.parquet as pq
import pytest

from permstudy.rollout_identity import (
    attach_permutation_identity,
    merge_identity_into_extra_infos,
    repeat_for_rollout,
)
from scripts_permstudy.data import _common as cli_io
from scripts_permstudy.data import build_permutations as permutation_cli
from scripts_permstudy.data import build_reasoning_pairs as pair_cli
from scripts_permstudy.data import plan_generation as generation_cli
from scripts_permstudy.data import run_fake_e2e as e2e
from scripts_permstudy.data import verify_candidates as verification_cli
from tests_permstudy.data_pipeline.test_trainer_export import FakeTrainerTokenizer
from verl import DataProto
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "data_pipeline"


@pytest.fixture(scope="module")
def synthetic_runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("task17-fake-e2e")
    return (
        e2e.run_fake_e2e(root / "first", FIXTURES),
        e2e.run_fake_e2e(root / "second", FIXTURES),
    )


def test_independent_fresh_roots_preserve_all_blocking_research_semantics(
    synthetic_runs,
):
    first, second = synthetic_runs
    assert first.private_root != second.private_root
    assert (first.private_root / "canonical/assignments.jsonl").read_bytes() == (
        second.private_root / "canonical/assignments.jsonl"
    ).read_bytes()
    semantic_fields = (
        "split_hash",
        "successful_keys",
        "semantic_candidate_set_hash",
        "pair_hash",
        "permutation_hash",
        "trainer_row_hash",
    )
    differing = [
        field
        for field in semantic_fields
        if getattr(first, field) != getattr(second, field)
    ]
    assert not differing, (
        f"Independent fresh roots changed semantic outputs: {differing}"
    )


def test_synthetic_full_flow_preserves_research_semantics_and_trainer_identity(
    synthetic_runs,
):
    # A missing stage, changed split, or lost resume/rollout identity breaks this path.
    assert callable(getattr(e2e, "run_fake_e2e", None)), (
        "Task 17 orchestration is missing"
    )
    summary = synthetic_runs[0]
    assert summary.question_count == 80
    assert summary.train_count == 72
    assert summary.holdout_count == 8
    assert summary.smoke_count == 40
    assert summary.planned_count == summary.successful_count == 240
    assert summary.historical_failure_count == 1
    assert summary.interrupted is True
    assert summary.successful_keys == summary.resumed_successful_keys
    assert len(set(summary.successful_keys)) == 240
    assert (
        summary.semantic_candidate_set_hash
        == summary.resumed_semantic_candidate_set_hash
    )
    assert summary.pair_hash == summary.resumed_pair_hash
    assert summary.permutation_hash == summary.resumed_permutation_hash
    assert summary.trainer_row_hash == summary.resumed_trainer_row_hash
    assert summary.functional_passed is True
    assert summary.audit_required_count == summary.audit_completed_count > 0
    assert summary.statistical_status == "FAIL"
    assert "production" not in summary.public_summary().values()

    def load_stage(stage):
        manifests = list(summary.private_root.glob(f"{stage}/*/manifest.json"))
        assert len(manifests) == 1
        relative = manifests[0].relative_to(summary.private_root).as_posix()
        return cli_io.load_manifest(summary.private_root, relative, stage)

    split_manifest = load_stage("split")
    split_result, formal_smoke = generation_cli.load_split(
        summary.private_root, split_manifest
    )
    assert len(split_result.assignments) == 80
    assert len(formal_smoke) == 40
    generation_manifest = load_stage("generation")
    formal_plan = generation_cli.load_plan(summary.private_root, generation_manifest)
    assert len(formal_plan.candidates) == 240
    verification_manifest = load_stage("verification")
    formal_verification = verification_cli.load_verification(
        summary.private_root, verification_manifest
    )
    assert len(formal_verification[2]) == 240
    pair_manifest = load_stage("pairs")
    assert len(pair_cli.load_pairs(summary.private_root, pair_manifest)) == 40
    permutation_manifest = load_stage("permutations")
    assert (
        len(
            permutation_cli.load_permutations(
                summary.private_root, permutation_manifest
            )
        )
        == 80
    )

    def records(name):
        return [
            json.loads(line)
            for line in (summary.private_root / name)
            .read_text(encoding="utf-8")
            .splitlines()
        ]

    questions = records("canonical/questions.jsonl")
    assignments = records("canonical/assignments.jsonl")
    smoke = records("canonical/smoke.jsonl")
    pairs = records("canonical/pairs.jsonl")
    permutations = records("canonical/permutations.jsonl")
    verified = records("canonical/verifications.jsonl")
    assert all(q["synthetic"] is True for q in questions)
    assert Counter(q["source"] for q in questions) == {"math": 40, "reclor": 40}
    assert Counter(q["gold_label"] for q in questions if q["source"] == "reclor") == {
        "A": 10,
        "B": 10,
        "C": 10,
        "D": 10,
    }
    assert Counter((a["source"], a["split"]) for a in assignments) == {
        ("math", "train"): 36,
        ("math", "internal_holdout"): 4,
        ("reclor", "train"): 36,
        ("reclor", "internal_holdout"): 4,
    }
    assert Counter(a["source"] for a in smoke) == {"math": 20, "reclor": 20}
    assert all(a["split"] == "train" for a in smoke)
    assert {v["verification_status"] for v in verified} == {
        "correct",
        "incorrect",
        "invalid",
        "ambiguous",
        "error",
    }
    assert len(pairs) == summary.pair_count == 40
    assert len({p["original_question_id"] for p in pairs}) == len(pairs)
    assert Counter(p["source"] for p in pairs) == {"math": 20, "reclor": 20}
    assert (
        len(permutations)
        == summary.permutation_count
        == summary.trainer_row_count
        == 80
    )
    assert all(
        count == 2 for count in Counter(p["pair_id"] for p in permutations).values()
    )

    def canonical(payload):
        return json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

    def payload_hash(rows):
        return hashlib.sha256(
            b"".join(
                canonical({k: v for k, v in row.items() if k != "record_hash"}) + b"\n"
                for row in rows
            )
        ).hexdigest()

    assert (
        summary.split_hash
        == hashlib.sha256(
            (summary.private_root / "canonical/split.canonical.jsonl").read_bytes()
        ).hexdigest()
    )
    assert summary.pair_hash == payload_hash(
        sorted(pairs, key=lambda p: p["original_question_id"])
    )
    assert summary.permutation_hash == payload_hash(
        sorted(
            permutations, key=lambda p: (p["original_question_id"], p["permutation_id"])
        )
    )
    parquet_rows = pq.read_table(
        summary.private_root / summary.trainer_relative_path
    ).to_pylist()
    assert (
        summary.trainer_row_hash == hashlib.sha256(canonical(parquet_rows)).hexdigest()
    )
    assert summary.successful_keys == tuple(
        sorted(
            (c["plan"]["generation_run_id"], c["plan"]["candidate_id"])
            for c in records("canonical/candidates.jsonl")
        )
    )

    dataset = RLHFDataset(
        data_files=str(summary.private_root / summary.trainer_relative_path),
        tokenizer=FakeTrainerTokenizer(),
        config=OmegaConf.create(
            {
                "prompt_key": "prompt",
                "max_prompt_length": 4096,
                "filter_overlong_prompts": False,
                "truncation": "error",
                "return_raw_chat": True,
                "cache_dir": str(summary.private_root / "trainer_cache"),
            }
        ),
    )
    assert len(dataset) == 80
    loaded = collate_fn([dataset[0], dataset[1]])
    batch = DataProto.from_single_dict(loaded)
    attach_permutation_identity(batch, identity_mode="explicit")
    repeated = repeat_for_rollout(batch, repeat_times=2, identity_mode="explicit")
    reward_extra = merge_identity_into_extra_infos(repeated)
    pair_id = pairs[0]["pair_id"]
    assert [entry["pair_id"] for entry in reward_extra] == [pair_id] * 4
    assert [entry["permutation_id"] for entry in reward_extra] == [0, 0, 1, 1]
    assert [entry["rollout_slot"] for entry in reward_extra] == [0, 1, 0, 1]
    assert [entry["ground_truth"] for entry in loaded["reward_model"]] == ["A", "B"]

    public = summary.public_summary()
    assert public["mode"] == "synthetic_fake"
    assert public["count"] == 240
    assert str(summary.private_root) not in json.dumps(public)
    assert "Synthetic" not in json.dumps(public)
