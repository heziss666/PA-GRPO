"""Tests for the Task 4 real-GPU smoke evidence verifier.

The verifier must decide PASS/FAIL strictly from artifacts the run produced, and
must never substitute an expected value for missing evidence.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts_permstudy" / "task4" / "verify_real_gpu_smoke.py"

PAIR_COUNT = 8
ROWS_PER_STEP = 32


def load_verifier():
    assert SCRIPT.is_file(), f"missing task4 evidence verifier: {SCRIPT}"
    spec = importlib.util.spec_from_file_location("task4_verify_real_gpu_smoke", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# synthetic evidence
# ---------------------------------------------------------------------------


def write_dataset_manifest(run_dir: Path, pairs: int = PAIR_COUNT, reverse: bool = False) -> dict:
    entries = []
    for index in range(pairs):
        pair_id = f"q{index:04d}"
        entries.append(
            {
                "pair_id": pair_id,
                "original_question_id": pair_id,
                "winner": "model_a",
                "max_prompt_tokens": 100 + index,
                "rows": [
                    {"permutation": 0, "row_index": 2 * index, "ground_truth": "A",
                     "prompt_tokens": 90 + index, "prompt_sha256": "0" * 64},
                    {"permutation": 1, "row_index": 2 * index + 1, "ground_truth": "B",
                     "prompt_tokens": 100 + index, "prompt_sha256": "1" * 64},
                ],
            }
        )
    if reverse:
        entries = entries[::-1]
    manifest = {
        "schema_version": "task4_tiny_pairwise_smoke_v1",
        "pairs": entries,
        "pair_count": pairs,
        "row_count": pairs * 2,
        "fixture_sha256": "a" * 64,
    }
    (run_dir / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def canonical_sequence(manifest: dict) -> list[tuple[str, int]]:
    sequence: list[tuple[str, int]] = []
    for pair in manifest["pairs"]:
        for permutation in (0, 1):
            sequence.extend([(pair["pair_id"], permutation)] * 2)
    return sequence


def write_train_log(
    run_dir: Path,
    *,
    steps=(1, 2),
    fallback: bool = False,
    grad_norm: float = 0.75,
    pair_baseline: bool = True,
    nan_metric: bool = False,
    vllm: bool = True,
    identity_error: bool = False,
) -> None:
    lines = []
    if vllm:
        lines.append("INFO 01-01 00:00:00 vllm_worker.py:88] vLLM engine v0.8.5 initialized (enforce_eager=True)")
    for step in steps:
        parts = [
            f"step:{step}",
            f"training/global_step:{step}",
            "actor/pg_loss:0.5000",
            f"actor/grad_norm:{grad_norm}",
            "actor/kl_loss:0.0100",
            "reward/consistency_unpaired_rate:0.0",
        ]
        if pair_baseline:
            parts += [
                "pair_baseline/mean_of_pair_means:0.100000",
                "pair_baseline/mean_of_pair_stds:0.200000",
                "pair_baseline/std_of_pair_stds:0.300000",
            ]
        if nan_metric:
            parts.append("actor/kl_loss:nan")
        lines.append(" - ".join(parts))
    if fallback:
        # the exact string the trainer prints at verl/trainer/ppo/ray_trainer.py:326
        lines.append("[Warning] pair-baseline GRPO failed with error RuntimeError('probe'), fallback to original GRPO.")
    if identity_error:
        lines.append("RolloutIdentityAlignmentError: generation output reordered rollout identity for pair_id")
    (run_dir / "train.log").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _log_line(body: str, index: object = 0) -> str:
    return f"2026-01-01 00:00:00.000000 | idx={index} | pid=1 | cuda=cuda:0 | {body}"


def write_reward_log(
    run_dir: Path,
    manifest: dict,
    *,
    reorder: bool = True,
    unpaired: bool = False,
    duplicate: bool = False,
    blocks: int = 2,
    batch_position_indices: bool = False,
) -> None:
    path = run_dir / "reward.log"
    # historical evidence that must never be admissible
    path.write_text(
        _log_line("PAIR_CHECK | pair_id=stale_pair | rollout_slot=0 | is_consistent=True") + "\n",
        encoding="utf-8",
    )
    start = path.stat().st_size

    canonical = canonical_sequence(manifest)
    pair_ids = [pair["pair_id"] for pair in manifest["pairs"]]
    row_index_by_key = {
        (pair["pair_id"], int(row["permutation"])): int(row["row_index"])
        for pair in manifest["pairs"]
        for row in pair["rows"]
    }
    lines: list[str] = []
    for block_index in range(blocks):
        order = list(canonical)
        if reorder and block_index == 0:
            order = order[::-1]
        for index, (pair_id, permutation) in enumerate(order):
            # ``my_reward/judge_qwen.py`` logs ``idx = extra.get("index", i)``: the fixture's
            # source row index when present, otherwise the post-reorder batch position.
            logged_index = index if batch_position_indices else row_index_by_key[(pair_id, permutation)]
            lines.append(
                _log_line(
                    f"BASE | i={index} | pair_id={pair_id} | perm={permutation} | ans=A | gt=A | "
                    f"confidence=1.00 | method=boxed | base_score=0.500 | corr=0.500 | fmt=0.000 | len=0.000",
                    index=logged_index,
                )
            )
        checks = []
        for pair_id in pair_ids:
            for slot in (0, 1):
                checks.append(
                    _log_line(
                        f"PAIR_CHECK | pair_id={pair_id} | rollout_slot={slot} | "
                        f"i0=0,idx0=0,ans0=A,conf0=1.00 | i1=1,idx1=1,ans1=A,conf1=1.00 | "
                        f"mapped(ans0)=A | is_consistent=True | pair_bonus=1.000 | "
                        f"final_score0=1.000 | final_score1=1.000"
                    )
                )
        if duplicate:
            checks.append(checks[0])
        lines.extend(checks)
        if unpaired and block_index == 0:
            lines.append(_log_line(f"PAIR_UNPAIRED | pair_id={pair_ids[0]} | rollout_slot=1 | i=9"))

    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    end = path.stat().st_size
    (run_dir / "reward_log_window.json").write_text(
        json.dumps({"reward_log_path": str(path), "start_offset_bytes": start, "_end": end}),
        encoding="utf-8",
    )


def write_rollouts(run_dir: Path, *, steps=(1, 2), rows: int = ROWS_PER_STEP) -> None:
    directory = run_dir / "rollouts"
    directory.mkdir(parents=True, exist_ok=True)
    for step in steps:
        with (directory / f"{step}.jsonl").open("w", encoding="utf-8") as handle:
            for _ in range(rows):
                handle.write(json.dumps({"input": "p", "output": "o", "gts": "A", "score": 1.0, "step": step}) + "\n")


def write_checkpoints(
    run_dir: Path,
    *,
    steps=(1, 2),
    latest: int = 2,
    adapter: bool = True,
    lora_b_zero: bool = False,
    non_finite_tensor: bool = False,
    include_lora_b: bool = True,
) -> None:
    checkpoints = run_dir / "checkpoints"
    for step in steps:
        (checkpoints / f"global_step_{step}").mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    (checkpoints / "latest_checkpointed_iteration.txt").write_text(f"{latest}\n", encoding="utf-8")
    target = checkpoints / "global_step_2" / "actor" / "hf"
    target.mkdir(parents=True, exist_ok=True)
    if not adapter:
        return
    if non_finite_tensor:
        b_value = np.full((2, 2), np.inf, dtype=np.float32)
    elif lora_b_zero:
        b_value = np.zeros((2, 2), dtype=np.float32)
    else:
        b_value = np.full((2, 2), 0.25, dtype=np.float32)
    tensors = {"base_model.model.q_proj.lora_A.weight": np.zeros((2, 2), dtype=np.float32)}
    if include_lora_b:
        tensors["base_model.model.q_proj.lora_B.weight"] = b_value
    save_file(tensors, str(target / "adapter_model.safetensors"))


def write_controlled_eval(
    run_dir: Path,
    *,
    contract: str = "controlled",
    num_options: int = 2,
    num_samples: int = 4,
    completed: bool = True,
) -> None:
    (run_dir / "controlled_eval.json").write_text(
        json.dumps(
            {
                "evaluation_contract": contract,
                "num_options": num_options,
                "num_samples": num_samples,
                "completed": completed,
            }
        ),
        encoding="utf-8",
    )


def build_run(tmp_path: Path, **overrides):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = write_dataset_manifest(run_dir, reverse=overrides.get("manifest_reverse", False))
    write_train_log(
        run_dir,
        steps=overrides.get("train_steps", overrides.get("steps", (1, 2))),
        fallback=overrides.get("fallback", False),
        grad_norm=overrides.get("grad_norm", 0.75),
        pair_baseline=overrides.get("pair_baseline", True),
        nan_metric=overrides.get("nan_metric", False),
        vllm=overrides.get("vllm", True),
        identity_error=overrides.get("identity_error", False),
    )
    if overrides.get("rollouts_present", True):
        write_rollouts(run_dir, steps=overrides.get("steps", (1, 2)), rows=overrides.get("rows", ROWS_PER_STEP))
    if overrides.get("reward_present", True):
        write_reward_log(
            run_dir,
            manifest,
            reorder=overrides.get("reorder", True),
            unpaired=overrides.get("unpaired", False),
            duplicate=overrides.get("duplicate", False),
            blocks=overrides.get("reward_blocks", 2),
            batch_position_indices=overrides.get("batch_position_indices", False),
        )
    else:
        (run_dir / "reward_log_window.json").write_text(
            json.dumps({"reward_log_path": str(run_dir / "missing.log"), "start_offset_bytes": 0}),
            encoding="utf-8",
        )
    write_checkpoints(
        run_dir,
        steps=overrides.get("steps", (1, 2)),
        latest=overrides.get("latest", 2),
        adapter=overrides.get("adapter", True),
        lora_b_zero=overrides.get("lora_b_zero", False),
        non_finite_tensor=overrides.get("non_finite_tensor", False),
        include_lora_b=overrides.get("include_lora_b", True),
    )
    write_controlled_eval(
        run_dir,
        contract=overrides.get("contract", "controlled"),
        num_options=overrides.get("num_options", 2),
        num_samples=overrides.get("num_samples", 4),
        completed=overrides.get("completed", True),
    )
    return run_dir


def paths_for(verifier, run_dir: Path):
    return verifier.VerifyPaths(
        run_dir=run_dir,
        dataset_manifest=run_dir / "dataset_manifest.json",
        reward_log_window=run_dir / "reward_log_window.json",
        train_log=run_dir / "train.log",
        rollout_dir=run_dir / "rollouts",
        checkpoint_dir=run_dir / "checkpoints",
        merged_adapter=run_dir / "checkpoints" / "global_step_2" / "actor" / "hf" / "adapter_model.safetensors",
        controlled_eval=run_dir / "controlled_eval.json",
    )


def run_verifier(tmp_path: Path, **overrides):
    verifier = load_verifier()
    run_dir = build_run(tmp_path, **overrides)
    summary = verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))
    return verifier, run_dir, summary


# ---------------------------------------------------------------------------
# success path
# ---------------------------------------------------------------------------


def test_valid_evidence_passes(tmp_path):
    verifier, run_dir, summary = run_verifier(tmp_path)

    assert summary["optimization_steps"] == [1, 2]
    assert summary["fallback_detected"] is False
    assert summary["identity"]["unpaired_count"] == 0
    assert summary["identity"]["reorder_observed"] is True
    assert summary["lora"]["nonzero_lora_b"] is True
    assert summary["passed"] is True
    assert summary["failures"] == []

    for name in ("metrics_summary.json", "identity_pairing_summary.json", "checkpoint_manifest.json", "lora_update.json"):
        assert (run_dir / name).is_file(), f"verifier must write {name}"
    written = json.loads((run_dir / "identity_pairing_summary.json").read_text(encoding="utf-8"))
    assert written["pair_check_count"] == PAIR_COUNT * 2 * 2


def test_historical_reward_lines_outside_the_window_are_ignored(tmp_path):
    _verifier, run_dir, summary = run_verifier(tmp_path)
    # the seeded historical PAIR_CHECK for stale_pair must not count
    assert summary["identity"]["pair_check_count"] == PAIR_COUNT * 2 * 2
    assert "stale_pair" not in json.dumps(summary["identity"])


def test_cli_returns_zero_on_success(tmp_path, capsys):
    verifier, run_dir, _summary = run_verifier(tmp_path)
    code = verifier.main(["--run-dir", str(run_dir), "--dataset-manifest", str(run_dir / "dataset_manifest.json")])
    assert code == 0
    assert "PASS" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# hard failures
# ---------------------------------------------------------------------------


def test_fallback_warning_fails(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path, fallback=True)
    with pytest.raises(verifier.VerificationFailed, match="fallback to original GRPO"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_missing_pair_baseline_metric_fails(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path, pair_baseline=False)
    with pytest.raises(verifier.VerificationFailed, match="pair-baseline metric"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_unpaired_rollout_key_fails(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path, unpaired=True)
    with pytest.raises(verifier.VerificationFailed, match="PAIR_UNPAIRED"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_duplicate_rollout_key_fails(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path, duplicate=True)
    with pytest.raises(verifier.VerificationFailed, match="duplicate PAIR_CHECK"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_missing_optimization_step_fails(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path, train_steps=(1,))
    with pytest.raises(verifier.VerificationFailed, match="optimization steps observed"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_zero_grad_norm_fails(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path, grad_norm=0.0)
    with pytest.raises(verifier.VerificationFailed, match="grad_norm is not positive"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_non_finite_metric_fails(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path, nan_metric=True)
    with pytest.raises(verifier.VerificationFailed, match="non-finite metric"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_absent_vllm_initialization_fails(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path, vllm=False)
    with pytest.raises(verifier.VerificationFailed, match="vLLM engine initialization"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_identity_error_in_train_log_fails(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path, identity_error=True)
    with pytest.raises(verifier.VerificationFailed, match="rollout identity error"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_no_observed_reorder_fails(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path, reorder=False)
    with pytest.raises(verifier.VerificationFailed, match="no balance_batch reorder"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_absent_checkpoint_fails(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path, steps=(1, 2), latest=1)
    with pytest.raises(verifier.VerificationFailed, match="latest_checkpointed_iteration"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_zero_lora_b_tensor_fails(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path, lora_b_zero=True)
    with pytest.raises(verifier.VerificationFailed, match="lora_B tensor is zero"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_missing_lora_b_tensor_fails(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path, include_lora_b=False)
    with pytest.raises(verifier.VerificationFailed, match="no lora_B tensor"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_non_finite_adapter_tensor_fails(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path, non_finite_tensor=True)
    with pytest.raises(verifier.VerificationFailed, match="non-finite tensors"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_wrong_evaluation_contract_fails(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path, contract="official")
    with pytest.raises(verifier.VerificationFailed, match="evaluation contract"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_wrong_num_options_fails(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path, num_options=4)
    with pytest.raises(verifier.VerificationFailed, match="num_options"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_incomplete_evaluation_fails(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path, completed=False)
    with pytest.raises(verifier.VerificationFailed, match="did not complete"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


# ---------------------------------------------------------------------------
# missing evidence must never be reconstructed
# ---------------------------------------------------------------------------


def test_absent_rollouts_are_rejected_rather_than_assumed(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path, rollouts_present=False)
    with pytest.raises(verifier.EvidenceError, match="rollout"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_short_rollout_file_is_observed_not_assumed(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path, rows=31)
    with pytest.raises(verifier.VerificationFailed, match="observed 31 rollout rows"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_missing_reward_log_is_rejected(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path, reward_present=False)
    with pytest.raises(verifier.EvidenceError, match="reward log"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_rotated_reward_log_window_is_rejected(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path)
    window_path = run_dir / "reward_log_window.json"
    window = json.loads(window_path.read_text(encoding="utf-8"))
    # simulate rotation/truncation: the recorded start is now past the file end
    window["start_offset_bytes"] = (run_dir / "reward.log").stat().st_size + 10
    window_path.write_text(json.dumps(window), encoding="utf-8")
    with pytest.raises(verifier.EvidenceError, match="rotated|truncated"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_missing_dataset_manifest_is_rejected(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path)
    (run_dir / "dataset_manifest.json").unlink()
    with pytest.raises(verifier.EvidenceError, match="dataset manifest"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_missing_merged_adapter_is_rejected(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path, adapter=False)
    with pytest.raises(verifier.EvidenceError, match="LoRA adapter"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_extra_reward_block_fails(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path, reward_blocks=3)
    with pytest.raises(verifier.VerificationFailed, match="invocation block"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_cli_exit_codes_distinguish_evidence_from_gate_failure(tmp_path, capsys):
    verifier = load_verifier()
    failing = build_run(tmp_path / "gate", fallback=True)
    assert verifier.main(["--run-dir", str(failing)]) == 3
    assert "verification failed" in capsys.readouterr().err

    broken = build_run(tmp_path / "evidence")
    (broken / "train.log").unlink()
    assert verifier.main(["--run-dir", str(broken)]) == 2
    assert "evidence error" in capsys.readouterr().err


def test_reordered_manifest_still_defines_the_canonical_order(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path, manifest_reverse=True)
    summary = verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))
    assert summary["identity"]["reorder_observed"] is True
    assert summary["passed"] is True
