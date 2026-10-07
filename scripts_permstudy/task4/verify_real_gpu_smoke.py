"""Verify the evidence produced by a Task 4 real vLLM / GPU / GRPO smoke run.

The verifier parses only artifacts that the run actually produced. It never
substitutes an expected value for missing evidence: absent or malformed inputs
raise ``EvidenceError``, and a present-but-failing gate raises
``VerificationFailed``. Both exit non-zero.

Evidence sources, all under ``--run-dir`` unless overridden:

``train.log``                trainer console output (metrics, vLLM init, warnings)
``reward_log_window.json``   {"reward_log_path", "start_offset_bytes"} written by the launcher
``dataset_manifest.json``    Task 1 fixture manifest (canonical pair/permutation order)
``rollouts/{step}.jsonl``    one file per optimization step, one JSON record per generated row
``checkpoints/``             ``global_step_N`` directories and ``latest_checkpointed_iteration.txt``
``checkpoints/global_step_2/actor/hf/adapter_model.safetensors``  merged LoRA adapter
``controlled_eval.json``     controlled-evaluation metadata produced by the eval step
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

FALLBACK_MARKER = "fallback to original GRPO"

IDENTITY_ERROR_MARKERS = (
    "RolloutIdentityAlignmentError",
    "IdentityMetadataError",
    "generation output reordered rollout identity",
    "generation output changed rollout batch size",
    "generation output did not preserve",
    "does not match rollout identity order",
    "requires extra_info metadata",
)

PAIR_BASELINE_METRICS = (
    "pair_baseline/mean_of_pair_means",
    "pair_baseline/mean_of_pair_stds",
    "pair_baseline/std_of_pair_stds",
)

PER_STEP_METRICS = ("actor/pg_loss", "actor/grad_norm", "actor/kl_loss")

VLLM_INIT_MARKERS = ("vLLM", "vllm")

_METRIC_RE = re.compile(
    r"([A-Za-z][A-Za-z0-9_./]*)\s*[:=]\s*(-?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?|nan|NaN|inf|-inf)\b"
)
_BASE_RE = re.compile(r"BASE \| i=(\d+) \| pair_id=(\S+?) \| perm=(\d+)")
_UNPAIRED_RE = re.compile(r"PAIR_UNPAIRED \| pair_id=(\S+?) \| rollout_slot=(\d+)")
_CHECK_RE = re.compile(r"PAIR_CHECK \| pair_id=(\S+?) \| rollout_slot=(\d+)")
# The log line prefix is "... | idx=<source index> | pid=... |"; note that PAIR_CHECK
# lines carry "idx0=" / "idx1=", which this pattern deliberately does not match.
_IDX_RE = re.compile(r"\bidx=(\d+)\b")


class EvidenceError(Exception):
    """Evidence is missing, unreadable, or internally inconsistent (contract error)."""


class VerificationFailed(Exception):
    """Evidence is readable but a hard PASS gate is not satisfied."""


# ---------------------------------------------------------------------------
# small readers
# ---------------------------------------------------------------------------


def _read_text(path: Path, label: str) -> str:
    if not path.is_file():
        raise EvidenceError(f"missing {label}: {path.name}")
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise EvidenceError(f"cannot read {label}: {path.name}") from exc


def _read_json(path: Path, label: str) -> Any:
    text = _read_text(path, label)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise EvidenceError(f"{label} is not valid JSON: {path.name}") from exc


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


# ---------------------------------------------------------------------------
# train log
# ---------------------------------------------------------------------------


def parse_train_log(text: str) -> dict[str, Any]:
    """Extract per-step metrics, fallback warnings, and identity failures."""
    metrics_by_step: dict[int, dict[str, float]] = {}
    current_step: int | None = None
    nan_or_inf: list[str] = []
    identity_errors: list[str] = []

    for line in text.splitlines():
        for marker in IDENTITY_ERROR_MARKERS:
            if marker in line:
                identity_errors.append(line.strip()[:240])
                break

        pairs = _METRIC_RE.findall(line)
        if not pairs:
            continue
        values: dict[str, float] = {}
        for key, raw in pairs:
            lowered = raw.lower()
            try:
                number = float(raw)
            except ValueError:
                continue
            if lowered in {"nan", "inf", "-inf"} or not math.isfinite(number):
                nan_or_inf.append(f"{key}={raw}")
            values[key] = number

        if "training/global_step" in values:
            current_step = int(values["training/global_step"])
        elif "step" in values:
            current_step = int(values["step"])
        if current_step is None:
            continue
        metrics_by_step.setdefault(current_step, {}).update(values)

    optimization_steps = sorted(
        step for step, values in metrics_by_step.items() if any(key in values for key in PER_STEP_METRICS)
    )
    return {
        "metrics_by_step": {str(step): metrics_by_step[step] for step in sorted(metrics_by_step)},
        "optimization_steps": optimization_steps,
        "fallback_detected": FALLBACK_MARKER in text,
        "nan_or_inf_metrics": nan_or_inf,
        "identity_errors": identity_errors,
        "vllm_mention_count": sum(text.count(marker) for marker in VLLM_INIT_MARKERS),
        "vllm_engine_initialized": ("vllm" in text.lower()) and ("engine" in text.lower()),
    }


# ---------------------------------------------------------------------------
# reward log window and identity
# ---------------------------------------------------------------------------


def read_reward_window(window_meta: Mapping[str, Any], run_dir: Path) -> tuple[str, int, int]:
    """Return (path, start, end) for the reward log, rejecting unusable windows."""
    path_value = window_meta.get("reward_log_path")
    start = window_meta.get("start_offset_bytes")
    if not isinstance(path_value, str) or not path_value:
        raise EvidenceError("reward_log_window.json has no reward_log_path")
    if not isinstance(start, int) or isinstance(start, bool) or start < 0:
        raise EvidenceError("reward_log_window.json has no valid start_offset_bytes")

    path = Path(path_value)
    if not path.is_absolute():
        path = run_dir / path
    if not path.is_file():
        raise EvidenceError(f"reward log recorded by the window does not exist: {path.name}")

    end = path.stat().st_size
    if end <= start:
        raise EvidenceError(
            f"reward log window is empty or was rotated/truncated: start={start} end={end}"
        )
    return str(path), start, end


def _reward_blocks(lines: Sequence[str]) -> list[dict[str, list[Any]]]:
    """Split the windowed reward log into one block per reward-function invocation."""
    blocks: list[dict[str, list[Any]]] = []
    current: dict[str, list[Any]] | None = None

    def ensure() -> dict[str, list[Any]]:
        nonlocal current
        if current is None:
            current = {"base": [], "checks": [], "unpaired": []}
        return current

    for line in lines:
        base = _BASE_RE.search(line)
        if base:
            index = int(base.group(1))
            if index == 0 and current is not None and current["base"]:
                blocks.append(current)
                current = None
            source_match = _IDX_RE.search(line)
            source_index = int(source_match.group(1)) if source_match else None
            ensure()["base"].append((index, base.group(2), int(base.group(3)), source_index))
            continue
        unpaired = _UNPAIRED_RE.search(line)
        if unpaired:
            ensure()["unpaired"].append((unpaired.group(1), int(unpaired.group(2))))
            continue
        check = _CHECK_RE.search(line)
        if check:
            ensure()["checks"].append((check.group(1), int(check.group(2))))

    if current is not None and any(current.values()):
        blocks.append(current)
    return blocks


def canonical_placement_sequence(dataset_manifest: Mapping[str, Any]) -> list[tuple[str, int]]:
    """The pre-balance batch order: each pair's rows, each repeated for ``rollout.n=2``."""
    pairs = dataset_manifest.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise EvidenceError("dataset manifest has no pairs")
    sequence: list[tuple[str, int]] = []
    for pair in pairs:
        pair_id = pair.get("pair_id")
        rows = pair.get("rows")
        if not isinstance(pair_id, str) or not isinstance(rows, list) or len(rows) != 2:
            raise EvidenceError("dataset manifest pair entries must carry pair_id and two rows")
        permutations = [row.get("permutation") for row in rows]
        if sorted(permutations) != [0, 1]:
            raise EvidenceError(f"dataset manifest pair {pair_id} does not carry permutations 0 and 1")
        for permutation in (0, 1):
            sequence.extend([(pair_id, permutation)] * 2)
    return sequence


def canonical_source_index_sequence(dataset_manifest: Mapping[str, Any]) -> list[int]:
    """The source row indices in pre-balance batch order, each repeated for ``rollout.n=2``."""
    pairs = dataset_manifest.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise EvidenceError("dataset manifest has no pairs")
    sequence: list[int] = []
    for pair in pairs:
        rows = pair.get("rows")
        if not isinstance(rows, list) or len(rows) != 2:
            raise EvidenceError("dataset manifest pair entries must carry pair_id and two rows")
        for row in sorted(rows, key=lambda entry: int(entry["permutation"])):
            row_index = row.get("row_index")
            if not isinstance(row_index, int) or isinstance(row_index, bool):
                raise EvidenceError(
                    f"dataset manifest row for pair {pair.get('pair_id')} has no integer row_index"
                )
            sequence.extend([row_index] * 2)
    return sequence


def analyse_identity(
    window_lines: Sequence[str],
    dataset_manifest: Mapping[str, Any],
    expected_blocks: int,
) -> dict[str, Any]:
    canonical = canonical_placement_sequence(dataset_manifest)
    canonical_sources = canonical_source_index_sequence(dataset_manifest)
    expected_keys = set(canonical)
    blocks = _reward_blocks(window_lines)

    unpaired_count = 0
    duplicate_keys = 0
    placement_reordered = False
    source_index_reordered = False
    source_indices_usable = bool(blocks)
    per_block: list[dict[str, Any]] = []

    for block in blocks:
        ordered = sorted(block["base"], key=lambda row: row[0])
        observed = [(pair_id, permutation) for _, pair_id, permutation, _ in ordered]
        observed_sources = [source for *_, source in ordered]
        checks = block["checks"]
        check_keys = list(checks)
        duplicates = len(check_keys) - len(set(check_keys))
        duplicate_keys += duplicates
        unpaired_count += len(block["unpaired"])

        # The source-index channel is only admissible when the logged indices are
        # exactly the fixture's own row indices. If the fixture dropped
        # ``extra_info.index``, ``my_reward/judge_qwen.py`` logs the post-reorder
        # batch position instead and this multiset cannot match.
        block_usable = (
            bool(observed_sources)
            and all(source is not None for source in observed_sources)
            and sorted(observed_sources) == sorted(canonical_sources)
        )
        source_indices_usable = source_indices_usable and block_usable
        block_source_reordered = block_usable and observed_sources != canonical_sources

        block_placement_reordered = bool(observed) and observed != canonical
        placement_reordered = placement_reordered or block_placement_reordered
        source_index_reordered = source_index_reordered or block_source_reordered

        per_block.append(
            {
                "base_rows": len(observed),
                "pair_checks": len(checks),
                "unpaired": len(block["unpaired"]),
                "duplicate_check_keys": duplicates,
                "missing_check_keys": sorted(str(key) for key in expected_keys - set(check_keys)),
                "unexpected_check_keys": sorted(str(key) for key in set(check_keys) - expected_keys),
                "placement_reordered": block_placement_reordered,
                "source_indices_usable": block_usable,
                "source_index_reordered": block_source_reordered,
            }
        )

    return {
        "reward_log_blocks": len(blocks),
        "expected_blocks": expected_blocks,
        "unpaired_count": unpaired_count,
        "duplicate_check_keys": duplicate_keys,
        "pair_check_count": sum(entry["pair_checks"] for entry in per_block),
        "expected_pair_check_count_per_block": len(expected_keys),
        # The plan's §3.2 hook: reward log source-index order must differ from the
        # canonical interleaved input order for at least one step.
        "reorder_observed": source_index_reordered,
        "placement_reordered": placement_reordered,
        "source_index_reordered": source_index_reordered,
        "source_indices_usable": source_indices_usable,
        "canonical_placement_count": len(canonical),
        "blocks": per_block,
    }


# ---------------------------------------------------------------------------
# rollouts, checkpoints, LoRA, controlled eval
# ---------------------------------------------------------------------------


def inspect_rollouts(rollout_dir: Path, expected_steps: Sequence[int]) -> dict[str, Any]:
    if not rollout_dir.is_dir():
        raise EvidenceError(f"missing rollout directory: {rollout_dir.name}")
    rows_per_step: dict[str, int] = {}
    records_seen = 0
    for step in expected_steps:
        path = rollout_dir / f"{step}.jsonl"
        if not path.is_file():
            raise EvidenceError(f"missing rollout file for step {step}: {path.name}")
        lines = [line for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
        if not lines:
            raise EvidenceError(f"rollout file for step {step} is empty")
        for line in lines:
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvidenceError(f"rollout file {path.name} contains a malformed record") from exc
            if int(record.get("step", -1)) != int(step):
                raise EvidenceError(f"rollout file {path.name} contains a record from another step")
            records_seen += 1
        rows_per_step[str(step)] = len(lines)
    return {"files": {str(step): f"{step}.jsonl" for step in expected_steps}, "rows_per_step": rows_per_step,
            "total_records": records_seen}


def inspect_checkpoints(checkpoint_dir: Path, expected_steps: Sequence[int]) -> dict[str, Any]:
    if not checkpoint_dir.is_dir():
        raise EvidenceError(f"missing checkpoint directory: {checkpoint_dir.name}")
    present = [step for step in expected_steps if (checkpoint_dir / f"global_step_{step}").is_dir()]
    latest_path = checkpoint_dir / "latest_checkpointed_iteration.txt"
    if not latest_path.is_file():
        raise EvidenceError("missing latest_checkpointed_iteration.txt")
    latest_raw = latest_path.read_text(encoding="utf-8", errors="replace").strip()
    try:
        latest = int(latest_raw)
    except ValueError as exc:
        raise EvidenceError(f"latest_checkpointed_iteration.txt is not an integer: {latest_raw!r}") from exc
    return {"expected_steps": list(expected_steps), "present_steps": present, "latest_iteration": latest}


def inspect_lora(adapter_path: Path) -> dict[str, Any]:
    if not adapter_path.is_file():
        raise EvidenceError(f"missing merged LoRA adapter: {adapter_path.name}")
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise EvidenceError("safetensors is required to inspect the merged adapter") from exc

    tensor_names: list[str] = []
    lora_b_names: list[str] = []
    all_finite = True
    nonzero_lora_b = False
    with safe_open(str(adapter_path), framework="numpy") as handle:
        for name in handle.keys():
            tensor_names.append(name)
            array = np.asarray(handle.get_tensor(name))
            if not np.all(np.isfinite(array)):
                all_finite = False
            if "lora_B" in name:
                lora_b_names.append(name)
                if array.size and float(np.abs(array).max()) > 0.0:
                    nonzero_lora_b = True
    return {
        "adapter": adapter_path.name,
        "tensor_count": len(tensor_names),
        "lora_b_tensor_count": len(lora_b_names),
        "lora_b_tensors": sorted(lora_b_names),
        "all_finite": all_finite,
        "nonzero_lora_b": nonzero_lora_b,
    }


def inspect_controlled_eval(
    summary_path: Path, expected_contract: str, expected_num_options: int, expected_num_samples: int
) -> dict[str, Any]:
    payload = _read_json(summary_path, "controlled-eval summary")
    if not isinstance(payload, Mapping):
        raise EvidenceError("controlled-eval summary must be a JSON object")
    for key in ("evaluation_contract", "num_options", "num_samples"):
        if key not in payload:
            raise EvidenceError(f"controlled-eval summary is missing {key}")
    return {
        "path": summary_path.name,
        "evaluation_contract": payload["evaluation_contract"],
        "num_options": payload["num_options"],
        "num_samples": payload["num_samples"],
        "expected_contract": expected_contract,
        "expected_num_options": expected_num_options,
        "expected_num_samples": expected_num_samples,
        "completed": bool(payload.get("completed", True)),
    }


# ---------------------------------------------------------------------------
# coordination
# ---------------------------------------------------------------------------


@dataclass
class VerifyPaths:
    run_dir: Path
    dataset_manifest: Path
    reward_log_window: Path
    train_log: Path
    rollout_dir: Path
    checkpoint_dir: Path
    merged_adapter: Path
    controlled_eval: Path


def verify_real_gpu_smoke(
    paths: VerifyPaths,
    *,
    expected_steps: Sequence[int] = (1, 2),
    expected_rows_per_step: int = 32,
    expected_eval_contract: str = "controlled",
    expected_num_options: int = 2,
    expected_num_samples: int = 4,
    write_summaries: bool = True,
) -> dict[str, Any]:
    run_dir = paths.run_dir
    if not run_dir.is_dir():
        raise EvidenceError(f"missing run directory: {run_dir.name}")

    dataset_manifest = _read_json(paths.dataset_manifest, "dataset manifest")
    if not isinstance(dataset_manifest, Mapping):
        raise EvidenceError("dataset manifest must be a JSON object")
    expected_pairs = len(dataset_manifest.get("pairs") or [])
    if expected_pairs <= 0:
        raise EvidenceError("dataset manifest contains no pairs")

    train_text = _read_text(paths.train_log, "train log")
    train = parse_train_log(train_text)

    window_meta = _read_json(paths.reward_log_window, "reward log window")
    if not isinstance(window_meta, Mapping):
        raise EvidenceError("reward_log_window.json must be a JSON object")
    reward_path, start, end = read_reward_window(window_meta, run_dir)
    with open(reward_path, "rb") as handle:
        handle.seek(start)
        window_bytes = handle.read(end - start)
    window_lines = window_bytes.decode("utf-8", errors="replace").splitlines()

    rollouts = inspect_rollouts(paths.rollout_dir, expected_steps)
    checkpoints = inspect_checkpoints(paths.checkpoint_dir, expected_steps)
    lora = inspect_lora(paths.merged_adapter)
    controlled_eval = inspect_controlled_eval(
        paths.controlled_eval, expected_eval_contract, expected_num_options, expected_num_samples
    )
    identity = analyse_identity(window_lines, dataset_manifest, expected_blocks=len(expected_steps))
    identity["window"] = {
        "start_offset_bytes": start,
        "end_offset_bytes": end,
        "window_bytes": end - start,
        "line_count": len(window_lines),
    }

    baseline = {
        name: next(
            (
                train["metrics_by_step"][step].get(name)
                for step in train["metrics_by_step"]
                if name in train["metrics_by_step"][step]
            ),
            None,
        )
        for name in PAIR_BASELINE_METRICS
    }

    failures: list[str] = []

    if train["optimization_steps"] != list(expected_steps):
        failures.append(
            f"optimization steps observed {train['optimization_steps']}, expected {list(expected_steps)}"
        )
    if train["fallback_detected"]:
        failures.append("trainer log reports fallback to original GRPO")
    if train["identity_errors"]:
        failures.append(f"{len(train['identity_errors'])} rollout identity error(s) in the trainer log")
    if train["nan_or_inf_metrics"]:
        failures.append(f"non-finite metric value(s): {sorted(set(train['nan_or_inf_metrics']))[:5]}")
    if not train["vllm_engine_initialized"]:
        failures.append("trainer log shows no vLLM engine initialization")
    for name, value in baseline.items():
        if value is None:
            failures.append(f"missing pair-baseline metric: {name}")

    for step in expected_steps:
        values = train["metrics_by_step"].get(str(step), {})
        for name in PER_STEP_METRICS:
            value = values.get(name)
            if value is None:
                failures.append(f"step {step}: missing metric {name}")
            elif not _finite(value):
                failures.append(f"step {step}: {name} is not finite ({value})")
        grad_norm = values.get("actor/grad_norm")
        if grad_norm is not None and _finite(grad_norm) and float(grad_norm) <= 0.0:
            failures.append(f"step {step}: actor/grad_norm is not positive ({grad_norm})")
        rows = rollouts["rows_per_step"].get(str(step))
        if rows != expected_rows_per_step:
            failures.append(f"step {step}: observed {rows} rollout rows, expected {expected_rows_per_step}")

    if identity["reward_log_blocks"] != identity["expected_blocks"]:
        failures.append(
            f"reward log window holds {identity['reward_log_blocks']} invocation block(s), "
            f"expected {identity['expected_blocks']}"
        )
    if identity["unpaired_count"]:
        failures.append(f"{identity['unpaired_count']} PAIR_UNPAIRED record(s) in the reward window")
    if identity["duplicate_check_keys"]:
        failures.append(f"{identity['duplicate_check_keys']} duplicate PAIR_CHECK key(s)")
    for index, block in enumerate(identity["blocks"]):
        if block["pair_checks"] != identity["expected_pair_check_count_per_block"]:
            failures.append(
                f"reward block {index}: {block['pair_checks']} PAIR_CHECK record(s), "
                f"expected {identity['expected_pair_check_count_per_block']}"
            )
        if block["missing_check_keys"]:
            failures.append(f"reward block {index}: unpaired rollout keys {block['missing_check_keys'][:4]}")
    if not identity["source_indices_usable"]:
        failures.append(
            "reward log source indices do not match the dataset manifest row indices, so the reorder evidence "
            "is vacuous: the fixture must carry extra_info.index, otherwise the logged idx= is the "
            "post-reorder batch position"
        )
    elif not identity["source_index_reordered"]:
        failures.append("no balance_batch reorder was observed in the reward log window")

    consistency_rate = train["metrics_by_step"].get(str(expected_steps[-1]), {}).get(
        "reward/consistency_unpaired_rate"
    )
    if consistency_rate is not None and _finite(consistency_rate) and float(consistency_rate) != 0.0:
        failures.append(f"reward/consistency_unpaired_rate is {consistency_rate}, expected 0")

    if checkpoints["present_steps"] != list(expected_steps):
        failures.append(
            f"checkpoint directories present {checkpoints['present_steps']}, expected {list(expected_steps)}"
        )
    if checkpoints["latest_iteration"] != int(expected_steps[-1]):
        failures.append(
            f"latest_checkpointed_iteration is {checkpoints['latest_iteration']}, "
            f"expected {int(expected_steps[-1])}"
        )

    if not lora["all_finite"]:
        failures.append("merged LoRA adapter contains non-finite tensors")
    if lora["lora_b_tensor_count"] == 0:
        failures.append("merged LoRA adapter contains no lora_B tensor")
    elif not lora["nonzero_lora_b"]:
        failures.append("every lora_B tensor is zero: no optimizer update was saved")

    if not controlled_eval["completed"]:
        failures.append("controlled evaluation did not complete")
    if controlled_eval["evaluation_contract"] != expected_eval_contract:
        failures.append(
            f"controlled evaluation contract is {controlled_eval['evaluation_contract']!r}, "
            f"expected {expected_eval_contract!r}"
        )
    if int(controlled_eval["num_options"]) != expected_num_options:
        failures.append(
            f"controlled evaluation num_options is {controlled_eval['num_options']}, expected {expected_num_options}"
        )
    if int(controlled_eval["num_samples"]) != expected_num_samples:
        failures.append(
            f"controlled evaluation num_samples is {controlled_eval['num_samples']}, "
            f"expected {expected_num_samples}"
        )

    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "dataset_manifest": {"path": paths.dataset_manifest.name, "pairs": expected_pairs},
        "optimization_steps": train["optimization_steps"],
        "expected_optimization_steps": list(expected_steps),
        "fallback_detected": train["fallback_detected"],
        "nan_or_inf_metrics": sorted(set(train["nan_or_inf_metrics"])),
        "identity_errors": train["identity_errors"],
        "vllm_engine_initialized": train["vllm_engine_initialized"],
        "pair_baseline_metrics": baseline,
        "metrics_by_step": train["metrics_by_step"],
        "rollouts": rollouts,
        "identity": identity,
        "checkpoints": checkpoints,
        "lora": lora,
        "controlled_eval": controlled_eval,
        "failures": failures,
        "passed": not failures,
    }

    if write_summaries:
        _write_json(run_dir / "metrics_summary.json", summary)
        _write_json(run_dir / "identity_pairing_summary.json", identity)
        _write_json(run_dir / "checkpoint_manifest.json", checkpoints)
        _write_json(run_dir / "lora_update.json", lora)

    if failures:
        raise VerificationFailed("; ".join(failures))
    return summary


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify Task 4 real-GPU smoke evidence.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset-manifest", default=None)
    parser.add_argument("--reward-log-window", default=None)
    parser.add_argument("--train-log", default=None)
    parser.add_argument("--rollout-dir", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--merged-adapter", default=None)
    parser.add_argument("--controlled-eval", default=None)
    parser.add_argument("--expected-eval-contract", default="controlled")
    parser.add_argument("--expected-num-options", type=int, default=2)
    parser.add_argument("--expected-num-samples", type=int, default=4)
    return parser


def resolve_paths(args: argparse.Namespace) -> VerifyPaths:
    run_dir = Path(args.run_dir)
    return VerifyPaths(
        run_dir=run_dir,
        dataset_manifest=Path(args.dataset_manifest) if args.dataset_manifest else run_dir / "dataset_manifest.json",
        reward_log_window=Path(args.reward_log_window)
        if args.reward_log_window
        else run_dir / "reward_log_window.json",
        train_log=Path(args.train_log) if args.train_log else run_dir / "train.log",
        rollout_dir=Path(args.rollout_dir) if args.rollout_dir else run_dir / "rollouts",
        checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else run_dir / "checkpoints",
        merged_adapter=Path(args.merged_adapter)
        if args.merged_adapter
        else run_dir / "checkpoints" / "global_step_2" / "actor" / "hf" / "adapter_model.safetensors",
        controlled_eval=Path(args.controlled_eval)
        if args.controlled_eval
        else run_dir / "controlled_eval.json",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        summary = verify_real_gpu_smoke(
            resolve_paths(args),
            expected_eval_contract=args.expected_eval_contract,
            expected_num_options=args.expected_num_options,
            expected_num_samples=args.expected_num_samples,
        )
    except EvidenceError as error:
        print(f"evidence error: {error}", file=sys.stderr)
        return 2
    except VerificationFailed as error:
        print(f"verification failed: {error}", file=sys.stderr)
        return 3
    print(
        "PASS steps={steps} pair_checks={checks} reorder_observed={reorder} nonzero_lora_b={lora}".format(
            steps=summary["optimization_steps"],
            checks=summary["identity"]["pair_check_count"],
            reorder=summary["identity"]["reorder_observed"],
            lora=summary["lora"]["nonzero_lora_b"],
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
