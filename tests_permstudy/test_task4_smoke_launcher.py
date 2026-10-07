"""Tests for the Task 4 foreground, fail-fast real-GPU smoke launcher.

The launcher is a Linux shell script executed on AutoDL, so these tests pin its
contract two ways: textual guarantees that specific fixed settings can never be
dropped, and a real Hydra composition that proves every override the launcher
passes actually resolves against the repository's `ppo_trainer` config.
"""

from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts" / "run_task4_real_gpu_smoke.sh"

OVERRIDE_BLOCK = re.compile(r"TRAINER_OVERRIDES=\(\n(.*?)\n\)\n", re.S)

# Values only ever set at runtime on the GPU host; replaced with inert paths for composition.
SUBSTITUTIONS = {
    "${TASK4_RUN_DIR}": "/tmp/task4_run_001",
    "${TASK4_RUN_ID}": "task4_run_001",
    "${TINY_DATASET}": "/tmp/task4_tiny_8pairs.parquet",
    "${MODEL_PATH}": "/tmp/Qwen3-8B",
    "${PROJECT_ROOT}": REPO_ROOT.as_posix(),
    "$TASK4_RUN_DIR": "/tmp/task4_run_001",
    "$TASK4_RUN_ID": "task4_run_001",
    "$TINY_DATASET": "/tmp/task4_tiny_8pairs.parquet",
    "$MODEL_PATH": "/tmp/Qwen3-8B",
    "$PROJECT_ROOT": REPO_ROOT.as_posix(),
}


def launcher_text() -> str:
    assert LAUNCHER.is_file(), f"missing task4 launcher: {LAUNCHER}"
    return LAUNCHER.read_text(encoding="utf-8")


def extract_overrides() -> list[str]:
    text = launcher_text()
    match = OVERRIDE_BLOCK.search(text)
    assert match is not None, "launcher must declare a TRAINER_OVERRIDES=( ... ) array"
    overrides = []
    for raw in match.group(1).splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if len(line) >= 2 and line[0] == line[-1] and line[0] in "\"'":
            line = line[1:-1]
        for source, replacement in SUBSTITUTIONS.items():
            line = line.replace(source, replacement)
        assert "$" not in line, f"unsubstituted variable left in override: {line}"
        overrides.append(line)
    return overrides


# ---------------------------------------------------------------------------
# foreground, fail-fast behaviour
# ---------------------------------------------------------------------------


def test_launcher_is_foreground_and_fail_fast():
    text = launcher_text()
    assert "set -euo pipefail" in text
    assert "nohup" not in text


def test_launcher_requires_every_run_input():
    text = launcher_text()
    for name in ("MODEL_PATH", "TINY_DATASET", "TASK4_RUN_ID", "TASK4_RUN_DIR"):
        assert f"${{{name}:?" in text, f"launcher must fail fast when {name} is unset"


def test_launcher_refuses_a_run_dir_with_prior_evidence():
    text = launcher_text()
    assert "TASK4_RUN_DIR" in text
    assert "refus" in text.lower()
    assert re.search(r"ls -A\s+\"?\$TASK4_RUN_DIR", text), "launcher must detect prior evidence in the run dir"


def test_launcher_creates_the_run_dir_before_writing_evidence():
    text = launcher_text()
    assert text.index("mkdir -p \"$TASK4_RUN_DIR\"") < text.index("reward_log_window.json")


def test_launcher_preserves_the_trainer_exit_code():
    text = launcher_text()
    assert "set -o pipefail" in text
    assert "PIPESTATUS[0]" in text
    assert re.search(r"exit\s+\"?\$\{?TRAINER", text), "launcher must exit with the trainer's own status"


def test_launcher_captures_train_log_and_keeps_running_in_the_foreground():
    text = launcher_text()
    assert "train.log" in text
    assert "tee" in text


def test_provenance_capture_cannot_abort_the_run_before_training():
    """A failing git/pip/python probe must be recorded, not kill the GPU run.

    Regression: under `set -e`, `git rev-parse HEAD` returning non-zero (for
    example git's safe.directory refusal) aborted the launcher before the
    trainer command was even written, masking the real exit status.
    """
    text = launcher_text()
    blocks = {
        "environment.txt": re.search(r"\{\n(.*?)\n\} > \"\$TASK4_RUN_DIR/environment\.txt\"", text, re.S),
        "git_state.txt": re.search(r"\{\n(.*?)\n\} > \"\$TASK4_RUN_DIR/git_state\.txt\"", text, re.S),
    }
    checked = 0
    for name, match in blocks.items():
        assert match is not None, f"missing provenance capture block for {name}"
        for raw in match.group(1).splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            probes = line.startswith(("git ", "pip ")) or "$(python" in line
            if probes:
                checked += 1
                assert "||" in line, (
                    f"{name} probe can abort the run under `set -e` before the trainer starts: {line}"
                )
    assert checked >= 4, f"expected to inspect several probes, saw {checked}"


# ---------------------------------------------------------------------------
# fixed smoke settings
# ---------------------------------------------------------------------------


def test_launcher_selects_explicit_identity_and_synchronous_vllm():
    text = launcher_text()
    assert "grouping.identity_mode=explicit" in text
    assert "actor_rollout_ref.rollout.name=vllm" in text
    assert "actor_rollout_ref.rollout.mode=sync" in text
    assert "identity_mode=legacy_index" not in text
    assert "rollout.mode=async" not in text


def test_launcher_pins_the_two_step_smoke_schedule():
    text = launcher_text()
    assert "trainer.total_training_steps=2" in text
    assert "trainer.save_freq=1" in text
    assert "data.train_batch_size=16" in text
    assert "actor_rollout_ref.rollout.n=2" in text


def test_launcher_isolates_tensorboard_per_run():
    text = launcher_text()
    assert 'TENSORBOARD_DIR="$TASK4_RUN_DIR/tensorboard"' in text
    assert "export TENSORBOARD_DIR" in text


def test_launcher_records_the_reward_log_window_before_training():
    text = launcher_text()
    offset = text.index("reward_log_window.json")
    assert offset < text.index("main_ppo"), "reward-log window must be recorded before the trainer starts"
    assert "judge_qwen3_8b.log" in text
    assert "start_offset_bytes" in text


def test_launcher_keeps_the_required_rollout_and_reward_overrides():
    overrides = extract_overrides()
    joined = "\n".join(overrides)
    for required in (
        "data.filter_overlong_prompts=false",
        "data.truncation=error",
        "data.shuffle=false",
        "trainer.balance_batch=true",
        "trainer.val_before_train=false",
        "trainer.resume_mode=disable",
        "trainer.test_freq=-1",
        "actor_rollout_ref.actor.strategy=fsdp2",
        "actor_rollout_ref.model.target_modules=all-linear",
        "reward_model.reward_manager=batch",
        "custom_reward_function.name=compute_score",
    ):
        assert required in joined, f"missing required override: {required}"


# ---------------------------------------------------------------------------
# real Hydra composition
# ---------------------------------------------------------------------------


@pytest.mark.skipif(importlib.util.find_spec("hydra") is None, reason="requires hydra-core")
def test_every_launcher_override_resolves_and_forces_explicit_identity():
    import hydra
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    overrides = extract_overrides()
    assert overrides, "launcher must pass overrides to the trainer"

    key_overrides = [override.split("=", 1)[0] for override in overrides]

    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config"), version_base=None):
            cfg = compose(config_name="ppo_trainer", overrides=overrides)
    finally:
        GlobalHydra.instance().clear()

    assert cfg.grouping.identity_mode == "explicit"
    for key in key_overrides:
        assert OmegaConf.select(cfg, key) is not None, f"launcher override does not resolve: {key}"
    assert hydra.__version__
