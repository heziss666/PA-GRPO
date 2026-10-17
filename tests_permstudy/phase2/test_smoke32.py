"""Small synthetic tests for the CPU-prepared 32GB execution view."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from permstudy.data_pipeline.canonical import sha256_hex
from permstudy.data_pipeline.trainer_export import build_trainer_rows
from scripts_permstudy.data import _common as io
from tests_permstudy.data_pipeline.test_trainer_export import _records


def synthetic_rows():
    original = build_trainer_rows(*_records())
    rows = []
    for i in range(4):
        for row in deepcopy(original):
            row["extra_info"]["original_question_id"] += f"-{i}"
            row["extra_info"]["pair_id"] = sha256_hex(
                (row["extra_info"]["source"] + str(i)).encode()
            )
            rows.append(row)
    return sorted(
        rows,
        key=lambda r: (
            r["extra_info"]["original_question_id"],
            r["extra_info"]["permutation_id"],
        ),
    )


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        text = messages[1]["content"]
        return [1] * (1025 if "OVERLONG" in text else 200)


def test_short_selection_is_grouped_balanced_and_does_not_truncate_or_relabel():
    from scripts_permstudy.phase2.smoke32 import select_short_pairs

    rows = synthetic_rows()
    rows[0]["prompt"][1]["content"] = "OVERLONG\n" + rows[0]["prompt"][1]["content"]
    before = deepcopy(rows)
    selected, lengths = select_short_pairs(rows, Tokenizer())
    assert rows == before
    assert len(selected) == 8 and len(lengths) == 8
    assert rows[0]["extra_info"]["pair_id"] not in {
        r["extra_info"]["pair_id"] for r in selected
    }
    for start in (0, 4):
        chunk = selected[start : start + 4]
        assert [r["extra_info"]["source"] for r in chunk] == [
            "math",
            "math",
            "reclor",
            "reclor",
        ]
        assert [r["extra_info"]["permutation_id"] for r in chunk] == [0, 1, 0, 1]
    by_identity = {
        (r["extra_info"]["pair_id"], r["extra_info"]["permutation_id"]): r for r in rows
    }
    for r in selected:
        source = by_identity[
            (r["extra_info"]["pair_id"], r["extra_info"]["permutation_id"])
        ]
        assert r["extra_info"] == source["extra_info"]
        assert r["reward_model"] == source["reward_model"]
        assert (
            r["prompt"][1]["content"].split("Which response", 1)[0]
            == source["prompt"][1]["content"].split("Which response", 1)[0]
        )
    assert select_short_pairs(rows, Tokenizer()) == (selected, lengths)


def test_not_enough_complete_short_pairs_stops():
    from scripts_permstudy.phase2.smoke32 import select_short_pairs

    with pytest.raises(io.IntegrityError):
        select_short_pairs(build_trainer_rows(*_records()), Tokenizer())


def test_config_keeps_fp32_actor_shared_reference_and_cpu_sdpa():
    from pathlib import Path
    from hydra import compose, initialize_config_dir
    from permstudy.trainer_advantage import validate_controlled_training_config
    from scripts_permstudy.phase2.smoke32 import training_overrides

    with initialize_config_dir(
        config_dir=str(Path("verl/trainer/config").resolve()), version_base=None
    ):
        cfg = compose(
            config_name="ppo_trainer",
            overrides=training_overrides("model", "data", "run"),
        )
    validate_controlled_training_config(cfg)
    assert cfg.data.train_batch_size == 4
    assert cfg.data.max_prompt_length == 1024
    assert cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 4
    assert cfg.actor_rollout_ref.actor.fsdp_config.model_dtype == "fp32"
    assert cfg.actor_rollout_ref.ref.fsdp_config.model_dtype == "bfloat16"
    assert cfg.actor_rollout_ref.model.override_config.attn_implementation == "sdpa"
    assert cfg.actor_rollout_ref.model.use_remove_padding is False
    assert cfg.actor_rollout_ref.model.lora_rank == 32
    assert cfg.actor_rollout_ref.rollout.n == 2
    assert cfg.trainer.total_training_steps == 2


def test_prepare_rejects_wrong_stage_A_hash_before_writing(tmp_path):
    from scripts_permstudy.phase2.smoke32 import prepare

    source = tmp_path / "source.parquet"
    source.write_bytes(b"synthetic wrong source")
    with pytest.raises(io.IntegrityError):
        prepare(SimpleNamespace(source="source.parquet"), tmp_path)
    assert list(tmp_path.iterdir()) == [source]


def test_launch_without_explicit_gpu_authorization_stops(tmp_path):
    from scripts_permstudy.phase2.smoke32 import launch

    with pytest.raises(io.PhaseBoundaryError):
        launch(SimpleNamespace(authorize_gpu=False), tmp_path)
    assert not list(tmp_path.iterdir())
