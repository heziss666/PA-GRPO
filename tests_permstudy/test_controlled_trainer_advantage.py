"""Controlled response-level signal survives the real Trainer/DataProto boundary."""

import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def _api():
    from permstudy import trainer_advantage
    return trainer_advantage


def _batch(lengths=(1, 3, 2, 4), mode="global_only"):
    base = torch.tensor([1.0, 1.0, -1.0, -1.0])
    consistency = torch.ones(4) if mode == "pa_controlled" else torch.zeros(4)
    mask = torch.zeros(4, max(lengths))
    rewards = torch.zeros_like(mask)
    for i, length in enumerate(lengths):
        mask[i, :length] = 1
        rewards[i, length - 1] = base[i] + consistency[i]
    return SimpleNamespace(
        batch={"response_mask": mask, "token_level_rewards": rewards},
        non_tensor_batch={
            "pair_id": np.array(["p"] * 4, dtype=object),
            "permutation_id": np.array([0, 0, 1, 1]),
            "rollout_slot": np.array([0, 1, 0, 1]),
            "base_reward": base.numpy(), "consistency": consistency.numpy(),
        },
        meta_info={},
    )


def _config(mode="global_only"):
    return {"mode": mode, "eps": 1e-6, "consistency_coef": 1.0,
            "sigma_gate_enabled": False, "sigma_gate_threshold": None,
            "sigma_gate_source": "unspecified"}


@pytest.mark.parametrize("mode", ["global_only", "eis", "alc", "pa_controlled"])
def test_scalar_signal_is_length_independent_and_masked(mode):
    data = _batch(mode=mode)
    result = _api().apply_controlled_advantage(data, _config(mode))
    scalar = result.batch["advantages"].sum(-1) / result.batch["response_mask"].sum(-1)
    torch.testing.assert_close(scalar, torch.tensor([1., 1., -1., -1.]), atol=2e-6, rtol=0)
    assert not result.batch["advantages"][result.batch["response_mask"] == 0].any()
    assert torch.isfinite(result.batch["advantages"]).all()
    assert result.meta_info["controlled_advantage_metrics"]["adv/final_rms"] > 0


def test_pa_default_records_nogate_and_original_reward_components():
    result = _api().apply_controlled_advantage(_batch(mode="pa_controlled"), _config("pa_controlled"))
    assert result.meta_info["controlled_method"] == "PA-Controlled-NoGate"
    assert result.meta_info["controlled_advantage_metrics"]["adv/gate_fraction"] == 0


@pytest.mark.parametrize("change", ["missing_component", "consistency_contamination", "total_mismatch", "identity_loss"])
def test_malformed_controlled_batches_fail_instead_of_fallback(change):
    data = _batch()
    if change == "missing_component":
        del data.non_tensor_batch["base_reward"]
    elif change == "consistency_contamination":
        data.non_tensor_batch["consistency"] = np.ones(4)
    elif change == "total_mismatch":
        data.batch["token_level_rewards"][0, 0] = 9
    else:
        del data.non_tensor_batch["rollout_slot"]
    with pytest.raises(ValueError):
        _api().apply_controlled_advantage(data, _config())


def test_reorder_keeps_identical_per_identity_advantages():
    data = _batch()
    order = np.array([3, 0, 2, 1])
    for key, values in data.batch.items():
        data.batch[key] = values[order]
    for key, values in data.non_tensor_batch.items():
        data.non_tensor_batch[key] = values[order]
    result = _api().apply_controlled_advantage(data, _config())
    scalar = result.batch["advantages"].sum(-1) / result.batch["response_mask"].sum(-1)
    torch.testing.assert_close(scalar, torch.tensor([-1., 1., -1., 1.]), atol=2e-6, rtol=0)


@pytest.mark.skipif(importlib.util.find_spec("ray") is None, reason="requires the existing verl environment")
def test_real_dataproto_dispatch_bypasses_official_two_stage_path():
    from verl import DataProto
    from verl.trainer.ppo.ray_trainer import compute_advantage
    data = _batch()
    proto = DataProto.from_dict(tensors=data.batch, non_tensors=data.non_tensor_batch)
    result = compute_advantage(proto, "grpo", config={}, permutation_config=_config())
    scalar = result.batch["advantages"].sum(-1) / result.batch["response_mask"].sum(-1)
    torch.testing.assert_close(scalar, torch.tensor([1., 1., -1., -1.]), atol=2e-6, rtol=0)
    assert result.meta_info["controlled_method"] == "Global-only"
    assert "pair_baseline_metrics" not in result.meta_info


def test_runtime_config_accepts_nogate_and_rejects_shuffle_or_reward_kl():
    config = {
        "permutation_training": _config("pa_controlled"),
        "algorithm": {"adv_estimator": "grpo", "use_kl_in_reward": False},
        "grouping": {"identity_mode": "explicit"},
        "actor_rollout_ref": {"rollout": {"name": "vllm", "mode": "sync"}},
        "reward_model": {"reward_manager": "batch"},
        "data": {"shuffle": False, "train_batch_size": 16},
    }
    _api().validate_controlled_training_config(config)
    config["data"]["shuffle"] = True
    with pytest.raises(ValueError, match="shuffle"):
        _api().validate_controlled_training_config(config)
    config["data"]["shuffle"] = False
    config["algorithm"]["use_kl_in_reward"] = True
    with pytest.raises(ValueError, match="KL"):
        _api().validate_controlled_training_config(config)
