"""Small DataProto adapter for the four controlled pairwise training modes."""

import math

import torch

from permstudy.advantages import compute_controlled_advantage


MODES = {"global_only", "eis", "alc", "pa_controlled"}
LABELS = {"global_only": "Global-only", "eis": "EIS", "alc": "ALC",
          "pa_controlled": "PA-Controlled-NoGate"}


def _gate(config):
    enabled = config.get("sigma_gate_enabled", False)
    threshold = config.get("sigma_gate_threshold")
    source = config.get("sigma_gate_source", "unspecified")
    if not isinstance(enabled, bool):
        raise ValueError("sigma_gate_enabled must be boolean")
    if enabled:
        if config.get("mode") != "pa_controlled":
            raise ValueError("sigma gate is only supported for pa_controlled")
        if threshold is None or not math.isfinite(float(threshold)) or float(threshold) <= 0:
            raise ValueError("enabled sigma gate requires a positive finite threshold")
        if not isinstance(source, str) or not source.strip() or source.strip().lower() == "unspecified":
            raise ValueError("enabled sigma gate requires an explicit source")
        return float(threshold), source
    if threshold is not None:
        raise ValueError("disabled sigma gate requires threshold=null")
    return None, None


def validate_controlled_training_config(config):
    """Reject incompatible runtime choices before the first generation/update."""
    method = config.get("permutation_training") or {}
    mode = method.get("mode", "official")
    if mode == "official":
        return
    if mode not in MODES:
        raise ValueError(f"unknown permutation_training.mode: {mode!r}")
    _gate(method)
    algorithm = config.get("algorithm") or {}
    estimator = algorithm.get("adv_estimator", "grpo")
    if getattr(estimator, "value", estimator) != "grpo":
        raise ValueError("controlled permutation training requires algorithm.adv_estimator=grpo")
    if algorithm.get("use_kl_in_reward", False):
        raise ValueError("controlled reward uses actor KL loss; algorithm.use_kl_in_reward must be false")
    if (config.get("grouping") or {}).get("identity_mode") != "explicit":
        raise ValueError("controlled permutation training requires grouping.identity_mode=explicit")
    rollout = ((config.get("actor_rollout_ref") or {}).get("rollout") or {})
    if rollout.get("name") != "vllm" or rollout.get("mode", "sync") != "sync":
        raise ValueError("controlled permutation training requires synchronous vLLM")
    if (config.get("reward_model") or {}).get("reward_manager") != "batch":
        raise ValueError("controlled permutation training requires the batch reward manager")
    data = config.get("data") or {}
    if data.get("shuffle", True):
        raise ValueError("controlled pairwise rows require data.shuffle=false to retain complete AB/BA batches")
    batch_size = data.get("train_batch_size", 0)
    if not isinstance(batch_size, int) or batch_size <= 0 or batch_size % 2:
        raise ValueError("controlled pairwise training requires a positive even train_batch_size")


def apply_controlled_advantage(data, config):
    """Use scalar outcomes once, then broadcast the result through token masks."""
    mode = config.get("mode")
    if mode not in MODES:
        raise ValueError(f"unknown controlled advantage mode: {mode!r}")
    threshold, source = _gate(config)
    missing = [key for key in ("pair_id", "permutation_id", "rollout_slot", "base_reward", "consistency")
               if key not in data.non_tensor_batch]
    if missing:
        raise ValueError(f"controlled advantage is missing identity/reward components: {missing}")
    rewards = data.batch["token_level_rewards"]
    mask = data.batch["response_mask"].to(device=rewards.device, dtype=rewards.dtype)
    if rewards.ndim != 2 or rewards.shape != mask.shape or not rewards.shape[0]:
        raise ValueError("rewards and response_mask must have the same nonempty [batch, tokens] shape")
    if not torch.isfinite(rewards).all() or not torch.isfinite(mask).all():
        raise ValueError("controlled rewards and mask must be finite")
    if not torch.all((mask == 0) | (mask == 1)) or not torch.all(mask.sum(-1) > 0):
        raise ValueError("response_mask must be binary with at least one token per response")
    if torch.any(rewards[mask == 0] != 0):
        raise ValueError("reward on a masked-out token is invalid")
    base = torch.as_tensor(data.non_tensor_batch["base_reward"], device=rewards.device, dtype=rewards.dtype)
    consistency = torch.as_tensor(data.non_tensor_batch["consistency"], device=rewards.device, dtype=rewards.dtype)
    total = (rewards * mask).sum(-1)
    if base.shape != total.shape or consistency.shape != total.shape:
        raise ValueError("reward component dimensions must match the response batch")
    if not torch.isfinite(base).all() or not torch.isfinite(consistency).all():
        raise ValueError("reward components must be finite")
    if mode != "pa_controlled" and torch.any(consistency != 0):
        raise ValueError("Global/EIS/ALC require zero consistency reward")
    if not torch.allclose(total, base + consistency, atol=1e-5, rtol=1e-5):
        raise ValueError("scalar outcomes disagree with base_reward + consistency")
    result = compute_controlled_advantage(
        total if mode == "pa_controlled" else base,
        data.non_tensor_batch["pair_id"], data.non_tensor_batch["permutation_id"],
        data.non_tensor_batch["rollout_slot"], mode=mode, eps=config.get("eps", 1e-6),
        sigma_gate_threshold=threshold, sigma_gate_source=source,
    )
    advantages = result.advantage.unsqueeze(-1) * mask
    if not torch.isfinite(advantages).all():
        raise ValueError("controlled advantages contain NaN/Inf")
    data.batch["advantages"] = advantages
    data.batch["returns"] = advantages.clone()
    data.meta_info["controlled_method"] = "PA-Controlled-Gated" if threshold is not None else LABELS[mode]
    data.meta_info["controlled_advantage_metrics"] = {
        "adv/global_std": float(result.global_advantage.std(correction=0).item()),
        "adv/local_std": float(result.local_advantage.std(correction=0).item()),
        "adv/final_std": float(result.advantage.std(correction=0).item()),
        "adv/final_rms": float(result.advantage.square().mean().sqrt().item()),
        "adv/local_weight_mean": float(result.local_weight.mean().item()),
        "adv/gate_fraction": float(result.gate_mask.float().mean().item()),
    }
    return data
