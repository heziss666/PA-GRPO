"""Pure response-level advantage functions used by fidelity audits."""

from dataclasses import dataclass
from math import isfinite
from numbers import Integral
from typing import Hashable, Sequence

import torch


@dataclass(frozen=True)
class AdvantageResult:
    """Per-sample advantage and the group statistics used to produce it."""

    advantage: torch.Tensor
    group_mean: torch.Tensor
    group_std: torch.Tensor
    gate_mask: torch.Tensor


@dataclass(frozen=True)
class ControlledAdvantageResult:
    """Response-level controlled advantage and its ungated diagnostics."""

    advantage: torch.Tensor
    global_advantage: torch.Tensor
    local_advantage: torch.Tensor
    local_weight: torch.Tensor
    group_mean: torch.Tensor
    group_std: torch.Tensor
    local_mean: torch.Tensor
    local_std: torch.Tensor
    gate_mask: torch.Tensor


def _prepare_inputs(
    rewards: torch.Tensor | Sequence[float], group_ids: Sequence[Hashable]
) -> tuple[torch.Tensor, list[Hashable]]:
    rewards_tensor = torch.as_tensor(rewards)
    if rewards_tensor.ndim != 1:
        raise ValueError(f"rewards must be one-dimensional, got shape {tuple(rewards_tensor.shape)}")
    if not rewards_tensor.is_floating_point():
        rewards_tensor = rewards_tensor.float()
    groups = list(group_ids)
    if len(groups) != rewards_tensor.numel():
        raise ValueError(f"rewards/group_ids length mismatch: {rewards_tensor.numel()} != {len(groups)}")
    return rewards_tensor, groups


def _positions_by_group(group_ids: Sequence[Hashable]) -> dict[Hashable, list[int]]:
    positions: dict[Hashable, list[int]] = {}
    for position, group_id in enumerate(group_ids):
        try:
            positions.setdefault(group_id, []).append(position)
        except TypeError as exc:
            raise TypeError(f"group_id at position {position} is not hashable: {group_id!r}") from exc
    return positions


def _population_stats(
    values: torch.Tensor, group_ids: Sequence[Hashable]
) -> tuple[torch.Tensor, torch.Tensor]:
    means = torch.empty_like(values)
    stds = torch.empty_like(values)
    for positions in _positions_by_group(group_ids).values():
        index = torch.tensor(positions, device=values.device, dtype=torch.long)
        group_values = values.index_select(0, index)
        mean = group_values.mean()
        variance = ((group_values - mean) ** 2).mean()
        means.index_fill_(0, index, mean)
        stds.index_fill_(0, index, torch.sqrt(torch.clamp_min(variance, 0.0)))
    return means, stds


def global_advantage_paper(
    rewards: torch.Tensor | Sequence[float],
    group_ids: Sequence[Hashable],
    *,
    eps: float = 1e-6,
    sigma_gate_threshold: float | None = None,
) -> AdvantageResult:
    """Compute the paper-form response-level global advantage.

    Standard deviation is the population standard deviation (``correction=0``
    semantics). When configured, the sigma gate is applied per group using the
    paper's strict ``sigma < threshold`` condition.
    """

    if eps <= 0:
        raise ValueError("eps must be positive")
    if sigma_gate_threshold is not None and sigma_gate_threshold < 0:
        raise ValueError("sigma_gate_threshold must be non-negative or None")

    rewards_tensor, groups = _prepare_inputs(rewards, group_ids)
    means, stds = _population_stats(rewards_tensor, groups)
    advantage = (rewards_tensor - means) / (stds + eps)

    if sigma_gate_threshold is None:
        gate_mask = torch.zeros_like(rewards_tensor, dtype=torch.bool)
    else:
        gate_mask = stds < sigma_gate_threshold
        advantage = torch.where(gate_mask, torch.zeros_like(advantage), advantage)

    return AdvantageResult(advantage=advantage, group_mean=means, group_std=stds, gate_mask=gate_mask)

def global_advantage_official_compatible(
    rewards: torch.Tensor | Sequence[float],
    group_ids: Sequence[Hashable],
    response_mask: torch.Tensor,
    *,
    variance_floor: float = 1e-6,
    clip_value: float = 5.0,
) -> AdvantageResult:
    """Reproduce the current PA repository's two-stage GRPO path.

    This is a characterization helper, not the controlled implementation. It
    models the current sequence: group-center scalar outcomes, broadcast through
    the response mask, mean over the padded token dimension, then apply the PA
    population-standardized group baseline and clipping.
    """

    if variance_floor < 0:
        raise ValueError("variance_floor must be non-negative")
    if clip_value <= 0:
        raise ValueError("clip_value must be positive")

    rewards_tensor, groups = _prepare_inputs(rewards, group_ids)
    mask = torch.as_tensor(response_mask, device=rewards_tensor.device, dtype=rewards_tensor.dtype)
    if mask.ndim != 2 or mask.shape[0] != rewards_tensor.numel():
        raise ValueError(
            "response_mask must have shape [num_rewards, response_length], "
            f"got {tuple(mask.shape)}"
        )

    first_stage_mean = torch.empty_like(rewards_tensor)
    for positions in _positions_by_group(groups).values():
        index = torch.tensor(positions, device=rewards_tensor.device, dtype=torch.long)
        group_rewards = rewards_tensor.index_select(0, index)
        mean = group_rewards.mean() if len(positions) > 1 else rewards_tensor.new_tensor(0.0)
        first_stage_mean.index_fill_(0, index, mean)

    centered_token_returns = (rewards_tensor - first_stage_mean).unsqueeze(-1) * mask
    collapsed_returns = centered_token_returns.reshape(rewards_tensor.shape[0], -1).mean(dim=-1)

    means, raw_stds = _population_stats(collapsed_returns, groups)
    stds = torch.sqrt(torch.clamp(raw_stds**2, min=variance_floor))
    advantage = torch.clamp((collapsed_returns - means) / stds.clamp_min(1e-6), -clip_value, clip_value)
    gate_mask = torch.zeros_like(rewards_tensor, dtype=torch.bool)
    return AdvantageResult(advantage=advantage, group_mean=means, group_std=stds, gate_mask=gate_mask)


def compute_controlled_advantage(
    rewards: torch.Tensor | Sequence[float],
    pair_ids: Sequence[Hashable],
    permutation_ids: Sequence[int],
    rollout_slots: Sequence[int],
    *,
    mode: str,
    eps: float = 1e-6,
    sigma_gate_threshold: float | None = None,
    sigma_gate_source: str | None = None,
) -> ControlledAdvantageResult:
    """Compute population-standardized Global, EIS, ALC, or controlled PA.

    Each pair must contain permutations 0 and 1 exactly once for each rollout
    slot, with slots contiguous from zero. Global statistics use ``pair_id``;
    local statistics use ``(pair_id, permutation_id)``. EIS adds unit-weight
    local advantage; ALC weights it by the clipped subgroup-mean gap divided
    by global sigma plus ``eps``.

    Callers supply base rewards for Global/EIS/ALC and total base-plus-
    consistency rewards for PA. PA is ungated when no threshold is supplied.
    An explicit gate requires a positive finite threshold and a nonblank,
    specified source; only the final advantage is zeroed when global
    ``sigma < threshold``. There is no response-length weighting or clipping
    of the resulting advantage.
    """

    if mode not in ("global_only", "eis", "alc", "pa_controlled"):
        raise ValueError(f"unsupported controlled advantage mode: {mode!r}")
    if not isfinite(eps) or eps <= 0:
        raise ValueError("eps must be positive and finite")
    if mode == "pa_controlled":
        if sigma_gate_threshold is not None:
            if not isfinite(sigma_gate_threshold) or sigma_gate_threshold <= 0:
                raise ValueError("pa_controlled requires a positive finite sigma_gate_threshold when gated")
            if (not isinstance(sigma_gate_source, str) or not sigma_gate_source.strip()
                    or sigma_gate_source.strip().lower() == "unspecified"):
                raise ValueError("gated pa_controlled requires a nonblank, specified sigma_gate_source")
    elif sigma_gate_threshold is not None or sigma_gate_source is not None:
        raise ValueError("sigma gate configuration is only supported for pa_controlled")

    rewards_tensor, pairs = _prepare_inputs(rewards, pair_ids)
    if not torch.isfinite(rewards_tensor).all():
        raise ValueError("rewards must be finite")
    permutations = list(permutation_ids)
    slots = list(rollout_slots)
    for name, identities in (("permutation_ids", permutations), ("rollout_slots", slots)):
        if len(identities) != rewards_tensor.numel():
            raise ValueError(f"rewards/{name} length mismatch: {rewards_tensor.numel()} != {len(identities)}")

    pair_positions = _positions_by_group(pairs)
    paired_slots: dict[tuple[Hashable, int], dict[int, int]] = {}
    slots_by_pair: dict[Hashable, set[int]] = {}
    for position, (pair_id, permutation_id, slot) in enumerate(zip(pairs, permutations, slots)):
        if not isinstance(permutation_id, Integral) or permutation_id not in (0, 1):
            raise ValueError(f"permutation_id at position {position} must be integer 0 or 1")
        if not isinstance(slot, Integral) or slot < 0:
            raise ValueError(f"rollout_slot at position {position} must be a non-negative integer")
        identities = paired_slots.setdefault((pair_id, slot), {})
        if permutation_id in identities:
            raise ValueError(f"duplicate rollout identity: {(pair_id, permutation_id, slot)!r}")
        identities[permutation_id] = position
        slots_by_pair.setdefault(pair_id, set()).add(slot)
    for key, identities in paired_slots.items():
        if set(identities) != {0, 1}:
            raise ValueError(f"missing permutation for pair/slot identity: {key!r}")
    for pair_id, pair_slots in slots_by_pair.items():
        if pair_slots != set(range(len(pair_slots))):
            raise ValueError(f"rollout slots must be contiguous from zero for pair_id={pair_id!r}")

    group_mean, group_std = _population_stats(rewards_tensor, pairs)
    local_mean, local_std = _population_stats(rewards_tensor, list(zip(pairs, permutations)))
    global_advantage = torch.where(
        group_std == 0, torch.zeros_like(rewards_tensor),
        (rewards_tensor - group_mean) / (group_std + eps),
    )
    local_advantage = torch.where(
        local_std == 0, torch.zeros_like(rewards_tensor),
        (rewards_tensor - local_mean) / (local_std + eps),
    )
    local_weight = torch.zeros_like(rewards_tensor)
    if mode == "eis":
        local_weight.fill_(1.0)
    elif mode == "alc":
        for pair_id, positions in pair_positions.items():
            zero_position, one_position = paired_slots[(pair_id, 0)][0], paired_slots[(pair_id, 0)][1]
            mean_gap = (local_mean[zero_position] - local_mean[one_position]).abs()
            weight = torch.where(
                mean_gap == 0, torch.zeros_like(mean_gap),
                torch.clamp(mean_gap / (group_std[zero_position] + eps), min=0.0, max=1.0),
            )
            index = torch.tensor(positions, device=rewards_tensor.device, dtype=torch.long)
            local_weight.index_fill_(0, index, weight)

    advantage = global_advantage + local_weight * local_advantage
    gate_mask = torch.zeros_like(rewards_tensor, dtype=torch.bool)
    if mode == "pa_controlled" and sigma_gate_threshold is not None:
        gate_mask = group_std < sigma_gate_threshold
        advantage = torch.where(gate_mask, torch.zeros_like(advantage), advantage)
    return ControlledAdvantageResult(
        advantage=advantage,
        global_advantage=global_advantage,
        local_advantage=local_advantage,
        local_weight=local_weight,
        group_mean=group_mean,
        group_std=group_std,
        local_mean=local_mean,
        local_std=local_std,
        gate_mask=gate_mask,
    )
