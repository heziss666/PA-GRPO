"""Focused scalar-reward checks for controlled Global/EIS/ALC/PA modes."""

from dataclasses import fields
from math import sqrt

import pytest
import torch

from permstudy import advantages


def _compute(rewards, pair_ids, permutation_ids, rollout_slots, **kwargs):
    assert hasattr(advantages, "compute_controlled_advantage"), "controlled advantage API is missing"
    return advantages.compute_controlled_advantage(
        rewards, pair_ids, permutation_ids, rollout_slots, **kwargs
    )


def _example(mode, **kwargs):
    return _compute(
        torch.tensor([-1.0, 1.0, 3.0, 5.0], dtype=torch.float64),
        ["pair"] * 4,
        [0, 0, 1, 1],
        [0, 1, 0, 1],
        mode=mode,
        **kwargs,
    )


def test_global_and_local_population_statistics_match_hand_calculation():
    result = _example("global_only", eps=1.0)

    assert isinstance(result, advantages.ControlledAdvantageResult)
    torch.testing.assert_close(result.group_mean, torch.full((4,), 2.0, dtype=torch.float64))
    torch.testing.assert_close(result.group_std, torch.full((4,), sqrt(5), dtype=torch.float64))
    torch.testing.assert_close(result.local_mean, torch.tensor([0.0, 0.0, 4.0, 4.0], dtype=torch.float64))
    torch.testing.assert_close(result.local_std, torch.ones(4, dtype=torch.float64))
    expected_global = torch.tensor([-3.0, -1.0, 1.0, 3.0], dtype=torch.float64) / (sqrt(5) + 1)
    torch.testing.assert_close(result.global_advantage, expected_global)
    torch.testing.assert_close(result.local_advantage, torch.tensor([-0.5, 0.5, -0.5, 0.5], dtype=torch.float64))
    torch.testing.assert_close(result.advantage, expected_global)
    torch.testing.assert_close(result.local_weight, torch.zeros(4, dtype=torch.float64))
    assert not result.gate_mask.any()


def test_eis_adds_local_advantage_with_fixed_unit_weight():
    result = _example("eis", eps=1.0)

    expected = torch.tensor(
        [-3 / (sqrt(5) + 1) - 0.5, -1 / (sqrt(5) + 1) + 0.5,
         1 / (sqrt(5) + 1) - 0.5, 3 / (sqrt(5) + 1) + 0.5],
        dtype=torch.float64,
    )
    torch.testing.assert_close(result.advantage, expected)
    torch.testing.assert_close(result.local_weight, torch.ones(4, dtype=torch.float64))


def test_alc_pair_weight_is_zero_partial_or_clipped_from_subgroup_imbalance():
    result = _compute(
        torch.tensor([-1.0, 1.0, -1.0, 1.0, -2.0, 2.0, -1.0, 3.0, -1.0, 1.0, 3.0, 5.0], dtype=torch.float64),
        ["balanced"] * 4 + ["partial"] * 4 + ["clipped"] * 4,
        [0, 0, 1, 1] * 3,
        [0, 1, 0, 1] * 3,
        mode="alc",
        eps=1e-6,
    )

    partial = 1 / (sqrt(4.25) + 1e-6)
    expected_weight = torch.tensor([0.0] * 4 + [partial] * 4 + [1.0] * 4, dtype=torch.float64)
    torch.testing.assert_close(result.local_weight, expected_weight)
    expected_partial = torch.tensor([-2.5, 1.5, -1.5, 2.5], dtype=torch.float64) / (sqrt(4.25) + 1e-6)
    expected_partial += partial * torch.tensor([-2.0, 2.0, -2.0, 2.0], dtype=torch.float64) / (2 + 1e-6)
    torch.testing.assert_close(result.advantage[4:8], expected_partial)
    torch.testing.assert_close(result.advantage[:4], torch.tensor([-1.0, 1.0, -1.0, 1.0], dtype=torch.float64) / (1 + 1e-6))


def test_reordering_preserves_each_identity_and_all_diagnostics():
    rewards = torch.tensor([-1.0, 1.0, 3.0, 5.0, -2.0, 2.0, -1.0, 3.0])
    pairs = ["a"] * 4 + ["b"] * 4
    permutations = [0, 0, 1, 1] * 2
    slots = [0, 1, 0, 1] * 2
    order = [7, 2, 4, 0, 5, 3, 1, 6]
    original = _compute(rewards, pairs, permutations, slots, mode="alc")
    reordered = _compute(
        rewards[order], [pairs[i] for i in order], [permutations[i] for i in order],
        [slots[i] for i in order], mode="alc",
    )

    for field in fields(original):
        torch.testing.assert_close(getattr(reordered, field.name), getattr(original, field.name)[order])


def test_zero_variance_keeps_every_mode_finite_and_zero():
    for dtype, eps in ((torch.float32, 1e-6), (torch.float32, 1e-50), (torch.float16, 1e-12)):
        for mode in ("global_only", "eis", "alc", "pa_controlled"):
            config = {"sigma_gate_threshold": 0.1, "sigma_gate_source": "explicit-test-choice"} if mode == "pa_controlled" else {}
            result = _compute(torch.full((4,), 3.0, dtype=dtype), ["pair"] * 4, [0, 0, 1, 1], [0, 1, 0, 1], mode=mode, eps=eps, **config)
            for name in ("advantage", "global_advantage", "local_advantage", "local_weight", "group_std", "local_std"):
                value = getattr(result, name)
                assert torch.isfinite(value).all()
                if name != "local_weight" or mode != "eis":
                    torch.testing.assert_close(value, torch.zeros(4, dtype=dtype))
            assert result.gate_mask.tolist() == [mode == "pa_controlled"] * 4


def test_rejects_mismatched_identity_lengths_non_scalar_rewards_and_invalid_mode():
    cases = (
        ([1.0, 2.0], ["pair"], [0, 1], [0, 0], "global_only"),
        ([1.0, 2.0], ["pair"] * 2, [0], [0, 0], "global_only"),
        ([1.0, 2.0], ["pair"] * 2, [0, 1], [0], "global_only"),
        ([[1.0], [2.0]], ["pair"] * 2, [0, 1], [0, 0], "global_only"),
        ([1.0, 2.0], ["pair"] * 2, [0, 2], [0, 0], "global_only"),
        ([1.0, 2.0], ["pair"] * 2, [0, 1], [0, 0], "unknown"),
    )
    for rewards, pairs, permutations, slots, mode in cases:
        with pytest.raises(ValueError):
            _compute(rewards, pairs, permutations, slots, mode=mode)


def test_rejects_duplicate_and_missing_pair_slot_identities():
    cases = (
        ([0, 0, 1, 1], [0, 0, 0, 1]),  # Duplicate permutation 0 / slot 0.
        ([0, 0, 1, 1], [0, 1, 0, 2]),  # Same counts, but unmatched slots.
        ([0, 0, 1], [0, 1, 0]),  # Missing permutation 1 / slot 1.
    )
    for permutations, slots in cases:
        with pytest.raises(ValueError):
            _compute([1.0] * len(slots), ["pair"] * len(slots), permutations, slots, mode="global_only")


def test_rollout_slots_must_be_nonnegative_contiguous_integers_per_pair():
    for slots in ([-1, -1], [0.5, 0.5], [1, 1], [0, 2, 0, 2]):
        permutations = [0, 1] if len(slots) == 2 else [0, 0, 1, 1]
        with pytest.raises(ValueError):
            _compute([1.0] * len(slots), ["pair"] * len(slots), permutations, slots, mode="eis")


def test_nonfinite_rewards_and_nonpositive_or_nonfinite_eps_fail_closed():
    for bad_reward in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            _compute([1.0, bad_reward], ["pair"] * 2, [0, 1], [0, 0], mode="global_only")
    for eps in (0.0, -1.0, float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            _example("global_only", eps=eps)


def test_pa_gate_uses_strict_population_sigma_and_retains_raw_diagnostics():
    for threshold, should_gate in ((0.5, False), (1.0, False), (1.1, True)):
        result = _compute(
            [-1.0, 1.0], ["pair"] * 2, [0, 1], [0, 0], mode="pa_controlled",
            sigma_gate_threshold=threshold, sigma_gate_source="explicit-test-choice",
        )
        torch.testing.assert_close(result.group_std, torch.ones(2))
        torch.testing.assert_close(result.global_advantage, torch.tensor([-1.0, 1.0]) / (1 + 1e-6))
        torch.testing.assert_close(result.advantage, torch.zeros(2) if should_gate else torch.tensor([-1.0, 1.0]) / (1 + 1e-6))
        assert result.gate_mask.tolist() == [should_gate] * 2


def test_pa_without_threshold_returns_ungated_global_advantage():
    expected = torch.tensor([-3.0, -1.0, 1.0, 3.0], dtype=torch.float64) / (sqrt(5) + 1)
    for source in (None, "unspecified"):
        result = _example("pa_controlled", eps=1.0, sigma_gate_source=source)
        torch.testing.assert_close(result.advantage, expected)
        torch.testing.assert_close(result.global_advantage, expected)
        torch.testing.assert_close(result.group_std, torch.full((4,), sqrt(5), dtype=torch.float64))
        torch.testing.assert_close(result.local_weight, torch.zeros(4, dtype=torch.float64))
        assert not result.gate_mask.any()


def test_explicit_gate_requires_positive_threshold_and_known_source_only_for_pa():
    invalid_pa = (
        {"sigma_gate_threshold": 0.1},
        {"sigma_gate_threshold": 0.1, "sigma_gate_source": "  "},
        {"sigma_gate_threshold": 0.1, "sigma_gate_source": "unspecified"},
        *({"sigma_gate_threshold": value, "sigma_gate_source": "explicit-choice"}
          for value in (0.0, -1.0, float("nan"), float("inf"), float("-inf"))),
    )
    for config in invalid_pa:
        with pytest.raises(ValueError):
            _example("pa_controlled", **config)
    for mode in ("global_only", "eis", "alc"):
        for config in ({"sigma_gate_threshold": 0.1}, {"sigma_gate_source": "explicit-choice"},
                       {"sigma_gate_threshold": 0.1, "sigma_gate_source": "explicit-choice"}):
            with pytest.raises(ValueError):
                _example(mode, **config)


def test_global_response_advantage_is_broadcast_without_response_length_weighting():
    expected = torch.tensor([1.0, 1.0, -1.0, -1.0]) / (1 + 1e-6)
    for lengths in ([8, 8, 8, 8], [2, 8, 4, 10]):
        result = _compute([1.0, 1.0, -1.0, -1.0], ["pair"] * 4, [0, 0, 1, 1], [0, 1, 0, 1], mode="global_only")
        mask = torch.arange(10).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
        broadcast = result.advantage.unsqueeze(1) * mask
        for row, length in enumerate(lengths):
            torch.testing.assert_close(broadcast[row, :length], expected[row].expand(length))
        torch.testing.assert_close(result.advantage, expected)
