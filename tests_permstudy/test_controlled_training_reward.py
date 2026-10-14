"""The controlled training contract preserves PA reward components and strict answers."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from my_reward import judge_qwen


class _NullWriter:
    def add_scalar(self, *_args, **_kwargs):
        pass


@pytest.fixture(autouse=True)
def quiet_reward_logs(monkeypatch):
    # Logging is the only external side effect; reward and loader remain real.
    monkeypatch.setattr(judge_qwen, "get_tb_writer", lambda: _NullWriter())
    monkeypatch.setattr(judge_qwen, "safe_log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(judge_qwen, "safe_log_output", lambda *_args, **_kwargs: None)
    monkeypatch.setitem(sys.modules, "custom_module", judge_qwen)


def _identities():
    return [
        {"pair_id": "pair", "permutation_id": 0, "rollout_slot": 0},
        {"pair_id": "pair", "permutation_id": 1, "rollout_slot": 0},
    ]


@pytest.mark.parametrize(
    "response, correctness, base_reward",
    [
        ("<answer>A</answer>", 1.0, 1.2),
        ("<answer>a</answer>", 1.0, 1.2),
        ("Final Answer: A", -1.0, -1.4),
        ("I choose A", -1.0, -1.4),
        ("<answer>A</answer><answer>B</answer>", -1.0, -0.8),
        ("<answer>C</answer>", -1.0, -0.8),
        (None, -1.0, -1.4),
    ],
)
def test_controlled_correctness_rejects_answers_outside_strict_contract(response, correctness, base_reward):
    """Catches permissive fallback parsing contaminating controlled correctness."""
    result = judge_qwen.compute_score(
        ["judge", "judge"],
        [response, "<answer>B</answer>"],
        ["A", "B"],
        extra_infos=_identities(),
        identity_mode="explicit",
        evaluation_contract="controlled",
        return_dict=True,
    )
    extra = result["reward_extra_info"]
    assert extra["correctness"][0] == correctness
    assert extra["base_reward"][0] == pytest.approx(base_reward)
    expected_consistency = 1.0 if correctness == 1.0 else 0.0
    assert extra["consistency"] == [expected_consistency, expected_consistency]


@pytest.mark.parametrize(
    "think_length, length_reward",
    [(499, -0.1), (500, 0.1), (1999, 0.1), (2000, 0.2), (3500, 0.2), (3501, 0.1), (5000, 0.1), (5001, -0.1)],
)
def test_controlled_reward_preserves_existing_pa_length_piecewise(think_length, length_reward):
    """Catches replacing the PA base reward with a correctness-only score."""
    result = judge_qwen.compute_score(
        ["judge", "judge"],
        ["<thinking>" + "x" * think_length + "</thinking><answer>A</answer>", "<answer>B</answer>"],
        ["A", "B"],
        extra_infos=_identities(),
        identity_mode="explicit",
        evaluation_contract="controlled",
        consistency_weight=0.0,
        return_dict=True,
    )
    extra = result["reward_extra_info"]
    assert extra["length"][0] == length_reward
    assert extra["base_reward"][0] == pytest.approx(1.3 + length_reward)
    assert result["reward_tensor"][0].item() == pytest.approx(1.3 + length_reward)


def test_official_default_keeps_permissive_answers_and_consistency():
    """Catches silently switching old callers to controlled parsing or weight zero."""
    result = judge_qwen.compute_score(
        ["judge", "judge"],
        ["Final Answer: A", "<answer>B</answer><answer>A</answer>"],
        ["A", "B"],
        extra_infos=[{"index": 0, "permutation": 0}, {"index": 1, "permutation": 1}],
    )
    torch.testing.assert_close(result, torch.tensor([1.6, 2.2]), atol=1e-6, rtol=0)


def _config(mode="global_only"):
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "permutation_training": {"mode": mode, "consistency_coef": 1.0},
            "grouping": {"identity_mode": "explicit"},
            "reward_model": {"reward_manager": "batch", "reward_kwargs": {}},
            "data": {"reward_fn_key": "data_source"},
            "custom_reward_function": {
                "path": str(Path(judge_qwen.__file__).resolve()),
                "name": "compute_score",
                "reward_kwargs": {},
            },
        }
    )


def _batch():
    from verl import DataProto

    return DataProto.from_single_dict(
        {
            "prompts": torch.tensor([[1], [1]]),
            "responses": torch.tensor([[2], [3]]),
            "attention_mask": torch.ones((2, 2), dtype=torch.long),
            "data_source": np.array(["judge", "judge"], dtype=object),
            "reward_model": np.array([{"ground_truth": "A"}, {"ground_truth": "B"}], dtype=object),
            "pair_id": np.array(["pair", "pair"], dtype=object),
            "permutation_id": np.array([0, 1], dtype=np.int64),
            "rollout_slot": np.array([0, 0], dtype=np.int64),
        }
    )


class _Tokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        return "<answer>A</answer>" if token_ids[0].item() == 2 else "<answer>B</answer>"


requires_verl = pytest.mark.skipif(importlib.util.find_spec("ray") is None, reason="requires real verl dependencies")


@requires_verl
@pytest.mark.parametrize("mode, consistency", [("global_only", 0.0), ("eis", 0.0), ("alc", 0.0), ("pa_controlled", 1.0)])
def test_loader_infers_controlled_components_and_batch_manager_preserves_base_reward(mode, consistency):
    """Catches controlled methods retaining PA consistency or losing reward components."""
    from verl.trainer.ppo.reward import compute_reward, load_reward_manager

    manager = load_reward_manager(_config(mode), tokenizer=_Tokenizer(), num_examine=0)
    reward, extra = compute_reward(_batch(), manager)

    assert manager.reward_kwargs["identity_mode"] == "explicit"
    assert manager.reward_kwargs["evaluation_contract"] == "controlled"
    assert manager.reward_kwargs["consistency_weight"] == consistency
    assert extra["base_reward"] == pytest.approx([1.2, 1.2])
    assert extra["consistency"] == [consistency, consistency]
    assert extra["correctness"] == [1.0, 1.0]
    assert extra["format"] == [0.3, 0.3]
    assert extra["length"] == [-0.1, -0.1]
    torch.testing.assert_close(reward[:, -1], torch.tensor([1.2 + consistency] * 2), atol=1e-6, rtol=0)


@requires_verl
def test_loader_uses_configurable_pa_controlled_consistency_coef():
    """Catches ignoring the PA controlled consistency coefficient."""
    from verl.trainer.ppo.reward import compute_reward, load_reward_manager

    config = _config("pa_controlled")
    config.permutation_training.consistency_coef = 0.25
    manager = load_reward_manager(config, tokenizer=_Tokenizer(), num_examine=0)
    reward, extra = compute_reward(_batch(), manager)
    assert extra["base_reward"] == pytest.approx([1.2, 1.2])
    assert extra["consistency"] == [0.25, 0.25]
    torch.testing.assert_close(reward[:, -1], torch.tensor([1.45, 1.45]), atol=1e-6, rtol=0)


@requires_verl
@pytest.mark.parametrize("scope", ["custom", "manager", "call"])
@pytest.mark.parametrize("key, value", [("identity_mode", "legacy_index"), ("evaluation_contract", "official"), ("consistency_weight", 1.0)])
def test_controlled_loader_rejects_conflicting_overrides(scope, key, value):
    """Catches a lower-level override changing the selected controlled method."""
    from verl.trainer.ppo.reward import load_reward_manager

    config = _config()
    kwargs = {}
    if scope == "custom":
        config.custom_reward_function.reward_kwargs[key] = value
    elif scope == "manager":
        config.reward_model.reward_kwargs[key] = value
    else:
        kwargs[key] = value
    with pytest.raises(ValueError, match=key):
        load_reward_manager(config, tokenizer=_Tokenizer(), num_examine=0, **kwargs)


@requires_verl
@pytest.mark.parametrize("invalid", ["grouping", "manager", "path", "function"])
def test_controlled_loader_requires_explicit_batch_qwen_reward(invalid):
    """Catches controlled mode dispatch through a different reward implementation."""
    from verl.trainer.ppo.reward import load_reward_manager

    config = _config()
    if invalid == "grouping":
        config.grouping.identity_mode = "legacy_index"
    elif invalid == "manager":
        config.reward_model.reward_manager = "naive"
    elif invalid == "path":
        config.custom_reward_function.path = str(Path(judge_qwen.__file__).with_name("judge_llama.py"))
    else:
        config.custom_reward_function.name = "extract_answer"
    with pytest.raises(ValueError):
        load_reward_manager(config, tokenizer=_Tokenizer(), num_examine=0)


@requires_verl
def test_controlled_reward_failure_is_not_retried_without_components(monkeypatch):
    """Catches compute_reward hiding an invalid controlled batch through scalar fallback."""
    from verl.trainer.ppo.reward import compute_reward, load_reward_manager

    manager = load_reward_manager(_config(), tokenizer=_Tokenizer(), num_examine=0)
    batch = _batch()
    batch.non_tensor_batch["permutation_id"] = np.array([0, 0], dtype=np.int64)
    with pytest.raises(ValueError, match="duplicate rollout identity"):
        compute_reward(batch, manager)


@requires_verl
def test_controlled_precomputed_reward_requires_base_reward_components():
    """Catches precomputed rm_scores silently bypassing controlled reward diagnostics."""
    from verl.trainer.ppo.reward import compute_reward, load_reward_manager

    manager = load_reward_manager(_config(), tokenizer=_Tokenizer(), num_examine=0)
    batch = _batch()
    batch.batch["rm_scores"] = torch.ones((2, 1))
    with pytest.raises(ValueError, match="base_reward"):
        compute_reward(batch, manager)


@requires_verl
@pytest.mark.parametrize("configured", [False, True])
def test_official_loader_keeps_existing_kwargs_without_controlled_inference(configured):
    """Catches controlled inference changing an absent or explicit official mode."""
    from verl.trainer.ppo.reward import load_reward_manager

    config = _config("official")
    if not configured:
        del config["permutation_training"]
    config.grouping.identity_mode = "legacy_index"
    manager = load_reward_manager(config, tokenizer=_Tokenizer(), num_examine=0)
    assert manager.reward_kwargs == {}
