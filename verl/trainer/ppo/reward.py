# Copyright 2025 Individual Contributor: Thibaut Barroyer
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import inspect
import math
import multiprocessing
import os
import sys
import warnings
from functools import partial
from typing import Any, Optional

import ray
import torch
from omegaconf import DictConfig

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.utils.transferqueue_utils import tqbridge
from verl.workers.reward_manager import get_reward_manager_cls
from verl.workers.reward_manager.abstract import AbstractRewardManager, RawRewardFn


def _call_with_kwargs(raw_fn, extra_kwargs, *args, **kwargs):
    """Calls `raw_fn` by merging `extra_kwargs` into call-time `kwargs`, with `extra_kwargs` taking precedence.

    This function is used to merge additional keyword arguments with the original function's arguments.
    """
    merged_kwargs = {**kwargs, **extra_kwargs}
    return raw_fn(*args, **merged_kwargs)


async def _call_with_kwargs_async(raw_fn, extra_kwargs, *args, **kwargs):
    """Calls `raw_fn` by merging `extra_kwargs` into call-time `kwargs`, with `extra_kwargs` taking precedence.

    This function is used to merge additional keyword arguments with the original function's arguments.
    """
    merged_kwargs = {**kwargs, **extra_kwargs}
    return await raw_fn(*args, **merged_kwargs)


def _controlled_reward_kwargs(config: DictConfig, call_kwargs=None) -> Optional[dict[str, Any]]:
    """Derive the selected controlled method and reject conflicting reward settings."""
    training_config = config.get("permutation_training") or {}
    mode = training_config.get("mode", "official")
    if mode == "official":
        return None
    if mode not in {"global_only", "eis", "alc", "pa_controlled"}:
        raise ValueError(f"unsupported permutation_training.mode: {mode!r}")
    if (config.get("grouping") or {}).get("identity_mode", "legacy_index") != "explicit":
        raise ValueError(f"permutation_training.mode={mode} requires grouping.identity_mode=explicit")
    if config.reward_model.get("reward_manager", "naive") != "batch":
        raise ValueError(f"permutation_training.mode={mode} requires reward_model.reward_manager=batch")

    custom_config = config.get("custom_reward_function") or {}
    expected_path = os.path.realpath(os.path.join(os.path.dirname(__file__), "../../../my_reward/judge_qwen.py"))
    file_path = custom_config.get("path")
    if not file_path or os.path.normcase(os.path.realpath(file_path)) != os.path.normcase(expected_path):
        raise ValueError(f"permutation_training.mode={mode} requires custom_reward_function.path={expected_path}")
    if custom_config.get("name") != "compute_score":
        raise ValueError(f"permutation_training.mode={mode} requires custom_reward_function.name=compute_score")

    consistency_weight = 0.0
    if mode == "pa_controlled":
        consistency_weight = float(training_config.get("consistency_coef", 1.0))
        if not math.isfinite(consistency_weight) or consistency_weight < 0:
            raise ValueError("permutation_training.consistency_coef must be finite and nonnegative")
    inferred = {
        "identity_mode": "explicit",
        "evaluation_contract": "controlled",
        "consistency_weight": consistency_weight,
    }
    sources = (
        ("custom_reward_function.reward_kwargs", custom_config.get("reward_kwargs") or {}),
        ("reward_model.reward_kwargs", config.reward_model.get("reward_kwargs") or {}),
        ("reward manager call kwargs", call_kwargs or {}),
    )
    for source, supplied in sources:
        for key, expected in inferred.items():
            if key in supplied and supplied[key] != expected:
                raise ValueError(f"{source}.{key} conflicts with permutation_training.mode={mode}: expected {expected!r}")
        if "return_dict" in supplied and supplied["return_dict"] is not True:
            raise ValueError(f"{source}.return_dict must be True for controlled reward components")
    return inferred


def _validate_controlled_reward_components(reward_tensor, extra_info):
    """Require intact row components before the controlled advantage adapter uses them."""
    if not isinstance(extra_info, dict):
        raise ValueError("controlled rewards require base_reward and consistency component mappings")
    rewards = torch.as_tensor(reward_tensor)
    if rewards.ndim < 1:
        raise ValueError("controlled reward_tensor must have a row dimension")
    components = []
    for key in ("base_reward", "consistency"):
        if key not in extra_info:
            raise ValueError(f"controlled rewards require reward_extra_info.{key}")
        values = torch.as_tensor(extra_info[key], dtype=rewards.dtype, device=rewards.device)
        if values.shape != (rewards.shape[0],) or not torch.isfinite(values).all():
            raise ValueError(f"controlled reward_extra_info.{key} must contain one finite value per row")
        components.append(values)
    totals = rewards.reshape(rewards.shape[0], -1).sum(dim=-1)
    if not torch.allclose(totals, components[0] + components[1], atol=1e-6, rtol=1e-6):
        raise ValueError("controlled reward total must equal base_reward + consistency")


def _controlled_batch_scores(raw_fn, *args, **kwargs):
    """Translate Judge's component dictionary into BatchRewardManager's row format."""
    kwargs["return_dict"] = True
    result = raw_fn(*args, **kwargs)
    rewards = result["reward_tensor"]
    extra_info = result.get("reward_extra_info", {})
    _validate_controlled_reward_components(rewards, extra_info)
    rows = []
    for position in range(len(rewards)):
        row = {key: values[position] for key, values in extra_info.items()}
        row["score"] = float(rewards[position])
        rows.append(row)
    return rows


def get_custom_reward_fn(config: DictConfig) -> Optional[RawRewardFn]:
    """Load and return a custom reward function from external file.

    Dynamically imports a reward function from a specified file path and wraps
    it with additional keyword arguments from the configuration.

    Args:
        config (dict): Configuration dictionary containing custom_reward_function
                      settings with 'path', 'name', and 'reward_kwargs' fields.

    Returns:
        callable or None: Wrapped reward function with merged kwargs, or None
                         if no custom reward function is configured.

    Raises:
        FileNotFoundError: If the specified reward function file doesn't exist.
        RuntimeError: If there's an error loading the module from file.
        AttributeError: If the specified function name isn't found in the module.
    """

    controlled_kwargs = _controlled_reward_kwargs(config)
    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    function_name = reward_fn_config.get("name")
    assert function_name is not None

    module = sys.modules.get("custom_module", None)
    if controlled_kwargs is not None and module is not None:
        # The official loader caches one module globally. A controlled run must
        # not reuse a previously loaded reward from a different file.
        module_path = getattr(module, "__file__", "")
        if os.path.normcase(os.path.realpath(module_path)) != os.path.normcase(os.path.realpath(file_path)):
            module = None
    if module is None:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

        spec = importlib.util.spec_from_file_location("custom_module", file_path)
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_module"] = module
            assert spec.loader is not None
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Error loading module from '{file_path}': {e}") from e

    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{module.__file__}'.")

    print(f"using customized reward function '{function_name}' from '{module.__file__}'")
    raw_fn = getattr(module, function_name)

    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))
    if controlled_kwargs is not None:
        reward_kwargs.update(controlled_kwargs)

    if not inspect.iscoroutinefunction(raw_fn):
        return partial(_call_with_kwargs, raw_fn, reward_kwargs)
    else:
        return partial(_call_with_kwargs_async, raw_fn, reward_kwargs)


def load_reward_manager(
    config: DictConfig, tokenizer: Any, num_examine: int, **reward_kwargs: Any
) -> AbstractRewardManager:
    """
    Load and initialize a reward manager based on the configuration.

    Args:
        config: PPO trainer configuration object containing reward_model fields.
        tokenizer: Tokenizer object used for processing text.
        num_examine: Number of samples to examine.
        **reward_kwargs: Additional keyword arguments for the reward manager.

    Returns:
        An instance of the specified reward manager class.
    """

    controlled_kwargs = _controlled_reward_kwargs(config, reward_kwargs)
    if controlled_kwargs is not None:
        reward_kwargs.update(controlled_kwargs)

    # Try to get a custom reward function based on the configuration
    # user defined reward manager can be registered in custom_reward_fn
    compute_score = get_custom_reward_fn(config)
    final_compute_score = compute_score
    if controlled_kwargs is not None:
        final_compute_score = partial(_controlled_batch_scores, compute_score)

    # The list of pre-defined reward managers are defined in `verl/workers/reward_manager/`:
    # naive: NaiveRewardManager
    # prime: PrimeRewardManager
    # batch: BatchRewardManager
    # dapo: DAPORewardManager
    # Note(haibin.lin): For custom reward managers, please make sure they are imported and
    # registered via `verl.workers.reward_manager.register`
    # By default reward_manager is set to naive (NaiveRewardManager)
    reward_manager_name = config.reward_model.get("reward_manager", "naive")
    identity_mode = config.get("grouping", {}).get("identity_mode", "legacy_index")
    reward_identity_mode = reward_kwargs.get("identity_mode")
    if reward_identity_mode is not None and reward_identity_mode != identity_mode:
        raise ValueError(
            "reward_model.reward_kwargs.identity_mode "
            f"({reward_identity_mode!r}) conflicts with grouping.identity_mode ({identity_mode!r})"
        )
    custom_reward_config = config.get("custom_reward_function") or {}
    custom_reward_kwargs = custom_reward_config.get("reward_kwargs") or {}
    custom_identity_mode = custom_reward_kwargs.get("identity_mode")
    if custom_identity_mode is not None and custom_identity_mode != identity_mode:
        raise ValueError(
            "custom_reward_function.reward_kwargs.identity_mode "
            f"({custom_identity_mode!r}) conflicts with grouping.identity_mode ({identity_mode!r})"
        )
    if identity_mode == "explicit":
        if reward_manager_name != "batch":
            raise ValueError("grouping.identity_mode=explicit requires reward_model.reward_manager=batch")
        reward_kwargs["identity_mode"] = identity_mode
    reward_manager_cls = get_reward_manager_cls(reward_manager_name)

    if compute_score is None:
        sandbox_config = config.reward_model.get("sandbox_fusion")
        sandbox_url = sandbox_config.get("url") if sandbox_config else None
        memory_limit_mb = sandbox_config.get("memory_limit_mb", 1024) if sandbox_config else 1024
        if sandbox_url:
            sandbox_manager = multiprocessing.Manager()
            # Create a semaphore to control concurrent access to the sandbox
            _concurrent_semaphore = sandbox_manager.Semaphore(sandbox_config.get("max_concurrent", 64))
            final_compute_score = partial(
                default_compute_score,
                sandbox_fusion_url=sandbox_url,
                concurrent_semaphore=_concurrent_semaphore,
                memory_limit_mb=memory_limit_mb,
            )
        else:
            final_compute_score = default_compute_score

    # Instantiate and return the reward manager with the specified parameters
    reward_manager = reward_manager_cls(
        tokenizer=tokenizer,
        num_examine=num_examine,
        compute_score=final_compute_score,
        reward_fn_key=config.data.reward_fn_key,
        **reward_kwargs,
    )
    if controlled_kwargs is not None:
        reward_manager.require_reward_components = True
    return reward_manager


@tqbridge(put_data=False)
def compute_reward(data: DataProto, reward_fn: AbstractRewardManager) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute reward for a batch of data.
    Args:
        data: DataProto object containing the input data.
        reward_fn: Reward function to compute the reward.
    Returns:
        Tuple of reward tensor and extra info dictionary.
    """
    try:
        reward_result = reward_fn(data, return_dict=True)
        reward_tensor = reward_result["reward_tensor"]
        reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
        if getattr(reward_fn, "require_reward_components", False):
            _validate_controlled_reward_components(reward_tensor, reward_extra_infos_dict)
    except Exception as e:
        if getattr(reward_fn, "require_reward_components", False):
            raise
        print(f"Error in reward_fn: {e}")
        reward_tensor = reward_fn(data)
        reward_extra_infos_dict = {}

    return reward_tensor, reward_extra_infos_dict


@ray.remote(num_cpus=1)
def compute_reward_async(data: DataProto, config=None, tokenizer=None, reward_fn=None):
    """
    Load the reward manager and compute the reward for a batch of data.
    This is meant to be run in a separate Ray worker.
    """
    if reward_fn is None:
        assert config is not None and tokenizer is not None, (
            "config and tokenizer must not be None when reward_fn is None"
        )

        warnings.warn("using config and tokenizer with compute_reward_async is deprecated", stacklevel=2)
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )

    return compute_reward(data, reward_fn)
