"""Generation planning and backend adapters with no eager vLLM dependency."""

from .base import (
    CANDIDATE_PLAN_SCHEMA,
    GENERATION_PLAN_SCHEMA,
    GENERATOR_REPOSITORIES,
    GenerationBackend,
    GenerationConfig,
    GenerationConfigurationError,
    GenerationPlan,
    GenerationRequest,
    GenerationResult,
    generation_stage_config,
    plan_generation,
)
from .fake import FakeGenerationBackend
from .runner import (
    GenerationShardPlan,
    RunMismatchError,
    ShardRunSummary,
    build_generation_shard_plan,
    run_generation_shard,
    semantic_candidate_set_hash,
    successful_candidate_keys,
)
from .vllm import VLLMGenerationBackend

__all__ = [
    "CANDIDATE_PLAN_SCHEMA",
    "GENERATION_PLAN_SCHEMA",
    "GENERATOR_REPOSITORIES",
    "FakeGenerationBackend",
    "GenerationBackend",
    "GenerationConfig",
    "GenerationConfigurationError",
    "GenerationPlan",
    "GenerationRequest",
    "GenerationResult",
    "GenerationShardPlan",
    "RunMismatchError",
    "ShardRunSummary",
    "VLLMGenerationBackend",
    "build_generation_shard_plan",
    "generation_stage_config",
    "plan_generation",
    "run_generation_shard",
    "semantic_candidate_set_hash",
    "successful_candidate_keys",
]
