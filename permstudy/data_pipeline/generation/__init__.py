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
    "VLLMGenerationBackend",
    "generation_stage_config",
    "plan_generation",
]
