"""Backend-neutral generation planning and immutable adapter contracts."""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
import math
import re
from types import MappingProxyType
from typing import Protocol

from ..canonical import canonical_json_bytes, sha256_hex
from ..ids import candidate_id as logical_candidate_id
from ..ids import run_id
from ..lineage import role_bound_stage_config
from ..schema import CandidatePlan, Split, SplitAssignment


GENERATION_PLAN_SCHEMA = "generation_plan_v1"
CANDIDATE_PLAN_SCHEMA = "candidate_plan_v1"

GENERATOR_REPOSITORIES = MappingProxyType(
    {
        "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
        "qwen2.5-32b-instruct": "Qwen/Qwen2.5-32B-Instruct",
        "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    }
)

_SHA40 = re.compile(r"[0-9a-f]{40}\Z")
_SHA64 = re.compile(r"[0-9a-f]{64}\Z")
_BACKENDS = frozenset({"fake", "vllm"})


class GenerationConfigurationError(ValueError):
    """A sanitized configuration/dependency failure at the backend boundary."""


@dataclass(frozen=True)
class GenerationConfig:
    backend: str
    generator_id: str
    model_repository: str
    model_revision: str
    prompt_template_revision: str
    prompt_template_hash: str
    temperature: float
    top_p: float
    max_new_tokens: int
    seed: int
    batch_size: int
    tensor_parallel_size: int
    max_model_len: int
    gpu_memory_utilization: float
    samples_per_question: int
    split_manifest_hash: str

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        if self.backend not in _BACKENDS:
            raise ValueError("backend must be fake or vllm")
        if self.generator_id not in GENERATOR_REPOSITORIES:
            raise ValueError("unknown generator_id")
        if self.model_repository != GENERATOR_REPOSITORIES[self.generator_id]:
            raise ValueError("model repository does not match generator_id")
        if not isinstance(self.model_revision, str) or not _SHA40.fullmatch(self.model_revision):
            raise ValueError("model_revision must be an immutable model revision SHA")
        if not isinstance(self.prompt_template_revision, str) or not self.prompt_template_revision.strip():
            raise ValueError("prompt_template_revision must be nonempty")
        if not isinstance(self.prompt_template_hash, str) or not _SHA64.fullmatch(self.prompt_template_hash):
            raise ValueError("prompt_template_hash must be a lowercase SHA256")
        if not isinstance(self.split_manifest_hash, str) or not _SHA64.fullmatch(self.split_manifest_hash):
            raise ValueError("split_manifest_hash must be a lowercase SHA256")
        if type(self.temperature) not in (int, float) or not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be a finite nonnegative number")
        if type(self.top_p) not in (int, float) or not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if (
            type(self.gpu_memory_utilization) not in (int, float)
            or not math.isfinite(self.gpu_memory_utilization)
            or not 0 < self.gpu_memory_utilization <= 1
        ):
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        for name in (
            "max_new_tokens",
            "batch_size",
            "tensor_parallel_size",
            "max_model_len",
            "samples_per_question",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "GenerationConfig":
        if not isinstance(payload, Mapping) or set(payload) != {field.name for field in fields(cls)}:
            raise ValueError("GenerationConfig fields do not match the schema")
        return cls(**dict(payload))


@dataclass(frozen=True)
class GenerationResult:
    plan: CandidatePlan
    response: str | None
    finish_reason: str | None
    generated_token_count: int
    error_type: str | None

    def __post_init__(self):
        self.plan.validate()
        if type(self.generated_token_count) is not int or self.generated_token_count < 0:
            raise ValueError("generated_token_count must be a nonnegative integer")
        if self.error_type is None:
            if not isinstance(self.response, str) or not isinstance(self.finish_reason, str) or not self.finish_reason:
                raise ValueError("successful generation requires response and finish_reason")
        elif (
            not isinstance(self.error_type, str)
            or not self.error_type.strip()
            or self.response is not None
            or self.finish_reason is not None
            or self.generated_token_count != 0
        ):
            raise ValueError("failed generation must contain only a sanitized error_type")


@dataclass(frozen=True)
class GenerationPlan:
    generation_run_id: str
    config_hash: str
    candidates: tuple[CandidatePlan, ...]
    shard_ids: tuple[str, ...]
    generator_configs: tuple[GenerationConfig, ...]
    shard_size: int
    split_upstream_manifest_hash: str


@dataclass(frozen=True)
class GenerationRequest:
    request_id: str
    plan: CandidatePlan


class GenerationBackend(Protocol):
    def generate(
        self,
        batch: Sequence[CandidatePlan],
        config: GenerationConfig,
    ) -> list[GenerationResult]: ...


def generation_stage_config(
    generators: Sequence[GenerationConfig],
    *,
    samples_per_question: int,
    shard_size: int,
    split_upstream_manifest_hash: str | None = None,
) -> dict[str, object]:
    """Bind the full generation recipe to its role-tagged split manifest."""
    configs = _validated_configs(generators, samples_per_question)
    split_hash = configs[0].split_manifest_hash if split_upstream_manifest_hash is None else split_upstream_manifest_hash
    return role_bound_stage_config(
        {
            "generation_plan_schema": GENERATION_PLAN_SCHEMA,
            "generators": [config.to_dict() for config in configs],
            "samples_per_question": samples_per_question,
            "shard_size": shard_size,
        },
        {"split": split_hash},
        {"split"},
    )


def plan_generation(
    smoke_assignments: Sequence[SplitAssignment],
    generators: Sequence[GenerationConfig],
    samples_per_question: int = 2,
    shard_size: int = 8,
    *,
    split_upstream_manifest_hash: str | None = None,
) -> GenerationPlan:
    """Create deterministic run-scoped candidate plans and per-generator shards."""
    if type(samples_per_question) is not int or samples_per_question <= 0:
        raise ValueError("samples_per_question must be a positive integer")
    if type(shard_size) is not int or shard_size <= 0:
        raise ValueError("shard_size must be a positive integer")
    configs = _validated_configs(generators, samples_per_question)
    assignments = _validated_assignments(smoke_assignments)

    typed_config = generation_stage_config(
        configs,
        samples_per_question=samples_per_question,
        shard_size=shard_size,
        split_upstream_manifest_hash=split_upstream_manifest_hash,
    )
    generation_run_id = run_id("generation", typed_config)
    config_hash = sha256_hex(canonical_json_bytes(typed_config))

    candidates: list[CandidatePlan] = []
    shard_ids: list[str] = []
    for config in configs:
        for position, (assignment, sampling_index) in enumerate(
            (assignment, sampling_index)
            for assignment in assignments
            for sampling_index in range(samples_per_question)
        ):
            shard_id = f"{position // shard_size:05d}"
            composite_shard_id = f"{config.generator_id}/{shard_id}"
            if not shard_ids or shard_ids[-1] != composite_shard_id:
                shard_ids.append(composite_shard_id)
            candidates.append(
                CandidatePlan(
                    schema_version=CANDIDATE_PLAN_SCHEMA,
                    generation_run_id=generation_run_id,
                    candidate_id=logical_candidate_id(
                        assignment.original_question_id,
                        config.generator_id,
                        sampling_index,
                    ),
                    original_question_id=assignment.original_question_id,
                    question_content_hash=assignment.question_content_hash,
                    source=assignment.source,
                    split=assignment.split,
                    split_manifest_hash=config.split_manifest_hash,
                    generator_id=config.generator_id,
                    sampling_index=sampling_index,
                    shard_id=shard_id,
                    prompt_hash=config.prompt_template_hash,
                )
            )
    return GenerationPlan(
        generation_run_id=generation_run_id,
        config_hash=config_hash,
        candidates=tuple(candidates),
        shard_ids=tuple(shard_ids),
        generator_configs=configs,
        shard_size=shard_size,
        split_upstream_manifest_hash=typed_config["upstream_bindings"]["split"],
    )


def _validated_configs(
    generators: Sequence[GenerationConfig],
    samples_per_question: int,
) -> tuple[GenerationConfig, ...]:
    if isinstance(generators, (str, bytes)) or not isinstance(generators, Sequence) or not generators:
        raise ValueError("generators must be a nonempty sequence")
    configs = tuple(generators)
    for config in configs:
        if not isinstance(config, GenerationConfig):
            raise TypeError("generators must contain GenerationConfig values")
        config.validate()
        if config.samples_per_question != samples_per_question:
            raise ValueError("samples_per_question does not match generator configuration")
    if len({config.generator_id for config in configs}) != len(configs):
        raise ValueError("generator_id values must be unique")
    if len({config.split_manifest_hash for config in configs}) != 1:
        raise ValueError("generator configs must bind the same split manifest")
    return tuple(sorted(configs, key=lambda item: item.generator_id))


def _validated_assignments(
    smoke_assignments: Sequence[SplitAssignment],
) -> tuple[SplitAssignment, ...]:
    if isinstance(smoke_assignments, (str, bytes)) or not isinstance(smoke_assignments, Sequence) or not smoke_assignments:
        raise ValueError("smoke_assignments must be a nonempty sequence")
    assignments = tuple(smoke_assignments)
    for assignment in assignments:
        if not isinstance(assignment, SplitAssignment):
            raise TypeError("smoke_assignments must contain SplitAssignment values")
        assignment.validate()
        if assignment.split is not Split.TRAIN:
            raise ValueError("smoke assignments must come from the internal training split")
    if len({assignment.original_question_id for assignment in assignments}) != len(assignments):
        raise ValueError("smoke assignments must have unique original_question_id values")
    return tuple(sorted(assignments, key=lambda item: item.original_question_id))
