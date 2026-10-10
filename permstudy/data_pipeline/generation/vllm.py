"""Lazy vLLM adapter boundary; Phase 1 never starts a real model."""

from collections.abc import Sequence
import importlib

from ..schema import CandidatePlan
from .base import (
    GenerationConfig,
    GenerationConfigurationError,
    GenerationRequest,
    GenerationResult,
)


def _request_id(plan: CandidatePlan) -> str:
    return f"{plan.generation_run_id}:{plan.candidate_id}"


class VLLMGenerationBackend:
    def __init__(self, engine_factory=None):
        self._engine_factory = engine_factory

    def generate(
        self,
        batch: Sequence[CandidatePlan],
        config: GenerationConfig,
    ) -> list[GenerationResult]:
        if config.backend != "vllm":
            raise GenerationConfigurationError("VLLMGenerationBackend requires backend='vllm'")
        try:
            vllm_module = importlib.import_module("vllm")
        except ImportError as error:
            raise GenerationConfigurationError("vLLM backend is unavailable in this environment") from error
        if self._engine_factory is None:
            raise GenerationConfigurationError("Phase 1 vLLM adapter requires an injected engine_factory")

        plans_by_request_id: dict[str, CandidatePlan] = {}
        requests: list[GenerationRequest] = []
        for plan in batch:
            plan.validate()
            if plan.generator_id != config.generator_id:
                raise GenerationConfigurationError("candidate generator does not match generation config")
            request_id = _request_id(plan)
            if request_id in plans_by_request_id:
                raise GenerationConfigurationError("duplicate generation request identity")
            plans_by_request_id[request_id] = plan
            requests.append(GenerationRequest(request_id=request_id, plan=plan))

        engine = self._engine_factory(vllm_module, config)
        raw_outputs = engine.generate(tuple(requests), config)
        results: list[GenerationResult] = []
        observed: set[str] = set()
        for raw_output in raw_outputs:
            request_id = getattr(raw_output, "request_id", None)
            if request_id not in plans_by_request_id or request_id in observed:
                raise GenerationConfigurationError("vLLM returned a foreign or duplicate request identity")
            completions = getattr(raw_output, "outputs", None)
            if not isinstance(completions, (list, tuple)) or len(completions) != 1:
                raise GenerationConfigurationError("vLLM must return exactly one completion per planned request")
            completion = completions[0]
            text = getattr(completion, "text", None)
            finish_reason = getattr(completion, "finish_reason", None)
            token_ids = getattr(completion, "token_ids", None)
            if not isinstance(text, str) or not isinstance(finish_reason, str) or not isinstance(token_ids, (list, tuple)):
                raise GenerationConfigurationError("vLLM returned a malformed completion")
            observed.add(request_id)
            results.append(
                GenerationResult(
                    plan=plans_by_request_id[request_id],
                    response=text,
                    finish_reason=finish_reason,
                    generated_token_count=len(token_ids),
                    error_type=None,
                )
            )
        if observed != set(plans_by_request_id):
            raise GenerationConfigurationError("vLLM did not return every planned request")
        return results
