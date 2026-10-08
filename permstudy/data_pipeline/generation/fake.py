"""Deterministic fake generation backend for Phase 1 tests and dry runs."""

from collections.abc import Mapping, Sequence
from types import MappingProxyType

from ..schema import CandidatePlan
from .base import GenerationConfig, GenerationResult


class FakeGenerationBackend:
    def __init__(self, response_map: Mapping[tuple[str, str], GenerationResult]):
        if not isinstance(response_map, Mapping):
            raise TypeError("response_map must be a mapping")
        copied = dict(response_map)
        for key, result in copied.items():
            if not isinstance(key, tuple) or len(key) != 2 or not all(isinstance(item, str) for item in key):
                raise ValueError("fake responses must use composite candidate keys")
            if not isinstance(result, GenerationResult) or result.plan.key != key:
                raise ValueError("fake response identity does not match its composite key")
        self._responses = MappingProxyType(copied)

    def generate(
        self,
        batch: Sequence[CandidatePlan],
        config: GenerationConfig,
    ) -> list[GenerationResult]:
        if config.backend != "fake":
            raise ValueError("FakeGenerationBackend requires backend='fake'")
        results = []
        for plan in batch:
            plan.validate()
            if plan.generator_id != config.generator_id:
                raise ValueError("candidate generator does not match generation config")
            try:
                result = self._responses[plan.key]
            except KeyError as error:
                raise ValueError("fake response is missing for a planned composite key") from error
            if result.plan != plan:
                raise ValueError("fake response plan does not match the requested candidate")
            results.append(result)
        return results
