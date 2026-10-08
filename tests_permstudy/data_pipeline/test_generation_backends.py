import importlib
from dataclasses import replace
from itertools import permutations
from types import SimpleNamespace

import pytest

from permstudy.data_pipeline.schema import Source, Split, SplitAssignment


REVISION_7B = "1" * 40
REVISION_32B = "2" * 40
REVISION_LLAMA = "3" * 40
SPLIT_HASH = "a" * 64
PROMPT_HASH = "b" * 64


def assignments():
    return tuple(
        SplitAssignment(
            original_question_id=f"{source.value}:train:{index:03d}",
            question_content_hash=f"{index + offset:064x}",
            source=source,
            split=Split.TRAIN,
            stratum=f"{source.value}-stratum",
        )
        for source, offset in ((Source.MATH, 1), (Source.RECLOR, 101))
        for index in range(20)
    )


def generator_configs(*, backend="fake", temperature=0.8, split_manifest_hash=SPLIT_HASH):
    from permstudy.data_pipeline.generation import GenerationConfig

    common = dict(
        backend=backend,
        prompt_template_revision="candidate_prompt_v1",
        prompt_template_hash=PROMPT_HASH,
        temperature=temperature,
        top_p=0.95,
        max_new_tokens=512,
        seed=42,
        batch_size=8,
        tensor_parallel_size=1,
        max_model_len=4096,
        gpu_memory_utilization=0.8,
        samples_per_question=2,
        split_manifest_hash=split_manifest_hash,
    )
    return (
        GenerationConfig(
            generator_id="qwen2.5-7b-instruct",
            model_repository="Qwen/Qwen2.5-7B-Instruct",
            model_revision=REVISION_7B,
            **common,
        ),
        GenerationConfig(
            generator_id="qwen2.5-32b-instruct",
            model_repository="Qwen/Qwen2.5-32B-Instruct",
            model_revision=REVISION_32B,
            **common,
        ),
        GenerationConfig(
            generator_id="llama-3.1-8b-instruct",
            model_repository="meta-llama/Llama-3.1-8B-Instruct",
            model_revision=REVISION_LLAMA,
            **common,
        ),
    )


def test_fixed_smoke_recipe_has_240_unique_composite_candidate_keys():
    from permstudy.data_pipeline.generation import plan_generation

    plan = plan_generation(assignments(), generator_configs())

    assert len(plan.candidates) == 240
    assert len({candidate.key for candidate in plan.candidates}) == 240
    assert len(plan.shard_ids) == 30
    assert plan.shard_ids[0] == "llama-3.1-8b-instruct/00000"


def test_candidate_ids_are_logical_while_generation_run_ids_bind_config():
    from permstudy.data_pipeline.generation import plan_generation

    first = plan_generation(assignments(), generator_configs(temperature=0.8))
    second = plan_generation(assignments(), generator_configs(temperature=0.7))

    first_ids = {
        (item.generator_id, item.original_question_id, item.sampling_index): item.candidate_id
        for item in first.candidates
    }
    second_ids = {
        (item.generator_id, item.original_question_id, item.sampling_index): item.candidate_id
        for item in second.candidates
    }
    assert first_ids == second_ids
    assert first.generation_run_id != second.generation_run_id
    assert first.config_hash != second.config_hash
    assert {item.generation_run_id for item in first.candidates} == {first.generation_run_id}
    assert {item.generation_run_id for item in second.candidates} == {second.generation_run_id}


def test_generation_plan_and_shards_are_input_order_invariant():
    from permstudy.data_pipeline.generation import plan_generation

    expected = plan_generation(assignments(), generator_configs())
    reordered = plan_generation(
        tuple(reversed(assignments())),
        tuple(reversed(generator_configs())),
    )

    assert reordered == expected
    sort_keys = [
        (item.generator_id, item.original_question_id, item.sampling_index)
        for item in expected.candidates
    ]
    assert sort_keys == sorted(sort_keys)
    for generator_id in {item.generator_id for item in expected.candidates}:
        generator_candidates = [item for item in expected.candidates if item.generator_id == generator_id]
        assert [item.shard_id for item in generator_candidates[:9]] == ["00000"] * 8 + ["00001"]

    small_assignments = assignments()[:3]
    small_generators = generator_configs()
    small_expected = plan_generation(small_assignments, small_generators)
    for assignment_order in permutations(small_assignments):
        for generator_order in permutations(small_generators):
            assert plan_generation(assignment_order, generator_order) == small_expected


def test_generation_run_lineage_changes_when_split_binding_changes():
    from permstudy.data_pipeline.generation import plan_generation

    first = plan_generation(assignments(), generator_configs(split_manifest_hash="a" * 64))
    second = plan_generation(assignments(), generator_configs(split_manifest_hash="c" * 64))

    assert first.generation_run_id != second.generation_run_id
    assert first.config_hash != second.config_hash


def test_generator_revisions_must_be_immutable_sha_values():
    from permstudy.data_pipeline.generation import GENERATOR_REPOSITORIES, GenerationConfig

    valid = generator_configs()[0]
    with pytest.raises(ValueError, match="immutable model revision"):
        replace(valid, model_revision="main")
    with pytest.raises(ValueError, match="repository"):
        replace(valid, model_repository="some/fork")
    with pytest.raises(ValueError, match="generator_id"):
        replace(valid, generator_id="unknown-generator")
    with pytest.raises(TypeError):
        GENERATOR_REPOSITORIES["new-generator"] = "new/repository"
    assert GenerationConfig.from_dict(valid.to_dict()) == valid


def test_fake_backend_emits_success_retryable_failure_and_length_result():
    from permstudy.data_pipeline.generation import FakeGenerationBackend, GenerationResult, plan_generation

    config = generator_configs()[0]
    candidates = plan_generation(assignments()[:2], (config,)).candidates[:3]
    mapped = {
        candidates[0].key: GenerationResult(candidates[0], "answer", "stop", 4, None),
        candidates[1].key: GenerationResult(candidates[1], None, None, 0, "timeout"),
        candidates[2].key: GenerationResult(candidates[2], "truncated", "length", 512, None),
    }
    backend = FakeGenerationBackend(dict(reversed(tuple(mapped.items()))))

    results = backend.generate((candidates[2], candidates[0], candidates[1]), config)

    assert [result.plan.key for result in results] == [
        candidate.key for candidate in (candidates[2], candidates[0], candidates[1])
    ]
    assert [(result.finish_reason, result.error_type) for result in results] == [
        ("length", None),
        ("stop", None),
        (None, "timeout"),
    ]


def test_importing_generation_does_not_attempt_to_import_vllm(monkeypatch):
    generation = importlib.import_module("permstudy.data_pipeline.generation")
    real_import = importlib.import_module

    def guarded_import(name, *args, **kwargs):
        if name == "vllm":
            raise AssertionError("package import attempted to load vLLM")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", guarded_import)
    importlib.reload(generation)


def test_vllm_constructor_is_lazy_and_generate_maps_outputs_by_request_id(monkeypatch):
    from permstudy.data_pipeline.generation import VLLMGenerationBackend, plan_generation

    config = replace(generator_configs(backend="vllm")[0], batch_size=2)
    candidates = plan_generation(assignments()[:1], (config,)).candidates
    imported = []
    real_import = importlib.import_module

    fake_vllm = SimpleNamespace(__name__="fake-vllm")

    def fake_import(name, *args, **kwargs):
        if name == "vllm":
            imported.append(name)
            return fake_vllm
        return real_import(name, *args, **kwargs)

    class Engine:
        def generate(self, requests, received_config):
            assert received_config is config
            return [
                SimpleNamespace(
                    request_id=request.request_id,
                    outputs=[SimpleNamespace(text=f"response-{index}", finish_reason="stop", token_ids=[1, 2, 3])],
                )
                for index, request in enumerate(reversed(requests))
            ]

    factories = []

    def engine_factory(module, received_config):
        factories.append((module, received_config))
        return Engine()

    monkeypatch.setattr(importlib, "import_module", fake_import)
    backend = VLLMGenerationBackend(engine_factory=engine_factory)
    assert imported == []

    results = backend.generate(candidates, config)

    assert imported == ["vllm"]
    assert factories == [(fake_vllm, config)]
    assert [result.plan for result in results] == list(reversed(candidates))
    assert [result.response for result in results] == ["response-0", "response-1"]
    assert [result.generated_token_count for result in results] == [3, 3]
    assert all(result.finish_reason == "stop" and result.error_type is None for result in results)


def test_vllm_missing_dependency_has_sanitized_configuration_error(monkeypatch):
    from permstudy.data_pipeline.generation import GenerationConfigurationError, VLLMGenerationBackend, plan_generation

    config = generator_configs(backend="vllm")[0]
    candidate = plan_generation(assignments()[:1], (config,)).candidates[0]
    real_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name == "vllm":
            raise ImportError("private third-party payload")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    with pytest.raises(GenerationConfigurationError, match="vLLM backend is unavailable") as caught:
        VLLMGenerationBackend(engine_factory=lambda *_: None).generate((candidate,), config)
    assert "private third-party payload" not in str(caught.value)


def test_vllm_phase1_adapter_requires_an_injected_engine_factory(monkeypatch):
    from permstudy.data_pipeline.generation import GenerationConfigurationError, VLLMGenerationBackend, plan_generation

    config = generator_configs(backend="vllm")[0]
    candidate = plan_generation(assignments()[:1], (config,)).candidates[0]
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: SimpleNamespace() if name == "vllm" else importlib.import_module(name),
    )

    with pytest.raises(GenerationConfigurationError, match="injected engine_factory"):
        VLLMGenerationBackend().generate((candidate,), config)
