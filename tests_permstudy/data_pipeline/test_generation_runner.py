import json
from dataclasses import replace

import pytest

from permstudy.data_pipeline.io import IntegrityError, scan_jsonl
from permstudy.data_pipeline.schema import CandidateRecord, FailureRecord, Source, Split, SplitAssignment


GENERATOR_ID = "qwen2.5-7b-instruct"
MODEL_REPOSITORY = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "1" * 40
SPLIT_HASH = "a" * 64
PROMPT_HASH = "b" * 64


def assignments(count=2):
    return tuple(
        SplitAssignment(
            original_question_id=f"math:train:{index:03d}",
            question_content_hash=f"{index + 1:064x}",
            source=Source.MATH,
            split=Split.TRAIN,
            stratum="source=math",
        )
        for index in range(count)
    )


def generation_config(**overrides):
    from permstudy.data_pipeline.generation import GenerationConfig

    values = dict(
        backend="fake",
        generator_id=GENERATOR_ID,
        model_repository=MODEL_REPOSITORY,
        model_revision=MODEL_REVISION,
        prompt_template_revision="candidate_prompt_v1",
        prompt_template_hash=PROMPT_HASH,
        temperature=0.8,
        top_p=0.95,
        max_new_tokens=512,
        seed=42,
        batch_size=1,
        tensor_parallel_size=1,
        max_model_len=4096,
        gpu_memory_utilization=0.8,
        samples_per_question=2,
        split_manifest_hash=SPLIT_HASH,
    )
    values.update(overrides)
    return GenerationConfig(**values)


def shard_plan(*, config=None, question_count=2):
    from permstudy.data_pipeline.generation import build_generation_shard_plan, plan_generation

    config = config or generation_config()
    generation_plan = plan_generation(assignments(question_count), (config,), shard_size=8)
    return generation_plan, build_generation_shard_plan(generation_plan, GENERATOR_ID, "00000")


def success(plan):
    from permstudy.data_pipeline.generation import GenerationResult

    return GenerationResult(
        plan=plan,
        response=f"response:{plan.candidate_id}",
        finish_reason="stop",
        generated_token_count=3,
        error_type=None,
    )


def failure(plan, error_type="timeout"):
    from permstudy.data_pipeline.generation import GenerationResult

    return GenerationResult(plan, None, None, 0, error_type)


class ScriptedBackend:
    def __init__(self, actions):
        self.actions = list(actions)
        self.calls = []
        self.configs = []

    def generate(self, batch, config):
        self.calls.extend(plan.key for plan in batch)
        self.configs.append(config)
        action = self.actions.pop(0)
        if isinstance(action, BaseException):
            raise action
        if action == "success":
            return [success(plan) for plan in reversed(batch)]
        if isinstance(action, tuple) and action[0] == "failure":
            return [failure(plan, action[1]) for plan in reversed(batch)]
        raise AssertionError(f"unknown test action: {action!r}")


class NoCallBackend:
    def generate(self, batch, config):
        raise AssertionError("backend must not be called")


def read_business_records(path, record_type):
    if not path.exists():
        return []
    return [
        record_type.from_dict({key: value for key, value in row.items() if key != "record_hash"})
        for row in scan_jsonl(path).records
    ]


def test_generation_plan_carries_complete_configs_and_builds_one_validated_shard():
    full_plan, shard = shard_plan()

    assert full_plan.generator_configs == (generation_config(),)
    assert full_plan.shard_size == 8
    assert shard.generation_run_id == full_plan.generation_run_id
    assert shard.config_hash == full_plan.config_hash
    assert shard.generator_config == generation_config()
    assert shard.generator_configs == full_plan.generator_configs
    assert shard.shard_id == "00000"
    assert shard.candidates == full_plan.candidates


@pytest.mark.parametrize(
    "mutate",
    [
        lambda plan: replace(plan, prompt_hash="c" * 64),
        lambda plan: replace(plan, split_manifest_hash="c" * 64),
        lambda plan: replace(plan, generator_id="qwen2.5-32b-instruct"),
        lambda plan: replace(plan, sampling_index=99),
        lambda plan: replace(plan, candidate_id="d" * 64),
        lambda plan: replace(plan, generation_run_id="e" * 64),
    ],
    ids=["prompt", "split", "generator", "sampling-index", "logical-id", "run-namespace"],
)
def test_runner_rejects_candidate_config_mismatch_before_any_write(tmp_path, mutate):
    from permstudy.data_pipeline.generation import RunMismatchError, run_generation_shard

    _, shard = shard_plan()
    bad = replace(shard, candidates=(mutate(shard.candidates[0]), *shard.candidates[1:]))
    output = tmp_path / "private" / "shard"

    with pytest.raises(RunMismatchError):
        run_generation_shard(bad, NoCallBackend(), output)
    assert not output.exists()


def test_interruption_preserves_appends_and_resume_only_calls_missing_candidates(tmp_path):
    from permstudy.data_pipeline.generation import run_generation_shard, successful_candidate_keys

    _, shard = shard_plan()
    output = tmp_path / "interrupted"
    first = ScriptedBackend(["success", "success", ("failure", "timeout"), KeyboardInterrupt()])

    with pytest.raises(KeyboardInterrupt):
        run_generation_shard(shard, first, output)

    stored_successes = read_business_records(output / "candidates.jsonl", CandidateRecord)
    stored_failures = read_business_records(output / "failures.jsonl", FailureRecord)
    assert [record.plan.key for record in stored_successes] == [plan.key for plan in shard.candidates[:2]]
    assert [(record.plan.key, record.retry_count) for record in stored_failures] == [
        (shard.candidates[2].key, 0)
    ]
    assert not (output / ".lock").exists()

    resume = ScriptedBackend(["success", "success"])
    summary = run_generation_shard(shard, resume, output)

    assert resume.calls == [plan.key for plan in shard.candidates[2:]]
    assert resume.configs == [shard.generator_config, shard.generator_config]
    assert summary.planned == 4
    assert summary.successful == 4
    assert summary.historical_failures == 1
    assert summary.missing == 0
    assert successful_candidate_keys(output) == frozenset(plan.key for plan in shard.candidates)
    assert len(read_business_records(output / "failures.jsonl", FailureRecord)) == 1


def test_interrupted_and_uninterrupted_runs_have_equal_semantic_results(tmp_path):
    from permstudy.data_pipeline.generation import run_generation_shard, successful_candidate_keys

    _, shard = shard_plan()
    interrupted = tmp_path / "interrupted"
    uninterrupted = tmp_path / "uninterrupted"

    with pytest.raises(KeyboardInterrupt):
        run_generation_shard(
            shard,
            ScriptedBackend(["success", "success", ("failure", "timeout"), KeyboardInterrupt()]),
            interrupted,
        )
    resumed = run_generation_shard(shard, ScriptedBackend(["success", "success"]), interrupted)
    direct = run_generation_shard(shard, ScriptedBackend(["success"] * 4), uninterrupted)

    assert successful_candidate_keys(interrupted) == successful_candidate_keys(uninterrupted)
    assert resumed.semantic_candidate_set_hash == direct.semantic_candidate_set_hash
    assert resumed.manifest_hash != direct.manifest_hash


@pytest.mark.parametrize(
    "changed",
    [
        {"temperature": 0.7},
        {"top_p": 0.9},
        {"max_new_tokens": 256},
        {"seed": 43},
        {"batch_size": 2},
        {"tensor_parallel_size": 2},
        {"max_model_len": 8192},
        {"gpu_memory_utilization": 0.7},
        {"model_revision": "2" * 40},
        {"prompt_template_hash": "c" * 64},
        {"split_manifest_hash": "c" * 64},
    ],
    ids=[
        "temperature",
        "top-p",
        "max-new-tokens",
        "seed",
        "batch-size",
        "tensor-parallel",
        "max-model-len",
        "memory-utilization",
        "model-revision",
        "prompt-hash",
        "split-hash",
    ],
)
def test_changed_config_and_run_cannot_resume_an_existing_shard(tmp_path, changed):
    from permstudy.data_pipeline.generation import RunMismatchError, run_generation_shard

    first_plan, first_shard = shard_plan(config=generation_config(temperature=0.8))
    output = tmp_path / "shard"
    run_generation_shard(first_shard, ScriptedBackend(["success"] * 4), output)
    before = (output / "candidates.jsonl").read_bytes()

    second_plan, second_shard = shard_plan(config=generation_config(**changed))
    assert [item.candidate_id for item in first_plan.candidates] == [item.candidate_id for item in second_plan.candidates]
    assert first_plan.generation_run_id != second_plan.generation_run_id

    with pytest.raises(RunMismatchError, match="manifest"):
        run_generation_shard(second_shard, NoCallBackend(), output)
    assert (output / "candidates.jsonl").read_bytes() == before


def test_manifest_pins_complete_config_and_execution_counts(tmp_path):
    from permstudy.data_pipeline.generation import run_generation_shard

    _, shard = shard_plan()
    output = tmp_path / "shard"
    summary = run_generation_shard(shard, ScriptedBackend(["success"] * 4), output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["generation_config"] == shard.generator_config.to_dict()
    assert manifest["generation_run_id"] == shard.generation_run_id
    assert manifest["config_hash"] == shard.config_hash
    assert manifest["generator_id"] == GENERATOR_ID
    assert manifest["shard_id"] == "00000"
    assert manifest["split_manifest_hash"] == SPLIT_HASH
    assert manifest["prompt_template_hash"] == PROMPT_HASH
    assert manifest["counts"] == {"historical_failures": 0, "missing": 0, "planned": 4, "successful": 4}
    assert manifest["output_manifest_hash"] == summary.manifest_hash


def test_runner_quarantines_an_incomplete_tail_before_resume(tmp_path):
    from permstudy.data_pipeline.generation import run_generation_shard

    _, shard = shard_plan(question_count=1)
    output = tmp_path / "shard"
    run_generation_shard(shard, ScriptedBackend(["success", "success"]), output)
    with (output / "candidates.jsonl").open("ab") as stream:
        stream.write(b'{"partial"')

    summary = run_generation_shard(shard, NoCallBackend(), output)

    assert summary.successful == 2
    quarantines = list(output.glob("candidates.*.quarantine"))
    assert len(quarantines) == 1
    assert quarantines[0].read_bytes() == b'{"partial"'
    assert (output / "candidates.jsonl").read_bytes().endswith(b"\n")


def test_runner_refuses_a_malformed_committed_middle_line(tmp_path):
    from permstudy.data_pipeline.generation import run_generation_shard

    _, shard = shard_plan(question_count=1)
    output = tmp_path / "shard"
    run_generation_shard(shard, ScriptedBackend(["success", "success"]), output)
    with (output / "candidates.jsonl").open("ab") as stream:
        stream.write(b"not-json\n")

    with pytest.raises(IntegrityError, match="middle line"):
        run_generation_shard(shard, NoCallBackend(), output)
    assert not (output / ".lock").exists()


def test_backend_oom_is_a_recoverable_failure_record(tmp_path):
    from permstudy.data_pipeline.generation import run_generation_shard

    class OutOfMemoryError(RuntimeError):
        pass

    _, shard = shard_plan(question_count=1)
    output = tmp_path / "shard"
    summary = run_generation_shard(
        shard,
        ScriptedBackend([OutOfMemoryError("private payload"), OutOfMemoryError("private payload")]),
        output,
    )

    failures = read_business_records(output / "failures.jsonl", FailureRecord)
    assert [record.error_type for record in failures] == ["oom", "oom"]
    assert summary.successful == 0
    assert summary.historical_failures == 2
    assert summary.missing == 2


def test_runner_rejects_an_unsanitized_backend_error_before_appending_it(tmp_path):
    from permstudy.data_pipeline.generation import RunMismatchError, run_generation_shard

    _, shard = shard_plan(question_count=1)
    output = tmp_path / "shard"
    backend = ScriptedBackend([("failure", "C:/private/model/error")])

    with pytest.raises(RunMismatchError, match="sanitized error type"):
        run_generation_shard(shard, backend, output)
    assert read_business_records(output / "failures.jsonl", FailureRecord) == []


def test_semantic_hash_ignores_record_order_but_binds_semantic_payload():
    from permstudy.data_pipeline.generation import semantic_candidate_set_hash

    _, shard = shard_plan(question_count=1)
    records = tuple(
        CandidateRecord(plan, f" response:{plan.candidate_id}\r\n", "stop", 3, MODEL_REVISION)
        for plan in shard.candidates
    )

    expected = semantic_candidate_set_hash(records)
    assert semantic_candidate_set_hash(reversed(records)) == expected
    changed = (replace(records[0], generated_token_count=4), *records[1:])
    assert semantic_candidate_set_hash(changed) != expected
