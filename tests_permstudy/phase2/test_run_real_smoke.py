"""Focused CPU-only tests for the Phase 2 real-smoke coordinator."""

import importlib
import json
from types import SimpleNamespace

import pytest

from permstudy.data_pipeline.schema import (
    CandidateRecord,
    QuestionRecord,
    Source,
    Split,
    SplitAssignment,
    VerificationStatus,
)


MODEL_REVISION = "1" * 40
SPLIT_HASH = "2" * 64
SPLIT_ENVELOPE_HASH = "3" * 64
TOKENIZER_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"


def reclor_question(index=1):
    return QuestionRecord(
        schema_version="question_v1",
        source=Source.RECLOR,
        original_question_id=f"reclor:train:synthetic_{index}",
        question_content_hash=f"{index:064x}",
        source_snapshot_id="synthetic-reclor",
        source_revision="synthetic-v1",
        source_row_id=str(index),
        context=f"Context {index}",
        question=f"Question {index}?",
        answers=("Alpha", "Beta", "Gamma", "Delta"),
        gold_label="A",
        problem=None,
        solution=None,
        category=None,
        level=None,
        synthetic=True,
    )


def math_question(index=2):
    return QuestionRecord(
        schema_version="question_v1",
        source=Source.MATH,
        original_question_id=f"math:train:synthetic_{index}",
        question_content_hash=f"{index:064x}",
        source_snapshot_id="synthetic-math",
        source_revision="synthetic-v1",
        source_row_id=str(index),
        context=None,
        question=None,
        answers=(),
        gold_label=None,
        problem=f"Compute {index}+{index}.",
        solution=f"Private solution {index + index}",
        category="algebra",
        level="1",
        synthetic=True,
    )


def assignment(question):
    return SplitAssignment(
        original_question_id=question.original_question_id,
        question_content_hash=question.question_content_hash,
        source=question.source,
        split=Split.TRAIN,
        stratum=f"source={question.source.value}",
    )


def generation_config(*, samples=1, batch_size=8):
    from permstudy.data_pipeline.generation import GenerationConfig
    from scripts_permstudy.phase2.run_real_smoke import (
        PROMPT_TEMPLATE_REVISION,
        prompt_contract_hash,
    )

    return GenerationConfig(
        backend="vllm",
        generator_id="qwen2.5-7b-instruct",
        model_repository="Qwen/Qwen2.5-7B-Instruct",
        model_revision=MODEL_REVISION,
        prompt_template_revision=PROMPT_TEMPLATE_REVISION,
        prompt_template_hash=prompt_contract_hash(),
        temperature=0.8,
        top_p=0.95,
        max_new_tokens=512,
        seed=42,
        batch_size=batch_size,
        tensor_parallel_size=1,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        samples_per_question=samples,
        split_manifest_hash=SPLIT_HASH,
    )


def generation_plan(questions, *, samples=1, shard_size=8, batch_size=8):
    from permstudy.data_pipeline.generation import plan_generation

    return plan_generation(
        tuple(assignment(question) for question in questions),
        (generation_config(samples=samples, batch_size=batch_size),),
        samples_per_question=samples,
        shard_size=shard_size,
        split_upstream_manifest_hash=SPLIT_ENVELOPE_HASH,
    )


class FakeTokenizer:
    def __init__(self):
        self.messages = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        self.messages.append(messages)
        return "\n".join(f"{item['role']}::{item['content']}" for item in messages)


class RecordingSamplingParams:
    def __init__(self, **values):
        self.values = values


class ScriptedLowLevelEngine:
    def __init__(self, *, mode="reverse", interrupt_on_step=None):
        self.tokenizer = FakeTokenizer()
        self.mode = mode
        self.interrupt_on_step = interrupt_on_step
        self.added = []
        self.pending = []
        self.step_count = 0

    def get_tokenizer(self):
        return self.tokenizer

    def add_request(self, request_id, prompt, params):
        self.added.append((request_id, prompt, params))
        self.pending.append((request_id, prompt))

    def has_unfinished_requests(self):
        return bool(self.pending)

    def step(self):
        self.step_count += 1
        if self.step_count == self.interrupt_on_step:
            raise KeyboardInterrupt()
        pending, self.pending = self.pending, []
        rows = [
            SimpleNamespace(
                request_id=request_id,
                finished=True,
                prompt=prompt,
                outputs=[
                    SimpleNamespace(
                        text=f"response-for:{request_id}",
                        finish_reason="stop",
                        token_ids=[11, 12],
                    )
                ],
            )
            for request_id, prompt in pending
        ]
        if self.mode == "reverse":
            return list(reversed(rows))
        if self.mode == "missing":
            return rows[:-1]
        if self.mode == "duplicate":
            return rows + rows[:1]
        if self.mode == "foreign":
            return [replace_namespace(rows[0], request_id="foreign")]
        return rows


def replace_namespace(value, **changes):
    payload = vars(value).copy()
    payload.update(changes)
    return SimpleNamespace(**payload)


def fake_vllm_module(engine, constructed):
    class EngineArgs:
        def __init__(self, **values):
            self.values = values

    class LLMEngine:
        @classmethod
        def from_engine_args(cls, args):
            constructed.append(args.values)
            return engine

    return SimpleNamespace(
        EngineArgs=EngineArgs,
        LLMEngine=LLMEngine,
        SamplingParams=RecordingSamplingParams,
    )


def install_fake_vllm(monkeypatch, module):
    real_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name == "vllm":
            return module
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import)


def test_fixed_real_recipe_builds_only_the_three_approved_vllm_configs():
    from scripts_permstudy.phase2.run_real_smoke import (
        build_real_generation_configs,
        prompt_contract_hash,
    )

    configs = build_real_generation_configs(SPLIT_HASH)

    assert [config.generator_id for config in configs] == [
        "qwen2.5-7b-instruct",
        "llama-3.1-8b-instruct",
        "qwen2.5-32b-instruct",
    ]
    assert [config.model_revision for config in configs] == [
        "a09a35458c702b33eeacc393d103063234e8bc28",
        "0e9e39f249a16976918f6564b8830bc894c89659",
        "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd",
    ]
    assert [config.batch_size for config in configs] == [8, 8, 2]
    assert [config.gpu_memory_utilization for config in configs] == [
        0.85,
        0.85,
        0.90,
    ]
    assert all(
        config.backend == "vllm"
        and config.samples_per_question == 2
        and config.seed == 42
        and config.prompt_template_hash == prompt_contract_hash()
        for config in configs
    )
    assert prompt_contract_hash("candidate_request_seed_sha256_v2") != (
        prompt_contract_hash()
    )


def test_native_adapter_keeps_request_identity_when_engine_finishes_out_of_order(
    monkeypatch,
):
    from permstudy.data_pipeline.generation import VLLMGenerationBackend
    from scripts_permstudy.phase2.run_real_smoke import CachedEngineFactory

    questions = (reclor_question(), math_question())
    plan = generation_plan(questions)
    low_level = ScriptedLowLevelEngine(mode="reverse")
    constructed = []
    install_fake_vllm(monkeypatch, fake_vllm_module(low_level, constructed))
    backend = VLLMGenerationBackend(engine_factory=CachedEngineFactory(questions))

    results = backend.generate(plan.candidates, plan.generator_configs[0])

    by_candidate = {result.plan.candidate_id: result.response for result in results}
    assert by_candidate == {
        candidate.candidate_id: f"response-for:{plan.generation_run_id}:{candidate.candidate_id}"
        for candidate in plan.candidates
    }
    assert len(constructed) == 1
    assert constructed[0]["model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert constructed[0]["revision"] == MODEL_REVISION
    assert len({item[2].values["seed"] for item in low_level.added}) == 2
    rendered = "\n".join(item[1] for item in low_level.added)
    assert "Context 1" in rendered and "Question 1?" in rendered
    assert "Compute 2+2." in rendered
    assert "Private solution" not in rendered
    assert "gold_label" not in rendered


@pytest.mark.parametrize("mode", ["missing", "duplicate", "foreign"])
def test_native_adapter_fails_closed_on_incomplete_or_corrupt_engine_identity(
    monkeypatch, mode
):
    from permstudy.data_pipeline.generation import (
        GenerationConfigurationError,
        VLLMGenerationBackend,
    )
    from scripts_permstudy.phase2.run_real_smoke import CachedEngineFactory

    question = reclor_question()
    plan = generation_plan((question,), samples=2)
    low_level = ScriptedLowLevelEngine(mode=mode)
    install_fake_vllm(monkeypatch, fake_vllm_module(low_level, []))

    with pytest.raises(GenerationConfigurationError):
        VLLMGenerationBackend(engine_factory=CachedEngineFactory((question,))).generate(
            plan.candidates, plan.generator_configs[0]
        )


def test_generator_reuses_one_engine_across_shards_and_resume_only_runs_missing(
    monkeypatch, tmp_path
):
    from scripts_permstudy.phase2.run_real_smoke import (
        CachedEngineFactory,
        run_generator_shards,
    )

    questions = (reclor_question(1), reclor_question(2))
    plan = generation_plan(questions, shard_size=1, batch_size=1)
    interrupted_engine = ScriptedLowLevelEngine(interrupt_on_step=2)
    first_constructed = []
    install_fake_vllm(
        monkeypatch, fake_vllm_module(interrupted_engine, first_constructed)
    )

    with pytest.raises(KeyboardInterrupt):
        run_generator_shards(
            plan,
            "qwen2.5-7b-instruct",
            tmp_path / "generation",
            questions,
            engine_factory=CachedEngineFactory(questions),
        )

    assert len(first_constructed) == 1
    resumed_engine = ScriptedLowLevelEngine()
    resumed_constructed = []
    install_fake_vllm(
        monkeypatch, fake_vllm_module(resumed_engine, resumed_constructed)
    )
    summaries = run_generator_shards(
        plan,
        "qwen2.5-7b-instruct",
        tmp_path / "generation",
        questions,
        engine_factory=CachedEngineFactory(questions),
    )

    assert len(resumed_constructed) == 1
    assert [item[0] for item in resumed_engine.added] == [
        f"{plan.generation_run_id}:{plan.candidates[1].candidate_id}"
    ]
    assert [summary.successful for summary in summaries] == [1, 1]


class UnitTokenizer:
    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return text.split()


def test_tiny_real_candidate_shape_reaches_existing_verification_pair_and_permutation_apis():
    from scripts_permstudy.phase2.run_real_smoke import (
        build_pairs_and_permutations,
        verify_candidate_records,
    )

    question = reclor_question()
    plan = generation_plan((question,), samples=2)
    candidates = (
        CandidateRecord(
            plan=plan.candidates[0],
            response="Reasoning.\nFinal Answer: A",
            finish_reason="stop",
            generated_token_count=4,
            model_revision=MODEL_REVISION,
        ),
        CandidateRecord(
            plan=plan.candidates[1],
            response="Different reasoning.\nFinal Answer: B",
            finish_reason="stop",
            generated_token_count=5,
            model_revision=MODEL_REVISION,
        ),
    )

    verification_run_id, gold, verified = verify_candidate_records(
        (question,),
        candidates,
        generation_manifest_hash="5" * 64,
        gold_timeout_seconds=30,
        candidate_timeout_seconds=15,
    )
    selected, permutations = build_pairs_and_permutations(
        (question,),
        candidates,
        verified,
        UnitTokenizer(),
        TOKENIZER_REVISION,
        generation_manifest_hash="5" * 64,
        verification_manifest_hash="6" * 64,
        pair_manifest_hash="7" * 64,
    )

    assert gold[0].verification_run_id == verification_run_id
    assert [item.verification_status for item in verified] == [
        VerificationStatus.CORRECT,
        VerificationStatus.INCORRECT,
    ]
    assert len(selected) == 1
    assert selected[0].verification_run_id == verification_run_id
    assert [(item.permutation_id, item.correct_surface) for item in permutations] == [
        (0, "A"),
        (1, "B"),
    ]
    assert all(item.pair_id == selected[0].pair_id for item in permutations)


def test_cli_generate_verify_finalize_writes_one_formal_resumable_chain(
    monkeypatch, tmp_path
):
    """Removing the CLI dispatcher must break the only executable P2 path."""
    from permstudy.data_pipeline import splitting
    from permstudy.data_pipeline.audit import AuditVerdict
    from permstudy.data_pipeline.canonical import sha256_hex
    from permstudy.data_pipeline.ids import reclor_content_hash, run_id
    from permstudy.data_pipeline.io import append_record, write_atomic_manifest
    from permstudy.data_pipeline.schema import AuditDecision, AuditSelectionRecord
    from scripts_permstudy.data import _common as io
    from scripts_permstudy.phase2 import run_real_smoke as cli

    answers = ("Alpha", "Beta", "Gamma", "Delta")
    question = QuestionRecord(
        schema_version="question_v1",
        source=Source.RECLOR,
        original_question_id="reclor:train:phase2_cli_fixture",
        question_content_hash=reclor_content_hash(
            "Fixture context", "Which option follows?", answers
        ),
        source_snapshot_id="phase2-cli-fixture",
        source_revision="fixture-v1",
        source_row_id="1",
        context="Fixture context",
        question="Which option follows?",
        answers=answers,
        gold_label="A",
        problem=None,
        solution=None,
        category=None,
        level=None,
        synthetic=True,
    )
    smoke = (assignment(question),)

    def envelope(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        write_atomic_manifest(path, payload)
        return json.loads(path.read_text(encoding="utf-8"))

    source_paths = {}
    source_hashes = {}
    for source in ("math", "reclor"):
        source_file = tmp_path / f"sources/{source}/questions.jsonl"
        source_file.parent.mkdir(parents=True, exist_ok=True)
        source_file.write_bytes(b"")
        payload = {
            "schema_version": "run_manifest_v1",
            "stage": "sources",
            "run_id": sha256_hex(source.encode()),
            "config_hash": "0" * 64,
            "upstream_manifest_hashes": [],
            "artifacts": [io.artifact(tmp_path, source_file, 0).to_dict()],
            "counts": {"questions": 0},
            "source": source,
        }
        path = tmp_path / f"sources/{source}/manifest.json"
        manifest = envelope(path, payload)
        source_paths[f"{source}_questions"] = path.relative_to(tmp_path).as_posix()
        source_hashes[f"{source}_questions"] = manifest["output_manifest_hash"]

    questions_ref = io.persist_records(
        tmp_path, "split/fixture/questions.jsonl", [question]
    )
    split_config = splitting.split_stage_config(42, source_hashes)
    split_payload = {
        "schema_version": "run_manifest_v1",
        "stage": "split",
        "run_id": run_id("split", split_config),
        "config_hash": sha256_hex(io.canonical_json_bytes(split_config)),
        "typed_config": split_config,
        "upstream_bindings": source_hashes,
        "upstream_manifest_hashes": sorted(source_hashes.values()),
        "upstream_manifests": {
            role: io.artifact(tmp_path, tmp_path / relative).to_dict()
            for role, relative in source_paths.items()
        },
        "artifacts": [questions_ref.to_dict()],
        "counts": {"questions": 1, "smoke": 1},
        "split_manifest_hash": SPLIT_HASH,
    }
    split_path = tmp_path / "split/fixture/manifest.json"
    envelope(split_path, split_payload)

    def tiny_split_loader(root, manifest):
        assert root == tmp_path
        assert manifest["split_manifest_hash"] == SPLIT_HASH
        return (question,), smoke

    class AnswerEngine:
        def generate(self, requests, config):
            return [
                SimpleNamespace(
                    request_id=request.request_id,
                    outputs=[
                        SimpleNamespace(
                            text=(
                                "Reasoning.\nFinal Answer: A"
                                if request.plan.sampling_index == 0
                                else "Reasoning.\nFinal Answer: B"
                            ),
                            finish_reason="stop",
                            token_ids=[1, 2, 3],
                        )
                    ],
                )
                for request in reversed(requests)
            ]

    install_fake_vllm(monkeypatch, SimpleNamespace())

    def factory_builder(questions):
        return lambda module, config: AnswerEngine()

    common = [
        "--data-root",
        str(tmp_path),
        "--split-manifest",
        split_path.relative_to(tmp_path).as_posix(),
    ]
    for generator_id in (
        "qwen2.5-7b-instruct",
        "llama-3.1-8b-instruct",
        "qwen2.5-32b-instruct",
    ):
        assert (
            cli.main(
                ["generate", *common, "--generator-id", generator_id],
                split_loader=tiny_split_loader,
                engine_factory_builder=factory_builder,
            )
            == 0
        )

    generation_path = next((tmp_path / "generation").glob("*/manifest.json"))
    assert (
        cli.main(
            [
                "generate",
                *common,
                "--generator-id",
                "qwen2.5-32b-instruct",
            ],
            split_loader=tiny_split_loader,
            engine_factory_builder=factory_builder,
        )
        == 0
    )
    assert (
        cli.main(
            [
                "verify",
                "--data-root",
                str(tmp_path),
                "--generation-manifest",
                generation_path.relative_to(tmp_path).as_posix(),
            ],
            split_loader=tiny_split_loader,
        )
        == 0
    )
    verification_path = next((tmp_path / "verification").glob("*/manifest.json"))
    audit_path = next((tmp_path / "audit").glob("*/manifest.json"))
    selection_path = next((tmp_path / "audit").glob("*/selection.jsonl"))
    selection = [
        AuditSelectionRecord.from_dict(
            {key: value for key, value in row.items() if key != "record_hash"}
        )
        for row in io.scan_jsonl(selection_path).records
    ]
    decisions_path = tmp_path / "human-decisions.jsonl"
    for selected in selection:
        append_record(
            decisions_path,
            AuditDecision(
                selected.audit_run_id,
                selected.generation_run_id,
                selected.verification_run_id,
                selected.record_kind,
                selected.original_question_id,
                selected.candidate_id,
                AuditVerdict.AGREE,
                selected.reason_code,
                False,
            ).to_dict(),
        )

    finalize_args = [
        "finalize",
        "--data-root",
        str(tmp_path),
        "--verification-manifest",
        verification_path.relative_to(tmp_path).as_posix(),
        "--audit-manifest",
        audit_path.relative_to(tmp_path).as_posix(),
        "--decisions",
        decisions_path.relative_to(tmp_path).as_posix(),
    ]
    assert (
        cli.main(
            [*finalize_args, "--tokenizer-revision", "4" * 40],
            split_loader=tiny_split_loader,
            tokenizer_loader=lambda revision, root: UnitTokenizer(),
        )
        == 2
    )
    assert (
        cli.main(
            [*finalize_args, "--tokenizer-revision", TOKENIZER_REVISION],
            split_loader=tiny_split_loader,
            tokenizer_loader=lambda revision, root: UnitTokenizer(),
        )
        == 0
    )
    export = next((tmp_path / "trainer_export").glob("*/manifest.json"))
    assert json.loads(export.read_text(encoding="utf-8"))["counts"] == {"rows": 2}
    assert io.load_manifest(
        tmp_path, export.relative_to(tmp_path).as_posix(), "trainer_export"
    )["run_id"]
    completed_audit = next((tmp_path / "audit").glob("*/completed.json"))
    assert io.load_manifest(
        tmp_path, completed_audit.relative_to(tmp_path).as_posix(), "audit"
    )["counts"]["completed"] == len(selection)
    report = next((tmp_path / "phase2").glob("*/gate-report.json"))
    assert json.loads(report.read_text(encoding="utf-8"))["status"] == "FAIL"
