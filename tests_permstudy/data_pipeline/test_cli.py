"""CLI boundaries use synthetic data only; no network or inference is exercised."""

import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from permstudy.data_pipeline.canonical import canonical_json_bytes, sha256_hex
from permstudy.data_pipeline.ids import run_id
from permstudy.data_pipeline.io import append_record, write_atomic_manifest
from permstudy.data_pipeline.schema import ArtifactRef


SCRIPTS = (
    "prepare_questions",
    "plan_generation",
    "generate_candidates",
    "verify_candidates",
    "build_reasoning_pairs",
    "build_permutations",
    "export_training_dataset",
    "audit_generator_distribution",
    "validate_dataset",
    "run_fake_e2e",
)
REPO = Path(__file__).resolve().parents[2]


def cli(name):
    path = REPO / "scripts_permstudy" / "data" / f"{name}.py"
    assert path.is_file(), f"missing CLI script: {name}"
    return importlib.import_module(f"scripts_permstudy.data.{name}")


@pytest.mark.parametrize("name", SCRIPTS)
def test_cli_help_and_import_need_no_vllm(name):
    module = cli(name)
    assert callable(module.main)
    assert module.build_parser().prog == f"{name}.py"
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts_permstudy/data" / f"{name}.py"), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "vllm" not in sys.modules


def test_fixed_parser_defaults_and_required_upstreams():
    prepare = cli("prepare_questions").build_parser()
    with pytest.raises(SystemExit):
        prepare.parse_args([])
    split = prepare.parse_args(
        [
            "build-split",
            "--data-root",
            "external",
            "--reclor-manifest",
            "r.json",
            "--math-manifest",
            "m.json",
        ]
    )
    assert split.split_seed == 42
    with pytest.raises(SystemExit):
        prepare.parse_args(
            ["acquire-reclor", "--data-root", "external", "--reclor-archive", "x.zip"]
        )
    gen = (
        cli("generate_candidates")
        .build_parser()
        .parse_args(
            [
                "--data-root",
                "external",
                "--generation-manifest",
                "g.json",
                "--generator-id",
                "g",
                "--shard-id",
                "00000",
            ]
        )
    )
    assert gen.backend == "fake"
    assert gen.recover_stale_lock is False
    export = cli("export_training_dataset").build_parser()
    with pytest.raises(SystemExit):
        export.parse_args(["--data-root", "external", "--split-manifest", "s.json"])


def test_vllm_boundary_precedes_reads_and_imports(tmp_path, capsys):
    result = cli("generate_candidates").main(
        [
            "--data-root",
            str(tmp_path),
            "--generation-manifest",
            "missing.json",
            "--generator-id",
            "g",
            "--shard-id",
            "00000",
            "--backend",
            "vllm",
        ]
    )
    assert result == 2
    assert "PhaseBoundaryError" in capsys.readouterr().out
    assert "vllm" not in sys.modules
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("ref", ["../SECRET.json", "C:/SECRET.json"])
def test_paths_cannot_escape_root_and_errors_never_echo_inputs(tmp_path, capsys, ref):
    result = cli("validate_dataset").main(
        ["--data-root", str(tmp_path), "--manifest", ref]
    )
    assert result == 2
    output = capsys.readouterr()
    assert "SECRET" not in output.out + output.err
    assert str(tmp_path) not in output.out + output.err


def test_parser_errors_and_missing_dependencies_have_fixed_sanitized_codes(
    tmp_path, capsys
):
    module = cli("validate_dataset")
    assert module.main(["--SECRET", str(tmp_path)]) == 2
    assert (
        module.main(["--data-root", str(tmp_path), "--manifest", "missing.json"]) == 4
    )
    output = capsys.readouterr()
    assert "SECRET" not in output.out + output.err
    assert str(tmp_path) not in output.out + output.err


def write_run(root, stage, config, records=(), *, extra=None, upstreams=None):
    """Construct real immutable envelopes, independently of CLI persistence."""
    identity = run_id(stage, config)
    directory = root / stage / identity
    directory.mkdir(parents=True, exist_ok=True)
    refs = []
    for name, rows in records:
        path = directory / name
        path.write_bytes(b"")
        for row in rows:
            append_record(path, row.to_dict() if hasattr(row, "to_dict") else row)
        refs.append(
            ArtifactRef(
                path.relative_to(root).as_posix(),
                sha256_hex(path.read_bytes()),
                len(rows),
            ).to_dict()
        )
    payload = dict(
        schema_version="run_manifest_v1",
        stage=stage,
        run_id=identity,
        config_hash=sha256_hex(canonical_json_bytes(config)),
        typed_config=config,
        upstream_bindings=config.get("upstream_bindings", {}),
        upstream_manifest_hashes=sorted(config.get("upstream_bindings", {}).values()),
        artifacts=refs,
        counts={"records": sum(ref["record_count"] for ref in refs)},
        upstream_manifests=upstreams or {},
    )
    payload.update(extra or {})
    path = directory / "manifest.json"
    write_atomic_manifest(path, payload)
    return path, json.loads(path.read_text())


def test_manifest_hash_artifact_hash_and_role_bindings_are_enforced(tmp_path):
    from permstudy.data_pipeline.permutations import permutation_stage_config

    module = cli("validate_dataset")
    config = permutation_stage_config({"pairs": "a" * 64})
    path, envelope = write_run(
        tmp_path, "permutations", config, [("permutations.jsonl", [])]
    )
    args = [
        "--data-root",
        str(tmp_path),
        "--manifest",
        path.relative_to(tmp_path).as_posix(),
    ]
    # A claimed role with no actual upstream manifest is not lineage proof.
    assert module.main(args) == 3
    envelope["counts"]["records"] = 123
    path.write_text(json.dumps(envelope))
    assert module.main(args) == 3
    write_atomic_manifest(path, envelope)
    (tmp_path / envelope["artifacts"][0]["relative_path"]).write_text("SECRET")
    assert module.main(args) == 3


def test_reclor_requires_acknowledgement_before_source_access(tmp_path):
    module = cli("prepare_questions")
    assert (
        module.main(
            ["acquire-reclor", "--data-root", str(tmp_path), "--reclor-dir", "missing"]
        )
        == 2
    )
    assert not list(tmp_path.iterdir())


def test_reclor_cli_accepts_an_absolute_external_source_directory(tmp_path):
    source = tmp_path / "reclor-source"
    data_root = tmp_path / "private-data"
    source.mkdir()
    (source / "train.json").write_text(
        json.dumps(
            [
                {
                    "id_string": "synthetic_0",
                    "context": "Synthetic context.",
                    "question": "Which follows?",
                    "answers": ["One", "Two", "Three", "Four"],
                    "label": 0,
                    "synthetic": True,
                }
            ]
        ),
        encoding="utf-8",
    )
    for name in ("val.json", "test.json", "use_items.txt"):
        (source / name).write_text("[]", encoding="utf-8")

    result = cli("prepare_questions").main(
        [
            "acquire-reclor",
            "--data-root",
            str(data_root),
            "--reclor-dir",
            str(source.resolve()),
            "--acknowledge-reclor-noncommercial",
        ]
    )

    assert result == 0
    assert len(list((data_root / "sources" / "reclor").glob("*/manifest.json"))) == 1


def test_unexpected_error_is_sanitized_and_traceback_is_opt_in(
    tmp_path, monkeypatch, capsys
):
    module = cli("validate_dataset")

    def failure(args, root):
        raise RuntimeError("SECRET private payload")

    monkeypatch.setattr(module, "dispatch", failure)
    args = ["--data-root", str(tmp_path), "--manifest", "unused.json"]
    assert module.main(args) == 4
    assert not list(tmp_path.iterdir())
    assert module.main(args + ["--private-log", "diagnostics/error.log"]) == 4
    assert "SECRET" in (tmp_path / "diagnostics/error.log").read_text()
    assert "SECRET" not in capsys.readouterr().out


def test_fake_e2e_exposes_boundary_without_claiming_task17_completion(tmp_path):
    module = cli("run_fake_e2e")
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(["--data-root", str(tmp_path)])
    assert module.main(["--data-root", str(tmp_path), "--fixture-root", "fixture"]) == 2


@pytest.fixture
def planned(tmp_path, monkeypatch):
    from permstudy.data_pipeline.sources import math as math_source
    from tests_permstudy.data_pipeline.test_generation_backends import generator_configs

    extracted = tmp_path / "reclor"
    extracted.mkdir()
    rows = [
        dict(
            id_string=f"synthetic_{i}",
            context=f"Synthetic context {i}.",
            question="Which follows?",
            answers=["One", "Two", "Three", "Four"],
            label=i % 2,
            synthetic=True,
        )
        for i in range(24)
    ]
    (extracted / "train.json").write_text(json.dumps(rows))
    for name in ("val.json", "test.json", "use_items.txt"):
        (extracted / name).write_text("[]")
    api = SimpleNamespace(get_dataset_config_names=lambda *args, **kwargs: ["algebra"])
    monkeypatch.setattr(math_source, "default_hf_api", lambda: api)
    monkeypatch.setattr(
        math_source,
        "hf_dataset_loader",
        lambda *args, **kwargs: [
            dict(
                problem=f"Synthetic item {i}: compute 1+1.",
                solution=r"Final Answer: \boxed{2}",
                level="Level 1",
                synthetic=True,
            )
            for i in range(24)
        ],
    )
    module = cli("prepare_questions")
    assert (
        module.main(
            [
                "acquire-reclor",
                "--data-root",
                str(tmp_path),
                "--reclor-dir",
                "reclor",
                "--acknowledge-reclor-noncommercial",
            ]
        )
        == 0
    )
    assert (
        module.main(
            ["acquire-math", "--data-root", str(tmp_path), "--math-revision", "1" * 40]
        )
        == 0
    )
    reclor = next((tmp_path / "sources/reclor").glob("*/manifest.json"))
    math = next((tmp_path / "sources/math").glob("*/manifest.json"))
    assert (
        module.main(
            [
                "build-split",
                "--data-root",
                str(tmp_path),
                "--reclor-manifest",
                reclor.relative_to(tmp_path).as_posix(),
                "--math-manifest",
                math.relative_to(tmp_path).as_posix(),
            ]
        )
        == 0
    )
    split = next((tmp_path / "split").glob("*/manifest.json"))
    split_payload = json.loads(split.read_text())
    argv = [
        "--data-root",
        str(tmp_path),
        "--split-manifest",
        split.relative_to(tmp_path).as_posix(),
        "--shard-size",
        "240",
    ]
    for index, config in enumerate(
        generator_configs(split_manifest_hash=split_payload["split_manifest_hash"])
    ):
        path = tmp_path / f"generator-{index}.json"
        path.write_text(json.dumps(config.to_dict()))
        argv += ["--generator-config", path.name]
    assert cli("plan_generation").main(argv) == 0
    return tmp_path, next((tmp_path / "generation").glob("*/plan.json")), split


def test_source_split_plan_and_fake_shard_dispatch_preserve_manifest_lineage(
    planned, capsys
):
    root, plan, split = planned
    envelope = json.loads(plan.read_text())
    split_envelope = json.loads(split.read_text())
    assert (
        envelope["upstream_bindings"]["split"] == split_envelope["output_manifest_hash"]
    )
    assert envelope["split_manifest_hash"] == split_envelope["split_manifest_hash"]
    assert envelope["split_manifest_hash"] != envelope["upstream_bindings"]["split"]
    args = [
        "--data-root",
        str(root),
        "--generation-manifest",
        plan.relative_to(root).as_posix(),
        "--generator-id",
        "qwen2.5-7b-instruct",
        "--shard-id",
        "00000",
    ]
    assert cli("generate_candidates").main(args) == 0
    assert cli("generate_candidates").main(args) == 0
    shard = next((plan.parent / "shards").glob("qwen2.5-7b-instruct/*/manifest.json"))
    shard_payload = json.loads(shard.read_text())
    assert shard_payload["counts"]["successful"] == 80
    assert (
        shard_payload["upstream_bindings"]["split"]
        == split_envelope["output_manifest_hash"]
    )
    printed = capsys.readouterr().out
    assert "Synthetic context" not in printed
    assert str(root) not in printed


def test_split_refuses_changed_source_artifact_before_writing(planned):
    root, _, _ = planned
    math = next((root / "sources/math").glob("*/manifest.json"))
    reclor = next((root / "sources/reclor").glob("*/manifest.json"))
    payload = json.loads(math.read_text())
    (root / payload["artifacts"][0]["relative_path"]).write_text("SECRET tampering")
    assert (
        cli("prepare_questions").main(
            [
                "build-split",
                "--data-root",
                str(root),
                "--split-seed",
                "43",
                "--math-manifest",
                math.relative_to(root).as_posix(),
                "--reclor-manifest",
                reclor.relative_to(root).as_posix(),
            ]
        )
        == 3
    )
    assert len(list((root / "split").iterdir())) == 1


@pytest.fixture
def generated(planned, monkeypatch):
    from permstudy.data_pipeline import generation
    from permstudy.data_pipeline.schema import Source

    root, plan, split = planned
    module = cli("generate_candidates")

    def backend(shard):
        return generation.FakeGenerationBackend(
            {
                p.key: generation.GenerationResult(
                    p,
                    (
                        f"Synthetic reasoning. Final Answer: {'A' if p.sampling_index == 0 else 'D'}"
                        if p.source is Source.RECLOR
                        else r"Synthetic reasoning. Final Answer: \boxed{2}"
                    ),
                    "stop",
                    0,
                    None,
                )
                for p in shard.candidates
            }
        )

    monkeypatch.setattr(module, "make_fake_backend", backend)
    for generator in (
        "qwen2.5-7b-instruct",
        "llama-3.1-8b-instruct",
        "qwen2.5-32b-instruct",
    ):
        assert (
            module.main(
                [
                    "--data-root",
                    str(root),
                    "--generation-manifest",
                    plan.relative_to(root).as_posix(),
                    "--generator-id",
                    generator,
                    "--shard-id",
                    "00000",
                ]
            )
            == 0
        )
    return root, plan.parent / "manifest.json", split


@pytest.fixture
def verified(generated, monkeypatch):
    from permstudy.data_pipeline.schema import QuestionVerificationRecord

    root, generation, split = generated
    module = cli("verify_candidates")

    def unavailable_gold(
        question,
        candidates,
        gold_timeout_seconds,
        candidate_timeout_seconds,
        *,
        verification_run_id,
    ):
        return QuestionVerificationRecord(
            verification_run_id,
            question.original_question_id,
            question.source,
            "gold_verification_error",
            None,
            "gold_parse_failure",
        ), []

    # Isolate only the expensive spawned MATH dependency; ReClor verification stays real.
    monkeypatch.setattr(module, "verify_math_question", unavailable_gold, raising=False)
    assert (
        module.main(
            [
                "--data-root",
                str(root),
                "--generation-manifest",
                generation.relative_to(root).as_posix(),
                "--gold-timeout-seconds",
                "2",
                "--candidate-timeout-seconds",
                "2",
            ]
        )
        == 0
    )
    return (
        root,
        next((root / "verification").glob("*/manifest.json")),
        generation,
        split,
    )


def test_verification_dispatch_writes_separate_records_and_rejects_bad_timeout(
    generated,
):
    root, generation, _ = generated
    module = cli("verify_candidates")
    assert (
        module.main(
            [
                "--data-root",
                str(root),
                "--generation-manifest",
                generation.relative_to(root).as_posix(),
                "--gold-timeout-seconds",
                "nan",
                "--candidate-timeout-seconds",
                "2",
            ]
        )
        == 2
    )


def patch_tokenizer_dependencies(
    monkeypatch, root, tokenizer, revision, *, resolved=...
):
    """Keep snapshot resolution and tokenizer loading offline at the dependency boundary."""
    cache = str((root / "cache/tokenizers").resolve())
    snapshot = str(Path(cache) / "snapshots" / revision / "tokenizer_config.json")
    expected_revision = revision

    def cached_file(repository, filename, *, revision, cache_dir):
        assert repository == "Qwen/Qwen2.5-7B-Instruct"
        assert filename == "tokenizer_config.json"
        assert revision == expected_revision
        assert cache_dir == cache
        return snapshot

    def extract_commit_hash(path, commit_hash):
        assert path == snapshot
        assert commit_hash is None
        return expected_revision if resolved is ... else resolved

    def from_pretrained(repository, *, revision, cache_dir, trust_remote_code):
        assert repository == "Qwen/Qwen2.5-7B-Instruct"
        assert revision == expected_revision
        assert cache_dir == cache
        assert trust_remote_code is False
        if resolved is not ... and resolved != expected_revision:
            pytest.fail("tokenizer must not load from an unverified snapshot")
        return tokenizer

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=from_pretrained)),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers.utils.hub",
        SimpleNamespace(
            cached_file=cached_file, extract_commit_hash=extract_commit_hash
        ),
    )


def test_tokenizer_accepts_verified_snapshot_without_private_commit_metadata(
    tmp_path, monkeypatch
):
    module = cli("build_reasoning_pairs")
    tokenizer = SimpleNamespace(
        name_or_path="Qwen/Qwen2.5-7B-Instruct",
        init_kwargs={},
        encode=lambda text, **kwargs: text.split(),
    )
    revision = "2" * 40
    patch_tokenizer_dependencies(monkeypatch, tmp_path, tokenizer, revision)

    loaded = module.load_tokenizer(revision, tmp_path)
    module.validate_tokenizer(loaded, revision)
    assert loaded.encode("one two", add_special_tokens=False) == ["one", "two"]


@pytest.mark.parametrize("resolved", [None, "3" * 40])
def test_tokenizer_rejects_unverified_snapshot_before_loading(
    tmp_path, monkeypatch, resolved
):
    module = cli("build_reasoning_pairs")
    patch_tokenizer_dependencies(
        monkeypatch, tmp_path, SimpleNamespace(), "2" * 40, resolved=resolved
    )
    with pytest.raises(module.io.DependencyContractError):
        module.load_tokenizer("2" * 40, tmp_path)


@pytest.mark.parametrize(
    "fields",
    [
        {"name_or_path": "other/repository"},
        {"encode": None},
        {"init_kwargs": {"_commit_hash": "3" * 40}},
        {"_commit_hash": "3" * 40},
    ],
)
def test_tokenizer_rejects_conflicting_identity_or_invalid_encode(fields):
    module = cli("build_reasoning_pairs")
    attributes = {
        "name_or_path": "Qwen/Qwen2.5-7B-Instruct",
        "init_kwargs": {"_commit_hash": "2" * 40},
        "encode": lambda text, **kwargs: text.split(),
    }
    attributes.update(fields)
    with pytest.raises(module.io.DependencyContractError):
        module.validate_tokenizer(SimpleNamespace(**attributes), "2" * 40)


def test_pair_dispatch_requires_pinned_actual_tokenizer_and_selects_real_pairs(
    verified, monkeypatch
):
    root, verification, _, _ = verified
    module = cli("build_reasoning_pairs")
    args = [
        "--data-root",
        str(root),
        "--verification-manifest",
        verification.relative_to(root).as_posix(),
    ]
    assert module.main(args + ["--tokenizer-revision", "main"]) == 2

    class Tokenizer:
        name_or_path = "Qwen/Qwen2.5-7B-Instruct"
        init_kwargs = {"_commit_hash": "2" * 40}

        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            return text.split()

    patch_tokenizer_dependencies(monkeypatch, root, Tokenizer(), "2" * 40)
    assert module.main(args + ["--tokenizer-revision", "2" * 40]) == 0
    path = next((root / "pairs").glob("*/manifest.json"))
    payload = json.loads(path.read_text())
    assert payload["counts"]["pairs"] == 10
    assert (
        payload["typed_config"]["parameters"]["tokenizer_repo"]
        == Tokenizer.name_or_path
    )
    assert (
        cli("validate_dataset").main(
            [
                "--data-root",
                str(root),
                "--pair-manifest",
                path.relative_to(root).as_posix(),
            ]
        )
        == 0
    )
    patch_tokenizer_dependencies(
        monkeypatch,
        root,
        SimpleNamespace(
            name_or_path="other/repository",
            init_kwargs={"_commit_hash": "3" * 40},
            encode=Tokenizer().encode,
        ),
        "3" * 40,
    )
    assert module.main(args + ["--tokenizer-revision", "3" * 40]) == 4


@pytest.fixture
def paired(verified, monkeypatch):
    root, verification, generation, split = verified
    module = cli("build_reasoning_pairs")
    tokenizer = SimpleNamespace(
        name_or_path="Qwen/Qwen2.5-7B-Instruct",
        init_kwargs={"_commit_hash": "2" * 40},
        encode=lambda text, **kwargs: text.split(),
    )
    patch_tokenizer_dependencies(monkeypatch, root, tokenizer, "2" * 40)
    assert (
        module.main(
            [
                "--data-root",
                str(root),
                "--verification-manifest",
                verification.relative_to(root).as_posix(),
                "--tokenizer-revision",
                "2" * 40,
            ]
        )
        == 0
    )
    return root, next((root / "pairs").glob("*/manifest.json")), generation, split


def test_permutation_and_export_dispatch_validate_four_cross_bound_inputs(paired):
    root, pair, generation, split = paired
    assert (
        cli("build_permutations").main(
            [
                "--data-root",
                str(root),
                "--pair-manifest",
                pair.relative_to(root).as_posix(),
            ]
        )
        == 0
    )
    permutation = next((root / "permutations").glob("*/manifest.json"))
    args = ["--data-root", str(root)]
    for role, path in (
        ("split", split),
        ("generation", generation),
        ("pair", pair),
        ("permutation", permutation),
    ):
        args += [f"--{role}-manifest", path.relative_to(root).as_posix()]
    assert cli("export_training_dataset").main(args) == 0
    export = next((root / "trainer_export").glob("*/manifest.json"))
    payload = json.loads(export.read_text())
    assert payload["counts"]["rows"] == 20
    assert (
        cli("validate_dataset").main(
            [
                "--data-root",
                str(root),
                "--export-manifest",
                export.relative_to(root).as_posix(),
            ]
        )
        == 0
    )
    raw = json.loads(permutation.read_text())
    raw["upstream_bindings"]["pairs"] = "f" * 64
    write_atomic_manifest(permutation, raw)
    assert cli("export_training_dataset").main(args) == 3


def test_audit_rebuilds_selection_and_requires_exact_decisions(verified):
    from permstudy.data_pipeline.schema import (
        AuditDecision,
        AuditSelectionRecord,
        AuditVerdict,
    )
    from permstudy.data_pipeline.io import scan_jsonl

    root, verification, _, _ = verified
    module = cli("audit_generator_distribution")
    args = [
        "--data-root",
        str(root),
        "--verification-manifest",
        verification.relative_to(root).as_posix(),
    ]
    assert module.main(args) == 0
    selection_path = next((root / "audit").glob("*/selection.jsonl"))
    selection = [
        AuditSelectionRecord.from_dict(
            {k: v for k, v in row.items() if k != "record_hash"}
        )
        for row in scan_jsonl(selection_path).records
    ]
    audit_manifest = json.loads(
        next((root / "audit").glob("*/manifest.json")).read_text()
    )
    assert {item.audit_run_id for item in selection} == {audit_manifest["run_id"]}
    assert (
        len(selection) == 50
    )  # 20 gold failures and 5 per binary source/generator/status cell.
    decisions_path = root / "decisions.jsonl"
    for item in selection:
        decision = AuditDecision(
            item.audit_run_id,
            item.generation_run_id,
            item.verification_run_id,
            item.record_kind,
            item.original_question_id,
            item.candidate_id,
            AuditVerdict.AGREE,
            item.reason_code,
            False,
        )
        append_record(decisions_path, decision.to_dict())
    assert module.main(args + ["--decisions", "decisions.jsonl"]) == 0
    summary = next((root / "audit").glob("*/summary.json"))
    assert json.loads(summary.read_text())["completed_count"] == 50
    completed = next((root / "audit").glob("*/completed.json"))
    # An attacker can recompute hashes; summary values still require exact decisions.
    forged = json.loads(summary.read_text())
    forged["confirmed_disagreements"] = 12
    summary.write_bytes(canonical_json_bytes(forged))
    manifest = json.loads(completed.read_text())
    for ref in manifest["artifacts"]:
        if ref["relative_path"].endswith("summary.json"):
            ref["sha256"] = sha256_hex(summary.read_bytes())
    write_atomic_manifest(completed, manifest)
    assert (
        cli("validate_dataset").main(
            [
                "--data-root",
                str(root),
                "--audit-manifest",
                completed.relative_to(root).as_posix(),
            ]
        )
        == 3
    )
    decisions_path.write_bytes(b"")
    assert module.main(args + ["--decisions", "decisions.jsonl"]) == 3


def test_named_validation_flag_rejects_wrong_stage(planned):
    root, generation, _ = planned
    assert (
        cli("validate_dataset").main(
            [
                "--data-root",
                str(root),
                "--pair-manifest",
                generation.relative_to(root).as_posix(),
            ]
        )
        == 3
    )


@pytest.mark.parametrize("payload", [[], None, {"upstream_bindings": ["SECRET"]}])
def test_malformed_manifest_is_integrity_error(tmp_path, payload):
    (tmp_path / "manifest.json").write_text(json.dumps(payload))
    assert (
        cli("validate_dataset").main(
            ["--data-root", str(tmp_path), "--manifest", "manifest.json"]
        )
        == 3
    )
