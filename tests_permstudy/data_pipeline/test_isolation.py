"""Synthetic runtime paths and payloads exercise the public privacy boundary."""

import dataclasses
import hashlib
import importlib
import os
from pathlib import Path
import subprocess

import pytest


def load_isolation():
    # A missing module is an explicit assertion failure in the initial RED run.
    assert importlib.util.find_spec("permstudy.data_pipeline.isolation") is not None
    return importlib.import_module("permstudy.data_pipeline.isolation")


def external_env(tmp_path):
    return {"HF_HOME": str(tmp_path / "cache")}


@pytest.mark.parametrize("relationship", ["child", "parent", "equal", "dotdot"])
def test_rejects_resolved_root_containment(tmp_path, relationship):
    isolation = load_isolation()
    repo = tmp_path / "repo"
    repo.mkdir()
    data = {"child": repo / "private", "parent": tmp_path,
            "equal": repo, "dotdot": repo / "other" / ".." / "private"}[relationship]
    with pytest.raises(isolation.IsolationError, match="must not contain one another"):
        isolation.validate_external_roots(repo, data, external_env(tmp_path))


def test_report_exposes_only_normalized_root_hashes_and_cache_kinds(tmp_path):
    isolation = load_isolation()
    repo, data = tmp_path / "repo", tmp_path / "data"
    report = isolation.validate_external_roots(repo, data, external_env(tmp_path))
    expected = hashlib.sha256(os.path.normcase(str(repo.resolve())).encode()).hexdigest()
    assert report.repo_root_hash == expected
    assert report.data_root_hash != report.repo_root_hash
    assert set(dataclasses.asdict(report)) == {"repo_root_hash", "data_root_hash", "effective_cache_kinds"}
    assert {"HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"} <= set(report.effective_cache_kinds)
    assert str(tmp_path) not in repr(report)
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.repo_root_hash = "changed"


def test_resolve_path_removes_dotdot(tmp_path):
    isolation = load_isolation()
    assert isolation.resolve_path(tmp_path / "a" / ".." / "b") == (tmp_path / "b").resolve()


@pytest.mark.skipif(os.name != "nt", reason="Windows normcase behavior")
def test_windows_case_alias_is_rejected(tmp_path):
    isolation = load_isolation()
    repo = tmp_path / "MiXeD"
    with pytest.raises(isolation.IsolationError):
        isolation.validate_external_roots(repo, Path(str(repo).swapcase()) / "private", external_env(tmp_path))


def test_resolved_symlink_cannot_hide_private_directory(tmp_path):
    isolation = load_isolation()
    repo = tmp_path / "repo"
    repo.mkdir()
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(repo, target_is_directory=True)
    except OSError:
        pytest.skip("OS does not permit symlinks")
    with pytest.raises(isolation.IsolationError):
        isolation.validate_external_roots(repo, alias / "private", external_env(tmp_path))


@pytest.mark.skipif(os.name != "nt", reason="Windows junction behavior")
def test_resolved_junction_cannot_hide_private_directory(tmp_path):
    isolation = load_isolation()
    repo = tmp_path / "repo"
    repo.mkdir()
    alias = tmp_path / "junction"
    created = subprocess.run(["cmd", "/c", "mklink", "/J", str(alias), str(repo)], capture_output=True)
    if created.returncode:
        pytest.skip("OS does not permit junctions")
    with pytest.raises(isolation.IsolationError):
        isolation.validate_external_roots(repo, alias / "private", external_env(tmp_path))


@pytest.mark.parametrize("cache_kind", ["HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "PYTORCH_TRANSFORMERS_CACHE", "PYTORCH_PRETRAINED_BERT_CACHE"])
def test_effective_cache_inside_repo_is_rejected(tmp_path, cache_kind):
    isolation = load_isolation()
    repo = tmp_path / "repo"
    env = external_env(tmp_path)
    env[cache_kind] = str(repo / "cache")
    with pytest.raises(isolation.IsolationError, match="cache"):
        isolation.validate_external_roots(repo, tmp_path / "data", env)


def test_default_cache_under_repo_home_is_rejected(tmp_path, monkeypatch):
    isolation = load_isolation()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "repo"))
    with pytest.raises(isolation.IsolationError, match="cache"):
        isolation.validate_external_roots(tmp_path / "repo", tmp_path / "data", {})


def test_xdg_cache_default_is_checked(tmp_path):
    isolation = load_isolation()
    with pytest.raises(isolation.IsolationError, match="cache"):
        isolation.validate_external_roots(tmp_path / "repo", tmp_path / "data", {"XDG_CACHE_HOME": str(tmp_path / "repo" / "cache")})


@pytest.mark.parametrize("kind", ["HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"])
@pytest.mark.parametrize("syntax", ["tilde", "dollar", "braces", "percent"])
def test_cache_expansion_cannot_hide_repository_containment(tmp_path, monkeypatch, kind, syntax):
    isolation = load_isolation()
    if syntax == "percent" and os.name != "nt":
        pytest.skip("percent environment expansion is Windows-specific")
    # The expansion variables are synthetic runtime inputs, never committed paths.
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PAGRPO_TEST_REPO", str(tmp_path / "repo"))
    value = {"tilde": "~/repo/cache", "dollar": "$PAGRPO_TEST_REPO/cache",
             "braces": "${PAGRPO_TEST_REPO}/cache", "percent": "%PAGRPO_TEST_REPO%/cache"}[syntax]
    env = external_env(tmp_path)
    env[kind] = value
    with pytest.raises(isolation.IsolationError, match="cache"):
        isolation.validate_external_roots(tmp_path / "repo", tmp_path / "data", env)


@pytest.mark.parametrize("kind", ["HF_ASSETS_CACHE", "HUGGINGFACE_ASSETS_CACHE", "HF_XET_CACHE"])
def test_assets_and_xet_cache_overrides_inside_repo_are_rejected(tmp_path, kind):
    isolation = load_isolation()
    env = external_env(tmp_path)
    env[kind] = str(tmp_path / "repo" / "cache")
    with pytest.raises(isolation.IsolationError, match="cache"):
        isolation.validate_external_roots(tmp_path / "repo", tmp_path / "data", env)


def test_default_assets_and_xet_caches_are_in_report(tmp_path):
    isolation = load_isolation()
    report = isolation.validate_external_roots(tmp_path / "repo", tmp_path / "data", external_env(tmp_path))
    assert {"HF_ASSETS_CACHE", "HF_XET_CACHE"} <= set(report.effective_cache_kinds)


def test_xet_literal_tilde_path_uses_pinned_unexpanded_semantics(tmp_path, monkeypatch):
    isolation = load_isolation()
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "external-home"))
    monkeypatch.setenv("HOME", str(tmp_path / "external-home"))
    env = external_env(tmp_path)
    # huggingface_hub 0.36.0 does not expand its HF_XET_CACHE override.
    env["HF_XET_CACHE"] = "~/literal-cache"
    with pytest.raises(isolation.IsolationError, match="cache"):
        isolation.validate_external_roots(repo, tmp_path / "data", env)


@pytest.mark.parametrize("name", ["assets", "xet"])
def test_default_assets_and_xet_alias_into_repo_is_rejected(tmp_path, name):
    isolation = load_isolation()
    repo, hf_home = tmp_path / "repo", tmp_path / "cache"
    repo.mkdir()
    hf_home.mkdir()
    alias = hf_home / name
    if os.name == "nt":
        created = subprocess.run(["cmd", "/c", "mklink", "/J", str(alias), str(repo)], capture_output=True)
        if created.returncode:
            pytest.skip("OS does not permit junctions")
    else:
        alias.symlink_to(repo, target_is_directory=True)
    with pytest.raises(isolation.IsolationError, match="cache"):
        isolation.validate_external_roots(repo, tmp_path / "data", external_env(tmp_path))


@pytest.mark.parametrize("key", ["LLMResponse", "QUESTIONText", "LLMGoldAnswer", "RAWPrompt", "HTTPContext"])
def test_acronym_sensitive_keys_are_rejected(key):
    isolation = load_isolation()
    with pytest.raises(isolation.SafeLoggingError):
        isolation.sanitize_public_manifest({"aggregate_statistics": {key: "synthetic-secret"}})


@pytest.mark.parametrize("value", ["failed:/private/user/data", "failed:C:/private/user/data",
                                    "failed:C:\\private\\user\\data", "failed://server/private/data"])
def test_public_log_rejects_embedded_paths(value):
    isolation = load_isolation()
    with pytest.raises(isolation.SafeLoggingError):
        isolation.safe_log_fields({"status": value})


@pytest.mark.parametrize("value", ["file:/private/user/data", "file:///private/user/data",
                                    "failed:C:/private/user/data", "file://server/private/data",
                                    "failed=/private/user/data"])
def test_public_manifest_rejects_embedded_paths_and_uris(value):
    isolation = load_isolation()
    with pytest.raises(isolation.SafeLoggingError):
        isolation.sanitize_public_manifest({"aggregate_statistics": {"mean": value}})


def test_aggregate_metadata_and_model_identifiers_remain_supported():
    isolation = load_isolation()
    payload = {"generator_model": "org/model", "aggregate_statistics": {"mean": 1.25, "method": "bootstrap_v1", "sample_count": 4}}
    assert isolation.sanitize_public_manifest(payload) == payload


@pytest.mark.parametrize("key", ["LLMRESPONSE", "QUESTIONTEXT", "RAWANSWER", "GOLDVALUE", "PROMPTLLM", "SOLUTIONRAW", "CONTEXTTEXT"])
def test_fused_uppercase_sensitive_keys_are_rejected(key):
    isolation = load_isolation()
    with pytest.raises(isolation.SafeLoggingError):
        isolation.sanitize_public_manifest({"aggregate_statistics": {key: "synthetic-secret"}})


@pytest.mark.parametrize("value", ["failed-/private/user/data", "prefix/private/user/data", "org/model",
                                    "prefix/C:/private/user/data", "prefix\\private\\user\\data", "prefix/file:/private/user/data"])
@pytest.mark.parametrize("field", ["aggregate_statistics", "phase_status"])
def test_general_metadata_fields_reject_embedded_paths(field, value):
    isolation = load_isolation()
    payload = {field: {"mean": value} if field == "aggregate_statistics" else value}
    with pytest.raises(isolation.SafeLoggingError):
        isolation.sanitize_public_manifest(payload)


def test_only_explicit_repository_fields_accept_repository_identifiers():
    isolation = load_isolation()
    manifest = {"source": "org/dataset", "generator_model": "org/model", "phase_status": "code_complete", "aggregate_statistics": {"mean": 1.25, "method": "bootstrap_v1"}}
    evidence = {"source_revisions": {"org/dataset": "a" * 40}, "completion_counts": {"ok": 4}, "stratification_level": {"math": "category_level"}}
    assert isolation.sanitize_public_manifest(manifest) == manifest
    assert isolation.sanitize_public_manifest(evidence) == evidence


@pytest.mark.parametrize("home_kind", ["HF_HOME", "XDG_CACHE_HOME"])
def test_datasets_default_keeps_environment_variable_syntax_literal(tmp_path, monkeypatch, home_kind):
    isolation = load_isolation()
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    monkeypatch.setenv("PAGRPO_REVIEW_EXTERNAL", str(tmp_path / "external"))
    env = {home_kind: "$PAGRPO_REVIEW_EXTERNAL/hf"}
    # Hub expands this to external storage; datasets only expands '~', so its
    # default remains relative to the repo's current working directory.
    with pytest.raises(isolation.IsolationError, match="cache"):
        isolation.validate_external_roots(repo, tmp_path / "data", env)


def test_explicit_datasets_override_remains_literal(tmp_path, monkeypatch):
    isolation = load_isolation()
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    monkeypatch.setenv("PAGRPO_REVIEW_EXTERNAL", str(tmp_path / "external"))
    env = external_env(tmp_path)
    env["HF_DATASETS_CACHE"] = "$PAGRPO_REVIEW_EXTERNAL/datasets"
    with pytest.raises(isolation.IsolationError, match="cache"):
        isolation.validate_external_roots(repo, tmp_path / "data", env)


@pytest.mark.parametrize("kind", ["HF_MODULES_CACHE", "HF_DATASETS_DOWNLOADED_DATASETS_PATH", "HF_DATASETS_EXTRACTED_DATASETS_PATH"])
def test_all_datasets_effective_path_overrides_inside_repo_are_rejected(tmp_path, kind):
    isolation = load_isolation()
    env = external_env(tmp_path)
    env[kind] = str(tmp_path / "repo" / "cache")
    with pytest.raises(isolation.IsolationError, match="cache"):
        isolation.validate_external_roots(tmp_path / "repo", tmp_path / "data", env)


@pytest.mark.parametrize("kind", ["HF_MODULES_CACHE", "HF_DATASETS_DOWNLOADED_DATASETS_PATH", "HF_DATASETS_EXTRACTED_DATASETS_PATH"])
def test_all_datasets_path_overrides_keep_variable_syntax_literal(tmp_path, monkeypatch, kind):
    isolation = load_isolation()
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    monkeypatch.setenv("PAGRPO_REVIEW_EXTERNAL", str(tmp_path / "external"))
    env = external_env(tmp_path)
    env[kind] = "$PAGRPO_REVIEW_EXTERNAL/cache"
    with pytest.raises(isolation.IsolationError, match="cache"):
        isolation.validate_external_roots(repo, tmp_path / "data", env)


@pytest.mark.parametrize("cache_shape", ["modules", "downloads", "extracted"])
def test_datasets_default_paths_resolve_repository_aliases(tmp_path, cache_shape):
    isolation = load_isolation()
    repo, cache = tmp_path / "repo", tmp_path / "cache"
    repo.mkdir()
    relative = {"modules": "modules", "downloads": "datasets/downloads", "extracted": "datasets/downloads/extracted"}[cache_shape]
    alias = cache / relative
    alias.parent.mkdir(parents=True)
    if os.name == "nt":
        created = subprocess.run(["cmd", "/c", "mklink", "/J", str(alias), str(repo)], capture_output=True)
        if created.returncode:
            pytest.skip("OS does not permit junctions")
    else:
        alias.symlink_to(repo, target_is_directory=True)
    with pytest.raises(isolation.IsolationError, match="cache"):
        isolation.validate_external_roots(repo, tmp_path / "data", external_env(tmp_path))


@pytest.mark.parametrize("derivation", ["config_default", "download_override"])
def test_datasets_extraction_derivations_resolve_repository_aliases(tmp_path, derivation):
    isolation = load_isolation()
    repo = tmp_path / "repo"
    repo.mkdir()
    env = external_env(tmp_path)
    if derivation == "download_override":
        # Pinned ExtractManager derives '<effective downloads>/extracted' whenever
        # a cache directory is passed, so redirecting downloads to an external root
        # moves the extraction directory with it. That derived location must also be
        # validated, or an external downloads override can smuggle a repository-local
        # extraction cache past the check while its parent looks external.
        env["HF_DATASETS_DOWNLOADED_DATASETS_PATH"] = str(tmp_path / "other-downloads")
        alias = tmp_path / "other-downloads" / "extracted"
    else:
        # The library default for config.EXTRACTED_DATASETS_PATH derives from the
        # default downloads path, so that derivation must stay validated on its own.
        alias = tmp_path / "cache" / "datasets" / "downloads" / "extracted"
    alias.parent.mkdir(parents=True)
    if os.name == "nt":
        created = subprocess.run(["cmd", "/c", "mklink", "/J", str(alias), str(repo)], capture_output=True)
        if created.returncode:
            pytest.skip("OS does not permit junctions")
    else:
        alias.symlink_to(repo, target_is_directory=True)
    with pytest.raises(isolation.IsolationError, match="cache"):
        isolation.validate_external_roots(repo, tmp_path / "data", env)


def test_datasets_extraction_derivations_are_reported_without_paths(tmp_path):
    isolation = load_isolation()
    env = external_env(tmp_path)
    env["HF_DATASETS_DOWNLOADED_DATASETS_PATH"] = str(tmp_path / "other-downloads")
    report = isolation.validate_external_roots(tmp_path / "repo", tmp_path / "data", env)
    kinds = set(report.effective_cache_kinds)
    # Both derivation models stay in the validated set: the library default that
    # follows the default downloads path, and the ExtractManager derivation that
    # follows the effective downloads path.
    assert {"HF_DATASETS_DEFAULT_EXTRACTED_DATASETS_PATH",
            "HF_DATASETS_EXTRACTED_DATASETS_PATH_FROM_DOWNLOADS"} <= kinds
    # An explicit extraction override joins the validated set as its own kind.
    env["HF_DATASETS_EXTRACTED_DATASETS_PATH"] = str(tmp_path / "external-extracted")
    overridden = isolation.validate_external_roots(tmp_path / "repo", tmp_path / "data", env)
    assert "HF_DATASETS_EXTRACTED_DATASETS_PATH" in set(overridden.effective_cache_kinds)
    assert str(tmp_path) not in repr(report) and str(tmp_path) not in repr(overridden)


def test_datasets_extraction_override_derivation_is_always_validated(tmp_path):
    isolation = load_isolation()
    # With both datasets path overrides external, the ExtractManager derivation and
    # the library default derivation must still be validated independently.
    repo = tmp_path / "repo"
    repo.mkdir()
    env = external_env(tmp_path)
    env["HF_DATASETS_DOWNLOADED_DATASETS_PATH"] = str(tmp_path / "other-downloads")
    env["HF_DATASETS_EXTRACTED_DATASETS_PATH"] = str(tmp_path / "external-extracted")
    alias = tmp_path / "cache" / "datasets" / "downloads" / "extracted"
    alias.parent.mkdir(parents=True)
    if os.name == "nt":
        created = subprocess.run(["cmd", "/c", "mklink", "/J", str(alias), str(repo)], capture_output=True)
        if created.returncode:
            pytest.skip("OS does not permit junctions")
    else:
        alias.symlink_to(repo, target_is_directory=True)
    with pytest.raises(isolation.IsolationError, match="cache"):
        isolation.validate_external_roots(repo, tmp_path / "data", env)


def test_pinned_tokenizers_version_is_approved_public_evidence():
    isolation = load_isolation()
    evidence = {"package_versions": {"tokenizers": "0.22.1", "torch": "2.5.1"}}
    assert isolation.sanitize_public_manifest(evidence) == evidence


@pytest.mark.parametrize("payload", [
    {"aggregate_statistics": {"tokenizers": "secret"}},
    {"package_versions": {"LLMRESPONSE": "0.22.1"}},
    {"package_versions": {"tokenizers": {"response": "secret"}}},
    {"package_versions": {"torch": {"tokenizers": "secret"}}},
])
def test_package_name_exception_does_not_escape_its_namespace(payload):
    isolation = load_isolation()
    with pytest.raises(isolation.SafeLoggingError):
        isolation.sanitize_public_manifest(payload)


@pytest.mark.parametrize("fields", [
    {"response": "synthetic secret"}, {"HF_TOKEN": "hf_syntheticsecret"},
    {"candidate_id": "/private/secret"}, {"candidate_id": "C:\\private\\secret"},
    {"candidate_id": "\\\\server\\private"}, {"status": "hf_syntheticsecret"},
    {"count": {"response": "secret"}}, {"retry_count": -1}, {"elapsed_seconds": float("nan")},
])
def test_public_log_rejects_unsafe_fields(fields):
    isolation = load_isolation()
    with pytest.raises(isolation.SafeLoggingError):
        isolation.safe_log_fields(fields)


def test_public_log_preserves_allowed_scalar_fields():
    isolation = load_isolation()
    fields = {"candidate_id": "cand-1", "original_question_id": "q-1", "generator_id": "g1", "shard_id": "s1", "status": "ok", "error_type": "timeout", "retry_count": 0, "count": 3, "elapsed_seconds": 1.25}
    assert isolation.safe_log_fields(fields) == fields


@pytest.mark.parametrize("error,want", [
    (TimeoutError("hf_secret /private"), "timeout"),
    (MemoryError("synthetic response"), "oom"),
    (ImportError("private dependency"), "dependency_error"),
    (PermissionError("C:\\private"), "access_denied"),
    (RuntimeError("CUDA out of memory hf_secret"), "unexpected_error"),
])
def test_exception_mapping_ignores_messages(error, want):
    isolation = load_isolation()
    assert isolation.sanitize_exception(error) == want


def test_third_party_type_is_sanitized_without_stringifying():
    isolation = load_isolation()
    class OutOfMemoryError(RuntimeError):
        def __str__(self):
            raise AssertionError("exception message accessed")
    assert isolation.sanitize_exception(OutOfMemoryError()) == "oom"


def test_public_manifest_accepts_safe_nested_evidence_and_copies_it():
    isolation = load_isolation()
    evidence = {"schema_version": "1", "package_versions": {"torch": "2.5.1"}, "run_ids": ["run-1"], "question_counts": {"math": 4}, "source_file_hashes": {"synthetic.jsonl": "a" * 64}, "fallback_reasons": ["small_stratum"], "real_generation_performed": False}
    sanitized = isolation.sanitize_public_manifest(evidence)
    assert sanitized == evidence
    sanitized["package_versions"]["torch"] = "changed"
    assert evidence["package_versions"]["torch"] == "2.5.1"
    assert isolation.sanitize_public_manifest({"run_id": "r1", "status_counts": {"ok": 3}, "generator_model": "org/model", "artifact_sha256": "b" * 64})["run_id"] == "r1"


@pytest.mark.parametrize("payload", [
    {"run_id": "r1", "schema_version": "1"},
    {"response": "synthetic secret"},
    {"aggregate_statistics": {"question": "secret"}},
    {"package_versions": {"torch": {"prompt": "secret"}}},
    {"artifact_hashes": {"answer": "a" * 64}},
    {"run_ids": ["/private/root"]},
    {"source_revision": "C:\\private\\root"},
    {"run_id": "hf_syntheticsecret"},
    {"question_count": -1}, {"artifact_sha256": "invalid"},
    {"question_counts": {"math": "private prose secret"}},
    {"aggregate_statistics": {"mean": float("inf")}},
    {"package_versions": {"gold_answer": "secret"}},
    {"aggregate_statistics": {"questionText": "secret"}},
    {"aggregate_statistics": {"responses": "secret"}},
    {"source_file_hashes": {"/private/root": "a" * 64}},
    {"question_count": {"math": 4}},
    {"run_id": ["r1"]},
])
def test_public_manifest_rejects_sensitive_or_invalid_structures(payload):
    isolation = load_isolation()
    with pytest.raises(isolation.SafeLoggingError):
        isolation.sanitize_public_manifest(payload)
