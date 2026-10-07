"""Focused tests for immutable MATH acquisition and source-local deduplication.

Every value used here is explicitly synthetic. The tracked fixture
``tests_permstudy/fixtures/data_pipeline/math_rows.jsonl`` is committed test data
read only by the test-owned injected row loader, so production code is never
handed a path inside the repository; the data root and the Hugging Face cache
directory are always external. Hugging Face access is fully injected: a fake
``HfApi`` supplies the resolved commit SHA and the configuration names, and a fake
``datasets`` module pins the exact production ``load_dataset`` call. The suite
never reaches the network and never resolves the real
``EleutherAI/hendrycks_math`` revision.
"""

import dataclasses
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import types

import pytest

from permstudy.data_pipeline import canonical, ids, io, isolation
from permstudy.data_pipeline.schema import ArtifactRef, QuestionRecord, RunManifest, Source
from permstudy.data_pipeline.sources import SourceSnapshot, math


REPO_ID = "EleutherAI/hendrycks_math"
PINNED_REVISION = "a" * 40
OTHER_REVISION = "b" * 40
# Deliberately unsorted, so sorting must happen inside the acquisition stage.
CONFIG_ORDER = ("prealgebra", "geometry", "counting_and_probability", "algebra")
SORTED_CONFIGS = tuple(sorted(CONFIG_ORDER))
COMMITTED_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "data_pipeline"
FIXTURE_FILE = COMMITTED_FIXTURES / "math_rows.jsonl"
NORMALIZATION_VERSION = "question_normalization_v1"
# The normalized form of fixture row index 4, written independently: NFC composed,
# LF only, ends stripped, interior doubled spaces preserved.
NORMALIZED_PROBLEM = "Synthetic normalization problem: Caf\u00e9\nkeeps  double  spaces."
ALGEBRA_PROBLEM = "Synthetic algebra problem one: solve 2x + 3 = 11 for x."
SOURCE_ROW_ID = re.compile(r"[a-z_]+:[0-9]{6}\Z")
# Every environment variable ``isolation._effective_caches`` may derive a path from.
CACHE_VARIABLES = (
    "XDG_CACHE_HOME",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "HUGGINGFACE_ASSETS_CACHE",
    "HF_ASSETS_CACHE",
    "HF_XET_CACHE",
    "TRANSFORMERS_CACHE",
    "PYTORCH_TRANSFORMERS_CACHE",
    "PYTORCH_PRETRAINED_BERT_CACHE",
    "HF_DATASETS_CACHE",
    "HF_MODULES_CACHE",
    "HF_DATASETS_DOWNLOADED_DATASETS_PATH",
    "HF_DATASETS_EXTRACTED_DATASETS_PATH",
)


@pytest.fixture(autouse=True)
def external_cache_environment(tmp_path, monkeypatch):
    """Keep every effective Hugging Face cache outside this repository."""
    for name in CACHE_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf_cache"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg_cache"))


class FakeHfApi:
    """A deterministic offline stand-in for ``huggingface_hub.HfApi``.

    The injected client is called with ``revision=`` as a keyword, so the fake
    keeps that parameter name.
    """

    def __init__(self, *, sha=PINNED_REVISION, configs=CONFIG_ORDER, info_error=None, configs_error=None):
        self.sha = sha
        self.configs = configs
        self.info_error = info_error
        self.configs_error = configs_error
        self.info_calls = []
        self.config_calls = []

    def dataset_info(self, repo_id, revision=None):
        self.info_calls.append((repo_id, revision))
        if self.info_error is not None:
            raise self.info_error
        return types.SimpleNamespace(sha=self.sha)

    def get_dataset_config_names(self, repo_id, revision=None):
        self.config_calls.append((repo_id, revision))
        if self.configs_error is not None:
            raise self.configs_error
        return self.configs


class NoInfoApi:
    """A hub client whose revision lookup returns nothing at all."""

    def dataset_info(self, repo_id, revision=None):
        return None


def load_fixture_rows():
    """Return fresh copies of the tracked synthetic rows, in file order."""
    return [json.loads(line) for line in FIXTURE_FILE.read_text(encoding="utf-8").splitlines()]


def fixture_rows_by_config():
    """Return fresh fixture rows routed by configuration, with ``config`` dropped."""
    rows_by_config = {}
    for row in load_fixture_rows():
        config = row.pop("config")
        rows_by_config.setdefault(config, []).append(row)
    return rows_by_config


def dict_loader(rows_by_config):
    """Build an injected loader over in-memory rows (used for malformed shapes)."""

    def load(repo_id, config, revision, cache_dir):
        return [dict(row) if isinstance(row, dict) else row for row in rows_by_config.get(config, ())]

    return load


def rows_loader(path, calls=None):
    """Build the injected row loader over a real JSONL file.

    The ``config`` field only routes a synthetic row to its dataset configuration
    and is dropped, so the acquisition stage sees MATH-shaped rows. The ``type``
    field is deliberately retained: real MATH rows carry it and it must be ignored.
    """

    def load(repo_id, config, revision, cache_dir):
        if calls is not None:
            calls.append((repo_id, config, revision, cache_dir))
        rows = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("config") != config:
                continue
            rows.append({key: value for key, value in row.items() if key != "config"})
        return rows

    return load


def derived_rows_file(tmp_path, transform, name="math_rows.jsonl"):
    """Write a deterministically mutated copy of the fixture below ``tmp_path``."""
    rows = load_fixture_rows()
    replacement = transform(rows)
    if replacement is not None:
        rows = replacement
    path = tmp_path / name
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    return path


def load(fixture_path, data_root, *, cache=None, revision=PINNED_REVISION, configs=CONFIG_ORDER, api=None):
    """Acquire a snapshot with the injected fake hub client and row loader."""
    cache_dir = cache if cache is not None else Path(data_root).parent / "hf_cache"
    return math.load_math_train(
        REPO_ID,
        revision,
        cache_dir,
        rows_loader(fixture_path),
        data_root=data_root,
        api=FakeHfApi(configs=configs) if api is None else api,
    )


def questions_path(data_root, snapshot_id):
    return Path(data_root) / "sources" / "math" / snapshot_id / "questions.jsonl"


def manifest_path(data_root, snapshot_id):
    return Path(data_root) / "sources" / "math" / snapshot_id / "manifest.json"


def _drop(index, field):
    def transform(rows):
        del rows[index][field]
        return rows

    return transform


def _set(index, field, value):
    def transform(rows):
        rows[index][field] = value
        return rows

    return transform


def _append(row):
    def transform(rows):
        rows.append(dict(row))
        return rows

    return transform


# --- the committed synthetic fixture -----------------------------------------


def test_committed_fixture_is_synthetic_lf_jsonl():
    raw = FIXTURE_FILE.read_bytes()
    assert b"\r" not in raw
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    rows = load_fixture_rows()
    assert len(rows) == 6
    for row in rows:
        assert row["synthetic"] is True
        assert {"config", "level", "problem", "solution", "type"} <= set(row)
    assert {row["config"] for row in rows} == set(SORTED_CONFIGS)


def test_tracked_fixture_lives_inside_the_repository_and_production_never_reads_it():
    """Ruling: the fixture is committed test data, read only by the injected loader."""
    assert FIXTURE_FILE.is_relative_to(math.repository_root())
    source = inspect.getsource(math)
    assert "fixtures" not in source
    assert "math_rows" not in source


# --- immutable revision resolution -------------------------------------------


def test_resolve_math_revision_returns_the_pinned_commit_sha():
    api = FakeHfApi(sha=PINNED_REVISION)
    assert math.resolve_math_revision(REPO_ID, "main", api) == PINNED_REVISION
    assert api.info_calls == [(REPO_ID, "main")]


@pytest.mark.parametrize(
    "sha",
    ["main", "1" * 39, "1" * 41, "Z" * 40, "A" * 40, "", "   ", "1" * 40 + "\n", 123, None, b"1" * 40],
)
def test_resolve_rejects_a_value_that_is_not_a_lowercase_commit_sha(sha):
    with pytest.raises(math.MathContractError, match="40-character lowercase"):
        math.resolve_math_revision(REPO_ID, "main", FakeHfApi(sha=sha))


def test_resolve_rejects_a_hub_client_that_returns_nothing():
    with pytest.raises(math.MathContractError, match="40-character lowercase"):
        math.resolve_math_revision(REPO_ID, "main", NoInfoApi())


@pytest.mark.parametrize(
    "repo_id",
    [
        "eleutherAI/hendrycks_math",
        "EleutherAI/Hendrycks_Math",
        "hendrycks_math",
        "EleutherAI/hendrycks_math_extra",
        "EleutherAI/math",
        "EleutherAI/hendrycks-math",
        "",
        None,
    ],
)
def test_foreign_repository_ids_are_rejected_before_any_lookup(repo_id):
    api = FakeHfApi()
    with pytest.raises(math.MathContractError, match="EleutherAI/hendrycks_math"):
        math.resolve_math_revision(repo_id, "main", api)
    assert api.info_calls == []


@pytest.mark.parametrize("requested", ["", "   ", None, 7, b"main", ["main"]])
def test_requested_revision_must_be_nonempty_text(requested):
    api = FakeHfApi()
    with pytest.raises(math.MathContractError, match="requested_revision"):
        math.resolve_math_revision(REPO_ID, requested, api)
    assert api.info_calls == []


def test_revision_lookup_failures_are_contract_errors_without_third_party_text():
    api = FakeHfApi(info_error=RuntimeError("https://example.invalid/private-hf-token-value"))
    with pytest.raises(math.MathContractError) as failure:
        math.resolve_math_revision(REPO_ID, "main", api)
    assert "token" not in str(failure.value)
    assert "example.invalid" not in str(failure.value)


def test_math_loader_rejects_a_foreign_repository_before_any_read(tmp_path):
    calls = []
    with pytest.raises(math.MathContractError, match="EleutherAI/hendrycks_math"):
        math.load_math_train(
            "EleutherAI/math",
            PINNED_REVISION,
            tmp_path / "hf",
            rows_loader(FIXTURE_FILE, calls),
            data_root=tmp_path / "external",
            api=FakeHfApi(),
        )
    assert calls == []
    assert not (tmp_path / "external").exists()
    assert not (tmp_path / "hf").exists()


# --- the revision is pinned before any row is read ---------------------------


def test_math_loader_pins_revision_before_reading(tmp_path):
    """The brief's Step 1 test: the loader only ever sees the pinned revision."""
    calls = []
    api = FakeHfApi(sha=PINNED_REVISION)
    snapshot = math.load_math_train(
        repo_id=REPO_ID,
        revision="main",
        cache_dir=tmp_path / "hf",
        dataset_loader=rows_loader(FIXTURE_FILE, calls),
        data_root=tmp_path / "external",
        api=api,
    )
    assert calls and {call[2] for call in calls} == {PINNED_REVISION}
    assert snapshot.source_revision == PINNED_REVISION
    assert api.info_calls == [(REPO_ID, "main")]
    assert api.config_calls == [(REPO_ID, PINNED_REVISION)]


@pytest.mark.parametrize("requested", ["main", "refs/heads/main", "v1.0", "abc123", "release"])
def test_mutable_refs_never_reach_the_row_loader(tmp_path, requested):
    calls = []
    snapshot = math.load_math_train(
        REPO_ID,
        requested,
        tmp_path / "hf",
        rows_loader(FIXTURE_FILE, calls),
        data_root=tmp_path / "external",
        api=FakeHfApi(sha=PINNED_REVISION),
    )
    assert {call[2] for call in calls} == {PINNED_REVISION}
    assert all(requested not in call[2] for call in calls)
    assert snapshot.source_revision == PINNED_REVISION
    assert all(record.source_revision == PINNED_REVISION for record in snapshot.questions)


def test_an_already_pinned_revision_is_used_without_a_revision_lookup(tmp_path):
    api = FakeHfApi(info_error=AssertionError("dataset_info must not be called for a pinned revision"))
    snapshot = math.load_math_train(
        REPO_ID,
        PINNED_REVISION,
        tmp_path / "hf",
        rows_loader(FIXTURE_FILE),
        data_root=tmp_path / "external",
        api=api,
    )
    assert api.info_calls == []
    assert snapshot.source_revision == PINNED_REVISION
    assert api.config_calls == [(REPO_ID, PINNED_REVISION)]


def test_loader_receives_the_repository_pinned_revision_and_cache_dir(tmp_path):
    cache = tmp_path / "hf"
    calls = []
    math.load_math_train(
        REPO_ID,
        "main",
        cache,
        rows_loader(FIXTURE_FILE, calls),
        data_root=tmp_path / "external",
        api=FakeHfApi(),
    )
    assert {call[0] for call in calls} == {REPO_ID}
    assert {call[2] for call in calls} == {PINNED_REVISION}
    assert {Path(call[3]) for call in calls} == {cache}


def test_official_loader_forwards_the_train_split_at_the_pinned_revision(monkeypatch, tmp_path):
    """The production seam must pass the pinned revision, never a mutable ref."""
    calls = []

    def load_dataset(repo_id, config, split=None, revision=None, cache_dir=None):
        calls.append((repo_id, config, split, revision, cache_dir))
        return [{"problem": "p", "solution": "s", "level": "Level 1"}]

    fake = types.ModuleType("datasets")
    fake.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake)
    cache = tmp_path / "hf"
    rows = math.hf_dataset_loader(REPO_ID, "algebra", PINNED_REVISION, cache)
    assert calls == [(REPO_ID, "algebra", "train", PINNED_REVISION, cache)]
    assert rows == [{"problem": "p", "solution": "s", "level": "Level 1"}]


def test_default_hf_api_is_the_pinned_hub_client():
    """The production client is the pinned hub client, used for the revision.

    Note the pinned fact this test records: ``huggingface_hub==0.36.0`` exposes
    ``dataset_info`` (the immutable revision lookup) but no configuration-name
    lookup, so configuration names come from ``datasets``' official
    ``get_dataset_config_names``; both paths are covered below.
    """
    hub = pytest.importorskip("huggingface_hub")
    api = math.default_hf_api()
    assert isinstance(api, hub.HfApi)
    assert callable(api.dataset_info)


def official_datasets_module(monkeypatch, *, config_names=True, rows_by_config=None, calls=None):
    """Install a fake ``datasets`` module for the official acquisition path."""
    module = types.ModuleType("datasets")
    if config_names:
        def get_dataset_config_names(path, revision=None):
            if calls is not None:
                calls.append(("config_names", path, revision))
            return list(CONFIG_ORDER)

        module.get_dataset_config_names = get_dataset_config_names

    def load_dataset(repo_id, config, split=None, revision=None, cache_dir=None):
        if calls is not None:
            calls.append(("load_dataset", repo_id, config, split, revision, cache_dir))
        source = fixture_rows_by_config() if rows_by_config is None else rows_by_config
        return [dict(row) for row in source.get(config, ())]

    module.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", module)
    return module


def test_the_official_datasets_path_runs_offline_for_a_pinned_revision(tmp_path, monkeypatch):
    """``api=None`` is the production client; a pinned revision needs no hub round trip."""
    calls = []
    official_datasets_module(monkeypatch, calls=calls)
    cache = tmp_path / "hf"
    snapshot = math.load_math_train(
        REPO_ID, PINNED_REVISION, cache, math.hf_dataset_loader,
        data_root=tmp_path / "external",
    )
    assert [call for call in calls if call[0] == "config_names"] == [("config_names", REPO_ID, PINNED_REVISION)]
    assert {call[2] for call in calls if call[0] == "load_dataset"} == set(SORTED_CONFIGS)
    assert {call[3] for call in calls if call[0] == "load_dataset"} == {"train"}
    assert {call[4] for call in calls if call[0] == "load_dataset"} == {PINNED_REVISION}
    assert {call[5] for call in calls if call[0] == "load_dataset"} == {cache}
    assert snapshot.source_revision == PINNED_REVISION
    assert len(snapshot.questions) == 5


def test_configuration_names_fall_back_to_datasets_when_the_client_cannot_list_them(tmp_path, monkeypatch):
    calls = []
    official_datasets_module(monkeypatch, calls=calls)

    class RevisionOnlyApi:
        def dataset_info(self, repo_id, revision=None):
            return types.SimpleNamespace(sha=PINNED_REVISION)

    snapshot = math.load_math_train(
        REPO_ID, PINNED_REVISION, tmp_path / "hf", rows_loader(FIXTURE_FILE),
        data_root=tmp_path / "external", api=RevisionOnlyApi(),
    )
    assert [call for call in calls if call[0] == "config_names"] == [("config_names", REPO_ID, PINNED_REVISION)]
    assert snapshot.private_manifest["dataset_configs"] == list(SORTED_CONFIGS)


def test_a_missing_configuration_name_provider_is_a_contract_error(tmp_path, monkeypatch):
    official_datasets_module(monkeypatch, config_names=False)

    class RevisionOnlyApi:
        def dataset_info(self, repo_id, revision=None):
            return types.SimpleNamespace(sha=PINNED_REVISION)

    with pytest.raises(math.MathContractError, match="configuration"):
        math.load_math_train(
            REPO_ID, PINNED_REVISION, tmp_path / "hf", rows_loader(FIXTURE_FILE),
            data_root=tmp_path / "external", api=RevisionOnlyApi(),
        )
    assert not (tmp_path / "external").exists()


# --- every train configuration, deterministically sorted ---------------------


def test_config_names_are_sorted_before_any_row_is_read(tmp_path):
    calls = []
    api = FakeHfApi(configs=CONFIG_ORDER)
    snapshot = math.load_math_train(
        REPO_ID, PINNED_REVISION, tmp_path / "hf", rows_loader(FIXTURE_FILE, calls),
        data_root=tmp_path / "external", api=api,
    )
    assert tuple(CONFIG_ORDER) != SORTED_CONFIGS
    assert [call[1] for call in calls] == list(SORTED_CONFIGS)
    assert snapshot.private_manifest["dataset_configs"] == list(SORTED_CONFIGS)
    assert snapshot.private_manifest["snapshot_manifest"]["configs"] == list(SORTED_CONFIGS)


def test_every_train_config_is_consumed_and_category_comes_from_the_config(tmp_path):
    calls = []
    snapshot = math.load_math_train(
        REPO_ID, PINNED_REVISION, tmp_path / "hf", rows_loader(FIXTURE_FILE, calls),
        data_root=tmp_path / "external", api=FakeHfApi(configs=CONFIG_ORDER),
    )
    assert {call[1] for call in calls} == set(SORTED_CONFIGS)
    assert {record.category for record in snapshot.questions} == set(SORTED_CONFIGS)
    # Fixture row 0 carries a decoy ``type``; the category must still be the config.
    decoys = {record.category for record in snapshot.questions if record.category == "synthetic-decoy-category"}
    assert decoys == set()
    algebra = [record for record in snapshot.questions if record.category == "algebra"]
    assert len(algebra) == 1
    assert algebra[0].problem == ALGEBRA_PROBLEM


def test_config_discovery_uses_the_pinned_revision(tmp_path):
    api = FakeHfApi()
    math.load_math_train(
        REPO_ID, "main", tmp_path / "hf", rows_loader(FIXTURE_FILE),
        data_root=tmp_path / "external", api=api,
    )
    assert api.config_calls == [(REPO_ID, PINNED_REVISION)]


@pytest.mark.parametrize(
    "configs",
    [[], (), ["algebra", "algebra"], ["", "geometry"], [" algebra"], [None], "algebra", None],
)
def test_invalid_configuration_names_are_rejected_without_writing(tmp_path, configs):
    data_root = tmp_path / "external"
    with pytest.raises(math.MathContractError):
        math.load_math_train(
            REPO_ID, PINNED_REVISION, tmp_path / "hf", rows_loader(FIXTURE_FILE),
            data_root=data_root, api=FakeHfApi(configs=configs),
        )
    assert not data_root.exists()
    assert not (tmp_path / "hf").exists()


def test_a_configuration_error_precedes_every_row_read(tmp_path):
    calls = []
    with pytest.raises(math.MathContractError):
        math.load_math_train(
            REPO_ID, PINNED_REVISION, tmp_path / "hf", rows_loader(FIXTURE_FILE, calls),
            data_root=tmp_path / "external", api=FakeHfApi(configs=[]),
        )
    assert calls == []
    assert not (tmp_path / "external").exists()


def test_config_lookup_failures_are_contract_errors(tmp_path):
    api = FakeHfApi(configs_error=RuntimeError("https://example.invalid/private-hf-token-value"))
    with pytest.raises(math.MathContractError, match="configuration names") as failure:
        math.load_math_train(
            REPO_ID, PINNED_REVISION, tmp_path / "hf", rows_loader(FIXTURE_FILE),
            data_root=tmp_path / "external", api=api,
        )
    assert "token" not in str(failure.value)
    assert not (tmp_path / "external").exists()


# --- train row validation ----------------------------------------------------


VALIDATION_CASES = [
    pytest.param(_drop(1, "problem"), id="missing-problem"),
    pytest.param(_drop(1, "solution"), id="missing-solution"),
    pytest.param(_drop(1, "level"), id="missing-level"),
    pytest.param(_set(1, "problem", None), id="problem-not-text"),
    pytest.param(_set(1, "problem", 17), id="problem-integer"),
    pytest.param(_set(1, "problem", ["text"]), id="problem-list"),
    pytest.param(_set(1, "problem", ""), id="problem-empty"),
    pytest.param(_set(1, "problem", "   "), id="problem-blank"),
    pytest.param(_set(1, "problem", "\r\n"), id="problem-only-line-breaks"),
    pytest.param(_set(1, "solution", ""), id="solution-empty"),
    pytest.param(_set(1, "solution", 4), id="solution-integer"),
    pytest.param(_set(1, "level", "   "), id="level-blank"),
    pytest.param(_set(1, "level", 3), id="level-integer"),
    pytest.param(_set(2, "synthetic", "true"), id="non-boolean-synthetic"),
    pytest.param(lambda rows: [], id="no-rows-at-all"),
]


@pytest.mark.parametrize("mutate", VALIDATION_CASES)
def test_invalid_training_rows_are_rejected_without_writing(tmp_path, mutate):
    data_root = tmp_path / "external"
    path = derived_rows_file(tmp_path, mutate)
    with pytest.raises(math.MathContractError):
        load(path, data_root)
    assert not data_root.exists()
    assert not (tmp_path / "hf_cache").exists()


def test_a_config_with_no_train_rows_is_recorded_rather_than_dropped(tmp_path):
    def transform(rows):
        return [row for row in rows if row["config"] != "geometry"]

    snapshot = load(derived_rows_file(tmp_path, transform), tmp_path / "external")
    counts = snapshot.private_manifest["counts"]
    assert counts["train_rows"] == 4
    assert counts["questions"] == 3
    assert snapshot.private_manifest["dataset_configs"] == list(SORTED_CONFIGS)
    assert snapshot.private_manifest["config_counts"]["geometry"] == {"train_rows": 0, "questions": 0}


@pytest.mark.parametrize("malformed", [["not-an-object"], "not-an-object", 17, None])
def test_a_row_that_is_not_a_mapping_is_rejected(tmp_path, malformed):
    rows_by_config = fixture_rows_by_config()
    rows_by_config["geometry"] = [malformed]
    with pytest.raises(math.MathContractError, match="row mapping"):
        math.load_math_train(
            REPO_ID, PINNED_REVISION, tmp_path / "hf", dict_loader(rows_by_config),
            data_root=tmp_path / "external", api=FakeHfApi(),
        )
    assert not (tmp_path / "external").exists()


def test_a_non_callable_loader_is_a_contract_error(tmp_path):
    with pytest.raises(math.MathContractError, match="dataset_loader"):
        math.load_math_train(
            REPO_ID, PINNED_REVISION, tmp_path / "hf", None,
            data_root=tmp_path / "external", api=FakeHfApi(),
        )


@pytest.mark.parametrize("produced", ["a string", {"problem": "p"}, 17, None])
def test_a_loader_that_does_not_produce_rows_is_a_contract_error(tmp_path, produced):
    def loader(repo_id, config, revision, cache_dir):
        return produced

    with pytest.raises(math.MathContractError):
        math.load_math_train(
            REPO_ID, PINNED_REVISION, tmp_path / "hf", loader,
            data_root=tmp_path / "external", api=FakeHfApi(),
        )
    assert not (tmp_path / "external").exists()


def test_a_failing_loader_creates_nothing(tmp_path):
    def loader(repo_id, config, revision, cache_dir):
        if config == "geometry":
            raise RuntimeError("synthetic upstream failure")
        return [{"problem": f"synthetic {config}", "solution": "s", "level": "Level 1"}]

    with pytest.raises(RuntimeError, match="synthetic upstream failure"):
        math.load_math_train(
            REPO_ID, PINNED_REVISION, tmp_path / "hf", loader,
            data_root=tmp_path / "external", api=FakeHfApi(),
        )
    assert not (tmp_path / "external").exists()
    assert not (tmp_path / "hf").exists()


# --- identity and normalization ----------------------------------------------


def test_question_identity_uses_the_shared_math_helper_and_the_full_hash(tmp_path):
    snapshot = load(FIXTURE_FILE, tmp_path / "external")
    assert len(snapshot.questions) == 5
    for record in snapshot.questions:
        expected_id, expected_hash = ids.math_question_identity(record.problem)
        assert record.original_question_id == expected_id
        assert record.question_content_hash == expected_hash
        # Independent literal payload: SHA256 over the UTF-8 bytes of the canonical problem.
        assert record.question_content_hash == canonical.sha256_hex(record.problem.encode("utf-8"))
        assert re.fullmatch(r"[0-9a-f]{64}", record.question_content_hash)
        assert re.fullmatch(r"math:train:[0-9a-f]{40}", record.original_question_id)
        assert record.original_question_id == f"math:train:{record.question_content_hash[:40]}"


def test_private_manifest_stores_the_full_64_hex_question_hash(tmp_path):
    snapshot = load(FIXTURE_FILE, tmp_path / "external")
    manifest = snapshot.private_manifest
    rows = {entry["source_row_id"]: entry for entry in manifest["snapshot_manifest"]["rows"]}
    for record in snapshot.questions:
        entry = rows[record.source_row_id]
        assert entry["question_content_hash"] == record.question_content_hash
        assert re.fullmatch(r"[0-9a-f]{64}", entry["question_content_hash"])
        assert record.original_question_id == f"math:train:{entry['question_content_hash'][:40]}"
    # The dropped duplicate keeps its full digest, not just the readable ID prefix.
    group = manifest["deduplication"]["groups"][0]
    assert group["question_content_hash"] == ids.math_question_identity(ALGEBRA_PROBLEM)[1]
    assert len(group["question_content_hash"]) == 64
    assert rows["algebra:000001"]["question_content_hash"] == group["question_content_hash"]


def test_two_distinct_problems_cannot_share_one_readable_id(tmp_path, monkeypatch):
    """Force the invariant guard: one readable ID must never carry two contents."""

    def colliding_identity(problem):
        digest = canonical.sha256_hex(canonical.canonical_json_bytes(problem))
        return "math:train:" + "0" * 40, digest

    monkeypatch.setattr(math, "_question_identity", colliding_identity)
    with pytest.raises(math.MathContractError, match="conflicting content"):
        load(FIXTURE_FILE, tmp_path / "external")
    assert not (tmp_path / "external").exists()
    assert not (tmp_path / "hf_cache").exists()


def test_question_records_are_valid_math_shapes(tmp_path):
    snapshot = load(FIXTURE_FILE, tmp_path / "external")
    for record in snapshot.questions:
        assert isinstance(record, QuestionRecord)
        record.validate()
        assert record.source is Source.MATH
        assert record.source_snapshot_id == snapshot.source_snapshot_id
        assert record.source_revision == snapshot.source_revision == PINNED_REVISION
        assert record.context is None and record.question is None
        assert record.answers == () and record.gold_label is None
        assert record.problem and record.solution and record.category and record.level
        assert record.synthetic is True
        assert SOURCE_ROW_ID.fullmatch(record.source_row_id)


def test_rows_are_normalized_with_version_v1(tmp_path):
    snapshot = load(FIXTURE_FILE, tmp_path / "external")
    by_row = {record.source_row_id: record for record in snapshot.questions}
    normalized = by_row["geometry:000001"]
    assert normalized.problem == NORMALIZED_PROBLEM
    assert "\r" not in normalized.problem
    assert normalized.problem.endswith("keeps  double  spaces.")
    assert normalized.level == "Level 4"
    assert snapshot.private_manifest["question_normalization_version"] == NORMALIZATION_VERSION


def test_interior_whitespace_is_preserved_in_the_content_hash(tmp_path):
    """Two problems that differ only by an interior space are different content."""

    def transform(rows):
        rows[3]["problem"] = rows[3]["problem"].replace("side length 5", "side  length 5")
        return rows

    baseline = load(FIXTURE_FILE, tmp_path / "baseline")
    changed = load(derived_rows_file(tmp_path, transform), tmp_path / "changed")
    baseline_hashes = {record.source_row_id: record.question_content_hash for record in baseline.questions}
    changed_hashes = {record.source_row_id: record.question_content_hash for record in changed.questions}
    assert changed_hashes["geometry:000000"] != baseline_hashes["geometry:000000"]
    assert changed_hashes["geometry:000001"] == baseline_hashes["geometry:000001"]


# --- source-local deduplication ----------------------------------------------


def test_repeated_problem_keeps_the_smallest_provenance_id(tmp_path):
    snapshot = load(FIXTURE_FILE, tmp_path / "external")
    duplicate_hash = ids.math_question_identity(ALGEBRA_PROBLEM)[1]
    matching = [record for record in snapshot.questions if record.question_content_hash == duplicate_hash]
    assert len(matching) == 1
    assert matching[0].source_row_id == "algebra:000000"
    assert matching[0].original_question_id == f"math:train:{duplicate_hash[:40]}"
    assert "algebra:000001" not in {record.source_row_id for record in snapshot.questions}


def test_dropped_duplicate_provenance_is_retained_in_the_private_manifest(tmp_path):
    snapshot = load(FIXTURE_FILE, tmp_path / "external")
    dedup = snapshot.private_manifest["deduplication"]
    assert dedup["group_count"] == 1
    assert dedup["dropped_row_count"] == 1
    assert len(dedup["groups"]) == 1
    group = dedup["groups"][0]
    assert group["question_content_hash"] == ids.math_question_identity(ALGEBRA_PROBLEM)[1]
    assert group["kept_source_row_id"] == "algebra:000000"
    assert group["dropped_source_row_ids"] == ["algebra:000001"]
    counts = snapshot.private_manifest["counts"]
    # Exactly the ReClor envelope's count keys: one shape for Task 7.
    assert set(counts) == {"train_rows", "questions", "duplicate_groups", "duplicate_rows_dropped"}
    assert counts["train_rows"] == 6
    assert counts["questions"] == 5
    assert counts["duplicate_groups"] == 1
    assert counts["duplicate_rows_dropped"] == 1
    config_counts = snapshot.private_manifest["config_counts"]
    assert config_counts["algebra"] == {"train_rows": 2, "questions": 1}
    assert config_counts["geometry"] == {"train_rows": 2, "questions": 2}


CONFLICT_CASES = [
    pytest.param(_set(1, "solution", "A different synthetic solution."), id="conflicting-solution"),
    pytest.param(_set(1, "level", "Level 5"), id="conflicting-level"),
    pytest.param(_set(1, "synthetic", False), id="conflicting-synthetic"),
]


@pytest.mark.parametrize("mutate", CONFLICT_CASES)
def test_identical_problem_with_conflicting_metadata_fails_fast(tmp_path, mutate):
    data_root = tmp_path / "external"
    path = derived_rows_file(tmp_path, mutate)
    with pytest.raises(math.MathContractError, match="conflicting metadata"):
        load(path, data_root)
    assert not data_root.exists()
    assert not (tmp_path / "hf_cache").exists()


def test_identical_problem_in_another_config_conflicts_on_category(tmp_path):
    """The category is the config name, so a cross-config duplicate must fail."""

    def transform(rows):
        cloned = dict(rows[0])
        cloned["config"] = "geometry"
        rows.append(cloned)
        return rows

    data_root = tmp_path / "external"
    path = derived_rows_file(tmp_path, transform)
    with pytest.raises(math.MathContractError, match="conflicting metadata"):
        load(path, data_root)
    assert not data_root.exists()


def test_duplicates_are_keyed_on_the_normalized_problem_text(tmp_path):
    """A differently encoded but canonically identical problem is a duplicate."""

    def transform(rows):
        appended = dict(rows[4])
        appended["problem"] = NORMALIZED_PROBLEM
        rows.append(appended)
        return rows

    snapshot = load(derived_rows_file(tmp_path, transform), tmp_path / "external")
    duplicates = [
        record for record in snapshot.questions if record.problem == NORMALIZED_PROBLEM
    ]
    assert len(duplicates) == 1
    assert duplicates[0].source_row_id == "geometry:000001"
    dedup = snapshot.private_manifest["deduplication"]
    assert dedup["group_count"] == 2
    assert dedup["dropped_row_count"] == 2
    geometry_group = [group for group in dedup["groups"] if group["kept_source_row_id"] == "geometry:000001"]
    assert len(geometry_group) == 1
    assert geometry_group[0]["dropped_source_row_ids"] == ["geometry:000002"]


# --- content-addressed snapshot identity -------------------------------------


def test_snapshot_id_is_the_hash_of_the_canonical_snapshot_payload(tmp_path):
    snapshot = load(FIXTURE_FILE, tmp_path / "external")
    payload = snapshot.private_manifest["snapshot_manifest"]
    expected = canonical.sha256_hex(canonical.canonical_json_bytes(payload))
    assert snapshot.source_snapshot_id == expected
    assert snapshot.private_manifest["snapshot_manifest_hash"] == expected
    assert re.fullmatch(r"[0-9a-f]{64}", snapshot.source_snapshot_id)
    assert payload["schema_version"] == "math_source_snapshot_v1"
    assert payload["source"] == "math"
    assert payload["repo_id"] == REPO_ID
    assert payload["revision"] == PINNED_REVISION
    assert payload["question_normalization_version"] == NORMALIZATION_VERSION
    assert payload["configs"] == list(SORTED_CONFIGS)
    assert [row["source_row_id"] for row in payload["rows"]] == [
        "algebra:000000", "algebra:000001", "counting_and_probability:000000",
        "geometry:000000", "geometry:000001", "prealgebra:000000",
    ]


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(_set(2, "solution", "Another synthetic solution."), id="changed-solution"),
        pytest.param(_set(2, "level", "Level 5"), id="changed-level"),
        pytest.param(_set(3, "problem", "Synthetic geometry problem two: a square has side length 6."), id="changed-problem"),
        pytest.param(_append({"config": "prealgebra", "level": "Level 2", "problem": "Synthetic prealgebra problem two: compute 1/2 + 1/3.", "solution": "The synthetic sum is 5/6.", "synthetic": True, "type": "prealgebra"}), id="added-row"),
        pytest.param(lambda rows: rows[:-1], id="removed-row"),
    ],
)
def test_snapshot_id_moves_when_any_row_changes(tmp_path, mutate):
    baseline = load(FIXTURE_FILE, tmp_path / "baseline")
    changed = load(derived_rows_file(tmp_path, mutate), tmp_path / "changed")
    assert changed.source_snapshot_id != baseline.source_snapshot_id


def test_snapshot_id_moves_when_only_a_solution_changes(tmp_path):
    """A solution-only change must move the snapshot identity, not the question identity."""

    def mutate(rows):
        rows[2]["solution"] = "A rewritten synthetic solution for the same problem."
        return rows

    baseline = load(FIXTURE_FILE, tmp_path / "baseline")
    changed = load(derived_rows_file(tmp_path, mutate), tmp_path / "changed")
    assert [record.original_question_id for record in changed.questions] == [
        record.original_question_id for record in baseline.questions
    ]
    assert [record.question_content_hash for record in changed.questions] == [
        record.question_content_hash for record in baseline.questions
    ]
    assert changed.source_snapshot_id != baseline.source_snapshot_id
    baseline_rows = baseline.private_manifest["snapshot_manifest"]["rows"]
    changed_rows = changed.private_manifest["snapshot_manifest"]["rows"]
    baseline_hashes = {row["source_row_id"]: row["row_content_hash"] for row in baseline_rows}
    changed_hashes = {row["source_row_id"]: row["row_content_hash"] for row in changed_rows}
    assert changed_hashes["counting_and_probability:000000"] != baseline_hashes["counting_and_probability:000000"]
    assert {key: value for key, value in changed_hashes.items() if key != "counting_and_probability:000000"} == {
        key: value for key, value in baseline_hashes.items() if key != "counting_and_probability:000000"
    }


def test_snapshot_identity_is_invariant_to_fixture_line_endings(tmp_path):
    """The snapshot is content-addressed over parsed rows, never over file bytes.

    Git materializes the committed LF fixture as CRLF on a fresh Windows checkout;
    because MATH identity comes from normalized row text, the snapshot id and the
    records are unchanged either way (unlike ReClor's raw-byte tree hash).
    """
    crlf = tmp_path / "math_rows_crlf.jsonl"
    crlf.write_bytes(FIXTURE_FILE.read_bytes().replace(b"\n", b"\r\n"))
    assert b"\r\n" in crlf.read_bytes()
    baseline = load(FIXTURE_FILE, tmp_path / "baseline")
    materialized = load(crlf, tmp_path / "materialized")
    assert materialized.source_snapshot_id == baseline.source_snapshot_id
    assert materialized.questions == baseline.questions


def test_snapshot_id_moves_when_the_revision_moves(tmp_path):
    first = load(FIXTURE_FILE, tmp_path / "first", revision=PINNED_REVISION)
    second = load(FIXTURE_FILE, tmp_path / "second", revision=OTHER_REVISION)
    assert first.source_snapshot_id != second.source_snapshot_id
    assert first.source_revision == PINNED_REVISION and second.source_revision == OTHER_REVISION
    assert [record.original_question_id for record in first.questions] == [
        record.original_question_id for record in second.questions
    ]
    assert [record.question_content_hash for record in first.questions] == [
        record.question_content_hash for record in second.questions
    ]


def test_snapshot_payload_is_path_free_and_cache_independent(tmp_path):
    first = load(FIXTURE_FILE, tmp_path / "first" / "data", cache=tmp_path / "cache-a")
    second = load(FIXTURE_FILE, tmp_path / "second" / "data", cache=tmp_path / "cache-b" / "deeper")
    assert first.source_snapshot_id == second.source_snapshot_id
    payload = canonical.canonical_json_bytes(dict(first.private_manifest["snapshot_manifest"])).decode("utf-8")
    assert "\\" not in payload
    assert not re.search(r"[A-Za-z]:[\\/]", payload)
    for raw in (tmp_path, tmp_path / "first" / "data", tmp_path / "cache-a", tmp_path / "cache-b", FIXTURE_FILE):
        assert str(raw) not in payload
        assert raw.as_posix() not in payload


def test_independent_runs_are_deterministic(tmp_path):
    first = load(FIXTURE_FILE, tmp_path / "first", cache=tmp_path / "hf-one")
    second = load(FIXTURE_FILE, tmp_path / "second", cache=tmp_path / "hf-two")
    assert first.questions == second.questions
    assert first.source_snapshot_id == second.source_snapshot_id
    left = dict(first.private_manifest)
    right = dict(second.private_manifest)
    assert left["output_manifest_hash"] == right["output_manifest_hash"]
    left.pop("created_at_utc")
    right.pop("created_at_utc")
    assert left == right
    assert questions_path(tmp_path / "first", first.source_snapshot_id).read_bytes() == questions_path(
        tmp_path / "second", second.source_snapshot_id
    ).read_bytes()


_CHILD_SCRIPT = """
import json
import sys
import types
from pathlib import Path

from permstudy.data_pipeline.sources import math

fixture, data_root, cache_dir, pinned = sys.argv[1:5]
rows_by_config = {}
for line in Path(fixture).read_text(encoding="utf-8").splitlines():
    row = json.loads(line)
    rows_by_config.setdefault(row.pop("config"), []).append(row)


class Api:
    def dataset_info(self, repo_id, revision=None):
        return types.SimpleNamespace(sha=pinned)

    def get_dataset_config_names(self, repo_id, revision=None):
        # Reverse order on purpose: the stage must sort the config names itself.
        return sorted(rows_by_config, reverse=True)


def load(repo_id, config, revision, cache_dir):
    return [dict(row) for row in rows_by_config.get(config, ())]


snapshot = math.load_math_train(
    "EleutherAI/hendrycks_math", "main", Path(cache_dir), load,
    data_root=Path(data_root), api=Api(),
)
manifest = dict(snapshot.private_manifest)
print(json.dumps({
    "source_snapshot_id": snapshot.source_snapshot_id,
    "source_revision": snapshot.source_revision,
    "output_manifest_hash": manifest["output_manifest_hash"],
    "question_ids": [record.original_question_id for record in snapshot.questions],
    "question_hashes": [record.question_content_hash for record in snapshot.questions],
}))
"""


def test_two_independent_processes_produce_identical_snapshot_identity(tmp_path):
    root = Path(__file__).resolve().parents[2]
    results = []
    for index in (1, 2):
        completed = subprocess.run(
            [
                sys.executable, "-c", _CHILD_SCRIPT, str(FIXTURE_FILE),
                str(tmp_path / f"process-{index}" / "data"),
                str(tmp_path / f"process-{index}" / "cache"),
                PINNED_REVISION,
            ],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(root)},
        )
        assert completed.returncode == 0, completed.stderr
        results.append(json.loads(completed.stdout.strip().splitlines()[-1]))

    reference = load(FIXTURE_FILE, tmp_path / "reference" / "data", cache=tmp_path / "reference" / "cache")
    expected = {
        "source_snapshot_id": reference.source_snapshot_id,
        "source_revision": PINNED_REVISION,
        "output_manifest_hash": reference.private_manifest["output_manifest_hash"],
        "question_ids": [record.original_question_id for record in reference.questions],
        "question_hashes": [record.question_content_hash for record in reference.questions],
    }
    assert results[0] == expected
    assert results[1] == expected


# --- the RunManifest-shaped private manifest ---------------------------------


EXPECTED_MANIFEST_KEYS = frozenset({
    "schema_version", "stage", "run_id", "config_hash", "upstream_manifest_hashes",
    "artifacts", "counts", "created_at_utc", "output_manifest_hash",
    "source", "source_revision", "source_snapshot_id", "repo_id", "dataset_configs",
    "question_normalization_version", "snapshot_manifest", "snapshot_manifest_hash",
    "config_counts", "deduplication",
})


def test_private_manifest_carries_the_full_run_manifest_shape(tmp_path):
    snapshot = load(FIXTURE_FILE, tmp_path / "external")
    manifest = snapshot.private_manifest
    assert set(manifest) == EXPECTED_MANIFEST_KEYS
    RunManifest(
        schema_version=manifest["schema_version"],
        stage=manifest["stage"],
        run_id=manifest["run_id"],
        config_hash=manifest["config_hash"],
        upstream_manifest_hashes=tuple(manifest["upstream_manifest_hashes"]),
        artifacts=tuple(ArtifactRef(**artifact) for artifact in manifest["artifacts"]),
        counts=manifest["counts"],
        created_at_utc=manifest["created_at_utc"],
        output_manifest_hash=manifest["output_manifest_hash"],
    ).validate()
    assert manifest["schema_version"] == "math_source_manifest_v1"
    assert manifest["stage"] == "sources"
    assert manifest["source"] == "math"
    assert manifest["repo_id"] == REPO_ID
    assert manifest["source_revision"] == PINNED_REVISION
    assert manifest["upstream_manifest_hashes"] == [manifest["snapshot_manifest_hash"]]
    assert manifest["snapshot_manifest_hash"] == manifest["source_snapshot_id"] == snapshot.source_snapshot_id
    expected_config = {
        "dataset_configs": list(SORTED_CONFIGS),
        "question_normalization_version": NORMALIZATION_VERSION,
        "repo_id": REPO_ID,
        "source": "math",
        "source_revision": PINNED_REVISION,
        "source_snapshot_id": snapshot.source_snapshot_id,
    }
    assert manifest["config_hash"] == canonical.sha256_hex(canonical.canonical_json_bytes(expected_config))
    assert manifest["run_id"] == ids.run_id("sources", expected_config)


def test_output_manifest_hash_excludes_the_envelope_fields(tmp_path):
    snapshot = load(FIXTURE_FILE, tmp_path / "external")
    manifest = dict(snapshot.private_manifest)
    envelope = {key: value for key, value in manifest.items() if key not in {"created_at_utc", "output_manifest_hash"}}
    assert manifest["output_manifest_hash"] == canonical.sha256_hex(canonical.canonical_json_bytes(envelope))
    assert manifest["created_at_utc"].endswith("Z")


def test_private_manifest_never_contains_source_text_or_user_paths(tmp_path):
    snapshot = load(FIXTURE_FILE, tmp_path / "external")
    blob = canonical.canonical_json_bytes(dict(snapshot.private_manifest)).decode("utf-8")
    assert snapshot.questions
    for record in snapshot.questions:
        assert record.problem not in blob
        assert record.solution not in blob
    assert "\\" not in blob
    for raw_path in (tmp_path, FIXTURE_FILE):
        assert str(raw_path) not in blob
        assert raw_path.as_posix() not in blob


def test_public_evidence_projection_is_allowlisted_and_text_free(tmp_path):
    snapshot = load(FIXTURE_FILE, tmp_path / "external")
    manifest = dict(snapshot.private_manifest)
    evidence = {
        "schema_version": "math_source_evidence_v1",
        "source_revisions": {"math": snapshot.source_revision},
        "question_counts": {"math": manifest["counts"]["questions"]},
        "duplicate_counts": {"math": manifest["counts"]["duplicate_rows_dropped"]},
        "artifact_hashes": {"math": manifest["artifacts"][0]["sha256"]},
    }
    assert isolation.sanitize_public_manifest(evidence) == evidence
    blob = canonical.canonical_json_bytes(evidence).decode("utf-8")
    for record in snapshot.questions:
        assert record.problem not in blob
        assert record.solution not in blob
    with pytest.raises(isolation.SafeLoggingError):
        isolation.sanitize_public_manifest(manifest)


# --- approved external output layout ----------------------------------------


def test_outputs_are_written_only_below_the_approved_external_layout(tmp_path):
    data_root = tmp_path / "external"
    snapshot = load(FIXTURE_FILE, data_root)
    target = questions_path(data_root, snapshot.source_snapshot_id)
    manifest_file = manifest_path(data_root, snapshot.source_snapshot_id)
    assert target.is_file() and manifest_file.is_file()
    assert target.parents[3] == data_root
    scan = io.scan_jsonl(target)
    assert scan.quarantined_tail_path is None
    assert len(scan.records) == len(snapshot.questions) == 5
    assert all(record["record_hash"] for record in scan.records)
    artifacts = snapshot.private_manifest["artifacts"]
    assert len(artifacts) == 1
    ref = ArtifactRef(**artifacts[0])
    assert ref.relative_path == f"sources/math/{snapshot.source_snapshot_id}/questions.jsonl"
    assert ref.record_count == 5
    assert ref.sha256 == scan.file_sha256
    assert io.verify_artifact_ref(data_root, ref) is None
    assert "\\" not in ref.relative_path and not ref.relative_path.startswith("/")


def test_rerunning_into_the_same_root_does_not_append_duplicates(tmp_path):
    data_root = tmp_path / "external"
    first = load(FIXTURE_FILE, data_root)
    target = questions_path(data_root, first.source_snapshot_id)
    before = target.read_bytes()
    second = load(FIXTURE_FILE, data_root)
    assert target.read_bytes() == before
    assert second.questions == first.questions
    assert second.private_manifest["output_manifest_hash"] == first.private_manifest["output_manifest_hash"]


def test_existing_corrupted_snapshot_artifact_fails_fast(tmp_path):
    data_root = tmp_path / "external"
    snapshot = load(FIXTURE_FILE, data_root)
    target = questions_path(data_root, snapshot.source_snapshot_id)
    corrupted = target.read_bytes() + b"{}\n"
    target.write_bytes(corrupted)
    with pytest.raises(io.IntegrityError):
        load(FIXTURE_FILE, data_root)
    assert target.read_bytes() == corrupted


def test_existing_snapshot_artifact_with_extra_records_fails_fast(tmp_path):
    data_root = tmp_path / "external"
    snapshot = load(FIXTURE_FILE, data_root)
    target = questions_path(data_root, snapshot.source_snapshot_id)
    payload = canonical.canonical_json_bytes({})
    extra = canonical.canonical_json_bytes({"record_hash": canonical.sha256_hex(payload)}) + b"\n"
    target.write_bytes(target.read_bytes() + extra)
    with pytest.raises(io.IntegrityError, match="content-addressed"):
        load(FIXTURE_FILE, data_root)


# --- isolation: everything is validated before the revision is resolved -------


def test_cache_dir_inside_the_repository_is_rejected_before_any_write(tmp_path):
    cache_dir = math.repository_root() / "artifacts" / "task6-math-cache-must-not-exist"
    data_root = tmp_path / "external"
    calls = []
    api = FakeHfApi()
    with pytest.raises(isolation.IsolationError, match="must not contain one another"):
        math.load_math_train(
            REPO_ID, "main", cache_dir, rows_loader(FIXTURE_FILE, calls), data_root=data_root, api=api
        )
    assert api.info_calls == [] and api.config_calls == []
    assert calls == []
    assert not data_root.exists()
    assert not cache_dir.exists()


def test_cache_dir_reached_through_dotdot_is_rejected_before_any_write(tmp_path):
    cache_dir = math.repository_root() / "permstudy" / ".." / "artifacts" / "task6-in-repo-cache-must-not-exist"
    data_root = tmp_path / "external"
    with pytest.raises(isolation.IsolationError, match="must not contain one another"):
        math.load_math_train(
            REPO_ID, PINNED_REVISION, cache_dir, rows_loader(FIXTURE_FILE),
            data_root=data_root, api=FakeHfApi(),
        )
    assert not data_root.exists()
    assert not cache_dir.resolve().exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows junction behavior")
def test_cache_dir_reached_through_a_junction_is_rejected_before_any_write(tmp_path):
    alias = tmp_path / "junction"
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(alias), str(math.repository_root() / "permstudy")],
        capture_output=True,
    )
    if created.returncode:
        pytest.skip("OS does not permit junctions")
    data_root = tmp_path / "external"
    with pytest.raises(isolation.IsolationError, match="must not contain one another"):
        math.load_math_train(
            REPO_ID, PINNED_REVISION, alias, rows_loader(FIXTURE_FILE),
            data_root=data_root, api=FakeHfApi(),
        )
    assert not data_root.exists()


def test_data_root_inside_the_repository_is_rejected_before_writing(tmp_path):
    data_root = math.repository_root() / "artifacts" / "task6-math-must-not-exist"
    calls = []
    api = FakeHfApi()
    with pytest.raises(isolation.IsolationError, match="must not contain one another"):
        math.load_math_train(
            REPO_ID, "main", tmp_path / "hf", rows_loader(FIXTURE_FILE, calls), data_root=data_root, api=api
        )
    assert api.info_calls == [] and api.config_calls == []
    assert calls == []
    assert not data_root.exists()
    assert not (tmp_path / "hf").exists()


def test_data_root_that_contains_the_repository_is_rejected_before_writing(tmp_path):
    data_root = math.repository_root().parent
    with pytest.raises(isolation.IsolationError, match="must not contain one another"):
        math.load_math_train(
            REPO_ID, PINNED_REVISION, tmp_path / "hf", rows_loader(FIXTURE_FILE),
            data_root=data_root, api=FakeHfApi(),
        )


def test_effective_hf_cache_inside_the_repository_is_rejected_before_writing(tmp_path, monkeypatch):
    data_root = tmp_path / "external"
    monkeypatch.setenv("HF_HOME", str(math.repository_root() / "artifacts" / "task6-rejected-hf-home"))
    with pytest.raises(isolation.IsolationError, match="cache"):
        math.load_math_train(
            REPO_ID, PINNED_REVISION, tmp_path / "hf", rows_loader(FIXTURE_FILE),
            data_root=data_root, api=FakeHfApi(),
        )
    assert not data_root.exists()
    assert not (tmp_path / "hf").exists()


def test_isolation_precedes_revision_resolution_and_every_row_read(tmp_path):
    cache_dir = math.repository_root() / "artifacts" / "task6-order-must-not-exist"
    api = FakeHfApi()
    calls = []
    with pytest.raises(isolation.IsolationError):
        math.load_math_train(
            REPO_ID, "main", cache_dir, rows_loader(FIXTURE_FILE, calls),
            data_root=tmp_path / "external", api=api,
        )
    assert api.info_calls == []
    assert api.config_calls == []
    assert calls == []
    assert not cache_dir.exists()
    assert not (tmp_path / "external").exists()


# --- result type -------------------------------------------------------------


def test_source_snapshot_is_frozen_and_self_consistent(tmp_path):
    snapshot = load(FIXTURE_FILE, tmp_path / "external")
    snapshot.validate()
    assert isinstance(snapshot.questions, tuple)
    assert snapshot.source is Source.MATH
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.source_snapshot_id = "changed"
    with pytest.raises(TypeError):
        snapshot.private_manifest["repo_id"] = "changed"


def test_source_snapshot_validate_rejects_records_from_another_snapshot(tmp_path):
    snapshot = load(FIXTURE_FILE, tmp_path / "external")
    foreign = dataclasses.replace(snapshot.questions[0], source_snapshot_id="0" * 64)
    broken = SourceSnapshot(
        source=snapshot.source,
        source_revision=snapshot.source_revision,
        source_snapshot_id=snapshot.source_snapshot_id,
        questions=(foreign,),
        private_manifest={},
    )
    with pytest.raises(ValueError, match="snapshot"):
        broken.validate()
