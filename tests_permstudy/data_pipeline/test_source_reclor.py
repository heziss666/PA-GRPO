"""Focused tests for explicit ReClor acquisition and source-local deduplication.

Every value used here is explicitly synthetic. The committed fixture directory
is the only tree of official-shaped files these tests read, and derived trees
are deterministic copies written below ``tmp_path``.
"""

import dataclasses
import json
from pathlib import Path
import re
import shutil

import pytest

from permstudy.data_pipeline import canonical, ids, io, isolation
from permstudy.data_pipeline.schema import ArtifactRef, QuestionRecord, RunManifest, Source
from permstudy.data_pipeline.sources import SourceSnapshot, reclor


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "data_pipeline" / "reclor"
OFFICIAL_FILES = ("test.json", "train.json", "use_items.txt", "val.json")
TRAIN_IDS = (
    "synthetic-train-001",
    "synthetic-train-002",
    "synthetic-train-003",
    "synthetic-train-004",
)
EXPECTED_IDS = tuple(f"reclor:train:{name}" for name in TRAIN_IDS)
HELD_OUT_IDS = (
    "synthetic-val-001",
    "synthetic-val-002",
    "synthetic-test-001",
    "synthetic-test-002",
)
# Text that exists only inside the synthetic fixture files.
SOURCE_TEXT = (
    "Synthetic scenario one: a laboratory keeps four labelled samples in a single row.",
    "Which sample is leftmost?",
    "First synthetic option",
    "A second synthetic line follows.",
    "interior spacing is preserved here",
)
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


def load(source_path, data_root):
    """Call the acquisition entry point with the mandatory acknowledgement."""
    return reclor.load_reclor_train(source_path, data_root, acknowledge_noncommercial=True)


def derived_tree(tmp_path, mutate=None):
    """Copy the committed synthetic snapshot and optionally rewrite one file."""
    tree = tmp_path / "reclor"
    shutil.copytree(FIXTURE_DIR, tree)
    if mutate is not None:
        mutate(tree)
    return tree


def load_train_rows(tree):
    return json.loads((tree / "train.json").read_text(encoding="utf-8"))


def write_train_rows(tree, rows):
    payload = json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
    (tree / "train.json").write_text(payload, encoding="utf-8")


def mutate_rows(transform):
    """Build a tree mutator that rewrites ``train.json`` through ``transform``."""

    def mutate(tree):
        write_train_rows(tree, transform(load_train_rows(tree)))

    return mutate


def write_raw_train(text):
    def mutate(tree):
        (tree / "train.json").write_text(text, encoding="utf-8")

    return mutate


def questions_path(data_root, snapshot_id):
    return Path(data_root) / "sources" / "reclor" / snapshot_id / "questions.jsonl"


def manifest_path(data_root, snapshot_id):
    return Path(data_root) / "sources" / "reclor" / snapshot_id / "manifest.json"


# --- contract: non-commercial acknowledgement and inputs ---------------------


def test_omitting_the_acknowledgement_keyword_is_not_callable(tmp_path):
    with pytest.raises(TypeError):
        reclor.load_reclor_train(FIXTURE_DIR, tmp_path / "external")


@pytest.mark.parametrize("value", [False, None, 0, 1, "true"])
def test_only_explicit_true_acknowledges_the_noncommercial_contract(tmp_path, value):
    with pytest.raises(reclor.ReclorContractError, match="non-commercial"):
        reclor.load_reclor_train(FIXTURE_DIR, tmp_path / "external", acknowledge_noncommercial=value)
    assert not (tmp_path / "external").exists()


def test_source_path_must_be_an_extracted_directory(tmp_path):
    absent = tmp_path / "absent"
    with pytest.raises(reclor.ReclorContractError, match="directory"):
        load(absent, tmp_path / "external-a")
    plain_file = tmp_path / "train.json"
    plain_file.write_text("[]", encoding="utf-8")
    with pytest.raises(reclor.ReclorContractError, match="directory"):
        load(plain_file, tmp_path / "external-b")
    assert not (tmp_path / "external-a").exists()
    assert not (tmp_path / "external-b").exists()


@pytest.mark.parametrize("missing", OFFICIAL_FILES)
def test_every_official_snapshot_file_is_required(tmp_path, missing):
    tree = derived_tree(tmp_path)
    (tree / missing).unlink()
    with pytest.raises(reclor.ReclorContractError, match=re.escape(missing)):
        load(tree, tmp_path / "external")
    assert not (tmp_path / "external").exists()


# --- label canonicalization --------------------------------------------------


def test_reclor_label_comes_from_official_field(tmp_path):
    snapshot = load(FIXTURE_DIR, tmp_path / "external")
    assert {record.gold_label for record in snapshot.questions} == {"A", "B", "C", "D"}


@pytest.mark.parametrize("value,want", [(0, "A"), (1, "B"), (2, "C"), (3, "D"), ("A", "A"), ("d", "D"), ("2", "C")])
def test_canonical_label_accepts_official_encodings(value, want):
    assert reclor.canonical_reclor_label(value) == want


@pytest.mark.parametrize("value", [4, -1, True, False, None, 1.0, "E", "", "   ", "10", ["A"], {"label": 0}])
def test_canonical_label_rejects_unsupported_values(value):
    with pytest.raises(reclor.ReclorContractError):
        reclor.canonical_reclor_label(value)


# --- train row validation ----------------------------------------------------


def _drop(field):
    def transform(rows):
        del rows[1][field]
        return rows

    return transform


def _set(field, value):
    def transform(rows):
        rows[1][field] = value
        return rows

    return transform


VALIDATION_CASES = [
    pytest.param(mutate_rows(_drop("label")), id="missing-label"),
    pytest.param(mutate_rows(_drop("id_string")), id="missing-id"),
    pytest.param(mutate_rows(_drop("context")), id="missing-context"),
    pytest.param(mutate_rows(_drop("question")), id="missing-question"),
    pytest.param(mutate_rows(_drop("answers")), id="missing-answers"),
    pytest.param(mutate_rows(_set("label", 4)), id="label-above-range"),
    pytest.param(mutate_rows(_set("label", -1)), id="label-below-range"),
    pytest.param(mutate_rows(_set("label", "Z")), id="label-unknown-letter"),
    pytest.param(mutate_rows(_set("label", True)), id="label-boolean"),
    pytest.param(mutate_rows(_set("label", 1.0)), id="label-float"),
    pytest.param(mutate_rows(_set("answers", ["a", "b", "c", "d", "e"])), id="five-answers"),
    pytest.param(mutate_rows(_set("answers", ["a", "b", "c"])), id="three-answers"),
    pytest.param(mutate_rows(_set("answers", "abcd")), id="answers-not-a-list"),
    pytest.param(mutate_rows(_set("answers", ["a", "b", "c", 4])), id="answer-not-text"),
    pytest.param(mutate_rows(_set("answers", ["a", "b", "c", "   "])), id="answer-blank"),
    pytest.param(mutate_rows(_set("id_string", "")), id="blank-id"),
    pytest.param(mutate_rows(_set("id_string", 17)), id="non-text-id"),
    pytest.param(mutate_rows(_set("context", "   ")), id="blank-context"),
    pytest.param(mutate_rows(_set("question", "")), id="blank-question"),
    pytest.param(mutate_rows(_set("synthetic", "true")), id="non-boolean-synthetic"),
    pytest.param(mutate_rows(lambda rows: rows[:1] + ["not-an-object"] + rows[2:]), id="row-not-an-object"),
    pytest.param(write_raw_train('{"rows": []}'), id="train-not-an-array"),
    pytest.param(write_raw_train("[{\"id_string\": \"x\""), id="train-not-json"),
    pytest.param(write_raw_train("[]"), id="train-empty"),
    pytest.param(
        write_raw_train(
            '[{"id_string":"synthetic-dup-001","id_string":"synthetic-dup-002",'
            '"context":"Synthetic duplicate-key scenario.","question":"Which synthetic key wins?",'
            '"answers":["One","Two","Three","Four"],"label":1,"synthetic":true}]'
        ),
        id="repeated-object-key",
    ),
    pytest.param(
        write_raw_train(
            '[{"id_string":"synthetic-nan-001","context":"Synthetic NaN scenario.",'
            '"question":"Which synthetic constant appears?","answers":["One","Two","Three","Four"],'
            '"label":NaN,"synthetic":true}]'
        ),
        id="non-standard-constant",
    ),
]


@pytest.mark.parametrize("mutate", VALIDATION_CASES)
def test_invalid_training_rows_are_rejected_without_writing(tmp_path, mutate):
    tree = derived_tree(tmp_path, mutate)
    with pytest.raises(reclor.ReclorContractError):
        load(tree, tmp_path / "external")
    assert not (tmp_path / "external").exists()


# --- snapshot integrity over all four official files -------------------------


def test_private_manifest_hashes_exactly_the_four_official_files(tmp_path):
    snapshot = load(FIXTURE_DIR, tmp_path / "external")
    manifest = snapshot.private_manifest
    assert set(manifest["source_file_hashes"]) == set(OFFICIAL_FILES)
    for name in OFFICIAL_FILES:
        expected = canonical.sha256_hex((FIXTURE_DIR / name).read_bytes())
        assert manifest["source_file_hashes"][name] == expected
    entries = manifest["tree_manifest"]["files"]
    assert [entry["relative_path"] for entry in entries] == sorted(OFFICIAL_FILES)
    for entry in entries:
        name = entry["relative_path"]
        assert entry["sha256"] == manifest["source_file_hashes"][name]
        assert entry["size_bytes"] == (FIXTURE_DIR / name).stat().st_size
    expected_tree_hash = canonical.sha256_hex(canonical.canonical_json_bytes(manifest["tree_manifest"]))
    assert manifest["tree_manifest_hash"] == expected_tree_hash
    assert manifest["source_snapshot_id"] == expected_tree_hash
    assert snapshot.source_snapshot_id == expected_tree_hash
    assert snapshot.source_revision == expected_tree_hash
    assert manifest["source_revision"] == expected_tree_hash
    assert manifest["license_scope"] == "non_commercial_research"
    assert snapshot.source is Source.RECLOR


@pytest.mark.parametrize("mutated", OFFICIAL_FILES)
def test_mutating_any_one_official_file_changes_the_tree_hash(tmp_path, mutated):
    baseline = load(FIXTURE_DIR, tmp_path / "baseline")

    def mutate(tree):
        # Touch this file's own bytes, whatever its type: a trailing space keeps
        # the JSON files valid and the held-out files are never parsed at all.
        target = tree / mutated
        target.write_bytes(target.read_bytes() + b" ")

    changed = load(derived_tree(tmp_path, mutate), tmp_path / "changed")
    baseline_hashes = {
        entry["relative_path"]: entry["sha256"] for entry in baseline.private_manifest["tree_manifest"]["files"]
    }
    changed_hashes = {
        entry["relative_path"]: entry["sha256"] for entry in changed.private_manifest["tree_manifest"]["files"]
    }
    assert changed_hashes[mutated] != baseline_hashes[mutated]
    assert {name: value for name, value in changed_hashes.items() if name != mutated} == {
        name: value for name, value in baseline_hashes.items() if name != mutated
    }
    assert changed.private_manifest["tree_manifest_hash"] != baseline.private_manifest["tree_manifest_hash"]
    assert changed.source_snapshot_id != baseline.source_snapshot_id
    assert changed.private_manifest["source_snapshot_id"] != baseline.private_manifest["source_snapshot_id"]


def test_snapshot_id_is_the_hash_of_the_canonical_tree_manifest_bytes(tmp_path):
    snapshot = load(FIXTURE_DIR, tmp_path / "external")
    manifest = snapshot.private_manifest
    assert snapshot.source_snapshot_id == canonical.sha256_hex(
        canonical.canonical_json_bytes(manifest["tree_manifest"])
    )
    assert re.fullmatch(r"[0-9a-f]{64}", snapshot.source_snapshot_id)


# The final private-manifest key list: the RunManifest-shaped envelope plus the
# source-acquisition fields the brief requires. Task 6 mirrors this shape.
EXPECTED_MANIFEST_KEYS = frozenset({
    "schema_version", "stage", "run_id", "config_hash", "upstream_manifest_hashes",
    "artifacts", "counts", "created_at_utc", "output_manifest_hash",
    "source", "source_revision", "source_snapshot_id", "license_scope",
    "noncommercial_acknowledged", "question_normalization_version",
    "source_file_hashes", "tree_manifest", "tree_manifest_hash", "deduplication",
})


def test_private_manifest_carries_the_full_run_manifest_shape(tmp_path):
    snapshot = load(FIXTURE_DIR, tmp_path / "external")
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
    assert manifest["stage"] == "sources"
    # This stage's only manifest-shaped upstream is its own canonical source tree.
    assert manifest["upstream_manifest_hashes"] == [manifest["tree_manifest_hash"]]
    # Ruling 4: the added envelope fields must not move the snapshot identity.
    assert manifest["source_snapshot_id"] == manifest["tree_manifest_hash"] == snapshot.source_snapshot_id
    expected_config = {
        "license_scope": "non_commercial_research",
        "question_normalization_version": "question_normalization_v1",
        "required_source_files": sorted(OFFICIAL_FILES),
        "source": "reclor",
        "source_snapshot_id": snapshot.source_snapshot_id,
    }
    assert manifest["config_hash"] == canonical.sha256_hex(canonical.canonical_json_bytes(expected_config))
    assert manifest["run_id"] == ids.run_id("sources", expected_config)


def test_output_manifest_hash_excludes_the_envelope_fields(tmp_path):
    snapshot = load(FIXTURE_DIR, tmp_path / "external")
    manifest = dict(snapshot.private_manifest)
    envelope = {key: value for key, value in manifest.items() if key not in {"created_at_utc", "output_manifest_hash"}}
    assert manifest["output_manifest_hash"] == canonical.sha256_hex(canonical.canonical_json_bytes(envelope))
    assert manifest["created_at_utc"].endswith("Z")


def test_crlf_materialization_changes_the_snapshot_id_but_not_record_content(tmp_path):
    """Git may materialize these committed LF fixtures with CRLF on Windows.

    The snapshot is content-addressed over exact file bytes, so a materialized
    checkout names a different snapshot; the parsed questions must not change,
    because no JSON string value contains a literal newline.
    """

    def mutate(tree):
        for name in OFFICIAL_FILES:
            raw = (tree / name).read_bytes()
            (tree / name).write_bytes(raw.replace(b"\n", b"\r\n"))

    baseline = load(FIXTURE_DIR, tmp_path / "baseline")
    materialized = load(derived_tree(tmp_path, mutate), tmp_path / "materialized")
    assert materialized.source_snapshot_id != baseline.source_snapshot_id
    assert [record.original_question_id for record in materialized.questions] == [
        record.original_question_id for record in baseline.questions
    ]
    assert [record.question_content_hash for record in materialized.questions] == [
        record.question_content_hash for record in baseline.questions
    ]


# --- only official train rows become questions -------------------------------


def test_val_test_and_use_items_never_become_questions(tmp_path):
    snapshot = load(FIXTURE_DIR, tmp_path / "external")
    assert tuple(record.original_question_id for record in snapshot.questions) == EXPECTED_IDS
    joined = " ".join(record.original_question_id + record.source_row_id for record in snapshot.questions)
    for held_out in HELD_OUT_IDS:
        assert held_out not in joined
    assert len(snapshot.questions) == 4


def test_question_records_are_valid_reclor_shapes(tmp_path):
    snapshot = load(FIXTURE_DIR, tmp_path / "external")
    for record in snapshot.questions:
        assert isinstance(record, QuestionRecord)
        record.validate()
        assert record.source is Source.RECLOR
        assert record.source_snapshot_id == snapshot.source_snapshot_id
        assert record.source_revision == snapshot.source_revision
        assert len(record.answers) == 4
        assert record.gold_label in ("A", "B", "C", "D")
        assert record.problem is None and record.solution is None
        assert record.category is None and record.level is None
        assert record.synthetic is True


def test_content_hash_uses_the_shared_structured_helper(tmp_path):
    snapshot = load(FIXTURE_DIR, tmp_path / "external")
    for record in snapshot.questions:
        expected = ids.reclor_content_hash(record.context, record.question, record.answers)
        assert record.question_content_hash == expected
        assert expected == canonical.sha256_hex(canonical.canonical_json_bytes({
            "answers": list(record.answers),
            "context": record.context,
            "question": record.question,
            "schema": "reclor_question_content_v1",
        }))


def test_text_is_normalized_with_version_v1_on_ingestion(tmp_path):
    snapshot = load(FIXTURE_DIR, tmp_path / "external")
    by_id = {record.source_row_id: record for record in snapshot.questions}
    assert "\r" not in by_id["synthetic-train-003"].context
    assert by_id["synthetic-train-003"].context == (
        "Synthetic scenario three: line endings are exercised here.\nA second synthetic line follows."
    )
    assert by_id["synthetic-train-004"].question == "Which synthetic conclusion holds?"
    assert by_id["synthetic-train-004"].context.endswith("keeps  double  spaces.")


def test_answer_order_changes_the_content_hash(tmp_path):
    baseline = load(FIXTURE_DIR, tmp_path / "baseline")
    baseline_hash = {record.source_row_id: record.question_content_hash for record in baseline.questions}

    def mutate(tree):
        rows = load_train_rows(tree)
        first, second = rows[2]["answers"][0], rows[2]["answers"][1]
        rows[2]["answers"][0], rows[2]["answers"][1] = second, first
        write_train_rows(tree, rows)

    reordered = load(derived_tree(tmp_path, mutate), tmp_path / "reordered")
    reordered_hash = {record.source_row_id: record.question_content_hash for record in reordered.questions}
    assert reordered_hash["synthetic-train-003"] != baseline_hash["synthetic-train-003"]
    assert reordered_hash["synthetic-train-001"] == baseline_hash["synthetic-train-001"]


# --- source-local deduplication ---------------------------------------------


def _duplicate_content_hash():
    rows = json.loads((FIXTURE_DIR / "train.json").read_text(encoding="utf-8"))
    return ids.reclor_content_hash(rows[1]["context"], rows[1]["question"], rows[1]["answers"])


def test_repeated_content_keeps_the_smallest_provenance_id(tmp_path):
    snapshot = load(FIXTURE_DIR, tmp_path / "external")
    duplicate_hash = _duplicate_content_hash()
    matching = [record for record in snapshot.questions if record.question_content_hash == duplicate_hash]
    assert len(matching) == 1
    assert matching[0].source_row_id == "synthetic-train-002"
    assert matching[0].original_question_id == "reclor:train:synthetic-train-002"
    kept_ids = {record.source_row_id for record in snapshot.questions}
    assert "synthetic-train-005" not in kept_ids


def test_dropped_duplicate_provenance_is_retained_in_the_private_manifest(tmp_path):
    snapshot = load(FIXTURE_DIR, tmp_path / "external")
    dedup = snapshot.private_manifest["deduplication"]
    assert dedup["group_count"] == 1
    assert dedup["dropped_row_count"] == 1
    assert len(dedup["groups"]) == 1
    group = dedup["groups"][0]
    assert group["question_content_hash"] == _duplicate_content_hash()
    assert group["kept_source_row_id"] == "synthetic-train-002"
    assert group["dropped_source_row_ids"] == ["synthetic-train-005"]
    counts = snapshot.private_manifest["counts"]
    assert counts["train_rows"] == 5
    assert counts["questions"] == 4
    assert counts["duplicate_groups"] == 1
    assert counts["duplicate_rows_dropped"] == 1


def test_identical_content_with_conflicting_labels_fails_fast(tmp_path):
    tree = derived_tree(tmp_path, mutate_rows(_set("label", 2)))
    with pytest.raises(reclor.ReclorContractError, match="conflicting metadata"):
        load(tree, tmp_path / "external")
    assert not (tmp_path / "external").exists()


def test_one_official_id_with_conflicting_content_fails_fast(tmp_path):
    def transform(rows):
        clone = json.loads(json.dumps(rows[3]))
        clone["id_string"] = rows[0]["id_string"]
        return rows + [clone]

    tree = derived_tree(tmp_path, mutate_rows(transform))
    with pytest.raises(reclor.ReclorContractError, match="conflicting content"):
        load(tree, tmp_path / "external")
    assert not (tmp_path / "external").exists()


# --- approved external output layout ----------------------------------------


def test_outputs_are_written_only_below_the_approved_external_layout(tmp_path):
    data_root = tmp_path / "external"
    snapshot = load(FIXTURE_DIR, data_root)
    target = questions_path(data_root, snapshot.source_snapshot_id)
    manifest_file = manifest_path(data_root, snapshot.source_snapshot_id)
    assert target.is_file() and manifest_file.is_file()
    assert target.parents[3] == data_root
    scan = io.scan_jsonl(target)
    assert scan.quarantined_tail_path is None
    assert len(scan.records) == len(snapshot.questions) == 4
    assert all(record["record_hash"] for record in scan.records)
    artifacts = snapshot.private_manifest["artifacts"]
    assert len(artifacts) == 1
    ref = ArtifactRef(**artifacts[0])
    assert ref.relative_path == f"sources/reclor/{snapshot.source_snapshot_id}/questions.jsonl"
    assert ref.record_count == 4
    assert ref.sha256 == scan.file_sha256
    assert io.verify_artifact_ref(data_root, ref) is None
    assert "\\" not in ref.relative_path and not ref.relative_path.startswith("/")


def test_private_manifest_never_contains_source_text_or_user_paths(tmp_path):
    data_root = tmp_path / "external"
    snapshot = load(FIXTURE_DIR, data_root)
    blob = canonical.canonical_json_bytes(dict(snapshot.private_manifest)).decode("utf-8")
    for text in SOURCE_TEXT:
        assert text not in blob
    for raw_path in (str(FIXTURE_DIR), str(data_root)):
        assert json.dumps(raw_path)[1:-1] not in blob
    assert "\\\\" not in blob
    assert snapshot.private_manifest["noncommercial_acknowledged"] is True


def test_public_evidence_projection_is_allowlisted_and_free_of_source_text(tmp_path):
    snapshot = load(FIXTURE_DIR, tmp_path / "external")
    evidence = reclor.public_reclor_evidence(snapshot)
    assert isolation.sanitize_public_manifest(evidence) == evidence
    blob = canonical.canonical_json_bytes(evidence).decode("utf-8")
    for text in SOURCE_TEXT:
        assert text not in blob
    assert evidence["tree_manifest_hash"] == snapshot.source_snapshot_id
    assert evidence["question_counts"] == {"reclor": 4}
    with pytest.raises(isolation.SafeLoggingError):
        isolation.sanitize_public_manifest(dict(snapshot.private_manifest))


# --- isolation and determinism ----------------------------------------------


def test_data_root_inside_the_repository_is_rejected_before_writing(tmp_path):
    data_root = reclor.repository_root() / "artifacts" / "task5-reclor-must-not-exist"
    assert not data_root.exists()
    with pytest.raises(isolation.IsolationError, match="must not contain one another"):
        load(FIXTURE_DIR, data_root)
    assert not data_root.exists()


def test_huggingface_cache_inside_the_repository_is_rejected_before_writing(tmp_path, monkeypatch):
    data_root = tmp_path / "external"
    monkeypatch.setenv("HF_HOME", str(reclor.repository_root() / "artifacts" / "task5-cache-must-not-exist"))
    with pytest.raises(isolation.IsolationError, match="cache"):
        load(FIXTURE_DIR, data_root)
    assert not data_root.exists()


def test_independent_runs_are_deterministic(tmp_path):
    first = load(FIXTURE_DIR, tmp_path / "first")
    second = load(FIXTURE_DIR, tmp_path / "second")
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


def test_rerunning_into_the_same_root_does_not_append_duplicates(tmp_path):
    data_root = tmp_path / "external"
    first = load(FIXTURE_DIR, data_root)
    target = questions_path(data_root, first.source_snapshot_id)
    before = target.read_bytes()
    second = load(FIXTURE_DIR, data_root)
    assert target.read_bytes() == before
    assert second.questions == first.questions
    assert second.private_manifest["output_manifest_hash"] == first.private_manifest["output_manifest_hash"]


def test_existing_corrupted_snapshot_artifact_fails_fast(tmp_path):
    data_root = tmp_path / "external"
    snapshot = load(FIXTURE_DIR, data_root)
    target = questions_path(data_root, snapshot.source_snapshot_id)
    corrupted = target.read_bytes() + b"{}\n"
    target.write_bytes(corrupted)
    with pytest.raises(io.IntegrityError):
        load(FIXTURE_DIR, data_root)
    assert target.read_bytes() == corrupted


def test_existing_snapshot_artifact_with_extra_records_fails_fast(tmp_path):
    data_root = tmp_path / "external"
    snapshot = load(FIXTURE_DIR, data_root)
    target = questions_path(data_root, snapshot.source_snapshot_id)
    payload = canonical.canonical_json_bytes({})
    extra = canonical.canonical_json_bytes({"record_hash": canonical.sha256_hex(payload)}) + b"\n"
    target.write_bytes(target.read_bytes() + extra)
    with pytest.raises(io.IntegrityError, match="content-addressed"):
        load(FIXTURE_DIR, data_root)


# --- result type -------------------------------------------------------------


def test_source_snapshot_is_frozen_and_self_consistent(tmp_path):
    snapshot = load(FIXTURE_DIR, tmp_path / "external")
    snapshot.validate()
    assert isinstance(snapshot.questions, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.source_snapshot_id = "changed"
    with pytest.raises(TypeError):
        snapshot.private_manifest["tree_manifest_hash"] = "changed"


def test_source_snapshot_validate_rejects_records_from_another_snapshot(tmp_path):
    snapshot = load(FIXTURE_DIR, tmp_path / "external")
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
