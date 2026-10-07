"""Official ReClor acquisition, snapshot hashing, and source-local deduplication.

Phase 1 accepts only an already extracted ReClor directory: archive and password
handling stay explicitly out of scope. The four official files (``test.json``,
``train.json``, ``use_items.txt``, ``val.json``) are hashed into a canonical
tree manifest before a single training row is read, and the SHA256 of those
canonical tree-manifest bytes is the content-addressed ``source_snapshot_id``.
Only ``train.json`` produces ``QuestionRecord`` values; the validation, test,
and use-item files participate in snapshot integrity alone.

Every canonical byte sequence and digest comes from ``data_pipeline.canonical``
and every question identity and run identity from ``data_pipeline.ids``. Records
are appended through ``data_pipeline.io``'s append-only JSONL helper and the
private manifest is swapped in atomically, so one snapshot always produces the
same bytes. The private manifest is a ``RunManifest``-shaped envelope
(``schema_version``, ``stage``, ``run_id``, ``config_hash``,
``upstream_manifest_hashes``, ``artifacts``, ``counts``, ``created_at_utc``,
``output_manifest_hash``) extended with the source-acquisition fields, and it
carries hashes, counts, and provenance identifiers only: never source text and
never user absolute paths.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path

import permstudy

from ..canonical import canonical_json_bytes, normalize_text_v1, sha256_hex
from .. import ids
from ..io import IntegrityError, append_record, scan_jsonl, verify_artifact_ref, write_atomic_manifest
from ..isolation import validate_external_roots, validate_external_to_repository
from ..schema import ArtifactRef, QuestionRecord, Source
from . import QUESTION_RECORD_SCHEMA_VERSION, SourceSnapshot


class ReclorContractError(ValueError):
    """The ReClor source tree or the acquisition request violates the contract."""


QUESTION_ID_PREFIX = "reclor:train:"
LICENSE_SCOPE = "non_commercial_research"
NORMALIZATION_VERSION = "question_normalization_v1"
MANIFEST_SCHEMA_VERSION = "reclor_source_manifest_v1"
MANIFEST_STAGE = "sources"
TREE_MANIFEST_SCHEMA_VERSION = "source_tree_v1"
PUBLIC_EVIDENCE_SCHEMA_VERSION = "reclor_source_evidence_v1"
OFFICIAL_TRAIN_FILE = "train.json"
# Exactly the four official snapshot files, named here in POSIX sort order.
REQUIRED_SOURCE_FILES = ("test.json", "train.json", "use_items.txt", "val.json")
_ANSWER_COUNT = 4
_LABELS = ("A", "B", "C", "D")
_REQUIRED_ROW_FIELDS = ("answers", "context", "id_string", "label", "question")
_ACKNOWLEDGEMENT_ERROR = "ReClor acquisition requires an explicit non-commercial acknowledgement"
_JSON_ERRORS = (UnicodeDecodeError, json.JSONDecodeError, OSError)


def repository_root() -> Path:
    """Derive the repository root from the location of the ``permstudy`` package.

    The data root must stay external to this directory, so it is derived from the
    importable package rather than from the current working directory.
    """
    location = getattr(permstudy, "__file__", None)
    if not location:
        raise ReclorContractError("the permstudy package location cannot be resolved")
    return Path(location).resolve().parent.parent


def canonical_reclor_label(label) -> str:
    """Map an official ReClor label onto ``A``-``D``.

    Official files carry an integer ``0``-``3``; the single-letter and digit
    forms are accepted for callers that already canonicalized the field.
    Booleans are rejected even though Python treats them as integers.
    """
    if type(label) is int:
        if 0 <= label < len(_LABELS):
            return _LABELS[label]
        raise ReclorContractError(f"ReClor label {label} is outside the official 0-3 range")
    if isinstance(label, str):
        text = label.strip()
        if len(text) == 1 and text.upper() in _LABELS:
            return text.upper()
        if text in ("0", "1", "2", "3"):
            return _LABELS[int(text)]
    raise ReclorContractError("ReClor label must be an official integer 0-3 or one of A-D")


def load_reclor_train(source_path, data_root, *, acknowledge_noncommercial: bool) -> SourceSnapshot:
    """Acquire the official ReClor training rows from an extracted directory.

    ``acknowledge_noncommercial`` is required and must be literally ``True``:
    ReClor is restricted to non-commercial research use, and a package caller
    must not be able to reach the source without that acknowledgement.

    Isolation is validated first: the extracted source directory and the data
    root must both stay outside the repository, and an in-repository path or an
    effective Hugging Face cache inside the repository fails before the source is
    read and before anything is written. Validation, test, and use-item files are
    hashed but never parsed into questions.
    """
    if acknowledge_noncommercial is not True:
        raise ReclorContractError(_ACKNOWLEDGEMENT_ERROR)
    source_directory = Path(source_path)
    data_root = Path(data_root)
    repository = repository_root()
    # The source check precedes the directory check so a path inside the
    # repository is rejected as an isolation violation whatever it holds.
    validate_external_to_repository(repository, source_directory)
    if not source_directory.is_dir():
        raise ReclorContractError("source_path must be an extracted ReClor directory")
    validate_external_roots(repository, data_root, os.environ)

    tree_manifest, file_hashes, snapshot_id = _tree_manifest(source_directory)
    rows = _validated_rows(source_directory / OFFICIAL_TRAIN_FILE)
    records, deduplication = _deduplicate(rows, snapshot_id)
    relative_questions, questions_sha256, record_count = _write_questions(data_root, snapshot_id, records)
    artifact = ArtifactRef(relative_questions, questions_sha256, record_count)
    verify_artifact_ref(data_root, artifact)
    config = _acquisition_config(snapshot_id)
    payload = {
        # The RunManifest-shaped envelope every layer records. ``created_at_utc``
        # and ``output_manifest_hash`` are stamped by write_atomic_manifest and
        # excluded from the payload it digests.
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "stage": MANIFEST_STAGE,
        "run_id": ids.run_id(MANIFEST_STAGE, config),
        "config_hash": sha256_hex(canonical_json_bytes(config)),
        # The direct upstream of this stage is the canonical source tree itself.
        "upstream_manifest_hashes": [snapshot_id],
        "artifacts": [{
            "relative_path": artifact.relative_path,
            "sha256": artifact.sha256,
            "record_count": artifact.record_count,
        }],
        "counts": {
            "train_rows": len(rows),
            "questions": record_count,
            "duplicate_groups": deduplication["group_count"],
            "duplicate_rows_dropped": deduplication["dropped_row_count"],
        },
        # Source-acquisition fields: the brief's contents are a minimum, not an exclusion.
        "source": Source.RECLOR.value,
        "source_revision": snapshot_id,
        "source_snapshot_id": snapshot_id,
        "license_scope": LICENSE_SCOPE,
        "noncommercial_acknowledged": True,
        "question_normalization_version": NORMALIZATION_VERSION,
        "source_file_hashes": file_hashes,
        "tree_manifest": tree_manifest,
        "tree_manifest_hash": snapshot_id,
        "deduplication": deduplication,
    }
    manifest_file = data_root / "sources" / "reclor" / snapshot_id / "manifest.json"
    output_manifest_hash = write_atomic_manifest(manifest_file, payload)
    snapshot = SourceSnapshot(
        source=Source.RECLOR,
        source_revision=snapshot_id,
        source_snapshot_id=snapshot_id,
        questions=records,
        private_manifest=_read_manifest(manifest_file, output_manifest_hash),
    )
    snapshot.validate()
    return snapshot


def _acquisition_config(snapshot_id: str) -> dict[str, object]:
    """Return the canonical configuration of one acquisition run.

    The configuration is deliberately machine-independent: the source directory
    path must never reach a manifest, and a path-free configuration keeps the
    manifest payload (and therefore ``output_manifest_hash``) reproducible
    across Windows and WSL.
    """
    return {
        "license_scope": LICENSE_SCOPE,
        "question_normalization_version": NORMALIZATION_VERSION,
        "required_source_files": list(REQUIRED_SOURCE_FILES),
        "source": Source.RECLOR.value,
        "source_snapshot_id": snapshot_id,
    }


def public_reclor_evidence(snapshot: SourceSnapshot) -> dict[str, object]:
    """Project one ReClor snapshot onto the allowlisted public evidence schema.

    The projection carries hashes and counts only, so it crosses the public
    boundary through ``isolation.sanitize_public_manifest`` without source text,
    provenance text, or user paths.
    """
    manifest = snapshot.private_manifest
    counts = manifest["counts"]
    return {
        "schema_version": PUBLIC_EVIDENCE_SCHEMA_VERSION,
        "source_file_hashes": dict(manifest["source_file_hashes"]),
        "tree_manifest_hash": snapshot.source_snapshot_id,
        "question_counts": {Source.RECLOR.value: counts["questions"]},
        "duplicate_counts": {Source.RECLOR.value: counts["duplicate_rows_dropped"]},
        "artifact_hashes": {Source.RECLOR.value: manifest["artifacts"][0]["sha256"]},
    }


@dataclass(frozen=True)
class _TrainingRow:
    """One validated, normalized official training row and its content identity."""

    position: int
    row_id: str
    context: str
    question: str
    answers: tuple[str, ...]
    gold_label: str
    synthetic: bool
    content_hash: str

    @property
    def original_question_id(self) -> str:
        return f"{QUESTION_ID_PREFIX}{self.row_id}"


def _tree_manifest(source_directory: Path) -> tuple[dict[str, object], dict[str, str], str]:
    """Hash the four official files into a canonical relative-file tree manifest."""
    entries = []
    file_hashes = {}
    for name in REQUIRED_SOURCE_FILES:
        path = source_directory / name
        if not path.is_file():
            raise ReclorContractError(f"required ReClor source file is missing: {name}")
        data = path.read_bytes()
        file_hashes[name] = sha256_hex(data)
        entries.append({"relative_path": name, "sha256": file_hashes[name], "size_bytes": len(data)})
    entries.sort(key=lambda entry: entry["relative_path"])
    tree_manifest = {"schema_version": TREE_MANIFEST_SCHEMA_VERSION, "files": entries}
    return tree_manifest, file_hashes, sha256_hex(canonical_json_bytes(tree_manifest))


def _validated_rows(train_path: Path) -> tuple[_TrainingRow, ...]:
    try:
        payload = json.loads(
            train_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except _JSON_ERRORS as error:
        raise ReclorContractError(f"{OFFICIAL_TRAIN_FILE} is not readable JSON") from error
    if not isinstance(payload, list):
        raise ReclorContractError(f"{OFFICIAL_TRAIN_FILE} must be a JSON array of training rows")
    if not payload:
        raise ReclorContractError(f"{OFFICIAL_TRAIN_FILE} contains no question rows")
    rows = tuple(_validated_row(row, position) for position, row in enumerate(payload))
    _require_one_content_per_id(rows)
    return rows


def _validated_row(row: object, position: int) -> _TrainingRow:
    if not isinstance(row, Mapping):
        raise ReclorContractError(f"train row {position} is not a JSON object")
    missing = tuple(name for name in _REQUIRED_ROW_FIELDS if name not in row)
    if missing:
        raise ReclorContractError(f"train row {position} is missing required fields: {', '.join(missing)}")
    row_id = row["id_string"]
    if not isinstance(row_id, str) or not row_id.strip():
        raise ReclorContractError(f"train row {position} has an invalid official id_string")
    context = _required_text(row["context"], position, "context")
    question = _required_text(row["question"], position, "question")
    answers = row["answers"]
    if not isinstance(answers, list) or len(answers) != _ANSWER_COUNT:
        raise ReclorContractError(f"train row {position} must carry exactly {_ANSWER_COUNT} ordered answers")
    ordered = tuple(_required_text(answer, position, "answers") for answer in answers)
    synthetic = row.get("synthetic", False)
    if type(synthetic) is not bool:
        raise ReclorContractError(f"train row {position} has a non-boolean synthetic flag")
    try:
        gold_label = canonical_reclor_label(row["label"])
    except ReclorContractError as error:
        raise ReclorContractError(f"train row {position}: {error}") from error
    return _TrainingRow(
        position=position,
        row_id=row_id,
        context=context,
        question=question,
        answers=ordered,
        gold_label=gold_label,
        synthetic=synthetic,
        content_hash=ids.reclor_content_hash(context, question, ordered),
    )


def _required_text(value: object, position: int, field: str) -> str:
    if not isinstance(value, str):
        raise ReclorContractError(f"train row {position} field {field} must be text")
    text = normalize_text_v1(value)
    if not text:
        raise ReclorContractError(f"train row {position} field {field} is empty after normalization")
    return text


def _require_one_content_per_id(rows: Sequence[_TrainingRow]) -> None:
    """Reject one official ID that maps to more than one question content."""
    known: dict[str, str] = {}
    for row in rows:
        recorded = known.setdefault(row.row_id, row.content_hash)
        if recorded != row.content_hash:
            raise ReclorContractError(
                f"official id_string {row.row_id} maps to conflicting content within the snapshot"
            )


def _deduplicate(rows: Sequence[_TrainingRow], snapshot_id: str) -> tuple[tuple[QuestionRecord, ...], dict[str, object]]:
    """Keep the smallest provenance ID per content hash and record the dropped rows.

    Identical content with consistent metadata keeps one deterministic
    representative. Identical content whose official label or synthetic flag
    disagrees is a snapshot integrity failure, never a silent choice.
    """
    groups: dict[str, list[_TrainingRow]] = {}
    for row in rows:
        groups.setdefault(row.content_hash, []).append(row)
    records = []
    duplicate_groups = []
    for content_hash in sorted(groups):
        members = sorted(groups[content_hash], key=lambda row: (row.row_id, row.position))
        kept = members[0]
        if len({(member.gold_label, member.synthetic) for member in members}) > 1:
            raise ReclorContractError(f"duplicate question content {content_hash} has conflicting metadata")
        if len(members) > 1:
            duplicate_groups.append({
                "question_content_hash": content_hash,
                "kept_source_row_id": kept.row_id,
                "dropped_source_row_ids": [member.row_id for member in members[1:]],
            })
        records.append(_question_record(kept, snapshot_id))
    records.sort(key=lambda record: record.original_question_id)
    deduplication = {
        "group_count": len(duplicate_groups),
        "dropped_row_count": sum(len(group["dropped_source_row_ids"]) for group in duplicate_groups),
        "groups": duplicate_groups,
    }
    return tuple(records), deduplication


def _question_record(row: _TrainingRow, snapshot_id: str) -> QuestionRecord:
    return QuestionRecord(
        schema_version=QUESTION_RECORD_SCHEMA_VERSION,
        source=Source.RECLOR,
        original_question_id=row.original_question_id,
        question_content_hash=row.content_hash,
        source_snapshot_id=snapshot_id,
        source_revision=snapshot_id,
        source_row_id=row.row_id,
        context=row.context,
        question=row.question,
        answers=row.answers,
        gold_label=row.gold_label,
        problem=None,
        solution=None,
        category=None,
        level=None,
        synthetic=row.synthetic,
    )


def _write_questions(data_root: Path, snapshot_id: str, records: Sequence[QuestionRecord]) -> tuple[str, str, int]:
    """Append the canonical records once, or verify the already-written artifact.

    The artifact path is content-addressed, so an identical rerun reuses the
    existing bytes instead of appending a second copy, and any other content at
    that path is an integrity failure.
    """
    directory = data_root / "sources" / "reclor" / snapshot_id
    questions_file = directory / "questions.jsonl"
    payloads = [record.to_dict() for record in records]
    expected = b"".join(
        canonical_json_bytes({**payload, "record_hash": sha256_hex(canonical_json_bytes(payload))}) + b"\n"
        for payload in payloads
    )
    directory.mkdir(parents=True, exist_ok=True)
    if questions_file.exists():
        scan = scan_jsonl(questions_file)
        if questions_file.read_bytes() != expected:
            raise IntegrityError("existing questions.jsonl does not match the content-addressed snapshot")
        record_count = len(scan.records)
    else:
        for payload in payloads:
            append_record(questions_file, payload)
        if questions_file.read_bytes() != expected:
            raise IntegrityError("written questions.jsonl does not match the canonical record bytes")
        record_count = len(payloads)
    return f"sources/reclor/{snapshot_id}/questions.jsonl", sha256_hex(expected), record_count


def _read_manifest(path: Path, expected_hash: str) -> dict[str, object]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except _JSON_ERRORS as error:
        raise IntegrityError("written source manifest is not readable JSON") from error
    if not isinstance(manifest, dict) or manifest.get("output_manifest_hash") != expected_hash:
        raise IntegrityError("written source manifest does not match its payload digest")
    return manifest


def _reject_duplicate_keys(pairs):
    """Reject duplicate object keys so a parsed row has exactly one interpretation."""
    row = {}
    for key, value in pairs:
        if key in row:
            raise ReclorContractError("a ReClor source row repeats an object key")
        row[key] = value
    return row


def _reject_constant(value):
    raise ReclorContractError(f"a ReClor source row contains the non-standard JSON constant {value}")
