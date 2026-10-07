"""Official MATH acquisition at an immutable revision, with source-local deduplication.

``EleutherAI/hendrycks_math`` is consumed exactly as the hub exposes it: a
requested ref (``main``, a branch, or a tag) is resolved once to a 40-character
commit SHA, every train configuration is read at that pinned SHA, and no row is
ever read through a mutable ref. Configuration names come from the injected hub
client when it can list them, and otherwise from ``datasets``' official
``get_dataset_config_names(repo_id, revision=...)``, which is what the pinned
``huggingface_hub==0.36.0`` client requires. All Hugging Face access is injected,
so the suite runs offline and never resolves the real dataset revision.

Rows are normalized with ``question_normalization_v1`` from
``data_pipeline.canonical``, identified through ``ids.math_question_identity``
(``math:train:`` plus the first 20 bytes of the canonical problem SHA256, while
the full 64-hex digest is retained as ``question_content_hash``), deduplicated
source-locally by that canonical problem hash, and appended through
``data_pipeline.io``.

The content-addressed ``source_snapshot_id`` is the SHA256 of the canonical JSON
of this snapshot's own payload: the resolved revision, the sorted configuration
names, the normalization version, the repository id, and one text-free provenance
entry per upstream row (category, level, the problem and whole-row digests, the
upstream row position, and the derived ``source_row_id``). Because the payload
holds hashes, identifiers, and counts only, and no path, host, clock, or
environment value, the identity is byte-identical on Windows and WSL, and both a
changed revision and any changed row move it.

The private manifest is a ``RunManifest``-shaped envelope (``schema_version``,
``stage``, ``run_id``, ``config_hash``, ``upstream_manifest_hashes``,
``artifacts``, ``counts``, ``created_at_utc``, ``output_manifest_hash``) extended
with the MATH acquisition fields. It carries hashes, counts, and provenance
identifiers only: never problem text, never solution text, and never user
absolute paths.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re

import permstudy

from ..canonical import canonical_json_bytes, normalize_text_v1, sha256_hex
from .. import ids
from ..io import IntegrityError, append_record, scan_jsonl, verify_artifact_ref, write_atomic_manifest
from ..isolation import validate_external_roots, validate_external_to_repository
from ..schema import ArtifactRef, QuestionRecord, Source
from . import QUESTION_RECORD_SCHEMA_VERSION, SourceSnapshot


class MathContractError(ValueError):
    """The MATH acquisition request or an upstream row violates the contract."""


MATH_REPO_ID = "EleutherAI/hendrycks_math"
NORMALIZATION_VERSION = "question_normalization_v1"
MANIFEST_SCHEMA_VERSION = "math_source_manifest_v1"
MANIFEST_STAGE = "sources"
SNAPSHOT_SCHEMA_VERSION = "math_source_snapshot_v1"
ROW_CONTENT_SCHEMA_VERSION = "math_row_content_v1"
# The official dataset exposes one train split per configuration.
TRAIN_SPLIT = "train"
# Only a resolved 40-character lowercase commit SHA is an immutable revision.
_COMMIT_SHA = re.compile(r"[0-9a-f]{40}\Z")
_REQUIRED_ROW_FIELDS = ("level", "problem", "solution")
# Six digits keeps ``source_row_id``'s lexicographic order equal to the upstream
# row order for any configuration far below a million rows.
_ROW_ID_WIDTH = 6
_REPOSITORY_ERROR = f"the MATH acquisition requires exactly {MATH_REPO_ID}"
_JSON_ERRORS = (UnicodeDecodeError, json.JSONDecodeError, OSError)


def repository_root() -> Path:
    """Derive the repository root from the location of the ``permstudy`` package.

    The data root and the Hugging Face cache directory must both stay external to
    this directory, so it is derived from the importable package rather than from
    the current working directory.
    """
    location = getattr(permstudy, "__file__", None)
    if not location:
        raise MathContractError("the permstudy package location cannot be resolved")
    return Path(location).resolve().parent.parent


def default_hf_api():
    """Return the production Hugging Face client, imported only when it is used.

    The client is used for the immutable revision lookup (``dataset_info``).
    Offline tests never reach this function: they inject a fake client through
    ``load_math_train``'s ``api`` keyword. Note that the pinned
    ``huggingface_hub==0.36.0`` client has no configuration-name lookup, so
    configuration names come from ``datasets`` (see ``_dataset_config_names``).
    """
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise MathContractError("huggingface_hub is required to resolve the MATH dataset revision") from error
    return HfApi()


def hf_dataset_loader(repo_id, config, revision, cache_dir):
    """The official row source: one configuration's train split at a pinned revision.

    This is the production implementation of the injected ``dataset_loader``
    seam. ``datasets`` is imported here so importing this module stays cheap and
    the dependency is only required when real acquisition happens.
    """
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise MathContractError("datasets is required to load the MATH dataset") from error
    return load_dataset(repo_id, config, split=TRAIN_SPLIT, revision=revision, cache_dir=cache_dir)


def resolve_math_revision(repo_id, requested_revision, api) -> str:
    """Resolve a requested MATH ref to the immutable commit SHA actually used.

    The repository must be exactly ``EleutherAI/hendrycks_math`` and the requested
    revision must be nonempty text. The hub answer is accepted only as a
    40-character lowercase commit SHA: anything shorter, longer, non-hex, or
    differently cased is rejected rather than recorded, so a mutable ref can never
    become a snapshot revision.
    """
    if repo_id != MATH_REPO_ID:
        raise MathContractError(_REPOSITORY_ERROR)
    if not isinstance(requested_revision, str) or not requested_revision.strip():
        raise MathContractError("requested_revision must be a nonempty text ref")
    try:
        info = api.dataset_info(repo_id, revision=requested_revision)
    except MathContractError:
        raise
    except Exception as error:
        # Never forward a third-party message: it can carry a URL or a token.
        raise MathContractError("the MATH dataset revision could not be resolved") from error
    sha = getattr(info, "sha", None)
    if not isinstance(sha, str) or not _COMMIT_SHA.fullmatch(sha):
        raise MathContractError("a resolved MATH revision must be a 40-character lowercase commit SHA")
    return sha


def load_math_train(repo_id, revision, cache_dir, dataset_loader, *, data_root, api=None) -> SourceSnapshot:
    """Acquire every MATH train configuration at one immutable revision.

    ``dataset_loader`` is the injected row source with the fixed signature
    ``(repo_id, config, revision, cache_dir) -> Iterable[Mapping]``; ``api`` is the
    injected Hugging Face client and defaults to the production client. ``data_root``
    is where the private snapshot is written and must stay outside the repository.

    Order of operations, which is the whole point of the stage:

    1. the repository id, the external cache directory, and the external data root
       (with every effective Hugging Face cache) are validated first, so isolation
       fails before any network access and before anything is written;
    2. the requested revision is resolved to a 40-character commit SHA, or accepted
       as-is when it already is one;
    3. every configuration name is discovered at that SHA and sorted;
    4. every train row of every configuration is read through the loader with the
       pinned SHA only, validated, normalized, and deduplicated;
    5. only then are the canonical records and the private manifest written below
       ``<data_root>/sources/math/<source_snapshot_id>/``.

    An identical rerun reuses the content-addressed artifact instead of appending a
    second copy, and any other content at that path is an integrity failure.
    """
    if repo_id != MATH_REPO_ID:
        raise MathContractError(_REPOSITORY_ERROR)
    cache = Path(cache_dir)
    root = Path(data_root)
    repository = repository_root()
    # The explicit cache path is checked first, exactly as the ReClor stage checks
    # its external source directory first: isolation is a property of the path, not
    # of what the path currently holds, so an in-repository cache is rejected
    # whatever it contains and nothing is created.
    validate_external_to_repository(repository, cache)
    validate_external_roots(repository, root, os.environ)

    connection = api if api is not None else default_hf_api()
    resolved_revision = revision if _is_commit_sha(revision) else resolve_math_revision(repo_id, revision, connection)
    config_names = _dataset_config_names(repo_id, resolved_revision, connection)
    rows = _validated_rows(repo_id, config_names, resolved_revision, cache, dataset_loader)
    _require_one_content_per_id(rows)
    snapshot_manifest = _snapshot_manifest(resolved_revision, config_names, rows)
    snapshot_id = sha256_hex(canonical_json_bytes(snapshot_manifest))
    records, deduplication = _deduplicate(rows, snapshot_id, resolved_revision)
    config_counts = _config_counts(config_names, rows, records)
    relative_questions, questions_sha256, record_count = _write_questions(root, snapshot_id, records)
    artifact = ArtifactRef(relative_questions, questions_sha256, record_count)
    verify_artifact_ref(root, artifact)
    config = _acquisition_config(resolved_revision, snapshot_id, config_names)
    payload = {
        # The RunManifest-shaped envelope every layer records. ``created_at_utc``
        # and ``output_manifest_hash`` are stamped by write_atomic_manifest and
        # excluded from the payload it digests.
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "stage": MANIFEST_STAGE,
        "run_id": ids.run_id(MANIFEST_STAGE, config),
        "config_hash": sha256_hex(canonical_json_bytes(config)),
        # The direct upstream of this stage is the canonical snapshot payload
        # itself, whose hash is the snapshot identity.
        "upstream_manifest_hashes": [snapshot_id],
        "artifacts": [{
            "relative_path": artifact.relative_path,
            "sha256": artifact.sha256,
            "record_count": artifact.record_count,
        }],
        "counts": {
            # Exactly the ReClor envelope's four count keys, so Task 7 reads one
            # shape from both sources; per-configuration detail lives in
            # ``config_counts``.
            "train_rows": len(rows),
            "questions": record_count,
            "duplicate_groups": deduplication["group_count"],
            "duplicate_rows_dropped": deduplication["dropped_row_count"],
        },
        # Source-acquisition fields: the brief's contents are a minimum, not an exclusion.
        "source": Source.MATH.value,
        "source_revision": resolved_revision,
        "source_snapshot_id": snapshot_id,
        "repo_id": MATH_REPO_ID,
        "dataset_configs": list(config_names),
        "question_normalization_version": NORMALIZATION_VERSION,
        "snapshot_manifest": snapshot_manifest,
        "snapshot_manifest_hash": snapshot_id,
        "config_counts": config_counts,
        "deduplication": deduplication,
    }
    manifest_file = root / "sources" / "math" / snapshot_id / "manifest.json"
    output_manifest_hash = write_atomic_manifest(manifest_file, payload)
    snapshot = SourceSnapshot(
        source=Source.MATH,
        source_revision=resolved_revision,
        source_snapshot_id=snapshot_id,
        questions=records,
        private_manifest=_read_manifest(manifest_file, output_manifest_hash),
    )
    snapshot.validate()
    return snapshot


@dataclass(frozen=True)
class _TrainingRow:
    """One validated, normalized MATH train row and its provenance identity."""

    position: int
    config: str
    row_position: int
    source_row_id: str
    original_question_id: str
    problem: str
    solution: str
    level: str
    synthetic: bool
    question_content_hash: str
    row_content_hash: str


def _is_commit_sha(value) -> bool:
    return isinstance(value, str) and bool(_COMMIT_SHA.fullmatch(value))


def _acquisition_config(revision: str, snapshot_id: str, config_names) -> dict[str, object]:
    """Return the canonical configuration of one acquisition run.

    The configuration is deliberately machine-independent: the cache directory,
    the data root, and the fixture location must never reach a manifest, and a
    path-free configuration keeps the manifest payload (and therefore
    ``output_manifest_hash``) reproducible on Windows and WSL. Like the ReClor
    stage's configuration it embeds the content-addressed snapshot identity, so it
    is a function of configuration *and* content rather than a pure configuration
    identity; that is deliberate and keeps ``run_id`` content-addressed.
    """
    return {
        "dataset_configs": list(config_names),
        "question_normalization_version": NORMALIZATION_VERSION,
        "repo_id": MATH_REPO_ID,
        "source": Source.MATH.value,
        "source_revision": revision,
        "source_snapshot_id": snapshot_id,
    }


def _dataset_config_names(repo_id: str, revision: str, api) -> tuple[str, ...]:
    """Return every dataset configuration name at the pinned revision, sorted.

    An injected client that can list configurations is used first, which is the
    seam the offline tests exercise. The pinned ``huggingface_hub==0.36.0`` client
    cannot list them, so the production fallback is ``datasets``' official
    ``get_dataset_config_names(repo_id, revision=...)``, the same library that
    supplies the rows.
    """
    provider = getattr(api, "get_dataset_config_names", None)
    if provider is not None:
        try:
            names = provider(repo_id, revision=revision)
        except MathContractError:
            raise
        except Exception as error:
            raise MathContractError("the MATH dataset configuration names could not be resolved") from error
    else:
        names = _datasets_config_names(repo_id, revision)
    return _validated_config_names(names)


def _datasets_config_names(repo_id: str, revision: str):
    """The official configuration-name lookup, imported only when it is needed."""
    try:
        from datasets import get_dataset_config_names
    except ImportError as error:
        raise MathContractError(
            "datasets.get_dataset_config_names is required to list the MATH dataset configurations"
        ) from error
    try:
        return get_dataset_config_names(repo_id, revision=revision)
    except MathContractError:
        raise
    except Exception as error:
        raise MathContractError("the MATH dataset configuration names could not be resolved") from error


def _validated_config_names(names) -> tuple[str, ...]:
    """Validate and sort configuration names, rejecting a list that cannot be used."""
    if isinstance(names, (str, bytes, bytearray, Mapping)) or not isinstance(names, Iterable):
        raise MathContractError("dataset configuration names must be an iterable of text")
    collected = []
    for name in names:
        if not isinstance(name, str) or not name.strip():
            raise MathContractError("dataset configuration names must be nonempty text")
        if name != name.strip():
            raise MathContractError("dataset configuration names must not carry surrounding whitespace")
        collected.append(name)
    if not collected:
        raise MathContractError("the MATH dataset exposes no train configurations")
    if len(set(collected)) != len(collected):
        raise MathContractError("the MATH dataset repeats a configuration name")
    return tuple(sorted(collected))


def _validated_rows(repo_id, config_names, revision, cache, dataset_loader) -> tuple[_TrainingRow, ...]:
    """Read every configuration's train rows through the injected loader."""
    if not callable(dataset_loader):
        raise MathContractError("dataset_loader must be a callable row source")
    rows = []
    position = 0
    for config in config_names:
        produced = dataset_loader(repo_id, config, revision, cache)
        if isinstance(produced, (str, bytes, bytearray, Mapping)):
            raise MathContractError(f"dataset_loader must return an iterable of row mappings for config {config}")
        try:
            iterator = iter(produced)
        except TypeError as error:
            raise MathContractError(
                f"dataset_loader must return an iterable of row mappings for config {config}"
            ) from error
        for row_position, raw in enumerate(iterator):
            rows.append(_validated_row(raw, config, position, row_position))
            position += 1
    if not rows:
        raise MathContractError("the MATH train split contains no rows")
    return tuple(rows)


def _validated_row(raw, config: str, position: int, row_position: int) -> _TrainingRow:
    """Validate and normalize one upstream row, ignoring fields MATH does not use."""
    if not isinstance(raw, Mapping):
        raise MathContractError(f"train row {row_position} of config {config} is not a row mapping")
    missing = tuple(name for name in _REQUIRED_ROW_FIELDS if name not in raw)
    if missing:
        raise MathContractError(
            f"train row {row_position} of config {config} is missing required fields: {', '.join(missing)}"
        )
    problem = _required_text(raw["problem"], config, row_position, "problem")
    solution = _required_text(raw["solution"], config, row_position, "solution")
    level = _required_text(raw["level"], config, row_position, "level")
    synthetic = raw.get("synthetic", False)
    if type(synthetic) is not bool:
        raise MathContractError(f"train row {row_position} of config {config} has a non-boolean synthetic flag")
    original_question_id, content_hash = _question_identity(problem)
    return _TrainingRow(
        position=position,
        config=config,
        row_position=row_position,
        source_row_id=f"{config}:{row_position:0{_ROW_ID_WIDTH}d}",
        original_question_id=original_question_id,
        problem=problem,
        solution=solution,
        level=level,
        synthetic=synthetic,
        question_content_hash=content_hash,
        row_content_hash=_row_content_hash(config, level, problem, solution, synthetic),
    )


def _required_text(value, config: str, position: int, field: str) -> str:
    if not isinstance(value, str):
        raise MathContractError(f"train row {position} of config {config} field {field} must be text")
    text = normalize_text_v1(value)
    if not text:
        raise MathContractError(
            f"train row {position} of config {config} field {field} is empty after normalization"
        )
    return text


def _question_identity(problem: str) -> tuple[str, str]:
    """Return the readable MATH ID and the full problem digest.

    Routed through Task 2's shared helper so the readable 40-hex ID, the retained
    64-hex digest, and the canonical payload cannot drift from the ID contract.
    """
    return ids.math_question_identity(problem)


def _row_content_hash(category: str, level: str, problem: str, solution: str, synthetic: bool) -> str:
    """Hash every canonical field of one upstream row, so any change moves it."""
    return sha256_hex(canonical_json_bytes({
        "category": category,
        "level": level,
        "problem": problem,
        "schema": ROW_CONTENT_SCHEMA_VERSION,
        "solution": solution,
        "synthetic": synthetic,
    }))


def _require_one_content_per_id(rows) -> None:
    """Reject one readable ID that maps to more than one canonical problem.

    The readable ID is the first 20 bytes of the problem digest, so this is a
    fail-closed invariant guard: two rows that share an ID but not a digest would
    otherwise reach the split stage as two records for one ``original_question_id``.
    """
    known: dict[str, str] = {}
    for row in rows:
        original_question_id = row.original_question_id
        recorded = known.setdefault(original_question_id, row.question_content_hash)
        if recorded != row.question_content_hash:
            raise MathContractError(f"readable question id {original_question_id} maps to conflicting content")


def _snapshot_manifest(revision: str, config_names, rows) -> dict[str, object]:
    """Build the text-free payload whose canonical bytes name this snapshot."""
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "source": Source.MATH.value,
        "repo_id": MATH_REPO_ID,
        "revision": revision,
        "question_normalization_version": NORMALIZATION_VERSION,
        "configs": list(config_names),
        "rows": [
            {
                "category": row.config,
                "level": row.level,
                "question_content_hash": row.question_content_hash,
                "row_content_hash": row.row_content_hash,
                "row_position": row.row_position,
                "source_row_id": row.source_row_id,
            }
            for row in sorted(rows, key=lambda row: (row.config, row.row_position))
        ],
    }


def _deduplicate(rows, snapshot_id: str, revision: str):
    """Keep the smallest provenance ID per canonical problem hash.

    Identical problem content with consistent metadata keeps one deterministic
    representative; identical content whose solution, category, level, or
    synthetic flag disagrees is a snapshot integrity failure, never a silent
    choice. Because the readable ID is derived from the content hash itself, every
    member of a duplicate group shares one ``original_question_id``, so the
    retained provenance is the lexicographically smallest ``source_row_id``, which
    the padded upstream row position makes the earliest upstream occurrence.
    """
    groups: dict[str, list[_TrainingRow]] = {}
    for row in rows:
        groups.setdefault(row.question_content_hash, []).append(row)
    records = []
    duplicate_groups = []
    for content_hash in sorted(groups):
        members = sorted(groups[content_hash], key=lambda row: (row.source_row_id, row.position))
        kept = members[0]
        if len({(member.solution, member.level, member.config, member.synthetic) for member in members}) > 1:
            raise MathContractError(f"duplicate problem content {content_hash} has conflicting metadata")
        if len(members) > 1:
            duplicate_groups.append({
                "question_content_hash": content_hash,
                "kept_source_row_id": kept.source_row_id,
                "dropped_source_row_ids": [member.source_row_id for member in members[1:]],
            })
        records.append(_question_record(kept, snapshot_id, revision))
    records.sort(key=lambda record: record.original_question_id)
    deduplication = {
        "group_count": len(duplicate_groups),
        "dropped_row_count": sum(len(group["dropped_source_row_ids"]) for group in duplicate_groups),
        "groups": duplicate_groups,
    }
    return tuple(records), deduplication


def _config_counts(config_names, rows, records) -> dict[str, dict[str, int]]:
    """Report per-configuration row and kept-question counts, including empty ones."""
    return {
        config: {
            "train_rows": sum(1 for row in rows if row.config == config),
            "questions": sum(1 for record in records if record.category == config),
        }
        for config in config_names
    }


def _question_record(row: _TrainingRow, snapshot_id: str, revision: str) -> QuestionRecord:
    return QuestionRecord(
        schema_version=QUESTION_RECORD_SCHEMA_VERSION,
        source=Source.MATH,
        original_question_id=row.original_question_id,
        question_content_hash=row.question_content_hash,
        source_snapshot_id=snapshot_id,
        source_revision=revision,
        source_row_id=row.source_row_id,
        context=None,
        question=None,
        answers=(),
        gold_label=None,
        problem=row.problem,
        solution=row.solution,
        category=row.config,
        level=row.level,
        synthetic=row.synthetic,
    )


def _write_questions(data_root: Path, snapshot_id: str, records) -> tuple[str, str, int]:
    """Append the canonical records once, or verify the already-written artifact.

    The artifact path is content-addressed, so an identical rerun reuses the
    existing bytes instead of appending a second copy, and any other content at
    that path is an integrity failure.
    """
    directory = data_root / "sources" / "math" / snapshot_id
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
    return f"sources/math/{snapshot_id}/questions.jsonl", sha256_hex(expected), record_count


def _read_manifest(path: Path, expected_hash: str) -> dict[str, object]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except _JSON_ERRORS as error:
        raise IntegrityError("written source manifest is not readable JSON") from error
    if not isinstance(manifest, dict) or manifest.get("output_manifest_hash") != expected_hash:
        raise IntegrityError("written source manifest does not match its payload digest")
    return manifest
