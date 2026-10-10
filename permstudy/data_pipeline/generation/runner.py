"""Crash-safe append-only execution and resume for one generation shard."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re

from ..canonical import canonical_json_bytes, normalize_text_v1, sha256_hex
from ..ids import candidate_id as logical_candidate_id
from ..ids import run_id
from ..io import IntegrityError, ShardLock, append_record, scan_jsonl, write_atomic_manifest
from ..isolation import sanitize_exception, validate_external_to_repository
from ..schema import CandidatePlan, CandidateRecord, FailureRecord
from .base import (
    GenerationBackend,
    GenerationConfig,
    GenerationConfigurationError,
    GenerationPlan,
    GenerationResult,
    generation_stage_config,
)


SHARD_PLAN_SCHEMA = "generation_shard_plan_v1"
SHARD_MANIFEST_SCHEMA = "generation_shard_manifest_v1"
SEMANTIC_SET_SCHEMA = "semantic_candidate_set_v1"
RESPONSE_NORMALIZATION_VERSION = "response_normalization_v1"

_HASH = re.compile(r"[0-9a-f]{64}\Z")
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_ENVELOPE_FIELDS = frozenset({"created_at_utc", "output_manifest_hash"})
_ERROR_TYPES = frozenset({"timeout", "oom", "dependency_error", "access_denied", "unexpected_error"})


class RunMismatchError(ValueError):
    """A shard plan, persisted manifest, or candidate identity does not match."""


@dataclass(frozen=True)
class GenerationShardPlan:
    schema_version: str
    generation_run_id: str
    config_hash: str
    generator_configs: tuple[GenerationConfig, ...]
    shard_size: int
    generator_config: GenerationConfig
    shard_id: str
    candidates: tuple[CandidatePlan, ...]
    split_upstream_manifest_hash: str


@dataclass(frozen=True)
class ShardRunSummary:
    planned: int
    successful: int
    historical_failures: int
    missing: int
    manifest_hash: str
    semantic_candidate_set_hash: str


def build_generation_shard_plan(
    plan: GenerationPlan,
    generator_id: str,
    shard_id: str,
) -> GenerationShardPlan:
    """Select one immutable generator/shard after validating the parent namespace."""
    _validate_generation_plan(plan)
    configs = {config.generator_id: config for config in plan.generator_configs}
    if generator_id not in configs:
        raise RunMismatchError("generator is not part of the generation plan")
    candidates = tuple(
        candidate
        for candidate in plan.candidates
        if candidate.generator_id == generator_id and candidate.shard_id == shard_id
    )
    if not candidates or f"{generator_id}/{shard_id}" not in plan.shard_ids:
        raise RunMismatchError("shard is not part of the generation plan")
    shard = GenerationShardPlan(
        schema_version=SHARD_PLAN_SCHEMA,
        generation_run_id=plan.generation_run_id,
        config_hash=plan.config_hash,
        generator_configs=plan.generator_configs,
        shard_size=plan.shard_size,
        generator_config=configs[generator_id],
        shard_id=shard_id,
        candidates=candidates,
        split_upstream_manifest_hash=plan.split_upstream_manifest_hash,
    )
    _validate_shard_plan(shard)
    return shard


def successful_candidate_keys(shard_dir) -> frozenset[tuple[str, str]]:
    """Read validated successful composite keys from one committed shard."""
    path = Path(shard_dir) / "candidates.jsonl"
    if not path.exists():
        return frozenset()
    records = _candidate_records(scan_jsonl(path).records)
    keys = [record.plan.key for record in records]
    if len(keys) != len(set(keys)):
        raise IntegrityError("duplicate successful composite candidate key")
    return frozenset(keys)


def semantic_candidate_set_hash(records: Iterable[CandidateRecord]) -> str:
    """Hash semantic successful-candidate content independently of execution history."""
    materialized = list(records)
    by_key: dict[tuple[str, str], CandidateRecord] = {}
    for record in materialized:
        if not isinstance(record, CandidateRecord):
            raise TypeError("records must contain CandidateRecord values")
        record.validate()
        if record.plan.key in by_key:
            raise IntegrityError("duplicate successful composite candidate key")
        by_key[record.plan.key] = record
    payloads = []
    for key in sorted(by_key):
        record = by_key[key]
        normalized = normalize_text_v1(record.response)
        payloads.append(
            {
                "candidate_id": record.plan.candidate_id,
                "finish_reason": record.finish_reason,
                "generated_token_count": record.generated_token_count,
                "generation_run_id": record.plan.generation_run_id,
                "model_revision": record.model_revision,
                "normalized_response_hash": sha256_hex(normalized.encode("utf-8")),
                "response_normalization_version": RESPONSE_NORMALIZATION_VERSION,
            }
        )
    return sha256_hex(canonical_json_bytes({"records": payloads, "schema": SEMANTIC_SET_SCHEMA}))


def run_generation_shard(
    plan: GenerationShardPlan,
    backend: GenerationBackend,
    output_dir,
    recover_stale_lock: bool = False,
) -> ShardRunSummary:
    """Execute or resume one shard while retaining every committed success/failure."""
    _validate_shard_plan(plan)
    if type(recover_stale_lock) is not bool:
        raise TypeError("recover_stale_lock must be a bool")
    output = Path(output_dir)
    validate_external_to_repository(_REPOSITORY_ROOT, output)

    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".lock"
    with ShardLock.acquire(
        lock_path,
        {"run_id": plan.generation_run_id, "shard_id": plan.shard_id},
        recover_stale=recover_stale_lock,
    ):
        candidates_path = output / "candidates.jsonl"
        failures_path = output / "failures.jsonl"
        manifest_path = output / "manifest.json"
        existing_manifest = _validate_existing_manifest(manifest_path, plan, candidates_path, failures_path)
        _ensure_file(candidates_path)
        _ensure_file(failures_path)

        candidate_scan = scan_jsonl(candidates_path, recover_incomplete_tail=True)
        failure_scan = scan_jsonl(failures_path, recover_incomplete_tail=True)
        if existing_manifest is not None:
            _validate_manifest_confirmed_prefix(
                existing_manifest,
                candidates_path,
                failures_path,
            )
        candidates = _candidate_records(candidate_scan.records)
        failures = _failure_records(failure_scan.records)
        _validate_history(plan, candidates, failures)

        successful = {record.plan.key: record for record in candidates}
        failure_counts: dict[tuple[str, str], int] = {}
        for record in failures:
            failure_counts[record.plan.key] = failure_counts.get(record.plan.key, 0) + 1

        manifest_hash = _write_manifest(output, plan, candidates, failures)
        missing = [candidate for candidate in plan.candidates if candidate.key not in successful]
        for start in range(0, len(missing), plan.generator_config.batch_size):
            batch = tuple(missing[start : start + plan.generator_config.batch_size])
            try:
                results = backend.generate(batch, plan.generator_config)
            except GenerationConfigurationError:
                raise
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as error:
                error_type = sanitize_exception(error)
                for candidate in batch:
                    failure = FailureRecord(
                        plan=candidate,
                        error_type=error_type,
                        retry_count=failure_counts.get(candidate.key, 0),
                    )
                    append_record(failures_path, failure.to_dict())
                    failures.append(failure)
                    failure_counts[candidate.key] = failure_counts.get(candidate.key, 0) + 1
                    manifest_hash = _write_manifest(output, plan, candidates, failures)
                continue

            ordered = _validated_results(batch, results)
            for result in ordered:
                if result.error_type is None:
                    record = CandidateRecord(
                        plan=result.plan,
                        response=result.response,
                        finish_reason=result.finish_reason,
                        generated_token_count=result.generated_token_count,
                        model_revision=plan.generator_config.model_revision,
                    )
                    append_record(candidates_path, record.to_dict())
                    candidates.append(record)
                    successful[record.plan.key] = record
                else:
                    failure = FailureRecord(
                        plan=result.plan,
                        error_type=result.error_type,
                        retry_count=failure_counts.get(result.plan.key, 0),
                    )
                    append_record(failures_path, failure.to_dict())
                    failures.append(failure)
                    failure_counts[result.plan.key] = failure_counts.get(result.plan.key, 0) + 1
                manifest_hash = _write_manifest(output, plan, candidates, failures)

        manifest_hash = _write_manifest(output, plan, candidates, failures)
        return _summary(plan, candidates, failures, manifest_hash)


def _validate_generation_plan(plan: GenerationPlan) -> None:
    if not isinstance(plan, GenerationPlan):
        raise TypeError("plan must be a GenerationPlan")
    if type(plan.shard_size) is not int or plan.shard_size <= 0:
        raise RunMismatchError("generation plan has an invalid shard size")
    if not plan.generator_configs:
        raise RunMismatchError("generation plan has no complete generator configs")
    samples = plan.generator_configs[0].samples_per_question
    try:
        typed = generation_stage_config(
            plan.generator_configs,
            samples_per_question=samples,
            shard_size=plan.shard_size,
            split_upstream_manifest_hash=plan.split_upstream_manifest_hash,
        )
    except (TypeError, ValueError) as error:
        raise RunMismatchError("generation plan config is invalid") from error
    if run_id("generation", typed) != plan.generation_run_id:
        raise RunMismatchError("generation run namespace does not match complete config")
    if sha256_hex(canonical_json_bytes(typed)) != plan.config_hash:
        raise RunMismatchError("generation config hash does not match complete config")
    expected_shards = {f"{candidate.generator_id}/{candidate.shard_id}" for candidate in plan.candidates}
    if tuple(sorted(expected_shards)) != tuple(sorted(plan.shard_ids)):
        raise RunMismatchError("generation shard identities do not match candidates")
    configs = {config.generator_id: config for config in plan.generator_configs}
    if len(configs) != len(plan.generator_configs):
        raise RunMismatchError("generation configs contain duplicate generators")
    keys: set[tuple[str, str]] = set()
    for candidate in plan.candidates:
        config = configs.get(candidate.generator_id)
        _validate_candidate(candidate, config, plan.generation_run_id)
        if candidate.key in keys:
            raise RunMismatchError("generation plan contains a duplicate composite key")
        keys.add(candidate.key)


def _validate_shard_plan(plan: GenerationShardPlan) -> None:
    if not isinstance(plan, GenerationShardPlan):
        raise TypeError("plan must be a GenerationShardPlan")
    if plan.schema_version != SHARD_PLAN_SCHEMA:
        raise RunMismatchError("unknown shard plan schema")
    parent = GenerationPlan(
        generation_run_id=plan.generation_run_id,
        config_hash=plan.config_hash,
        candidates=plan.candidates,
        shard_ids=(f"{plan.generator_config.generator_id}/{plan.shard_id}",),
        generator_configs=plan.generator_configs,
        shard_size=plan.shard_size,
        split_upstream_manifest_hash=plan.split_upstream_manifest_hash,
    )
    # Validate the run/config namespace against the complete recipe. Candidate
    # coverage is shard-local, so validate its identity separately below.
    samples = plan.generator_configs[0].samples_per_question if plan.generator_configs else -1
    try:
        typed = generation_stage_config(
            plan.generator_configs,
            samples_per_question=samples,
            shard_size=plan.shard_size,
            split_upstream_manifest_hash=plan.split_upstream_manifest_hash,
        )
    except (TypeError, ValueError) as error:
        raise RunMismatchError("shard plan config is invalid") from error
    if run_id("generation", typed) != parent.generation_run_id or sha256_hex(canonical_json_bytes(typed)) != parent.config_hash:
        raise RunMismatchError("shard run namespace does not match complete config")
    matching = [config for config in plan.generator_configs if config.generator_id == plan.generator_config.generator_id]
    if matching != [plan.generator_config]:
        raise RunMismatchError("selected generator config is not part of the complete recipe")
    if not plan.candidates:
        raise RunMismatchError("shard plan has no candidates")
    if tuple(plan.candidates) != tuple(
        sorted(plan.candidates, key=lambda item: (item.original_question_id, item.sampling_index))
    ):
        raise RunMismatchError("shard candidates are not in deterministic order")
    keys: set[tuple[str, str]] = set()
    for candidate in plan.candidates:
        if candidate.shard_id != plan.shard_id:
            raise RunMismatchError("candidate shard identity does not match shard plan")
        _validate_candidate(candidate, plan.generator_config, plan.generation_run_id)
        if candidate.key in keys:
            raise RunMismatchError("shard plan contains a duplicate composite key")
        keys.add(candidate.key)


def _validate_candidate(
    candidate: CandidatePlan,
    config: GenerationConfig | None,
    generation_run_id: str,
) -> None:
    try:
        candidate.validate()
    except (TypeError, ValueError) as error:
        raise RunMismatchError("candidate plan is invalid") from error
    if config is None or candidate.generator_id != config.generator_id:
        raise RunMismatchError("candidate generator does not match config")
    if candidate.generation_run_id != generation_run_id:
        raise RunMismatchError("candidate generation run namespace does not match")
    if candidate.split_manifest_hash != config.split_manifest_hash:
        raise RunMismatchError("candidate split manifest does not match config")
    if candidate.prompt_hash != config.prompt_template_hash:
        raise RunMismatchError("candidate prompt hash does not match config")
    if not 0 <= candidate.sampling_index < config.samples_per_question:
        raise RunMismatchError("candidate sampling index is outside the configured range")
    expected = logical_candidate_id(
        candidate.original_question_id,
        candidate.generator_id,
        candidate.sampling_index,
    )
    if candidate.candidate_id != expected:
        raise RunMismatchError("candidate logical identity does not match its fields")


def _validate_existing_manifest(
    manifest_path: Path,
    plan: GenerationShardPlan,
    candidates_path: Path,
    failures_path: Path,
) -> dict[str, object] | None:
    if not manifest_path.exists():
        if any(path.exists() and path.stat().st_size for path in (candidates_path, failures_path)):
            raise RunMismatchError("generation manifest is missing for existing shard records")
        return None
    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        if not isinstance(manifest, dict):
            raise ValueError("not an object")
        stored_hash = manifest["output_manifest_hash"]
        computed_hash = sha256_hex(
            canonical_json_bytes({key: value for key, value in manifest.items() if key not in _ENVELOPE_FIELDS})
        )
        if stored_hash != computed_hash:
            raise ValueError("hash mismatch")
    except (KeyError, TypeError, UnicodeDecodeError, ValueError) as error:
        raise RunMismatchError("generation manifest is malformed or has invalid integrity") from error
    expected = _manifest_identity(plan)
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise RunMismatchError("generation manifest does not match the complete shard config")
    return manifest


def _validate_manifest_confirmed_prefix(
    manifest: Mapping[str, object],
    candidates_path: Path,
    failures_path: Path,
) -> None:
    """Require each manifest-confirmed JSONL prefix to remain byte-identical."""
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise IntegrityError("manifest-confirmed prefix metadata is malformed")
    expected_paths = {
        "candidates.jsonl": candidates_path,
        "failures.jsonl": failures_path,
    }
    by_path: dict[str, Mapping[str, object]] = {}
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise IntegrityError("manifest-confirmed prefix metadata is malformed")
        relative_path = artifact.get("relative_path")
        if relative_path not in expected_paths or relative_path in by_path:
            raise IntegrityError("manifest-confirmed prefix metadata is malformed")
        by_path[relative_path] = artifact
    if set(by_path) != set(expected_paths):
        raise IntegrityError("manifest-confirmed prefix metadata is malformed")

    for relative_path, path in expected_paths.items():
        artifact = by_path[relative_path]
        record_count = artifact.get("record_count")
        stored_sha256 = artifact.get("sha256")
        if (
            type(record_count) is not int
            or record_count < 0
            or not isinstance(stored_sha256, str)
            or not _HASH.fullmatch(stored_sha256)
        ):
            raise IntegrityError("manifest-confirmed prefix metadata is malformed")
        data = path.read_bytes()
        lines = [raw + b"\n" for raw in data.split(b"\n")[:-1]]
        if len(lines) < record_count:
            raise IntegrityError("manifest-confirmed prefix was truncated")
        prefix = b"".join(lines[:record_count])
        if sha256_hex(prefix) != stored_sha256:
            raise IntegrityError("manifest-confirmed prefix was modified")


def _validate_history(
    plan: GenerationShardPlan,
    candidates: list[CandidateRecord],
    failures: list[FailureRecord],
) -> None:
    planned = {candidate.key: candidate for candidate in plan.candidates}
    seen: set[tuple[str, str]] = set()
    for record in candidates:
        expected = planned.get(record.plan.key)
        if expected is None or record.plan != expected or record.model_revision != plan.generator_config.model_revision:
            raise IntegrityError("stored candidate does not match the immutable shard plan")
        if record.plan.key in seen:
            raise IntegrityError("duplicate successful composite candidate key")
        seen.add(record.plan.key)
    retry_counts: dict[tuple[str, str], int] = {}
    for record in failures:
        expected = planned.get(record.plan.key)
        if expected is None or record.plan != expected:
            raise IntegrityError("stored failure does not match the immutable shard plan")
        expected_retry = retry_counts.get(record.plan.key, 0)
        if record.retry_count != expected_retry:
            raise IntegrityError("stored failure retry history is not contiguous")
        retry_counts[record.plan.key] = expected_retry + 1


def _validated_results(
    batch: tuple[CandidatePlan, ...],
    results,
) -> tuple[GenerationResult, ...]:
    if not isinstance(results, (list, tuple)):
        raise RunMismatchError("backend results must be a concrete sequence")
    by_key: dict[tuple[str, str], GenerationResult] = {}
    for result in results:
        if not isinstance(result, GenerationResult):
            raise RunMismatchError("backend returned a non-GenerationResult value")
        if result.error_type is not None and result.error_type not in _ERROR_TYPES:
            raise RunMismatchError("backend returned an unsanitized error type")
        if result.plan.key in by_key:
            raise RunMismatchError("backend returned a duplicate composite key")
        by_key[result.plan.key] = result
    expected = {candidate.key: candidate for candidate in batch}
    if set(by_key) != set(expected):
        raise RunMismatchError("backend result identities do not match the requested batch")
    if any(by_key[key].plan != candidate for key, candidate in expected.items()):
        raise RunMismatchError("backend result plans do not match the requested batch")
    return tuple(by_key[candidate.key] for candidate in batch)


def _candidate_records(rows: Iterable[Mapping[str, object]]) -> list[CandidateRecord]:
    return [
        CandidateRecord.from_dict({key: value for key, value in row.items() if key != "record_hash"})
        for row in rows
    ]


def _failure_records(rows: Iterable[Mapping[str, object]]) -> list[FailureRecord]:
    return [
        FailureRecord.from_dict({key: value for key, value in row.items() if key != "record_hash"})
        for row in rows
    ]


def _ensure_file(path: Path) -> None:
    if path.exists():
        return
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0), 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.flush()
        os.fsync(stream.fileno())


def _planned_keys_hash(plan: GenerationShardPlan) -> str:
    return sha256_hex(canonical_json_bytes([list(candidate.key) for candidate in plan.candidates]))


def _manifest_identity(plan: GenerationShardPlan) -> dict[str, object]:
    config = plan.generator_config
    return {
        "schema_version": SHARD_MANIFEST_SCHEMA,
        "stage": "generation",
        "generation_run_id": plan.generation_run_id,
        "config_hash": plan.config_hash,
        "generator_id": config.generator_id,
        "shard_id": plan.shard_id,
        "shard_size": plan.shard_size,
        "generation_config": config.to_dict(),
        "generation_configs": [item.to_dict() for item in plan.generator_configs],
        "model_repository": config.model_repository,
        "model_revision": config.model_revision,
        "backend": config.backend,
        "prompt_template_revision": config.prompt_template_revision,
        "prompt_template_hash": config.prompt_template_hash,
        "split_manifest_hash": config.split_manifest_hash,
        "upstream_bindings": {"split": plan.split_upstream_manifest_hash},
        "upstream_manifest_hashes": [plan.split_upstream_manifest_hash],
        "planned_candidate_keys_hash": _planned_keys_hash(plan),
    }


def _write_manifest(
    output: Path,
    plan: GenerationShardPlan,
    candidates: list[CandidateRecord],
    failures: list[FailureRecord],
) -> str:
    candidates_scan = scan_jsonl(output / "candidates.jsonl")
    failures_scan = scan_jsonl(output / "failures.jsonl")
    successful = len(candidates)
    payload = {
        **_manifest_identity(plan),
        "artifacts": [
            {
                "record_count": len(candidates_scan.records),
                "relative_path": "candidates.jsonl",
                "sha256": candidates_scan.file_sha256,
            },
            {
                "record_count": len(failures_scan.records),
                "relative_path": "failures.jsonl",
                "sha256": failures_scan.file_sha256,
            },
        ],
        "counts": {
            "historical_failures": len(failures),
            "missing": len(plan.candidates) - successful,
            "planned": len(plan.candidates),
            "successful": successful,
        },
        "semantic_candidate_set_hash": semantic_candidate_set_hash(candidates),
    }
    return write_atomic_manifest(output / "manifest.json", payload)


def _summary(
    plan: GenerationShardPlan,
    candidates: list[CandidateRecord],
    failures: list[FailureRecord],
    manifest_hash: str,
) -> ShardRunSummary:
    return ShardRunSummary(
        planned=len(plan.candidates),
        successful=len(candidates),
        historical_failures=len(failures),
        missing=len(plan.candidates) - len(candidates),
        manifest_hash=manifest_hash,
        semantic_candidate_set_hash=semantic_candidate_set_hash(candidates),
    )


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate object key")
        value[key] = item
    return value


def _reject_constant(value):
    raise ValueError(f"non-standard JSON constant {value}")
