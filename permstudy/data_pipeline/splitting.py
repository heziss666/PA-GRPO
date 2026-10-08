"""Deterministic question-level split, smoke selection, and lineage checks."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from types import MappingProxyType

from .canonical import canonical_json_bytes, sha256_hex
from .io import IntegrityError, scan_jsonl, verify_artifact_ref
from .lineage import role_bound_stage_config
from .schema import ArtifactRef, QuestionRecord, Source, Split, SplitAssignment
from .sources import SourceSnapshot
from .sources.math import NORMALIZATION_VERSION as MATH_NORMALIZATION_VERSION
from .sources.reclor import NORMALIZATION_VERSION as RECLOR_NORMALIZATION_VERSION


SPLIT_SCHEMA = "split_manifest_v1"
SPLIT_ALGORITHM = "deterministic_stratified_question_split_v1"
QUESTION_NORMALIZATION_VERSIONS = MappingProxyType(
    {
        Source.MATH.value: MATH_NORMALIZATION_VERSION,
        Source.RECLOR.value: RECLOR_NORMALIZATION_VERSION,
    }
)

_HASH = re.compile(r"[0-9a-f]{64}\Z")
_SOURCE_BY_ROLE = {
    "math_questions": Source.MATH,
    "reclor_questions": Source.RECLOR,
}
_MANIFEST_ENVELOPE_FIELDS = frozenset({"created_at_utc", "output_manifest_hash"})


@dataclass(frozen=True)
class SplitBuildResult:
    assignments: tuple[SplitAssignment, ...]
    stratification_level_by_source: Mapping[str, str]
    fallback_reasons: Mapping[str, tuple[str, ...]]
    split_manifest_hash: str

    def __post_init__(self):
        object.__setattr__(
            self,
            "stratification_level_by_source",
            MappingProxyType(dict(self.stratification_level_by_source)),
        )
        object.__setattr__(
            self,
            "fallback_reasons",
            MappingProxyType({key: tuple(value) for key, value in self.fallback_reasons.items()}),
        )


def split_stage_config(split_seed: int, upstream_bindings: Mapping[str, str]) -> dict[str, object]:
    """Bind the unified split run to its named MATH and ReClor manifests."""
    if type(split_seed) is not int or split_seed < 0:
        raise ValueError("split_seed must be a nonnegative integer")
    return role_bound_stage_config(
        {"split_seed": split_seed},
        upstream_bindings,
        {"math_questions", "reclor_questions"},
    )


def build_bound_internal_split(
    source_snapshots: Mapping[str, SourceSnapshot],
    upstream_bindings: Mapping[str, str],
    data_root,
    split_seed: int = 42,
) -> SplitBuildResult:
    """Verify named source manifests and artifacts, then build the fixed split."""
    split_stage_config(split_seed, upstream_bindings)
    if not isinstance(source_snapshots, Mapping) or set(source_snapshots) != set(_SOURCE_BY_ROLE):
        raise ValueError("source manifest lineage mismatch")

    questions: list[QuestionRecord] = []
    for role in sorted(_SOURCE_BY_ROLE):
        snapshot = source_snapshots[role]
        if not isinstance(snapshot, SourceSnapshot) or snapshot.source is not _SOURCE_BY_ROLE[role]:
            raise ValueError("source manifest lineage mismatch")
        snapshot.validate()
        manifest = snapshot.private_manifest
        try:
            recorded_hash = manifest["output_manifest_hash"]
            computed_hash = sha256_hex(
                canonical_json_bytes(
                    {
                        key: value
                        for key, value in manifest.items()
                        if key not in _MANIFEST_ENVELOPE_FIELDS
                    }
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("source manifest lineage mismatch") from error
        if recorded_hash != computed_hash or recorded_hash != upstream_bindings[role]:
            raise ValueError("source manifest lineage mismatch")
        if (
            manifest.get("stage") != "sources"
            or manifest.get("source") != snapshot.source.value
            or manifest.get("source_revision") != snapshot.source_revision
            or manifest.get("source_snapshot_id") != snapshot.source_snapshot_id
        ):
            raise ValueError("source manifest lineage mismatch")
        if manifest.get("question_normalization_version") != QUESTION_NORMALIZATION_VERSIONS[snapshot.source.value]:
            raise ValueError("source normalization contract mismatch")

        artifacts = manifest.get("artifacts")
        counts = manifest.get("counts")
        if not isinstance(artifacts, list) or len(artifacts) != 1 or not isinstance(counts, Mapping):
            raise ValueError("source manifest lineage mismatch")
        try:
            artifact = ArtifactRef.from_dict(artifacts[0])
            verify_artifact_ref(data_root, artifact)
            scan = scan_jsonl(Path(data_root) / artifact.relative_path)
        except (IntegrityError, TypeError, ValueError, OSError) as error:
            raise ValueError("source artifact integrity mismatch") from error
        if artifact.record_count != len(snapshot.questions) or counts.get("questions") != len(snapshot.questions):
            raise ValueError("source artifact integrity mismatch")
        stored = sorted(
            ({key: value for key, value in record.items() if key != "record_hash"} for record in scan.records),
            key=lambda record: record["original_question_id"],
        )
        expected = sorted(
            (question.to_dict() for question in snapshot.questions),
            key=lambda record: record["original_question_id"],
        )
        if stored != expected:
            raise ValueError("source artifact integrity mismatch")
        questions.extend(snapshot.questions)
    return build_internal_split(questions, split_seed=split_seed)


def _score(seed: int, question_id: str, *, smoke: bool = False) -> str:
    marker = "\0smoke\0" if smoke else "\0"
    return hashlib.sha256(f"{seed}{marker}{question_id}".encode("utf-8")).hexdigest()


def _canonical_split_manifest_bytes(
    assignments: Sequence[SplitAssignment],
    provenance_by_source: Mapping[Source, tuple[str, str]],
    levels: Mapping[str, str],
    reasons: Mapping[str, tuple[str, ...]],
    split_seed: int,
) -> bytes:
    metadata = {
        "fallback_reasons": dict(reasons),
        "question_normalization_versions": dict(QUESTION_NORMALIZATION_VERSIONS),
        "record_type": "split_metadata",
        "schema_version": SPLIT_SCHEMA,
        "source_provenance": {
            source.value: {
                "source_revision": provenance_by_source[source][0],
                "source_snapshot_id": provenance_by_source[source][1],
            }
            for source in sorted(Source, key=lambda item: item.value)
        },
        "split_algorithm": SPLIT_ALGORITHM,
        "split_seed": split_seed,
        "stratification_level_by_source": dict(levels),
    }
    rows = (
        {"record_type": "split_assignment", **assignment.to_dict()}
        for assignment in sorted(assignments, key=lambda item: item.original_question_id)
    )
    return canonical_json_bytes(metadata) + b"\n" + b"".join(
        canonical_json_bytes(row) + b"\n" for row in rows
    )


def split_manifest_bytes(
    questions: Iterable[QuestionRecord],
    result: SplitBuildResult,
    split_seed: int = 42,
) -> bytes:
    """Render the exact canonical JSONL bytes whose digest is the split identity."""
    if not isinstance(result, SplitBuildResult):
        raise TypeError("result must be a SplitBuildResult")
    if type(split_seed) is not int or split_seed < 0:
        raise ValueError("split_seed must be a nonnegative integer")

    records = list(questions)
    questions_by_id: dict[str, QuestionRecord] = {}
    provenance_by_source: dict[Source, tuple[str, str]] = {}
    for question in records:
        if not isinstance(question, QuestionRecord):
            raise TypeError("questions must contain QuestionRecord values")
        question.validate()
        if question.original_question_id in questions_by_id:
            raise ValueError("duplicate original_question_id in split manifest input")
        provenance = (question.source_revision, question.source_snapshot_id)
        previous = provenance_by_source.setdefault(question.source, provenance)
        if previous != provenance:
            raise ValueError("source provenance mismatch in split manifest input")
        questions_by_id[question.original_question_id] = question
    if set(provenance_by_source) != set(Source):
        raise ValueError("split manifest input must contain both MATH and ReClor")

    assignment_ids: set[str] = set()
    for assignment in result.assignments:
        assignment.validate()
        question = questions_by_id.get(assignment.original_question_id)
        if (
            question is None
            or assignment.original_question_id in assignment_ids
            or assignment.question_content_hash != question.question_content_hash
            or assignment.source is not question.source
        ):
            raise ValueError("split manifest assignments do not match their questions")
        assignment_ids.add(assignment.original_question_id)
    if assignment_ids != set(questions_by_id):
        raise ValueError("split manifest assignments do not match their questions")

    payload = _canonical_split_manifest_bytes(
        result.assignments,
        provenance_by_source,
        result.stratification_level_by_source,
        result.fallback_reasons,
        split_seed,
    )
    if sha256_hex(payload) != result.split_manifest_hash:
        raise ValueError("split manifest bytes do not match split_manifest_hash")
    return payload


def _holdout_count(total: int) -> int:
    if total < 2:
        raise ValueError("each source requires at least two questions")
    return min(total - 1, max(1, (total + 5) // 10))


def _groups(questions: Sequence[QuestionRecord], level: str) -> dict[str, list[QuestionRecord]]:
    grouped: dict[str, list[QuestionRecord]] = {}
    for question in questions:
        if question.source is Source.RECLOR:
            key = f"gold_label={question.gold_label}"
        elif level == "category+level":
            key = f"category={question.category}|level={question.level}"
        elif level == "category":
            key = f"category={question.category}"
        elif level == "source":
            key = "source=math"
        else:  # pragma: no cover - internal invariant
            raise ValueError("unknown stratification level")
        grouped.setdefault(key, []).append(question)
    return grouped


def _feasible(grouped: Mapping[str, Sequence[QuestionRecord]], holdout_count: int, total: int) -> bool:
    group_count = len(grouped)
    return (
        bool(grouped)
        and all(len(group) >= 2 for group in grouped.values())
        and group_count <= holdout_count <= total - group_count
    )


def _allocate_holdout(grouped: Mapping[str, Sequence[QuestionRecord]], count: int) -> dict[str, int]:
    quotas = {key: 1 for key in grouped}
    while sum(quotas.values()) < count:
        best_key = None
        for key in sorted(grouped):
            if quotas[key] >= len(grouped[key]) - 1:
                continue
            if best_key is None:
                best_key = key
                continue
            left = quotas[key] * len(grouped[best_key])
            right = quotas[best_key] * len(grouped[key])
            if left < right or (left == right and key < best_key):
                best_key = key
        if best_key is None:  # pragma: no cover - guarded by feasibility
            raise ValueError("holdout allocation has no remaining capacity")
        quotas[best_key] += 1
    return quotas


def _source_plan(questions: Sequence[QuestionRecord], seed: int):
    total = len(questions)
    holdout_count = _holdout_count(total)
    source = questions[0].source
    fallback: tuple[str, ...] = ()
    if source is Source.RECLOR:
        level = "gold_label"
        grouped = _groups(questions, level)
        if not _feasible(grouped, holdout_count, total):
            raise ValueError("ReClor gold_label stratification is infeasible")
    else:
        grouped = _groups(questions, "category+level")
        if _feasible(grouped, holdout_count, total):
            level = "category+level"
        else:
            fallback = ("category+level_infeasible",)
            grouped = _groups(questions, "category")
            if _feasible(grouped, holdout_count, total):
                level = "category"
            else:
                fallback += ("category_infeasible",)
                grouped = _groups(questions, "source")
                if not _feasible(grouped, holdout_count, total):
                    raise ValueError("MATH source stratification is infeasible")
                level = "source"

    quotas = _allocate_holdout(grouped, holdout_count)
    holdout_ids: set[str] = set()
    stratum_by_id: dict[str, str] = {}
    for key in sorted(grouped):
        ranked = sorted(
            grouped[key],
            key=lambda question: (_score(seed, question.original_question_id), question.original_question_id),
        )
        holdout_ids.update(question.original_question_id for question in ranked[: quotas[key]])
        stratum_by_id.update((question.original_question_id, key) for question in grouped[key])
    return level, fallback, holdout_ids, stratum_by_id


def build_internal_split(questions: Iterable[QuestionRecord], split_seed: int = 42) -> SplitBuildResult:
    """Build one stable 90/10 question-level split for MATH and ReClor."""
    if type(split_seed) is not int or split_seed < 0:
        raise ValueError("split_seed must be a nonnegative integer")
    records = list(questions)
    by_id: dict[str, QuestionRecord] = {}
    by_source: dict[Source, list[QuestionRecord]] = {source: [] for source in Source}
    provenance_by_source: dict[Source, tuple[str, str]] = {}
    for question in records:
        if not isinstance(question, QuestionRecord):
            raise TypeError("questions must contain QuestionRecord values")
        question.validate()
        if question.original_question_id in by_id:
            raise ValueError("duplicate original_question_id in split input")
        provenance = (question.source_revision, question.source_snapshot_id)
        previous = provenance_by_source.setdefault(question.source, provenance)
        if previous != provenance:
            raise ValueError("source provenance mismatch in split input")
        by_id[question.original_question_id] = question
        by_source[question.source].append(question)
    if any(not by_source[source] for source in Source):
        raise ValueError("split input must contain both MATH and ReClor")

    assignments: list[SplitAssignment] = []
    levels: dict[str, str] = {}
    reasons: dict[str, tuple[str, ...]] = {}
    for source in sorted(Source, key=lambda item: item.value):
        source_questions = sorted(by_source[source], key=lambda question: question.original_question_id)
        level, fallback, holdout_ids, strata = _source_plan(source_questions, split_seed)
        levels[source.value] = level
        if fallback:
            reasons[source.value] = fallback
        for question in source_questions:
            assignment = SplitAssignment(
                original_question_id=question.original_question_id,
                question_content_hash=question.question_content_hash,
                source=source,
                split=(
                    Split.INTERNAL_HOLDOUT
                    if question.original_question_id in holdout_ids
                    else Split.TRAIN
                ),
                stratum=strata[question.original_question_id],
            )
            assignment.validate()
            assignments.append(assignment)

    ordered = tuple(sorted(assignments, key=lambda assignment: assignment.original_question_id))
    manifest_bytes = _canonical_split_manifest_bytes(
        ordered,
        provenance_by_source,
        levels,
        reasons,
        split_seed,
    )
    return SplitBuildResult(
        assignments=ordered,
        stratification_level_by_source=levels,
        fallback_reasons=reasons,
        split_manifest_hash=sha256_hex(manifest_bytes),
    )


def select_smoke_questions(
    assignments: Iterable[SplitAssignment],
    per_source: int = 20,
    split_seed: int = 42,
) -> list[SplitAssignment]:
    """Select a proportional deterministic smoke subset from training only."""
    if type(per_source) is not int or per_source <= 0:
        raise ValueError("per_source must be a positive integer")
    if type(split_seed) is not int or split_seed < 0:
        raise ValueError("split_seed must be a nonnegative integer")

    seen: set[str] = set()
    train_by_source: dict[Source, list[SplitAssignment]] = {source: [] for source in Source}
    for assignment in assignments:
        if not isinstance(assignment, SplitAssignment):
            raise TypeError("assignments must contain SplitAssignment values")
        assignment.validate()
        if assignment.original_question_id in seen:
            raise ValueError("duplicate original_question_id in split assignments")
        seen.add(assignment.original_question_id)
        if assignment.split is Split.TRAIN:
            train_by_source[assignment.source].append(assignment)

    selected: list[SplitAssignment] = []
    for source in sorted(Source, key=lambda item: item.value):
        source_rows = train_by_source[source]
        if len(source_rows) < per_source:
            raise ValueError(f"{source.value} has fewer than {per_source} train questions")
        grouped: dict[str, list[SplitAssignment]] = {}
        for assignment in source_rows:
            grouped.setdefault(assignment.stratum, []).append(assignment)

        total = len(source_rows)
        quotas = {key: len(group) * per_source // total for key, group in grouped.items()}
        remaining = per_source - sum(quotas.values())
        candidates = sorted(
            (
                ((len(grouped[key]) * per_source) % total, key)
                for key in grouped
                if quotas[key] < len(grouped[key])
            ),
            key=lambda item: (-item[0], item[1]),
        )
        if remaining > len(candidates):  # pragma: no cover - floor allocation invariant
            raise ValueError("smoke allocation has insufficient stratum capacity")
        for _, key in candidates[:remaining]:
            quotas[key] += 1

        for key in sorted(grouped):
            ranked = sorted(
                grouped[key],
                key=lambda assignment: (
                    _score(split_seed, assignment.original_question_id, smoke=True),
                    assignment.original_question_id,
                ),
            )
            selected.extend(ranked[: quotas[key]])
    return sorted(selected, key=lambda assignment: (assignment.source.value, assignment.original_question_id))


def validate_split_inheritance(
    assignments: Iterable[SplitAssignment],
    records: Iterable[object],
    split_manifest_hash: str,
) -> None:
    """Reject candidate/pair/permutation records detached from the fixed split."""
    if not isinstance(split_manifest_hash, str) or not _HASH.fullmatch(split_manifest_hash):
        raise ValueError("split lineage mismatch")
    by_id: dict[str, SplitAssignment] = {}
    for assignment in assignments:
        assignment.validate()
        if assignment.original_question_id in by_id:
            raise ValueError("split lineage mismatch")
        by_id[assignment.original_question_id] = assignment

    for record in records:
        question_id = getattr(record, "original_question_id", None)
        assignment = by_id.get(question_id)
        if assignment is None:
            raise ValueError("split lineage mismatch")
        checks = (
            getattr(record, "split", None) == assignment.split,
            getattr(record, "split_manifest_hash", None) == split_manifest_hash,
        )
        if hasattr(record, "source"):
            checks += (record.source == assignment.source,)
        if hasattr(record, "question_content_hash"):
            checks += (record.question_content_hash == assignment.question_content_hash,)
        if not all(checks):
            raise ValueError("split lineage mismatch")
