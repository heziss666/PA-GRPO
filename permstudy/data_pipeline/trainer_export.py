"""Fail-closed export of immutable pairs and permutations to trainer rows."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import re
import tempfile
from types import MappingProxyType

from .canonical import canonical_json_bytes, normalize_text_v1, sha256_hex
from .ids import pair_id as make_pair_id
from .ids import reclor_content_hash, run_id
from .io import IntegrityError, scan_jsonl, verify_artifact_ref, write_atomic_manifest
from .isolation import validate_external_to_repository
from .lineage import role_bound_stage_config
from .schema import (
    ArtifactRef,
    CandidateRecord,
    PairRecord,
    PermutationRecord,
    QuestionRecord,
    RunManifest,
    Source,
)


DATA_SOURCE = "permstudy_pairwise_judge_v1"
PROMPT_TEMPLATE_VERSION = "pairwise_judge_direct_v1"
ROW_SCHEMA_VERSION = "trainer_pairwise_parquet_v1"
EXPORT_CONFIG_SCHEMA = "trainer_export_config_v1"

_EXPORT_UPSTREAM_ROLES = frozenset({"split", "generation", "pairs", "permutations"})
_ARTIFACT_BASENAMES = {
    "split": "questions.jsonl",
    "generation": "candidates.jsonl",
    "pairs": "pairs.jsonl",
    "permutations": "permutations.jsonl",
}
_RECORD_TYPES = {
    "split": QuestionRecord,
    "generation": CandidateRecord,
    "pairs": PairRecord,
    "permutations": PermutationRecord,
}
_ROW_FIELDS = frozenset(
    {"data_source", "prompt", "ability", "reward_model", "extra_info"}
)
_EXTRA_INFO_FIELDS = frozenset(
    {
        "pair_id",
        "original_question_id",
        "permutation_id",
        "permutation_label",
        "source",
        "split",
        "split_manifest_hash",
        "generation_run_id",
        "pair_run_id",
        "permutation_run_id",
    }
)
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_PROMPT_TEMPLATE_CONTRACT = {
    "system": "Reply with only A or B.",
    "user_sections": (
        "Question",
        "Response A",
        "Response B",
        "Which response is more correct?",
        "Answer with A or B only.",
    ),
    "version": PROMPT_TEMPLATE_VERSION,
}
PROMPT_TEMPLATE_HASH = sha256_hex(canonical_json_bytes(_PROMPT_TEMPLATE_CONTRACT))


class TrainerExportIntegrityError(ValueError):
    """Raised when an export join or immutable identity cannot be trusted."""


@dataclass(frozen=True)
class TrainerExportSummary:
    """Sanitized identity and artifact evidence for one trainer export."""

    export_run_id: str
    dataset_artifact: ArtifactRef
    row_count: int
    prompt_template_hash: str
    upstream_bindings: Mapping[str, str]
    output_manifest_hash: str

    def __post_init__(self):
        object.__setattr__(
            self, "upstream_bindings", MappingProxyType(dict(self.upstream_bindings))
        )


@dataclass(frozen=True)
class _BoundManifest:
    run_id: str
    output_manifest_hash: str
    artifacts: tuple[ArtifactRef, ...]


def export_stage_config(upstream_bindings: Mapping[str, str]) -> dict[str, object]:
    """Bind trainer export identity to all canonical upstream roles."""
    return role_bound_stage_config(
        {
            "export_config_schema": EXPORT_CONFIG_SCHEMA,
            "prompt_template_hash": PROMPT_TEMPLATE_HASH,
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            "row_schema_version": ROW_SCHEMA_VERSION,
        },
        upstream_bindings,
        _EXPORT_UPSTREAM_ROLES,
    )


def render_pairwise_judge_prompt(
    question: QuestionRecord,
    response_a: str,
    response_b: str,
) -> list[dict[str, str]]:
    """Render the exact direct A/B judge template over canonical text."""
    _validate_question_content(question)
    if not isinstance(response_a, str) or not isinstance(response_b, str):
        raise TypeError("responses must be strings")
    rendered_question = _render_question(question)
    return [
        {"role": "system", "content": "Reply with only A or B."},
        {
            "role": "user",
            "content": (
                f"Question:\n{rendered_question}\n\n"
                f"Response A:\n{normalize_text_v1(response_a)}\n\n"
                f"Response B:\n{normalize_text_v1(response_b)}\n\n"
                "Which response is more correct?\n"
                "Answer with A or B only."
            ),
        },
    ]


def build_trainer_rows(
    questions: Sequence[QuestionRecord],
    candidates: Sequence[CandidateRecord],
    pairs: Sequence[PairRecord],
    permutations: Sequence[PermutationRecord],
) -> list[dict[str, object]]:
    """Join canonical records and render validated trainer rows."""
    question_by_id = _index_questions(questions)
    candidate_by_key = _index_candidates(candidates)
    pair_by_id = _index_pairs(pairs)
    permutation_groups = _group_permutations(permutations)
    if set(permutation_groups) != set(pair_by_id):
        raise TrainerExportIntegrityError(
            "permutation and pair manifests do not contain identical pair identities"
        )

    rows: list[dict[str, object]] = []
    for pair in pair_by_id.values():
        question = question_by_id.get(pair.original_question_id)
        if question is None:
            raise TrainerExportIntegrityError(
                "pair question is missing from the split artifact"
            )
        _validate_pair_question(pair, question)
        selected = {
            pair.positive_candidate_id: _selected_candidate(
                candidate_by_key, pair, pair.positive_candidate_id
            ),
            pair.negative_candidate_id: _selected_candidate(
                candidate_by_key, pair, pair.negative_candidate_id
            ),
        }
        _validate_selected_candidate(
            pair, selected[pair.positive_candidate_id], positive=True
        )
        _validate_selected_candidate(
            pair, selected[pair.negative_candidate_id], positive=False
        )

        group = sorted(
            permutation_groups[pair.pair_id], key=lambda record: record.permutation_id
        )
        _validate_permutation_group(pair, group)
        for permutation in group:
            response_a = selected[permutation.surface_a_candidate_id].response
            response_b = selected[permutation.surface_b_candidate_id].response
            rows.append(
                {
                    "data_source": DATA_SOURCE,
                    "prompt": render_pairwise_judge_prompt(
                        question, response_a, response_b
                    ),
                    "ability": "math"
                    if question.source is Source.MATH
                    else "logical_reasoning",
                    "reward_model": {
                        "ground_truth": permutation.correct_surface,
                        "style": "rule",
                    },
                    "extra_info": {
                        "pair_id": pair.pair_id,
                        "original_question_id": pair.original_question_id,
                        "permutation_id": permutation.permutation_id,
                        "permutation_label": permutation.permutation_label,
                        "source": pair.source.value,
                        "split": pair.split.value,
                        "split_manifest_hash": pair.split_manifest_hash,
                        "generation_run_id": pair.generation_run_id,
                        "pair_run_id": pair.pair_run_id,
                        "permutation_run_id": permutation.permutation_run_id,
                    },
                }
            )
    rows.sort(
        key=lambda row: (
            row["extra_info"]["original_question_id"],
            row["extra_info"]["permutation_id"],
        )
    )
    validate_trainer_rows(rows)
    return rows


def validate_trainer_rows(rows: Sequence[Mapping[str, object]]) -> None:
    """Enforce the formal trainer row contract before any Parquet write."""
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError("trainer rows must be a sequence")
    identities: set[tuple[str, int]] = set()
    by_pair: dict[str, list[Mapping[str, object]]] = {}
    previous: tuple[str, int] | None = None
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != _ROW_FIELDS:
            raise TrainerExportIntegrityError(
                "trainer row fields do not match the row schema"
            )
        if row["data_source"] != DATA_SOURCE or row["ability"] not in {
            "math",
            "logical_reasoning",
        }:
            raise TrainerExportIntegrityError(
                "trainer row source or ability is invalid"
            )
        prompt = row["prompt"]
        if (
            not isinstance(prompt, list)
            or len(prompt) != 2
            or prompt[0] != {"role": "system", "content": "Reply with only A or B."}
            or not isinstance(prompt[1], dict)
            or set(prompt[1]) != {"role", "content"}
            or prompt[1]["role"] != "user"
            or not isinstance(prompt[1]["content"], str)
        ):
            raise TrainerExportIntegrityError(
                "trainer prompt does not match the template contract"
            )
        reward = row["reward_model"]
        if not isinstance(reward, Mapping) or dict(reward) not in (
            {"ground_truth": "A", "style": "rule"},
            {"ground_truth": "B", "style": "rule"},
        ):
            raise TrainerExportIntegrityError("trainer reward_model is invalid")
        extra = row["extra_info"]
        if not isinstance(extra, Mapping) or set(extra) != _EXTRA_INFO_FIELDS:
            if not isinstance(extra, Mapping) or "pair_id" not in extra:
                raise TrainerExportIntegrityError("trainer extra_info requires pair_id")
            raise TrainerExportIntegrityError(
                "trainer extra_info fields do not match the row schema"
            )
        pair_id = extra["pair_id"]
        permutation_id = extra["permutation_id"]
        if not isinstance(pair_id, str) or not _HASH.fullmatch(pair_id):
            raise TrainerExportIntegrityError("trainer extra_info pair_id is invalid")
        if type(permutation_id) is not int or permutation_id not in (0, 1):
            raise TrainerExportIntegrityError(
                "trainer permutation_id must be integer 0 or 1"
            )
        expected_label = "AB" if permutation_id == 0 else "BA"
        expected_surface = "A" if permutation_id == 0 else "B"
        if (
            extra["permutation_label"] != expected_label
            or reward["ground_truth"] != expected_surface
        ):
            raise TrainerExportIntegrityError(
                "trainer permutation surface contract is invalid"
            )
        for hash_field in (
            "split_manifest_hash",
            "generation_run_id",
            "pair_run_id",
            "permutation_run_id",
        ):
            if not isinstance(extra[hash_field], str) or not _HASH.fullmatch(
                extra[hash_field]
            ):
                raise TrainerExportIntegrityError(f"trainer {hash_field} is invalid")
        identity = (pair_id, permutation_id)
        if identity in identities:
            raise TrainerExportIntegrityError(
                "trainer rows contain duplicate permutation identity"
            )
        identities.add(identity)
        by_pair.setdefault(pair_id, []).append(row)
        order = (extra["original_question_id"], permutation_id)
        if not isinstance(order[0], str) or not order[0]:
            raise TrainerExportIntegrityError("trainer original_question_id is invalid")
        if previous is not None and order < previous:
            raise TrainerExportIntegrityError("trainer rows are not in canonical order")
        previous = order
    for group in by_pair.values():
        if len(group) != 2 or {
            row["extra_info"]["permutation_id"] for row in group
        } != {0, 1}:
            raise TrainerExportIntegrityError(
                "trainer pair must contain exactly AB and BA rows"
            )
        ab, ba = sorted(group, key=lambda row: row["extra_info"]["permutation_id"])
        shared = _EXTRA_INFO_FIELDS - {"permutation_id", "permutation_label"}
        if any(ab["extra_info"][field] != ba["extra_info"][field] for field in shared):
            raise TrainerExportIntegrityError("trainer AB and BA lineage differs")


def export_trainer_parquet(
    upstream_manifests: Mapping[str, RunManifest | Mapping[str, object]],
    output_dir: Path,
) -> TrainerExportSummary:
    """Resolve hash-pinned canonical artifacts and atomically export Parquet."""
    root = Path(output_dir).resolve()
    validate_external_to_repository(_REPOSITORY_ROOT, root)
    manifests = _validate_upstream_manifests(upstream_manifests)
    records = {
        role: _read_role_records(root, role, manifest)
        for role, manifest in manifests.items()
    }
    _validate_record_run_lineage(records, manifests)
    rows = build_trainer_rows(
        records["split"],
        records["generation"],
        records["pairs"],
        records["permutations"],
    )
    bindings = {
        role: manifests[role].output_manifest_hash for role in sorted(manifests)
    }
    typed_config = export_stage_config(bindings)
    export_identity = run_id("trainer_export", typed_config)
    export_directory = root / "trainer_exports" / export_identity
    export_directory.mkdir(parents=True, exist_ok=True)
    parquet_path = export_directory / "train.parquet"
    _write_parquet_atomically(parquet_path, rows)
    dataset_hash = sha256_hex(parquet_path.read_bytes())
    relative_path = parquet_path.relative_to(root).as_posix()
    artifact = ArtifactRef(relative_path, dataset_hash, len(rows))
    artifact.validate()
    verify_artifact_ref(root, artifact)

    payload = {
        "schema_version": "run_manifest_v1",
        "stage": "trainer_export",
        "run_id": export_identity,
        "config_hash": sha256_hex(canonical_json_bytes(typed_config)),
        "upstream_manifest_hashes": [bindings[role] for role in sorted(bindings)],
        "artifacts": [artifact.to_dict()],
        "counts": {"rows": len(rows)},
        "upstream_bindings": bindings,
        "prompt_template_hash": PROMPT_TEMPLATE_HASH,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "row_schema_version": ROW_SCHEMA_VERSION,
        "dataset_sha256": dataset_hash,
    }
    manifest_path = export_directory / "manifest.json"
    output_manifest_hash = write_atomic_manifest(manifest_path, payload)
    return TrainerExportSummary(
        export_run_id=export_identity,
        dataset_artifact=artifact,
        row_count=len(rows),
        prompt_template_hash=PROMPT_TEMPLATE_HASH,
        upstream_bindings=bindings,
        output_manifest_hash=output_manifest_hash,
    )


def _index_questions(records: Sequence[QuestionRecord]) -> dict[str, QuestionRecord]:
    indexed: dict[str, QuestionRecord] = {}
    for record in records:
        if not isinstance(record, QuestionRecord):
            raise TypeError("questions must contain QuestionRecord values")
        _validate_question_content(record)
        if record.original_question_id in indexed:
            raise TrainerExportIntegrityError(
                "split artifact contains duplicate question identity"
            )
        indexed[record.original_question_id] = record
    return indexed


def _index_candidates(
    records: Sequence[CandidateRecord],
) -> dict[tuple[str, str], CandidateRecord]:
    indexed: dict[tuple[str, str], CandidateRecord] = {}
    for record in records:
        if not isinstance(record, CandidateRecord):
            raise TypeError("candidates must contain CandidateRecord values")
        record.validate()
        if record.plan.key in indexed:
            raise TrainerExportIntegrityError(
                "generation artifact contains duplicate composite candidate identity"
            )
        indexed[record.plan.key] = record
    return indexed


def _index_pairs(records: Sequence[PairRecord]) -> dict[str, PairRecord]:
    indexed: dict[str, PairRecord] = {}
    question_ids: set[str] = set()
    for record in records:
        if not isinstance(record, PairRecord):
            raise TypeError("pairs must contain PairRecord values")
        record.validate()
        expected = make_pair_id(
            record.generation_run_id,
            record.original_question_id,
            record.positive_candidate_id,
            record.negative_candidate_id,
            record.schema_version,
        )
        if record.pair_id != expected:
            raise TrainerExportIntegrityError(
                "pair identity does not match selected candidates"
            )
        if record.pair_id in indexed or record.original_question_id in question_ids:
            raise TrainerExportIntegrityError(
                "pair artifact contains duplicate identity"
            )
        indexed[record.pair_id] = record
        question_ids.add(record.original_question_id)
    return indexed


def _group_permutations(
    records: Sequence[PermutationRecord],
) -> dict[str, list[PermutationRecord]]:
    grouped: dict[str, list[PermutationRecord]] = {}
    for record in records:
        if not isinstance(record, PermutationRecord):
            raise TypeError("permutations must contain PermutationRecord values")
        record.validate()
        grouped.setdefault(record.pair_id, []).append(record)
    return grouped


def _validate_question_content(question: QuestionRecord) -> None:
    if not isinstance(question, QuestionRecord):
        raise TypeError("question must be a QuestionRecord")
    question.validate()
    if question.source is Source.MATH:
        expected = sha256_hex(normalize_text_v1(question.problem).encode("utf-8"))
    else:
        expected = reclor_content_hash(
            question.context, question.question, question.answers
        )
    if question.question_content_hash != expected:
        raise TrainerExportIntegrityError(
            "question content hash does not match canonical question text"
        )


def _render_question(question: QuestionRecord) -> str:
    if question.source is Source.MATH:
        return normalize_text_v1(question.problem)
    answers = "\n".join(
        f"{label}. {normalize_text_v1(answer)}"
        for label, answer in zip("ABCD", question.answers, strict=True)
    )
    return (
        f"Context:\n{normalize_text_v1(question.context)}\n\n"
        f"Question:\n{normalize_text_v1(question.question)}\n\n"
        f"{answers}"
    )


def _validate_pair_question(pair: PairRecord, question: QuestionRecord) -> None:
    if (
        pair.question_content_hash != question.question_content_hash
        or pair.source is not question.source
        or pair.original_question_id != question.original_question_id
    ):
        raise TrainerExportIntegrityError(
            "pair and question content hash or identity differs"
        )


def _selected_candidate(
    candidates: Mapping[tuple[str, str], CandidateRecord],
    pair: PairRecord,
    candidate_id: str,
) -> CandidateRecord:
    candidate = candidates.get((pair.generation_run_id, candidate_id))
    if candidate is None:
        raise TrainerExportIntegrityError(
            "selected candidate is missing from the generation artifact"
        )
    return candidate


def _validate_selected_candidate(
    pair: PairRecord, candidate: CandidateRecord, *, positive: bool
) -> None:
    plan = candidate.plan
    if (
        plan.original_question_id != pair.original_question_id
        or plan.question_content_hash != pair.question_content_hash
        or plan.source is not pair.source
    ):
        raise TrainerExportIntegrityError(
            "selected candidate question identity differs from pair"
        )
    if (
        plan.split is not pair.split
        or plan.split_manifest_hash != pair.split_manifest_hash
    ):
        raise TrainerExportIntegrityError(
            "selected candidate split lineage differs from pair"
        )
    expected_hash = sha256_hex(normalize_text_v1(candidate.response).encode("utf-8"))
    recorded_hash = (
        pair.positive_response_hash if positive else pair.negative_response_hash
    )
    recorded_generator = (
        pair.positive_generator_id if positive else pair.negative_generator_id
    )
    if expected_hash != recorded_hash:
        raise TrainerExportIntegrityError(
            "selected candidate response hash differs from pair"
        )
    if plan.generator_id != recorded_generator:
        raise TrainerExportIntegrityError(
            "selected candidate generator differs from pair"
        )


def _validate_permutation_group(
    pair: PairRecord, group: Sequence[PermutationRecord]
) -> None:
    if len(group) != 2 or {record.permutation_id for record in group} != {0, 1}:
        raise TrainerExportIntegrityError(
            "each pair must contain exactly AB and BA permutations"
        )
    ab, ba = sorted(group, key=lambda record: record.permutation_id)
    for record in group:
        if (
            record.original_question_id != pair.original_question_id
            or record.generation_run_id != pair.generation_run_id
            or record.split is not pair.split
            or record.split_manifest_hash != pair.split_manifest_hash
        ):
            raise TrainerExportIntegrityError(
                "permutation split lineage or pair provenance differs"
            )
    if not (
        ab.surface_a_candidate_id == pair.positive_candidate_id
        and ab.surface_b_candidate_id == pair.negative_candidate_id
        and ba.surface_a_candidate_id == pair.negative_candidate_id
        and ba.surface_b_candidate_id == pair.positive_candidate_id
    ):
        raise TrainerExportIntegrityError(
            "permutation surface mapping does not match the semantic pair"
        )


def _validate_upstream_manifests(
    upstream_manifests: Mapping[str, RunManifest | Mapping[str, object]],
) -> dict[str, _BoundManifest]:
    if (
        not isinstance(upstream_manifests, Mapping)
        or set(upstream_manifests) != _EXPORT_UPSTREAM_ROLES
    ):
        raise TrainerExportIntegrityError(
            "export requires split, generation, pairs, and permutations manifests"
        )
    validated: dict[str, _BoundManifest] = {}
    for role in sorted(upstream_manifests):
        manifest = upstream_manifests[role]
        if isinstance(manifest, RunManifest):
            payload = manifest.to_dict()
        elif isinstance(manifest, Mapping):
            payload = dict(manifest)
        else:
            raise TypeError("upstream manifests must be RunManifest or mapping values")
        try:
            core = RunManifest.from_dict(
                {field: payload[field] for field in RunManifest.__annotations__}
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise TrainerExportIntegrityError(
                f"{role} manifest envelope is invalid"
            ) from exc
        if core.stage != role:
            raise TrainerExportIntegrityError(
                f"{role} manifest stage does not match its role"
            )
        expected_hash = sha256_hex(
            canonical_json_bytes(
                {
                    key: value
                    for key, value in payload.items()
                    if key not in {"created_at_utc", "output_manifest_hash"}
                }
            )
        )
        if core.output_manifest_hash != expected_hash:
            raise TrainerExportIntegrityError(
                f"{role} manifest envelope hash is invalid"
            )
        validated[role] = _BoundManifest(
            core.run_id, core.output_manifest_hash, core.artifacts
        )
    return validated


def _validate_record_run_lineage(
    records: Mapping[str, tuple[object, ...]],
    manifests: Mapping[str, _BoundManifest],
) -> None:
    generation_runs = {
        record.plan.generation_run_id for record in records["generation"]
    }
    if generation_runs != {manifests["generation"].run_id}:
        raise TrainerExportIntegrityError(
            "generation run lineage does not match its manifest"
        )
    pair_runs = {record.pair_run_id for record in records["pairs"]}
    if pair_runs != {manifests["pairs"].run_id}:
        raise TrainerExportIntegrityError(
            "pair run lineage does not match its manifest"
        )
    permutation_runs = {record.permutation_run_id for record in records["permutations"]}
    if permutation_runs != {manifests["permutations"].run_id}:
        raise TrainerExportIntegrityError(
            "permutation run lineage does not match its manifest"
        )


def _read_role_records(
    root: Path, role: str, manifest: _BoundManifest
) -> tuple[object, ...]:
    basename = _ARTIFACT_BASENAMES[role]
    selected = [
        ref for ref in manifest.artifacts if Path(ref.relative_path).name == basename
    ]
    if not selected:
        raise TrainerExportIntegrityError(
            f"{role} manifest does not reference {basename}"
        )
    decoded: list[object] = []
    for ref in manifest.artifacts:
        try:
            verify_artifact_ref(root, ref)
        except IntegrityError as exc:
            raise TrainerExportIntegrityError(
                f"{role} artifact failed immutable hash validation"
            ) from exc
        if ref not in selected:
            continue
        try:
            scan = scan_jsonl(root / ref.relative_path)
        except (IntegrityError, OSError) as exc:
            raise TrainerExportIntegrityError(
                f"{role} artifact is not valid append-only JSONL"
            ) from exc
        if len(scan.records) != ref.record_count:
            raise TrainerExportIntegrityError(
                f"{role} artifact record count differs from its manifest"
            )
        record_type = _RECORD_TYPES[role]
        for stored in scan.records:
            payload = {
                key: value for key, value in stored.items() if key != "record_hash"
            }
            try:
                decoded.append(record_type.from_dict(payload))
            except (TypeError, ValueError) as exc:
                raise TrainerExportIntegrityError(
                    f"{role} artifact contains an invalid record"
                ) from exc
    return tuple(decoded)


def _write_parquet_atomically(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    # Imports stay inside the Parquet boundary so canonical audit/gate consumers
    # never acquire a pandas or PyArrow dependency through this module.
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - exercised by production preflight
        raise RuntimeError("trainer export requires pandas and pyarrow") from exc

    descriptor, name = tempfile.mkstemp(
        dir=path.parent, prefix="train.", suffix=".parquet.tmp"
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        pd.DataFrame(list(rows), columns=sorted(_ROW_FIELDS)).to_parquet(
            temporary,
            engine="pyarrow",
            index=False,
        )
        # Windows requires a writable descriptor for FlushFileBuffers/fsync.
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def export_run_id(upstream_bindings: Mapping[str, str]) -> str:
    """Return the deterministic export namespace for an immutable input set."""
    return run_id("trainer_export", export_stage_config(upstream_bindings))
