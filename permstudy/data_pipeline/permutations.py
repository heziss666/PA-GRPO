"""Exact AB/BA surface construction over immutable semantic pair records."""

from collections.abc import Mapping, Sequence

from .canonical import canonical_json_bytes, sha256_hex
from .ids import pair_id as make_pair_id, run_id
from .lineage import role_bound_stage_config
from .pairs import PAIR_SCHEMA_VERSION
from .schema import PairRecord, PermutationRecord


PERMUTATION_CONFIG_SCHEMA = "permutation_config_v1"
PERMUTATION_MAPPING_VERSION = "ab_ba_mapping_v1"

_PERMUTATION_UPSTREAM_ROLES = frozenset({"pairs"})


class PermutationIntegrityError(ValueError):
    """Raised when pair or permutation lineage cannot be trusted."""


def permutation_stage_config(upstream_bindings: Mapping[str, str]) -> dict[str, object]:
    """Bind the fixed AB/BA mapping to the immutable pair manifest."""
    return role_bound_stage_config(
        {
            "permutation_config_schema": PERMUTATION_CONFIG_SCHEMA,
            "permutation_mapping_version": PERMUTATION_MAPPING_VERSION,
        },
        upstream_bindings,
        _PERMUTATION_UPSTREAM_ROLES,
    )


def build_permutations(
    pair: PairRecord,
    *,
    upstream_bindings: Mapping[str, str],
) -> tuple[PermutationRecord, PermutationRecord]:
    """Construct canonical AB then BA records without re-deriving semantics."""
    _validate_pair(pair)
    permutation_run_id = run_id("permutations", permutation_stage_config(upstream_bindings))
    common = {
        "permutation_run_id": permutation_run_id,
        "pair_id": pair.pair_id,
        "generation_run_id": pair.generation_run_id,
        "original_question_id": pair.original_question_id,
        "split": pair.split,
        "split_manifest_hash": pair.split_manifest_hash,
    }
    ab = PermutationRecord(
        **common,
        permutation_id=0,
        permutation_label="AB",
        surface_a_candidate_id=pair.positive_candidate_id,
        surface_b_candidate_id=pair.negative_candidate_id,
        correct_surface="A",
    )
    ba = PermutationRecord(
        **common,
        permutation_id=1,
        permutation_label="BA",
        surface_a_candidate_id=pair.negative_candidate_id,
        surface_b_candidate_id=pair.positive_candidate_id,
        correct_surface="B",
    )
    ab.validate()
    ba.validate()
    return ab, ba


def permutation_manifest_bytes(records: Sequence[PermutationRecord]) -> bytes:
    """Validate complete mirrored pairs and render canonical private JSONL."""
    materialized = tuple(records)
    if not materialized:
        return b""
    by_pair: dict[str, list[PermutationRecord]] = {}
    permutation_run_ids: set[str] = set()
    for record in materialized:
        if not isinstance(record, PermutationRecord):
            raise TypeError("permutation manifest records must be PermutationRecord values")
        record.validate()
        by_pair.setdefault(record.pair_id, []).append(record)
        permutation_run_ids.add(record.permutation_run_id)
    if len(permutation_run_ids) != 1:
        raise PermutationIntegrityError("permutation manifest has mixed run lineage")

    question_ids: set[str] = set()
    for group in by_pair.values():
        _validate_mirrored_group(group)
        question_id = group[0].original_question_id
        if question_id in question_ids:
            raise PermutationIntegrityError("permutation manifest contains duplicate question identity")
        question_ids.add(question_id)

    ordered = sorted(materialized, key=lambda record: (record.original_question_id, record.permutation_id))
    return b"".join(canonical_json_bytes(record.to_dict()) + b"\n" for record in ordered)


def permutation_manifest_hash(records: Sequence[PermutationRecord]) -> str:
    """Hash the exact canonical permutation JSONL bytes."""
    return sha256_hex(permutation_manifest_bytes(records))


def _validate_pair(pair: PairRecord) -> None:
    if not isinstance(pair, PairRecord):
        raise TypeError("pair must be a PairRecord")
    pair.validate()
    if pair.schema_version != PAIR_SCHEMA_VERSION:
        raise PermutationIntegrityError("unknown pair schema version")
    expected_pair_id = make_pair_id(
        pair.generation_run_id,
        pair.original_question_id,
        pair.positive_candidate_id,
        pair.negative_candidate_id,
        pair.schema_version,
    )
    if pair.pair_id != expected_pair_id:
        raise PermutationIntegrityError("pair identity does not match semantic candidates")


def _validate_mirrored_group(group: Sequence[PermutationRecord]) -> None:
    if len(group) != 2 or {record.permutation_id for record in group} != {0, 1}:
        raise PermutationIntegrityError("each pair must contain exactly AB and BA permutations")
    ab, ba = sorted(group, key=lambda record: record.permutation_id)
    if ab.original_question_id != ba.original_question_id:
        raise PermutationIntegrityError("AB and BA question lineage differs")
    if ab.generation_run_id != ba.generation_run_id:
        raise PermutationIntegrityError("AB and BA generation lineage differs")
    if ab.split != ba.split or ab.split_manifest_hash != ba.split_manifest_hash:
        raise PermutationIntegrityError("AB and BA split lineage differs")
    if ab.permutation_run_id != ba.permutation_run_id:
        raise PermutationIntegrityError("AB and BA permutation run lineage differs")
    if not (
        ab.surface_a_candidate_id == ba.surface_b_candidate_id
        and ab.surface_b_candidate_id == ba.surface_a_candidate_id
    ):
        raise PermutationIntegrityError("AB and BA surface mappings are not mirrored")
