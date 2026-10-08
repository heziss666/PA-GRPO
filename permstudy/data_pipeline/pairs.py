"""Run-scoped response deduplication and deterministic verified pair selection."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re

from .canonical import canonical_jsonl_bytes, normalize_text_v1, sha256_hex
from .ids import pair_id as make_pair_id, run_id
from .lineage import role_bound_stage_config
from .schema import (
    CandidateRecord,
    PairRecord,
    QuestionRecord,
    VerificationRecord,
    VerificationStatus,
)


PAIR_SCHEMA_VERSION = "pair_v1"
PAIR_RUN_CONFIG_SCHEMA = "pair_selection_config_v1"
RESPONSE_NORMALIZATION_VERSION = "response_normalization_v1"
TOKENIZER_REPOSITORY = "Qwen/Qwen2.5-7B-Instruct"

_IMMUTABLE_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_PAIR_UPSTREAM_ROLES = frozenset({"generation", "verification"})
_PAIRABLE_STATUSES = frozenset({VerificationStatus.CORRECT, VerificationStatus.INCORRECT})

VerifiedCandidate = tuple[CandidateRecord, VerificationRecord]


class PairIntegrityError(ValueError):
    """Raised when candidate, verification, or pair lineage is inconsistent."""


@dataclass(frozen=True)
class ResponseGroup:
    response_hash: str
    representative_key: tuple[str, str]
    member_keys: tuple[tuple[str, str], ...]
    member_generators: tuple[str, ...]
    verification_status: VerificationStatus


@dataclass(frozen=True)
class _EligibleCandidate:
    candidate: CandidateRecord
    verification: VerificationRecord
    canonical_response: str
    response_hash: str

    @property
    def key(self) -> tuple[str, str]:
        return self.candidate.plan.key


def pair_stage_config(
    tokenizer_revision: str,
    upstream_bindings: Mapping[str, str],
) -> dict[str, object]:
    """Bind pair selection to its tokenizer contract and role-tagged upstreams."""
    _validate_tokenizer_revision(tokenizer_revision)
    return role_bound_stage_config(
        {
            "pair_run_config_schema": PAIR_RUN_CONFIG_SCHEMA,
            "pair_schema_version": PAIR_SCHEMA_VERSION,
            "response_normalization_version": RESPONSE_NORMALIZATION_VERSION,
            "tokenizer_repo": TOKENIZER_REPOSITORY,
            "tokenizer_revision": tokenizer_revision,
        },
        upstream_bindings,
        _PAIR_UPSTREAM_ROLES,
    )


def build_dedup_groups(records: Sequence[VerifiedCandidate]) -> list[ResponseGroup]:
    """Group eligible normalized responses within each generation run."""
    eligible = _eligible_candidates(records)
    grouped: dict[tuple[str, str, str], list[_EligibleCandidate]] = {}
    for item in eligible:
        grouped.setdefault(
            (
                item.candidate.plan.generation_run_id,
                item.candidate.plan.original_question_id,
                item.response_hash,
            ),
            [],
        ).append(item)

    groups: list[ResponseGroup] = []
    for (_generation_run_id, _original_question_id, response_hash), members in grouped.items():
        statuses = {member.verification.verification_status for member in members}
        if len(statuses) != 1:
            raise PairIntegrityError("one normalized response has conflicting verification statuses")
        ordered = sorted(members, key=lambda member: member.key)
        groups.append(
            ResponseGroup(
                response_hash=response_hash,
                representative_key=ordered[0].key,
                member_keys=tuple(member.key for member in ordered),
                member_generators=tuple(member.candidate.plan.generator_id for member in ordered),
                verification_status=ordered[0].verification.verification_status,
            )
        )
    return sorted(
        groups,
        key=lambda group: (
            group.response_hash,
            group.representative_key[0],
            group.representative_key[1],
        ),
    )


def select_pair(
    question: QuestionRecord,
    verified_candidates: Sequence[VerifiedCandidate],
    tokenizer: object,
    tokenizer_revision: str,
    *,
    upstream_bindings: Mapping[str, str],
) -> PairRecord | None:
    """Select the correct/incorrect representative pair with minimum token gap."""
    if not isinstance(question, QuestionRecord):
        raise TypeError("question must be a QuestionRecord")
    question.validate()
    typed_config = pair_stage_config(tokenizer_revision, upstream_bindings)
    materialized = tuple(verified_candidates)
    if not materialized:
        return None
    _validate_question_scope(question, materialized)

    eligible = _eligible_candidates(materialized)
    groups = build_dedup_groups(materialized)
    if not groups:
        return None
    by_key = {item.key: item for item in eligible}

    representatives: list[tuple[ResponseGroup, _EligibleCandidate, int]] = []
    for group in groups:
        representative = by_key[group.representative_key]
        tokens = _encode_response(tokenizer, representative.canonical_response)
        representatives.append((group, representative, len(tokens)))

    positives = [item for item in representatives if item[0].verification_status is VerificationStatus.CORRECT]
    negatives = [item for item in representatives if item[0].verification_status is VerificationStatus.INCORRECT]
    if not positives or not negatives:
        return None

    positive, negative = min(
        ((positive, negative) for positive in positives for negative in negatives),
        key=lambda pair: (
            abs(pair[0][2] - pair[1][2]),
            pair[0][1].candidate.plan.candidate_id,
            pair[1][1].candidate.plan.candidate_id,
        ),
    )
    positive_group, positive_record, positive_tokens = positive
    negative_group, negative_record, negative_tokens = negative
    plan = positive_record.candidate.plan
    verification_run_id = positive_record.verification.verification_run_id
    pair_run_id = run_id("pairs", typed_config)
    selected = PairRecord(
        schema_version=PAIR_SCHEMA_VERSION,
        pair_run_id=pair_run_id,
        pair_id=make_pair_id(
            plan.generation_run_id,
            question.original_question_id,
            plan.candidate_id,
            negative_record.candidate.plan.candidate_id,
            PAIR_SCHEMA_VERSION,
        ),
        generation_run_id=plan.generation_run_id,
        verification_run_id=verification_run_id,
        original_question_id=question.original_question_id,
        question_content_hash=question.question_content_hash,
        source=question.source,
        split=plan.split,
        split_manifest_hash=plan.split_manifest_hash,
        positive_candidate_id=plan.candidate_id,
        negative_candidate_id=negative_record.candidate.plan.candidate_id,
        positive_response_hash=positive_group.response_hash,
        negative_response_hash=negative_group.response_hash,
        positive_generator_id=plan.generator_id,
        negative_generator_id=negative_record.candidate.plan.generator_id,
        positive_token_count=positive_tokens,
        negative_token_count=negative_tokens,
        tokenizer_repo=TOKENIZER_REPOSITORY,
        tokenizer_revision=tokenizer_revision,
    )
    selected.validate()
    return selected


def pair_manifest_bytes(records: Sequence[PairRecord]) -> bytes:
    """Render the canonical, timestamp-free private pair JSONL payload."""
    materialized = tuple(records)
    question_ids: set[str] = set()
    payloads: list[dict[str, object]] = []
    for record in materialized:
        if not isinstance(record, PairRecord):
            raise TypeError("pair manifest records must be PairRecord values")
        record.validate()
        if record.original_question_id in question_ids:
            raise PairIntegrityError("pair manifest contains duplicate question identity")
        question_ids.add(record.original_question_id)
        payloads.append(record.to_dict())
    return canonical_jsonl_bytes(payloads, "original_question_id")


def pair_manifest_hash(records: Sequence[PairRecord]) -> str:
    """Hash the exact canonical pair manifest bytes."""
    return sha256_hex(pair_manifest_bytes(records))


def _eligible_candidates(records: Sequence[VerifiedCandidate]) -> list[_EligibleCandidate]:
    eligible: list[_EligibleCandidate] = []
    seen: set[tuple[str, str]] = set()
    for value in records:
        if not isinstance(value, tuple) or len(value) != 2:
            raise TypeError("verified candidates must be candidate/verification tuples")
        candidate, verification = value
        _validate_join(candidate, verification)
        if candidate.plan.key in seen:
            raise PairIntegrityError("verified candidates contain a duplicate composite identity")
        seen.add(candidate.plan.key)
        if verification.verification_status not in _PAIRABLE_STATUSES:
            continue
        if candidate.finish_reason.strip().lower() == "length":
            continue
        canonical_response = normalize_text_v1(candidate.response)
        if not canonical_response:
            continue
        if verification.prediction_parse_status != "parsed" or not verification.canonical_prediction:
            continue
        if verification.error_type is not None:
            raise PairIntegrityError("pairable verification status cannot carry an error")
        eligible.append(
            _EligibleCandidate(
                candidate=candidate,
                verification=verification,
                canonical_response=canonical_response,
                response_hash=sha256_hex(canonical_response.encode("utf-8")),
            )
        )
    return eligible


def _validate_join(candidate: CandidateRecord, verification: VerificationRecord) -> None:
    if not isinstance(candidate, CandidateRecord) or not isinstance(verification, VerificationRecord):
        raise TypeError("verified candidates must contain CandidateRecord and VerificationRecord")
    candidate.validate()
    verification.validate()
    plan = candidate.plan
    if (plan.generation_run_id, plan.candidate_id) != (
        verification.generation_run_id,
        verification.candidate_id,
    ):
        raise PairIntegrityError("candidate and verification composite identity mismatch")
    if (
        plan.original_question_id != verification.original_question_id
        or plan.source is not verification.source
        or plan.generator_id != verification.generator_id
    ):
        raise PairIntegrityError("candidate and verification provenance mismatch")


def _validate_question_scope(
    question: QuestionRecord,
    records: Sequence[VerifiedCandidate],
) -> None:
    generation_runs: set[str] = set()
    verification_runs: set[str] = set()
    splits = set()
    split_hashes: set[str] = set()
    for candidate, verification in records:
        _validate_join(candidate, verification)
        plan = candidate.plan
        if (
            plan.original_question_id != question.original_question_id
            or plan.question_content_hash != question.question_content_hash
            or plan.source is not question.source
        ):
            raise PairIntegrityError("verified candidate question identity mismatch")
        generation_runs.add(plan.generation_run_id)
        verification_runs.add(verification.verification_run_id)
        splits.add(plan.split)
        split_hashes.add(plan.split_manifest_hash)
    if len(generation_runs) != 1:
        raise PairIntegrityError("pair selection requires exactly one generation run")
    if len(verification_runs) != 1:
        raise PairIntegrityError("pair selection requires exactly one verification run")
    if len(splits) != 1 or len(split_hashes) != 1:
        raise PairIntegrityError("pair selection requires one immutable split lineage")


def _encode_response(tokenizer: object, canonical_response: str) -> Sequence[object]:
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise TypeError("tokenizer must provide encode")
    tokens = encode(canonical_response, add_special_tokens=False)
    if isinstance(tokens, (str, bytes)) or not isinstance(tokens, Sequence):
        raise TypeError("tokenizer encode must return a sequence")
    return tokens


def _validate_tokenizer_revision(tokenizer_revision: str) -> None:
    if not isinstance(tokenizer_revision, str) or not _IMMUTABLE_REVISION.fullmatch(tokenizer_revision):
        raise ValueError("tokenizer revision must be a lowercase immutable 40-hex SHA")
