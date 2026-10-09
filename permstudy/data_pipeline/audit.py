"""Deterministic verifier-audit selection and exact human-decision joins."""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from .canonical import canonical_json_bytes, sha256_hex
from .ids import run_id
from .lineage import role_bound_stage_config
from .schema import (
    AuditDecision,
    AuditSelectionRecord,
    AuditVerdict,
    QuestionVerificationRecord,
    VerificationRecord,
    VerificationStatus,
)


AUDIT_CONFIG_SCHEMA = "verifier_audit_config_v1"
AUDIT_SAMPLE_SCORE_SCHEMA = "audit_sample_score_v1"
AUDIT_VERIFICATION_SNAPSHOT_SCHEMA = "audit_verification_snapshot_v1"

_AUDIT_UPSTREAM_ROLES = frozenset({"verification"})
_BINARY_STATUSES = frozenset({VerificationStatus.CORRECT, VerificationStatus.INCORRECT})
_EXHAUSTIVE_STATUSES = frozenset(
    {
        VerificationStatus.INVALID,
        VerificationStatus.AMBIGUOUS,
        VerificationStatus.ERROR,
    }
)
_REASON_CODES = frozenset(
    {
        "missing_final_marker",
        "duplicate_same_marker",
        "conflicting_markers",
        "math_parse_failure",
        "gold_parse_failure",
        "timeout",
        "dependency_error",
        "prompt_contract_mismatch",
        "other",
    }
)
_SYSTEMATIC_REASON_CODES = frozenset(
    {
        "prompt_contract_mismatch",
        "math_parse_failure",
        "gold_parse_failure",
        "dependency_error",
    }
)


class AuditIntegrityError(ValueError):
    """Raised when audit coverage or a human-decision join is incomplete."""


@dataclass(frozen=True)
class AuditSummary:
    """Sanitized audit completion and hard-fail signals for Task 15."""

    required_count: int
    completed_count: int
    verdict_counts: Mapping[str, int]
    confirmed_disagreements: int
    systematic_issue: bool

    def __post_init__(self) -> None:
        if (
            type(self.required_count) is not int
            or type(self.completed_count) is not int
            or self.required_count < 0
            or self.completed_count < 0
            or type(self.confirmed_disagreements) is not int
            or self.confirmed_disagreements < 0
            or type(self.systematic_issue) is not bool
        ):
            raise ValueError("invalid audit summary")
        counts = dict(self.verdict_counts)
        if set(counts) != {verdict.value for verdict in AuditVerdict} or any(
            type(value) is not int or value < 0 for value in counts.values()
        ):
            raise ValueError("invalid audit verdict counts")
        object.__setattr__(self, "verdict_counts", MappingProxyType(counts))


def audit_stage_config(
    generation_run_id: str,
    verification_run_id: str,
    verification_snapshot_hash: str,
    audit_seed: int = 42,
    max_per_cell: int = 5,
) -> dict[str, object]:
    """Bind deterministic audit sampling to the exact verified record snapshot."""
    if not isinstance(generation_run_id, str) or not generation_run_id.strip():
        raise ValueError("generation_run_id must be nonempty text")
    if not isinstance(verification_run_id, str) or not verification_run_id.strip():
        raise ValueError("verification_run_id must be nonempty text")
    if type(audit_seed) is not int or audit_seed < 0:
        raise ValueError("audit_seed must be a nonnegative integer")
    if type(max_per_cell) is not int or max_per_cell <= 0:
        raise ValueError("max_per_cell must be a positive integer")
    return role_bound_stage_config(
        {
            "audit_config_schema": AUDIT_CONFIG_SCHEMA,
            "audit_seed": audit_seed,
            "generation_run_id": generation_run_id,
            "max_per_cell": max_per_cell,
            "verification_run_id": verification_run_id,
        },
        {"verification": verification_snapshot_hash},
        _AUDIT_UPSTREAM_ROLES,
    )


def build_audit_selection(
    candidate_records: Sequence[VerificationRecord],
    question_records: Sequence[QuestionVerificationRecord],
    generation_run_id: str,
    audit_seed: int = 42,
    max_per_cell: int = 5,
) -> list[AuditSelectionRecord]:
    """Select all exceptional outcomes and sampled binary outcomes."""
    candidates, questions, verification_run_id = _validate_verification_inputs(
        candidate_records,
        question_records,
        generation_run_id,
    )
    verification_snapshot_hash = _verification_snapshot_hash(
        candidates,
        tuple(questions.values()),
    )
    typed_config = audit_stage_config(
        generation_run_id,
        verification_run_id,
        verification_snapshot_hash,
        audit_seed,
        max_per_cell,
    )
    audit_run_id = run_id("audit", typed_config)

    chosen = [
        record
        for record in candidates
        if record.verification_status in _EXHAUSTIVE_STATUSES
    ]
    cells: dict[tuple[str, str, str], list[VerificationRecord]] = {}
    for record in candidates:
        if record.verification_status in _BINARY_STATUSES:
            cell = (
                record.source.value,
                record.generator_id,
                record.verification_status.value,
            )
            cells.setdefault(cell, []).append(record)
    for records in cells.values():
        ordered = sorted(
            records,
            key=lambda record: (
                _sample_score(record, audit_seed),
                record.candidate_id,
            ),
        )
        chosen.extend(ordered[:max_per_cell])

    selection = [_candidate_selection(record, audit_run_id) for record in chosen]
    selection.extend(
        _question_selection(record, generation_run_id, audit_run_id)
        for record in questions.values()
        if record.gold_parse_status == "gold_verification_error"
    )
    return sorted(selection, key=_selection_sort_key)


def validate_audit_decisions(
    selection: Sequence[AuditSelectionRecord],
    decisions: Sequence[AuditDecision],
) -> AuditSummary:
    """Require one exact human decision for every selected audit identity."""
    selected_by_identity: dict[tuple[object, ...], AuditSelectionRecord] = {}
    run_namespaces: set[tuple[str, str, str]] = set()
    for record in selection:
        if not isinstance(record, AuditSelectionRecord):
            raise TypeError("selection must contain AuditSelectionRecord values")
        record.validate()
        identity = _audit_identity(record)
        if identity in selected_by_identity:
            raise AuditIntegrityError("audit selection contains a duplicate identity")
        selected_by_identity[identity] = record
        run_namespaces.add(
            (
                record.audit_run_id,
                record.generation_run_id,
                record.verification_run_id,
            )
        )
    if len(run_namespaces) > 1:
        raise AuditIntegrityError("audit selection contains mixed audit run namespaces")

    decision_by_identity: dict[tuple[object, ...], AuditDecision] = {}
    for decision in decisions:
        if not isinstance(decision, AuditDecision):
            raise TypeError("decisions must contain AuditDecision values")
        decision.validate()
        identity = _audit_identity(decision)
        if identity in decision_by_identity:
            raise AuditIntegrityError("audit decisions contain a duplicate identity")
        if identity not in selected_by_identity:
            raise AuditIntegrityError("audit decision has a foreign identity")
        selected = selected_by_identity[identity]
        if (
            decision.audit_run_id != selected.audit_run_id
            or decision.original_question_id != selected.original_question_id
        ):
            raise AuditIntegrityError("audit decision has foreign selection metadata")
        if (
            decision.confirmed_disagree
            and selected.verification_status not in _BINARY_STATUSES
        ):
            raise AuditIntegrityError(
                "confirmed disagreement is limited to binary correct/incorrect labels"
            )
        decision_by_identity[identity] = decision

    missing = set(selected_by_identity) - set(decision_by_identity)
    if missing:
        raise AuditIntegrityError("audit decisions are missing required identities")

    verdict_counts = Counter(
        decision.verdict.value for decision in decision_by_identity.values()
    )
    counts = {verdict.value: verdict_counts[verdict.value] for verdict in AuditVerdict}
    confirmed = sum(
        decision.confirmed_disagree for decision in decision_by_identity.values()
    )
    systematic_issue = any(
        decision.verdict is AuditVerdict.DISAGREE
        and decision.reason_code in _SYSTEMATIC_REASON_CODES
        and (
            selected_by_identity[identity].verification_status not in _BINARY_STATUSES
            or decision.confirmed_disagree
        )
        for identity, decision in decision_by_identity.items()
    )
    return AuditSummary(
        required_count=len(selected_by_identity),
        completed_count=len(decision_by_identity),
        verdict_counts=counts,
        confirmed_disagreements=confirmed,
        systematic_issue=systematic_issue,
    )


def _validate_verification_inputs(
    candidate_records: Sequence[VerificationRecord],
    question_records: Sequence[QuestionVerificationRecord],
    generation_run_id: str,
) -> tuple[
    tuple[VerificationRecord, ...],
    dict[str, QuestionVerificationRecord],
    str,
]:
    if not isinstance(generation_run_id, str) or not generation_run_id.strip():
        raise AuditIntegrityError("generation run identity must be nonempty text")
    candidates = tuple(candidate_records)
    question_values = tuple(question_records)
    verification_runs: set[str] = set()
    candidate_ids: set[tuple[str, str, str]] = set()
    for record in candidates:
        if not isinstance(record, VerificationRecord):
            raise TypeError("candidate_records must contain VerificationRecord values")
        record.validate()
        if record.generation_run_id != generation_run_id:
            raise AuditIntegrityError(
                "candidate generation run does not match the audit request"
            )
        identity = (
            record.generation_run_id,
            record.verification_run_id,
            record.candidate_id,
        )
        if identity in candidate_ids:
            raise AuditIntegrityError("duplicate candidate audit identity")
        candidate_ids.add(identity)
        verification_runs.add(record.verification_run_id)

    questions: dict[str, QuestionVerificationRecord] = {}
    for record in question_values:
        if not isinstance(record, QuestionVerificationRecord):
            raise TypeError(
                "question_records must contain QuestionVerificationRecord values"
            )
        record.validate()
        if record.original_question_id in questions:
            raise AuditIntegrityError("duplicate question verification identity")
        questions[record.original_question_id] = record
        verification_runs.add(record.verification_run_id)

    if len(verification_runs) != 1:
        raise AuditIntegrityError("audit selection requires one verification run")
    verification_run_id = next(iter(verification_runs))
    for candidate in candidates:
        question = questions.get(candidate.original_question_id)
        if question is None:
            raise AuditIntegrityError(
                "candidate is missing its question verification record"
            )
        if question.verification_run_id != candidate.verification_run_id:
            raise AuditIntegrityError("candidate and question verification run differs")
        if question.source is not candidate.source:
            raise AuditIntegrityError("candidate and question source differs")
        if question.gold_parse_status != "ok":
            raise AuditIntegrityError(
                "gold verification errors cannot have candidate verification records"
            )
    return candidates, questions, verification_run_id


def _sample_score(record: VerificationRecord, audit_seed: int) -> str:
    return sha256_hex(
        canonical_json_bytes(
            {
                "audit_seed": audit_seed,
                "candidate_id": record.candidate_id,
                "generation_run_id": record.generation_run_id,
                "schema": AUDIT_SAMPLE_SCORE_SCHEMA,
                "verification_run_id": record.verification_run_id,
            }
        )
    )


def _verification_snapshot_hash(
    candidates: Sequence[VerificationRecord],
    questions: Sequence[QuestionVerificationRecord],
) -> str:
    candidate_payloads = [
        record.to_dict()
        for record in sorted(
            candidates,
            key=lambda record: (
                record.generation_run_id,
                record.verification_run_id,
                record.candidate_id,
            ),
        )
    ]
    question_payloads = [
        record.to_dict()
        for record in sorted(
            questions,
            key=lambda record: (
                record.verification_run_id,
                record.original_question_id,
            ),
        )
    ]
    return sha256_hex(
        canonical_json_bytes(
            {
                "candidate_records": candidate_payloads,
                "question_records": question_payloads,
                "schema": AUDIT_VERIFICATION_SNAPSHOT_SCHEMA,
            }
        )
    )


def _reason_code(error_type: str | None) -> str:
    return error_type if error_type in _REASON_CODES else "other"


def _candidate_selection(
    record: VerificationRecord,
    audit_run_id: str,
) -> AuditSelectionRecord:
    selected = AuditSelectionRecord(
        audit_run_id=audit_run_id,
        generation_run_id=record.generation_run_id,
        verification_run_id=record.verification_run_id,
        record_kind="candidate",
        original_question_id=record.original_question_id,
        candidate_id=record.candidate_id,
        source=record.source,
        generator_id=record.generator_id,
        verification_status=record.verification_status,
        gold_parse_status=None,
        reason_code=_reason_code(record.error_type),
    )
    selected.validate()
    return selected


def _question_selection(
    record: QuestionVerificationRecord,
    generation_run_id: str,
    audit_run_id: str,
) -> AuditSelectionRecord:
    selected = AuditSelectionRecord(
        audit_run_id=audit_run_id,
        generation_run_id=generation_run_id,
        verification_run_id=record.verification_run_id,
        record_kind="question_gold",
        original_question_id=record.original_question_id,
        candidate_id=None,
        source=record.source,
        generator_id=None,
        verification_status=None,
        gold_parse_status=record.gold_parse_status,
        reason_code=_reason_code(record.error_type),
    )
    selected.validate()
    return selected


def _audit_identity(record: AuditSelectionRecord | AuditDecision) -> tuple[object, ...]:
    subject_id = (
        record.candidate_id
        if record.record_kind == "candidate"
        else record.original_question_id
    )
    return (
        record.record_kind,
        record.generation_run_id,
        record.verification_run_id,
        subject_id,
    )


def _selection_sort_key(record: AuditSelectionRecord) -> tuple[object, ...]:
    return (
        record.record_kind,
        record.source.value,
        record.generator_id or "",
        record.verification_status.value if record.verification_status else "",
        record.original_question_id,
        record.candidate_id or "",
    )
