"""Deterministic gates over validated canonical records, without trainer imports.

Manifest envelopes and exact human-audit identities are validated by the I/O
and audit stages before these records and their sanitized summary are supplied.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from types import MappingProxyType

from .audit import AuditSummary
from .canonical import normalize_text_v1, sha256_hex
from .ids import candidate_id, pair_id
from .lineage import role_bound_stage_config
from .pairs import PAIR_SCHEMA_VERSION, build_dedup_groups, pair_stage_config
from .permutations import permutation_manifest_bytes
from .schema import (
    CandidatePlan,
    CandidateRecord,
    GateStatus,
    PairRecord,
    PermutationRecord,
    QuestionVerificationRecord,
    Source,
    VerificationRecord,
    VerificationStatus,
)


GATE_CONFIG_SCHEMA = "pipeline_gates_v1"
_UPSTREAM_ROLES = frozenset(
    {"generation", "verification", "audit", "pairs", "permutations"}
)
_LABELS = tuple(status.value for status in VerificationStatus)


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class GateInputs:
    plans: tuple[CandidatePlan, ...]
    candidates: tuple[CandidateRecord, ...]
    verifications: tuple[VerificationRecord, ...]
    question_verifications: tuple[QuestionVerificationRecord, ...]
    pairs: tuple[PairRecord, ...]
    permutations: tuple[PermutationRecord, ...]
    audit_summary: AuditSummary

    def __post_init__(self) -> None:
        for name in (
            "plans",
            "candidates",
            "verifications",
            "question_verifications",
            "pairs",
            "permutations",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not isinstance(self.audit_summary, AuditSummary):
            raise TypeError("audit_summary must be an AuditSummary")


@dataclass(frozen=True)
class FunctionalGateReport:
    passed: bool
    failures: tuple[str, ...]
    metrics: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "failures", tuple(self.failures))
        object.__setattr__(self, "metrics", _freeze(self.metrics))


@dataclass(frozen=True)
class StatisticalGateReport:
    status: GateStatus
    hard_failures: tuple[str, ...]
    warnings: tuple[str, ...]
    diagnostics: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "hard_failures", tuple(self.hard_failures))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "diagnostics", _freeze(self.diagnostics))


def gate_stage_config(upstream_bindings: Mapping[str, str]) -> dict[str, object]:
    """Bind all five upstream roles to the fixed, versioned gate rules."""
    return role_bound_stage_config(
        {"gate_config_schema": GATE_CONFIG_SCHEMA},
        upstream_bindings,
        _UPSTREAM_ROLES,
    )


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _index(records, record_type, identity, name, failures):
    indexed = {}
    for record in records:
        if not isinstance(record, record_type):
            failures.add(f"{name}:invalid_record_type")
            continue
        try:
            record.validate()
        except (TypeError, ValueError):
            failures.add(f"{name}:invalid_record")
            continue
        key = identity(record)
        if key in indexed:
            failures.add(f"{name}:duplicate_identity")
        else:
            indexed[key] = record
    return indexed


def _verification_key(record):
    return record.generation_run_id, record.candidate_id


def evaluate_functional_gate(inputs: GateInputs) -> FunctionalGateReport:
    """Require complete success, exact verification coverage, and semantic joins."""
    failures: set[str] = set()
    plans = _index(
        inputs.plans, CandidatePlan, lambda record: record.key, "plans", failures
    )
    candidates = _index(
        inputs.candidates,
        CandidateRecord,
        lambda record: record.plan.key,
        "candidates",
        failures,
    )
    verified = _index(
        inputs.verifications,
        VerificationRecord,
        _verification_key,
        "verifications",
        failures,
    )
    questions = _index(
        inputs.question_verifications,
        QuestionVerificationRecord,
        lambda record: record.original_question_id,
        "question_verifications",
        failures,
    )
    pairs = _index(
        inputs.pairs,
        PairRecord,
        lambda record: record.original_question_id,
        "pairs",
        failures,
    )
    successful = set(plans) & set(candidates)
    completion = _rate(len(successful), len(plans))
    if completion != 1.0:
        failures.add("generation_completion:below_100_percent")
    if set(candidates) - set(plans):
        failures.add("candidates:unplanned_identity")
    if len({plan.generation_run_id for plan in plans.values()}) > 1:
        failures.add("plans:mixed_generation_runs")
    if len({plan.split_manifest_hash for plan in plans.values()}) > 1:
        failures.add("plans:mixed_split_lineage")
    planned_questions = {}
    for plan in plans.values():
        if (
            plan.schema_version != "candidate_plan_v1"
            or plan.candidate_id
            != candidate_id(
                plan.original_question_id,
                plan.generator_id,
                plan.sampling_index,
            )
        ):
            failures.add("plans:invalid_candidate_identity")
        metadata = (
            plan.source,
            plan.question_content_hash,
            plan.split,
            plan.split_manifest_hash,
        )
        previous = planned_questions.setdefault(plan.original_question_id, metadata)
        if previous != metadata:
            failures.add("plans:conflicting_question_metadata")
    for key, candidate in candidates.items():
        if key in plans and candidate.plan != plans[key]:
            failures.add("candidates:plan_metadata_mismatch")
    if set(questions) != set(planned_questions):
        failures.add("question_verifications:incomplete_or_foreign_coverage")
    verification_runs = {record.verification_run_id for record in verified.values()}
    verification_runs.update(
        record.verification_run_id for record in questions.values()
    )
    if len(verification_runs) != 1:
        failures.add("verifications:requires_one_run")
    expected_verifications = set()
    for key in successful:
        plan = plans[key]
        question = questions.get(plan.original_question_id)
        if question is None:
            continue
        if question.source is not plan.source:
            failures.add("question_verifications:source_mismatch")
        if question.gold_parse_status == "ok":
            expected_verifications.add(key)
    if set(verified) != expected_verifications:
        failures.add("verifications:incomplete_or_foreign_coverage")
    for key, record in verified.items():
        plan = plans.get(key)
        if plan is None or key not in successful:
            failures.add("verifications:unknown_successful_candidate")
        elif (record.original_question_id, record.source, record.generator_id) != (
            plan.original_question_id,
            plan.source,
            plan.generator_id,
        ):
            failures.add("verifications:provenance_mismatch")
    audit = inputs.audit_summary
    if (
        audit.required_count != audit.completed_count
        or sum(audit.verdict_counts.values()) != audit.completed_count
    ):
        failures.add("audit:incomplete_decisions")
    gold_error_count = sum(
        record.gold_parse_status == "gold_verification_error"
        for record in questions.values()
    )
    if audit.required_count < gold_error_count:
        failures.add("audit:missing_gold_error_coverage")
    _check_pairs(pairs, planned_questions, candidates, verified, failures)
    _check_permutations(inputs.permutations, pairs, failures)
    metrics = {
        "planned_count": len(plans),
        "successful_count": len(successful),
        "missing_count": len(set(plans) - successful),
        "generation_completion": completion,
        "question_verification_count": len(questions),
        "candidate_verification_count": len(verified),
        "gold_verification_error_count": gold_error_count,
        "pair_count": len(pairs),
        "permutation_count": len(inputs.permutations),
        "audit_required_count": audit.required_count,
        "audit_completed_count": audit.completed_count,
    }
    return FunctionalGateReport(not failures, tuple(sorted(failures)), metrics)


def _check_pairs(pairs, planned_questions, candidates, verified, failures):
    if len({pair.pair_run_id for pair in pairs.values()}) > 1:
        failures.add("pairs:mixed_pair_runs")
    # The actual pair tokenizer is unavailable here; validate its pinned contract
    # and response identities, rather than re-tokenizing with generation counts.
    by_question = {}
    for key in set(candidates) & set(verified):
        by_question.setdefault(candidates[key].plan.original_question_id, []).append(
            (candidates[key], verified[key])
        )
    for question_id, pair in pairs.items():
        if planned_questions.get(question_id) != (
            pair.source,
            pair.question_content_hash,
            pair.split,
            pair.split_manifest_hash,
        ):
            failures.add("pairs:question_lineage_mismatch")
        if pair.schema_version != PAIR_SCHEMA_VERSION or pair.pair_id != pair_id(
            pair.generation_run_id,
            question_id,
            pair.positive_candidate_id,
            pair.negative_candidate_id,
            pair.schema_version,
        ):
            failures.add("pairs:semantic_identity_mismatch")
        try:
            config = pair_stage_config(
                pair.tokenizer_revision,
                {"generation": "0" * 64, "verification": "0" * 64},
            )
            if pair.tokenizer_repo != config["parameters"]["tokenizer_repo"]:
                failures.add("pairs:tokenizer_contract_mismatch")
            groups = build_dedup_groups(by_question.get(question_id, ()))
        except (TypeError, ValueError):
            failures.add("pairs:invalid_verified_candidates")
            continue
        eligible = {group.representative_key: group for group in groups}
        for side, status in (
            ("positive", VerificationStatus.CORRECT),
            ("negative", VerificationStatus.INCORRECT),
        ):
            key = pair.generation_run_id, getattr(pair, f"{side}_candidate_id")
            candidate, verification, group = (
                candidates.get(key),
                verified.get(key),
                eligible.get(key),
            )
            if candidate is None or verification is None or group is None:
                failures.add(f"pairs:{side}_not_eligible_representative")
                continue
            plan = candidate.plan
            if (
                plan.original_question_id,
                plan.source,
                plan.split,
                plan.split_manifest_hash,
                plan.question_content_hash,
                plan.generator_id,
            ) != (
                question_id,
                pair.source,
                pair.split,
                pair.split_manifest_hash,
                pair.question_content_hash,
                getattr(pair, f"{side}_generator_id"),
            ) or verification.verification_run_id != pair.verification_run_id:
                failures.add(f"pairs:{side}_provenance_mismatch")
            if verification.verification_status is not status:
                failures.add(f"pairs:{side}_binary_label_mismatch")
            digest = sha256_hex(normalize_text_v1(candidate.response).encode("utf-8"))
            if digest != getattr(pair, f"{side}_response_hash"):
                failures.add(f"pairs:{side}_response_hash_mismatch")


def _check_permutations(records, pairs, failures):
    by_pair = {pair.pair_id: pair for pair in pairs.values()}
    indexed = _index(
        records,
        PermutationRecord,
        lambda record: (record.pair_id, record.permutation_id),
        "permutations",
        failures,
    )
    expected = {(pair_id, index) for pair_id in by_pair for index in (0, 1)}
    if set(indexed) != expected:
        failures.add("permutations:incomplete_or_foreign_coverage")
    try:
        permutation_manifest_bytes(tuple(indexed.values()))
    except (TypeError, ValueError):
        failures.add("permutations:invalid_mirrored_contract")
    for (semantic_pair_id, index), record in indexed.items():
        pair = by_pair.get(semantic_pair_id)
        if pair is None:
            continue
        surfaces = (pair.positive_candidate_id, pair.negative_candidate_id)
        if index == 1:
            surfaces = surfaces[::-1]
        if (
            record.generation_run_id,
            record.original_question_id,
            record.split,
            record.split_manifest_hash,
            record.surface_a_candidate_id,
            record.surface_b_candidate_id,
        ) != (
            pair.generation_run_id,
            pair.original_question_id,
            pair.split,
            pair.split_manifest_hash,
            *surfaces,
        ):
            failures.add("permutations:pair_lineage_or_surface_mismatch")


def evaluate_statistical_gate(inputs: GateInputs) -> StatisticalGateReport:
    """Apply exact v1 thresholds; ambiguous outcomes contribute diagnostics only."""
    functional = evaluate_functional_gate(inputs)
    hard_failures = set(functional.failures)
    warnings: set[str] = set()
    # Invalid inputs cannot pass. Only planned unique successes and their joined
    # verification records enter statistics, never question-level gold errors.
    ignored: set[str] = set()
    plans = _index(
        inputs.plans, CandidatePlan, lambda record: record.key, "plans", ignored
    )
    candidates = _index(
        inputs.candidates,
        CandidateRecord,
        lambda record: record.plan.key,
        "candidates",
        ignored,
    )
    verified = _index(
        inputs.verifications,
        VerificationRecord,
        _verification_key,
        "verifications",
        ignored,
    )
    pairs = _index(
        inputs.pairs,
        PairRecord,
        lambda record: record.original_question_id,
        "pairs",
        ignored,
    )
    cells = {(plan.source.value, plan.generator_id) for plan in plans.values()}
    generators = sorted({generator for _, generator in cells})
    counts = {
        cell: Counter({"successful": 0, **dict.fromkeys(_LABELS, 0)}) for cell in cells
    }
    for key in set(plans) & set(candidates):
        plan = plans[key]
        counts[plan.source.value, plan.generator_id]["successful"] += 1
        record = verified.get(key)
        if record is not None:
            counts[plan.source.value, plan.generator_id][
                record.verification_status.value
            ] += 1
    invalid_rates, correct_rates, cell_counts = {}, {}, {}
    for source in Source:
        source_name = source.value
        (
            invalid_rates[source_name],
            correct_rates[source_name],
            cell_counts[source_name],
        ) = {}, {}, {}
        for generator in generators:
            cell = counts.get((source_name, generator), Counter())
            cell_counts[source_name][generator] = dict(cell)
            bad, total = cell["invalid"] + cell["error"], cell["successful"]
            invalid_rates[source_name][generator] = _rate(bad, total)
            if total and bad * 5 > total:
                hard_failures.add(
                    f"invalid_error_rate:{source_name}/{generator}:above_20_percent"
                )
            binary = cell["correct"] + cell["incorrect"]
            correct_rates[source_name][generator] = _rate(cell["correct"], binary)
            if binary and cell["correct"] in (0, binary):
                warnings.add(f"correct_rate:{source_name}/{generator}:binary_extreme")
    source_pairs = {}
    pair_yield = {}
    for source in Source:
        planned_questions = {
            plan.original_question_id
            for plan in plans.values()
            if plan.source is source
        }
        selected = [
            pair
            for question, pair in pairs.items()
            if pair.source is source and question in planned_questions
        ]
        source_pairs[source.value] = selected
        numerator, denominator = len(selected), len(planned_questions)
        pair_yield[source.value] = _rate(numerator, denominator)
        if not denominator or numerator * 5 < denominator:
            hard_failures.add(f"pair_yield:{source.value}:below_20_percent")
        if denominator and numerator * 10 < denominator * 3:
            warnings.add(f"pair_yield:{source.value}:below_30_percent")
    selected_pairs = [pair for source in Source for pair in source_pairs[source.value]]
    shares = {}
    scopes = {"pooled": selected_pairs, **source_pairs}
    for scope, selected in scopes.items():
        shares[scope] = {}
        for side in ("positive", "negative"):
            supplied = Counter(
                getattr(pair, f"{side}_generator_id") for pair in selected
            )
            shares[scope][side] = {
                generator: _rate(supplied[generator], len(selected))
                for generator in generators
            }
            for generator in generators:
                numerator, denominator = supplied[generator], len(selected)
                if not denominator:
                    continue
                prefix = f"generator_share:{scope}/{side}/{generator}"
                if scope == "pooled" and numerator * 10 >= denominator * 9:
                    hard_failures.add(f"{prefix}:at_least_90_percent")
                if numerator * 10 > denominator * 7:
                    warnings.add(f"{prefix}:above_70_percent")
                if scope != "pooled" and numerator * 10 >= denominator * 9:
                    warnings.add(f"{prefix}:strong_at_least_90_percent")
    audit = inputs.audit_summary
    if audit.confirmed_disagreements:
        hard_failures.add("audit:confirmed_binary_disagreement")
    if audit.systematic_issue:
        hard_failures.add("audit:systematic_issue")
    pooled_counts = Counter()
    for cell in counts.values():
        pooled_counts.update(cell)
    table = tuple(
        tuple(
            sum(
                counts.get((source.value, generator), Counter())[label]
                for source in Source
            )
            for label in _LABELS
        )
        for generator in generators
    )
    same_generator = sum(
        pair.positive_generator_id == pair.negative_generator_id
        for pair in selected_pairs
    )
    diagnostics = {
        "generation_completion": functional.metrics["generation_completion"],
        "pair_yield": pair_yield,
        "cell_counts": cell_counts,
        "invalid_error_rate": invalid_rates,
        "correct_rate": correct_rates,
        "generator_shares": shares,
        "gold_verification_error_count": functional.metrics[
            "gold_verification_error_count"
        ],
        "ambiguous_rate": _rate(
            pooled_counts["ambiguous"], pooled_counts["successful"]
        ),
        "ambiguous_rate_by_cell": {
            source.value: {
                generator: _rate(
                    counts.get((source.value, generator), Counter())["ambiguous"],
                    counts.get((source.value, generator), Counter())["successful"],
                )
                for generator in generators
            }
            for source in Source
        },
        "contingency_generators": tuple(generators),
        "contingency_labels": _LABELS,
        "contingency_table": table,
        "cramers_v": cramers_v(table),
        "same_generator_pair_proportion": _rate(same_generator, len(selected_pairs)),
        "cross_generator_pair_proportion": _rate(
            len(selected_pairs) - same_generator, len(selected_pairs)
        ),
        "source_correct_rate": {
            source.value: _binary_rate(
                [cell for (name, _), cell in counts.items() if name == source.value],
            )
            for source in Source
        },
        "generator_correct_rate": {
            generator: _binary_rate(
                [cell for (_, name), cell in counts.items() if name == generator],
            )
            for generator in generators
        },
        "response_token_lengths": tuple(
            sorted(
                candidates[key].generated_token_count
                for key in set(plans) & set(candidates)
            )
        ),
        "positive_negative_token_gaps": tuple(
            sorted(
                pair.positive_token_count - pair.negative_token_count
                for pair in selected_pairs
            )
        ),
    }
    status = (
        GateStatus.FAIL
        if hard_failures
        else GateStatus.PASS_WITH_WARNINGS
        if warnings
        else GateStatus.PASS
    )
    return StatisticalGateReport(
        status, tuple(sorted(hard_failures)), tuple(sorted(warnings)), diagnostics
    )


def _binary_rate(cells) -> float | None:
    correct = sum(cell["correct"] for cell in cells)
    incorrect = sum(cell["incorrect"] for cell in cells)
    return _rate(correct, correct + incorrect)


def cramers_v(table: Sequence[Sequence[int]]) -> float | None:
    """Return descriptive Cramer's V, dropping zero margins; never infer significance."""
    rows = [tuple(row) for row in table]
    if not rows:
        return None
    columns = len(rows[0])
    if any(len(row) != columns for row in rows) or any(
        type(value) is not int or value < 0 for row in rows for value in row
    ):
        raise ValueError(
            "contingency table must be rectangular nonnegative integer counts"
        )
    if not columns:
        return None
    rows = [row for row in rows if sum(row)]
    nonzero_columns = [
        index for index in range(columns) if any(row[index] for row in rows)
    ]
    rows = [tuple(row[index] for index in nonzero_columns) for row in rows]
    if len(rows) < 2 or len(nonzero_columns) < 2:
        return None
    row_totals = [sum(row) for row in rows]
    column_totals = [
        sum(row[index] for row in rows) for index in range(len(nonzero_columns))
    ]
    total = sum(row_totals)
    chi_squared = 0.0
    for row, row_total in zip(rows, row_totals, strict=True):
        for observed, column_total in zip(row, column_totals, strict=True):
            expected = row_total * column_total / total
            chi_squared += (observed - expected) ** 2 / expected
    return math.sqrt(
        chi_squared / (total * min(len(rows) - 1, len(nonzero_columns) - 1))
    )
