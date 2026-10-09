"""Deterministic human-verifier audit selection and decision validation."""

import subprocess
import sys
from dataclasses import replace

import pytest

from permstudy.data_pipeline.canonical import canonical_json_bytes, sha256_hex
from permstudy.data_pipeline.schema import (
    AuditDecision,
    AuditVerdict,
    QuestionVerificationRecord,
    Source,
    VerificationRecord,
    VerificationStatus,
)


GENERATION_RUN_ID = "1" * 64
VERIFICATION_RUN_ID = "2" * 64


def _candidate(
    candidate_id,
    status,
    *,
    source=Source.RECLOR,
    generator_id="qwen2.5-7b-instruct",
    question_id=None,
    error_type=None,
):
    question_id = question_id or f"{source.value}:train:{candidate_id}"
    parsed = status in (VerificationStatus.CORRECT, VerificationStatus.INCORRECT)
    return VerificationRecord(
        generation_run_id=GENERATION_RUN_ID,
        verification_run_id=VERIFICATION_RUN_ID,
        candidate_id=candidate_id,
        original_question_id=question_id,
        source=source,
        generator_id=generator_id,
        verification_status=status,
        prediction_parse_status="parsed" if parsed else status.value,
        canonical_prediction="A" if parsed else None,
        verifier_name="synthetic_verifier",
        verifier_version="synthetic_verifier_v1",
        parser_version="synthetic_parser_v1",
        verifier_timeout_seconds=1.0,
        error_type=error_type,
    )


def _question(candidate=None, *, question_id=None, source=None, gold_error=None):
    if candidate is not None:
        question_id = candidate.original_question_id
        source = candidate.source
    return QuestionVerificationRecord(
        verification_run_id=VERIFICATION_RUN_ID,
        original_question_id=question_id,
        source=source,
        gold_parse_status="gold_verification_error" if gold_error else "ok",
        canonical_gold=None if gold_error else "A",
        error_type=gold_error,
    )


def _selection_inputs():
    correct = [
        _candidate(f"correct_{index}", VerificationStatus.CORRECT) for index in range(7)
    ]
    incorrect = [
        _candidate(
            f"incorrect_{index}",
            VerificationStatus.INCORRECT,
            source=Source.MATH,
            generator_id="llama-3.1-8b-instruct",
        )
        for index in range(3)
    ]
    exceptional = [
        _candidate(
            "invalid_1",
            VerificationStatus.INVALID,
            error_type="missing_final_marker",
        ),
        _candidate(
            "ambiguous_1",
            VerificationStatus.AMBIGUOUS,
            error_type="conflicting_markers",
        ),
        _candidate(
            "error_1",
            VerificationStatus.ERROR,
            source=Source.MATH,
            error_type="timeout",
        ),
    ]
    candidates = correct + incorrect + exceptional
    questions = [_question(candidate) for candidate in candidates]
    questions.extend(
        [
            _question(
                question_id="math:train:gold_error_1",
                source=Source.MATH,
                gold_error="gold_parse_failure",
            ),
            _question(
                question_id="reclor:train:gold_error_2",
                source=Source.RECLOR,
                gold_error="dependency_error",
            ),
        ]
    )
    return candidates, questions


def _score(candidate_id, audit_seed=42):
    return sha256_hex(
        canonical_json_bytes(
            {
                "audit_seed": audit_seed,
                "candidate_id": candidate_id,
                "generation_run_id": GENERATION_RUN_ID,
                "schema": "audit_sample_score_v1",
                "verification_run_id": VERIFICATION_RUN_ID,
            }
        )
    )


def test_selection_is_exhaustive_for_exceptions_and_sha256_sampled_per_cell():
    from permstudy.data_pipeline.audit import build_audit_selection

    candidates, questions = _selection_inputs()

    selected = build_audit_selection(
        tuple(reversed(candidates)),
        tuple(reversed(questions)),
        GENERATION_RUN_ID,
        audit_seed=42,
        max_per_cell=5,
    )
    repeated = build_audit_selection(
        candidates,
        questions,
        GENERATION_RUN_ID,
        audit_seed=42,
        max_per_cell=5,
    )

    assert selected == repeated
    assert len(selected) == 13
    candidate_selections = [
        record for record in selected if record.record_kind == "candidate"
    ]
    gold_selections = [
        record for record in selected if record.record_kind == "question_gold"
    ]
    expected_correct = {
        record.candidate_id
        for record in sorted(
            candidates[:7],
            key=lambda record: (_score(record.candidate_id), record.candidate_id),
        )[:5]
    }
    selected_correct = {
        record.candidate_id
        for record in candidate_selections
        if record.verification_status is VerificationStatus.CORRECT
    }
    assert selected_correct == expected_correct
    assert {
        record.candidate_id
        for record in candidate_selections
        if record.verification_status is VerificationStatus.INCORRECT
    } == {"incorrect_0", "incorrect_1", "incorrect_2"}
    assert {record.candidate_id for record in candidate_selections} >= {
        "invalid_1",
        "ambiguous_1",
        "error_1",
    }
    assert {
        record.reason_code
        for record in candidate_selections
        if record.candidate_id in {"invalid_1", "ambiguous_1", "error_1"}
    } == {
        "missing_final_marker",
        "conflicting_markers",
        "timeout",
    }
    assert len(gold_selections) == 2
    assert {record.original_question_id for record in gold_selections} == {
        "math:train:gold_error_1",
        "reclor:train:gold_error_2",
    }
    assert all(
        record.candidate_id is None
        and record.generator_id is None
        and record.verification_status is None
        and record.gold_parse_status == "gold_verification_error"
        for record in gold_selections
    )
    assert len({record.audit_run_id for record in selected}) == 1
    assert {
        (record.generation_run_id, record.verification_run_id) for record in selected
    } == {(GENERATION_RUN_ID, VERIFICATION_RUN_ID)}


def test_selection_seed_and_cell_limit_are_bound_into_audit_run_identity():
    from permstudy.data_pipeline.audit import build_audit_selection

    candidates, questions = _selection_inputs()

    baseline = build_audit_selection(candidates, questions, GENERATION_RUN_ID)
    changed_seed = build_audit_selection(
        candidates, questions, GENERATION_RUN_ID, audit_seed=43
    )
    changed_limit = build_audit_selection(
        candidates, questions, GENERATION_RUN_ID, max_per_cell=4
    )

    assert baseline[0].audit_run_id != changed_seed[0].audit_run_id
    assert baseline[0].audit_run_id != changed_limit[0].audit_run_id


def test_audit_run_identity_binds_the_complete_verification_snapshot():
    from permstudy.data_pipeline.audit import build_audit_selection

    candidates, questions = _selection_inputs()
    baseline = build_audit_selection(candidates, questions, GENERATION_RUN_ID)
    selected_ids = {
        record.candidate_id for record in baseline if record.record_kind == "candidate"
    }
    omitted = next(
        record for record in candidates if record.candidate_id not in selected_ids
    )
    reduced_candidates = [record for record in candidates if record is not omitted]
    reduced_questions = [
        record
        for record in questions
        if record.original_question_id != omitted.original_question_id
    ]

    repeated = build_audit_selection(candidates, questions, GENERATION_RUN_ID)
    reduced = build_audit_selection(
        reduced_candidates,
        reduced_questions,
        GENERATION_RUN_ID,
    )

    assert baseline == repeated
    assert {record.candidate_id for record in baseline} == {
        record.candidate_id for record in reduced
    }
    assert baseline[0].audit_run_id != reduced[0].audit_run_id


def test_audit_stage_config_role_binds_verification_snapshot():
    from permstudy.data_pipeline.audit import audit_stage_config
    from permstudy.data_pipeline.ids import run_id

    snapshot_hash = "3" * 64
    baseline = audit_stage_config(
        GENERATION_RUN_ID,
        VERIFICATION_RUN_ID,
        snapshot_hash,
        audit_seed=42,
        max_per_cell=5,
    )
    repeated = audit_stage_config(
        GENERATION_RUN_ID,
        VERIFICATION_RUN_ID,
        snapshot_hash,
        audit_seed=42,
        max_per_cell=5,
    )
    changed_snapshot = audit_stage_config(
        GENERATION_RUN_ID,
        VERIFICATION_RUN_ID,
        "4" * 64,
        audit_seed=42,
        max_per_cell=5,
    )

    assert baseline["parameters"]["verification_run_id"] == VERIFICATION_RUN_ID
    assert baseline["upstream_bindings"] == {"verification": snapshot_hash}
    assert run_id("audit", baseline) == run_id("audit", repeated)
    assert run_id("audit", baseline) != run_id("audit", changed_snapshot)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("mixed_generation", "generation run"),
        ("mixed_verification", "verification run"),
        ("duplicate_candidate", "duplicate candidate"),
        ("duplicate_question", "duplicate question"),
        ("missing_question", "question verification"),
        ("source_mismatch", "source"),
    ],
)
def test_selection_fails_closed_on_identity_or_completeness_mismatch(mutation, message):
    from permstudy.data_pipeline.audit import AuditIntegrityError, build_audit_selection

    candidates, questions = _selection_inputs()
    if mutation == "mixed_generation":
        candidates[0] = replace(candidates[0], generation_run_id="3" * 64)
    elif mutation == "mixed_verification":
        candidates[0] = replace(candidates[0], verification_run_id="3" * 64)
    elif mutation == "duplicate_candidate":
        candidates.append(candidates[0])
    elif mutation == "duplicate_question":
        questions.append(questions[0])
    elif mutation == "missing_question":
        questions = [
            record
            for record in questions
            if record.original_question_id != candidates[0].original_question_id
        ]
    else:
        questions[0] = replace(questions[0], source=Source.MATH)

    with pytest.raises(AuditIntegrityError, match=message):
        build_audit_selection(candidates, questions, GENERATION_RUN_ID)


def _decision(selection, verdict, *, reason_code="other", confirmed=False):
    return AuditDecision(
        audit_run_id=selection.audit_run_id,
        generation_run_id=selection.generation_run_id,
        verification_run_id=selection.verification_run_id,
        record_kind=selection.record_kind,
        original_question_id=selection.original_question_id,
        candidate_id=selection.candidate_id,
        verdict=verdict,
        reason_code=reason_code,
        confirmed_disagree=confirmed,
    )


def test_decisions_match_selection_and_keep_disagree_unsure_and_confirmation_distinct():
    from permstudy.data_pipeline.audit import (
        build_audit_selection,
        validate_audit_decisions,
    )

    candidates = [
        _candidate("correct", VerificationStatus.CORRECT),
        _candidate("incorrect", VerificationStatus.INCORRECT, source=Source.MATH),
        _candidate(
            "invalid",
            VerificationStatus.INVALID,
            error_type="prompt_contract_mismatch",
        ),
    ]
    questions = [_question(candidate) for candidate in candidates]
    selection = build_audit_selection(candidates, questions, GENERATION_RUN_ID)
    by_id = {record.candidate_id: record for record in selection}
    decisions = [
        _decision(by_id["correct"], AuditVerdict.AGREE),
        _decision(
            by_id["incorrect"],
            AuditVerdict.DISAGREE,
            reason_code="other",
            confirmed=True,
        ),
        _decision(
            by_id["invalid"],
            AuditVerdict.UNSURE,
            reason_code="prompt_contract_mismatch",
        ),
    ]

    summary = validate_audit_decisions(selection, tuple(reversed(decisions)))

    assert summary.required_count == 3
    assert summary.completed_count == 3
    assert dict(summary.verdict_counts) == {"AGREE": 1, "DISAGREE": 1, "UNSURE": 1}
    assert summary.confirmed_disagreements == 1
    assert summary.systematic_issue is False


def test_confirmed_disagree_is_limited_to_binary_labels_and_systematic_disagree_is_separate():
    from permstudy.data_pipeline.audit import (
        AuditIntegrityError,
        build_audit_selection,
        validate_audit_decisions,
    )

    invalid = _candidate(
        "invalid",
        VerificationStatus.INVALID,
        error_type="prompt_contract_mismatch",
    )
    selection = build_audit_selection(
        [invalid], [_question(invalid)], GENERATION_RUN_ID
    )
    disagree = _decision(
        selection[0],
        AuditVerdict.DISAGREE,
        reason_code="prompt_contract_mismatch",
    )

    summary = validate_audit_decisions(selection, [disagree])
    assert summary.confirmed_disagreements == 0
    assert summary.systematic_issue is True

    with pytest.raises(AuditIntegrityError, match="binary"):
        validate_audit_decisions(
            selection,
            [replace(disagree, confirmed_disagree=True)],
        )

    binary = _candidate("binary", VerificationStatus.CORRECT)
    binary_selection = build_audit_selection(
        [binary], [_question(binary)], GENERATION_RUN_ID
    )
    first_review = _decision(
        binary_selection[0],
        AuditVerdict.DISAGREE,
        reason_code="prompt_contract_mismatch",
    )
    binary_summary = validate_audit_decisions(binary_selection, [first_review])
    assert binary_summary.confirmed_disagreements == 0
    assert binary_summary.systematic_issue is False


def test_candidate_identity_excludes_question_and_audit_run_metadata():
    from permstudy.data_pipeline.audit import (
        AuditIntegrityError,
        build_audit_selection,
        validate_audit_decisions,
    )

    candidate = _candidate("candidate", VerificationStatus.CORRECT)
    selection = build_audit_selection(
        [candidate], [_question(candidate)], GENERATION_RUN_ID
    )
    conflicting_selection = replace(
        selection[0],
        audit_run_id="f" * 64,
        original_question_id="reclor:train:different-question",
    )

    with pytest.raises(AuditIntegrityError, match="duplicate identity"):
        validate_audit_decisions([selection[0], conflicting_selection], [])


def test_decision_validation_rejects_mixed_audit_run_namespaces():
    from permstudy.data_pipeline.audit import (
        AuditIntegrityError,
        build_audit_selection,
        validate_audit_decisions,
    )

    first = _candidate("first", VerificationStatus.CORRECT)
    first_selection = build_audit_selection(
        [first], [_question(first)], GENERATION_RUN_ID
    )
    second_generation_run = "9" * 64
    second = replace(
        _candidate("second", VerificationStatus.INCORRECT),
        generation_run_id=second_generation_run,
    )
    second_selection = build_audit_selection(
        [second], [_question(second)], second_generation_run
    )
    combined = first_selection + second_selection
    decisions = [_decision(record, AuditVerdict.AGREE) for record in combined]

    with pytest.raises(AuditIntegrityError, match="mixed audit run"):
        validate_audit_decisions(combined, decisions)


def test_audit_import_is_independent_of_trainer_export_and_parquet_dependencies():
    script = """
import builtins

real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    blocked = ("permstudy.data_pipeline.trainer_export", "pandas", "pyarrow")
    if any(name == prefix or name.startswith(prefix + ".") for prefix in blocked):
        raise AssertionError(f"blocked dependency imported: {name}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from permstudy.data_pipeline.audit import build_audit_selection
assert callable(build_audit_selection)
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "missing"),
        ("duplicate", "duplicate"),
        ("foreign", "foreign"),
        ("wrong_run", "foreign"),
        ("bad_verdict", "verdict"),
        ("bad_reason", "reason_code"),
    ],
)
def test_decision_validation_rejects_missing_duplicate_foreign_or_invalid_entries(
    mutation, message
):
    from permstudy.data_pipeline.audit import (
        AuditIntegrityError,
        build_audit_selection,
        validate_audit_decisions,
    )

    candidates = [
        _candidate("first", VerificationStatus.CORRECT),
        _candidate("second", VerificationStatus.INCORRECT, source=Source.MATH),
    ]
    selection = build_audit_selection(
        candidates,
        [_question(candidate) for candidate in candidates],
        GENERATION_RUN_ID,
    )
    decisions = [_decision(record, AuditVerdict.AGREE) for record in selection]
    if mutation == "missing":
        decisions.pop()
    elif mutation == "duplicate":
        decisions.append(decisions[0])
    elif mutation == "foreign":
        decisions[0] = replace(decisions[0], candidate_id="foreign")
    elif mutation == "wrong_run":
        decisions[0] = replace(decisions[0], audit_run_id="f" * 64)
    elif mutation == "bad_verdict":
        decisions[0] = replace(decisions[0], verdict="MAYBE")
    else:
        decisions[0] = replace(decisions[0], reason_code="not_approved")

    with pytest.raises((AuditIntegrityError, ValueError), match=message):
        validate_audit_decisions(selection, decisions)
