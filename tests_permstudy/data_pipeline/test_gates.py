"""Canonical-record gate contracts, threshold boundaries, and diagnostics."""

from dataclasses import FrozenInstanceError, replace
import math
import subprocess
import sys

import pytest

from permstudy.data_pipeline.audit import AuditSummary
from permstudy.data_pipeline.canonical import normalize_text_v1, sha256_hex
from permstudy.data_pipeline.gates import (
    GateInputs,
    cramers_v,
    evaluate_functional_gate,
    evaluate_statistical_gate,
    gate_stage_config,
)
from permstudy.data_pipeline.ids import candidate_id, pair_id, run_id
from permstudy.data_pipeline.permutations import build_permutations
from permstudy.data_pipeline.schema import (
    CandidatePlan,
    CandidateRecord,
    GateStatus,
    PairRecord,
    QuestionVerificationRecord,
    Source,
    Split,
    VerificationRecord,
    VerificationStatus,
)


HASH = "1" * 64
VERIFY_RUN = "2" * 64
REVISION = "a" * 40


def audit_summary(**changes):
    values = dict(
        required_count=0,
        completed_count=0,
        verdict_counts={"AGREE": 0, "DISAGREE": 0, "UNSURE": 0},
        confirmed_disagreements=0,
        systematic_issue=False,
    )
    values.update(changes)
    return AuditSummary(**values)


def make_inputs(*, questions=100, selected=100, shares=None, ambiguous=0):
    """Each cell starts with two successful candidates per planned question."""
    plans, candidates, verifications, gold, pairs, permutations = [], [], [], [], [], []
    for source in Source:
        for index in range(questions):
            question = f"{source.value}:train:{index}"
            gold.append(
                QuestionVerificationRecord(
                    VERIFY_RUN, question, source, "ok", "A", None
                )
            )
            by_generator = {}
            for generator in ("g0", "g1"):
                records = []
                for sample in range(2):
                    plan = CandidatePlan(
                        "candidate_plan_v1",
                        HASH,
                        candidate_id(question, generator, sample),
                        question,
                        HASH,
                        source,
                        Split.TRAIN,
                        HASH,
                        generator,
                        sample,
                        "00000",
                        HASH,
                    )
                    generated = CandidateRecord(
                        plan,
                        f"{question} {generator} reasoning {sample}\nFinal Answer: A",
                        "stop",
                        10 + sample,
                        REVISION,
                    )
                    verified = VerificationRecord(
                        HASH,
                        VERIFY_RUN,
                        plan.candidate_id,
                        question,
                        source,
                        generator,
                        VerificationStatus.CORRECT
                        if sample == 0
                        else VerificationStatus.INCORRECT,
                        "parsed",
                        "A",
                        "test_verifier",
                        "v1",
                        "v1",
                        1.0,
                        None,
                    )
                    plans.append(plan)
                    candidates.append(generated)
                    verifications.append(verified)
                    records.append(generated)
                by_generator[generator] = records
            count = selected[source.value] if isinstance(selected, dict) else selected
            if index < count:
                cutoff = shares[source.value] if shares else count // 2
                pos_generator = "g0" if index < cutoff else "g1"
                neg_generator = "g1" if index < cutoff else "g0"
                positive = by_generator[pos_generator][0]
                negative = by_generator[neg_generator][1]
                pair = PairRecord(
                    "pair_v1",
                    "3" * 64,
                    pair_id(
                        HASH,
                        question,
                        positive.plan.candidate_id,
                        negative.plan.candidate_id,
                        "pair_v1",
                    ),
                    HASH,
                    VERIFY_RUN,
                    question,
                    HASH,
                    source,
                    Split.TRAIN,
                    HASH,
                    positive.plan.candidate_id,
                    negative.plan.candidate_id,
                    sha256_hex(normalize_text_v1(positive.response).encode()),
                    sha256_hex(normalize_text_v1(negative.response).encode()),
                    pos_generator,
                    neg_generator,
                    10,
                    11,
                    "Qwen/Qwen2.5-7B-Instruct",
                    REVISION,
                )
                pairs.append(pair)
                permutations.extend(
                    build_permutations(pair, upstream_bindings={"pairs": HASH})
                )
    # Add genuine non-binary candidates, without changing any binary denominator.
    for index in range(ambiguous):
        base = candidates[index % len(candidates)]
        plan = replace(
            base.plan,
            sampling_index=2 + index,
            candidate_id=candidate_id(
                base.plan.original_question_id, base.plan.generator_id, 2 + index
            ),
        )
        plans.append(plan)
        candidates.append(replace(base, plan=plan))
        verifications.append(
            replace(
                verifications[index % (questions * 8)],
                candidate_id=plan.candidate_id,
                verification_status=VerificationStatus.AMBIGUOUS,
                canonical_prediction=None,
            )
        )
    return GateInputs(
        plans, candidates, verifications, gold, pairs, permutations, audit_summary()
    )


def test_complete_canonical_pipeline_passes_both_gates():
    inputs = make_inputs()
    functional = evaluate_functional_gate(inputs)
    statistical = evaluate_statistical_gate(inputs)
    assert functional.passed
    assert functional.failures == ()
    assert functional.metrics["generation_completion"] == 1.0
    assert statistical.status is GateStatus.PASS
    assert statistical.hard_failures == statistical.warnings == ()


@pytest.mark.parametrize(
    "missing, completion, passed", [(0, 1.0, True), (8, 0.99, False)]
)
def test_completion_uses_final_successful_composite_keys(missing, completion, passed):
    inputs = make_inputs()
    if missing:
        inputs = replace(inputs, candidates=inputs.candidates[:-missing])
    report = evaluate_functional_gate(inputs)
    assert report.metrics["generation_completion"] == completion
    assert report.passed is passed
    assert (evaluate_statistical_gate(inputs).status is GateStatus.FAIL) is (not passed)


@pytest.mark.parametrize(
    "yield_count, status",
    [
        (19, GateStatus.FAIL),
        (20, GateStatus.PASS_WITH_WARNINGS),
        (29, GateStatus.PASS_WITH_WARNINGS),
        (30, GateStatus.PASS),
    ],
)
def test_source_pair_yield_thresholds(yield_count, status):
    report = evaluate_statistical_gate(
        make_inputs(selected={"math": yield_count, "reclor": 100})
    )
    assert report.status is status
    assert report.diagnostics["pair_yield"]["math"] == yield_count / 100


@pytest.mark.parametrize(
    "exceptional, status", [(40, GateStatus.PASS), (41, GateStatus.FAIL)]
)
def test_candidate_invalid_error_threshold(exceptional, status):
    inputs = make_inputs(selected=30)
    records = list(inputs.verifications)
    indices = [
        i
        for i, record in enumerate(records)
        if record.source is Source.MATH and record.generator_id == "g0"
    ][-exceptional:]
    for number, index in enumerate(indices):
        records[index] = replace(
            records[index],
            verification_status=(
                VerificationStatus.INVALID if number % 2 else VerificationStatus.ERROR
            ),
        )
    report = evaluate_statistical_gate(replace(inputs, verifications=records))
    assert report.status is status
    assert report.diagnostics["invalid_error_rate"]["math"]["g0"] == exceptional / 200
    assert report.diagnostics["cell_counts"]["math"]["g0"]["successful"] == 200


@pytest.mark.parametrize(
    "share, status", [(89, GateStatus.PASS_WITH_WARNINGS), (90, GateStatus.FAIL)]
)
def test_pooled_positive_and_negative_dominance_thresholds(share, status):
    report = evaluate_statistical_gate(
        make_inputs(shares={"math": share, "reclor": share})
    )
    assert report.status is status
    assert (
        report.diagnostics["generator_shares"]["pooled"]["positive"]["g0"]
        == share / 100
    )
    assert (
        report.diagnostics["generator_shares"]["pooled"]["negative"]["g1"]
        == share / 100
    )


@pytest.mark.parametrize("share, warned", [(70, False), (71, True)])
def test_pooled_share_warning_boundary_is_strict(share, warned):
    report = evaluate_statistical_gate(
        make_inputs(shares={"math": share, "reclor": share})
    )
    assert (
        any("generator_share:pooled" in warning for warning in report.warnings)
        is warned
    )
    assert report.status is (
        GateStatus.PASS_WITH_WARNINGS if warned else GateStatus.PASS
    )


@pytest.mark.parametrize(
    "share, status, strong",
    [
        (70, GateStatus.PASS, False),
        (71, GateStatus.PASS_WITH_WARNINGS, False),
        (90, GateStatus.PASS_WITH_WARNINGS, True),
    ],
)
def test_per_source_dominance_is_warning_only(share, status, strong):
    report = evaluate_statistical_gate(
        make_inputs(shares={"math": share, "reclor": 100 - share})
    )
    assert report.status is status
    assert any("strong" in warning for warning in report.warnings) is strong


@pytest.mark.parametrize(
    "state, expected",
    [
        (VerificationStatus.INCORRECT, 0.0),
        (VerificationStatus.CORRECT, 1.0),
        (VerificationStatus.AMBIGUOUS, None),
    ],
)
def test_correct_rate_binary_extremes_and_undefined(state, expected):
    inputs = make_inputs(selected={"math": 0, "reclor": 100})
    records = [
        replace(record, verification_status=state)
        if record.source is Source.MATH
        else record
        for record in inputs.verifications
    ]
    report = evaluate_statistical_gate(replace(inputs, verifications=records))
    assert report.diagnostics["correct_rate"]["math"]["g0"] == expected
    assert any("correct_rate:math/g0" in warning for warning in report.warnings) is (
        expected is not None
    )


def test_ambiguous_rate_never_changes_v1_gate_status():
    low = evaluate_statistical_gate(make_inputs(ambiguous=0))
    high = evaluate_statistical_gate(make_inputs(ambiguous=200))
    assert low.status == high.status == GateStatus.PASS
    assert high.diagnostics["ambiguous_rate"] == 0.2
    assert high.diagnostics["ambiguous_rate"] > low.diagnostics["ambiguous_rate"]
    assert high.diagnostics["correct_rate"]["math"]["g0"] == 0.5


def test_gold_errors_are_question_audit_data_and_never_candidate_rate_data():
    inputs = make_inputs(selected=30)
    question = inputs.question_verifications[-1]
    gold_error = replace(
        question,
        gold_parse_status="gold_verification_error",
        canonical_gold=None,
        error_type="gold_parse_failure",
    )
    inputs = replace(
        inputs,
        question_verifications=(*inputs.question_verifications[:-1], gold_error),
        verifications=tuple(
            record
            for record in inputs.verifications
            if record.original_question_id != question.original_question_id
        ),
        audit_summary=audit_summary(
            required_count=1,
            completed_count=1,
            verdict_counts={"AGREE": 1, "DISAGREE": 0, "UNSURE": 0},
        ),
    )
    functional = evaluate_functional_gate(inputs)
    report = evaluate_statistical_gate(inputs)
    assert functional.passed
    assert functional.metrics["gold_verification_error_count"] == 1
    assert functional.metrics["audit_completed_count"] == 1
    assert report.diagnostics["gold_verification_error_count"] == 1
    assert report.diagnostics["invalid_error_rate"]["reclor"]["g0"] == 0
    assert report.diagnostics["cell_counts"]["reclor"]["g0"]["successful"] == 200
    assert report.diagnostics["cell_counts"]["reclor"]["g0"]["error"] == 0
    assert report.status is GateStatus.PASS


def test_gold_error_does_not_push_candidate_error_rate_over_boundary():
    inputs = make_inputs(selected=30)
    gold = replace(
        inputs.question_verifications[-1],
        gold_parse_status="gold_verification_error",
        canonical_gold=None,
        error_type="gold_parse_failure",
    )
    records = [
        record
        for record in inputs.verifications
        if record.original_question_id != gold.original_question_id
    ]
    indices = [
        i
        for i, record in enumerate(records)
        if record.source is Source.RECLOR and record.generator_id == "g0"
    ][-40:]
    for index in indices:
        records[index] = replace(
            records[index], verification_status=VerificationStatus.ERROR
        )
    report = evaluate_statistical_gate(
        replace(
            inputs,
            verifications=records,
            question_verifications=(*inputs.question_verifications[:-1], gold),
            audit_summary=audit_summary(
                required_count=1,
                completed_count=1,
                verdict_counts={"AGREE": 1, "DISAGREE": 0, "UNSURE": 0},
            ),
        )
    )
    assert report.status is GateStatus.PASS
    assert report.diagnostics["invalid_error_rate"]["reclor"]["g0"] == 0.2
    assert report.diagnostics["cell_counts"]["reclor"]["g0"]["error"] == 40
    assert report.diagnostics["cell_counts"]["reclor"]["g0"]["successful"] == 200


def test_zero_selected_pairs_leave_generator_dominance_undefined():
    report = evaluate_statistical_gate(make_inputs(selected=0))
    assert report.status is GateStatus.FAIL
    assert report.diagnostics["generator_shares"]["pooled"]["positive"] == {
        "g0": None,
        "g1": None,
    }
    assert report.diagnostics["generator_shares"]["math"]["negative"] == {
        "g0": None,
        "g1": None,
    }
    assert report.diagnostics["same_generator_pair_proportion"] is None


@pytest.mark.parametrize(
    "changes",
    [
        {"confirmed_disagreements": 1},
        {"systematic_issue": True},
    ],
)
def test_audit_hard_failures_override_warnings(changes):
    report = evaluate_statistical_gate(
        replace(
            make_inputs(selected=20),
            audit_summary=audit_summary(**changes),
        )
    )
    assert report.status is GateStatus.FAIL
    assert report.hard_failures
    assert report.warnings


@pytest.mark.parametrize(
    "kind",
    [
        "missing_question",
        "duplicate_question",
        "missing_verification",
        "duplicate_verification",
        "gold_with_candidates",
        "foreign_verification",
        "foreign_candidate",
        "duplicate_plan",
        "duplicate_candidate",
        "duplicate_pair",
        "wrong_label",
        "wrong_response_hash",
        "wrong_pair_generator",
        "foreign_permutation",
        "missing_permutation",
        "wrong_surface",
        "incomplete_audit",
        "mixed_verification_run",
        "plan_metadata_mismatch",
        "wrong_pair_hash",
    ],
)
def test_functional_contract_corruption_fails(kind):
    inputs = make_inputs(questions=4, selected=4)
    if kind == "missing_question":
        inputs = replace(
            inputs, question_verifications=inputs.question_verifications[1:]
        )
    elif kind == "duplicate_question":
        inputs = replace(
            inputs,
            question_verifications=(
                *inputs.question_verifications,
                inputs.question_verifications[0],
            ),
        )
    elif kind == "missing_verification":
        inputs = replace(inputs, verifications=inputs.verifications[1:])
    elif kind == "duplicate_verification":
        inputs = replace(
            inputs, verifications=(*inputs.verifications, inputs.verifications[0])
        )
    elif kind == "gold_with_candidates":
        inputs = replace(
            inputs,
            question_verifications=(
                replace(
                    inputs.question_verifications[0],
                    gold_parse_status="gold_verification_error",
                    canonical_gold=None,
                    error_type="gold_parse_failure",
                ),
                *inputs.question_verifications[1:],
            ),
        )
    elif kind in {"foreign_verification", "mixed_verification_run", "wrong_label"}:
        changes = {
            "foreign_verification": {"generation_run_id": "f" * 64},
            "mixed_verification_run": {"verification_run_id": "f" * 64},
            "wrong_label": {"verification_status": VerificationStatus.INCORRECT},
        }[kind]
        inputs = replace(
            inputs,
            verifications=(
                replace(inputs.verifications[0], **changes),
                *inputs.verifications[1:],
            ),
        )
    elif kind in {"foreign_candidate", "plan_metadata_mismatch"}:
        changes = (
            {"generation_run_id": "f" * 64}
            if kind == "foreign_candidate"
            else {"prompt_hash": "f" * 64}
        )
        inputs = replace(
            inputs,
            candidates=(
                replace(inputs.candidates[0], plan=replace(inputs.plans[0], **changes)),
                *inputs.candidates[1:],
            ),
        )
    elif kind in {"duplicate_plan", "duplicate_candidate", "duplicate_pair"}:
        field = {
            "duplicate_plan": "plans",
            "duplicate_candidate": "candidates",
            "duplicate_pair": "pairs",
        }[kind]
        values = getattr(inputs, field)
        inputs = replace(inputs, **{field: (*values, values[0])})
    elif kind in {"wrong_response_hash", "wrong_pair_generator", "wrong_pair_hash"}:
        changes = {
            "wrong_response_hash": {"positive_response_hash": "f" * 64},
            "wrong_pair_generator": {"positive_generator_id": "foreign"},
            "wrong_pair_hash": {"pair_id": "f" * 64},
        }[kind]
        inputs = replace(
            inputs, pairs=(replace(inputs.pairs[0], **changes), *inputs.pairs[1:])
        )
    elif kind == "missing_permutation":
        inputs = replace(inputs, permutations=inputs.permutations[1:])
    elif kind in {"foreign_permutation", "wrong_surface"}:
        changes = (
            {"pair_id": "foreign"}
            if kind == "foreign_permutation"
            else {
                "surface_a_candidate_id": inputs.permutations[0].surface_b_candidate_id,
                "surface_b_candidate_id": inputs.permutations[0].surface_a_candidate_id,
            }
        )
        inputs = replace(
            inputs,
            permutations=(
                replace(inputs.permutations[0], **changes),
                *inputs.permutations[1:],
            ),
        )
    else:
        inputs = replace(inputs, audit_summary=audit_summary(required_count=1))
    report = evaluate_functional_gate(inputs)
    assert not report.passed
    assert report.failures


def test_undefined_empty_rates_are_none_and_empty_plan_does_not_pass():
    inputs = GateInputs((), (), (), (), (), (), audit_summary())
    report = evaluate_statistical_gate(inputs)
    assert report.status is GateStatus.FAIL
    assert report.diagnostics["ambiguous_rate"] is None
    assert report.diagnostics["pair_yield"] == {"math": None, "reclor": None}
    assert not evaluate_functional_gate(inputs).passed


@pytest.mark.parametrize(
    "table, expected",
    [
        ([], None),
        ([[]], None),
        ([[1, 2]], None),
        ([[1], [2]], None),
        ([[0, 0], [0, 0]], None),
        ([[0, 0], [3, 4]], None),
        ([[1, 0, 0], [0, 1, 0]], 1.0),
        ([[10, 10], [10, 10]], 0.0),
        ([[10, 0], [0, 10]], 1.0),
        ([[30, 10], [10, 30]], 0.5),
        ([[10, 0, 0], [0, 10, 0], [0, 0, 10]], 1.0),
    ],
)
def test_cramers_v_known_tables_and_zero_margins(table, expected):
    result = cramers_v(table)
    assert result is None if expected is None else math.isclose(result, expected)


@pytest.mark.parametrize(
    "table", [[[1, -1], [0, 1]], [[1, 2], [3]], [[1.0, 0], [0, 1]], [[True, 0], [0, 1]]]
)
def test_cramers_v_rejects_invalid_count_tables(table):
    with pytest.raises(ValueError):
        cramers_v(table)


def test_diagnostics_have_contingency_lengths_and_pair_proportions():
    report = evaluate_statistical_gate(make_inputs(questions=4, selected=4))
    assert report.diagnostics["contingency_table"] == ((8, 8, 0, 0, 0), (8, 8, 0, 0, 0))
    assert report.diagnostics["cramers_v"] == 0.0
    assert report.diagnostics["same_generator_pair_proportion"] == 0.0
    assert report.diagnostics["cross_generator_pair_proportion"] == 1.0
    assert report.diagnostics["source_correct_rate"] == {"math": 0.5, "reclor": 0.5}
    assert report.diagnostics["generator_correct_rate"] == {"g0": 0.5, "g1": 0.5}
    assert report.diagnostics["response_token_lengths"] == (10,) * 16 + (11,) * 16
    assert report.diagnostics["positive_negative_token_gaps"] == (-1,) * 8


def test_gate_results_and_input_sequences_are_immutable_and_order_independent():
    inputs = make_inputs(questions=4, selected=4)
    report = evaluate_statistical_gate(inputs)
    reversed_inputs = replace(
        inputs,
        **{
            name: tuple(reversed(getattr(inputs, name)))
            for name in (
                "plans",
                "candidates",
                "verifications",
                "question_verifications",
                "pairs",
                "permutations",
            )
        },
    )
    assert evaluate_statistical_gate(reversed_inputs) == report
    assert evaluate_functional_gate(reversed_inputs) == evaluate_functional_gate(inputs)
    with pytest.raises(FrozenInstanceError):
        inputs.plans = ()
    with pytest.raises(TypeError):
        report.diagnostics["cell_counts"]["math"]["g0"]["successful"] = 0


@pytest.mark.parametrize(
    "role", ["generation", "verification", "audit", "pairs", "permutations"]
)
def test_gate_run_config_binds_every_required_upstream_role(role):
    bindings = dict.fromkeys(
        ("generation", "verification", "audit", "pairs", "permutations"), HASH
    )
    before = run_id("gates", gate_stage_config(bindings))
    bindings[role] = "f" * 64
    assert run_id("gates", gate_stage_config(bindings)) != before


def test_gate_import_does_not_load_trainer_or_dataframe_dependencies():
    code = "import sys; import permstudy.data_pipeline.gates; assert not any(n in sys.modules for n in ('pandas', 'pyarrow', 'permstudy.data_pipeline.trainer_export'))"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
