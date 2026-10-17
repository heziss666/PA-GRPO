"""Synthetic-only tests for the bounded CPU smoke-release diagnostic."""

import numpy as np
import pytest


def test_format_features_do_not_encode_words_numbers_or_option_identity():
    from scripts_permstudy.phase2.smoke_release_gate import structure_features

    first = "Alpha beta.\n\n- Proof $123$!\nFinal Answer: A"
    second = "Gamma zeta.\n\n- Claim $987$!\nFinal Answer: D"
    assert structure_features(first, 15) == structure_features(second, 15)
    assert all(
        isinstance(v, (int, float)) for v in structure_features(first, 15).values()
    )
    assert structure_features("a\n\n- b", 3)["list_items"] == 1
    assert structure_features("a\n\nb", 3)["list_items"] == 0


def test_fixed_thresholds_and_diagnostic_only_models_cannot_release_export():
    from scripts_permstudy.phase2.smoke_release_gate import release_status, style_status

    assert [style_status(x) for x in (0.6499, 0.65, 0.748, 0.75, 0.752)] == [
        "PASS",
        "PASS_WITH_WARNINGS",
        "PASS_WITH_WARNINGS",
        "FAIL",
        "FAIL",
    ]
    assert release_status(True, [0.64, 0.64]) == "PASS"
    assert release_status(True, [0.64, 0.70]) == "PASS_WITH_WARNINGS"
    assert release_status(True, [0.64, 0.75]) == "FAIL"
    assert release_status(False, [0.5, 0.5]) == "FAIL"
    with pytest.raises(ValueError):
        style_status(float("nan"))


def test_group_folds_keep_all_rows_of_each_question_together():
    from scripts_permstudy.phase2.smoke_release_gate import grouped_folds

    groups = np.repeat([f"synthetic-{i:02}" for i in range(15)], 4)
    labels = np.tile([0, 1, 0, 1], 15)
    folds = grouped_folds(groups, labels)
    assert len(folds) == 5
    visited = []
    for train, test in folds:
        assert set(groups[train]).isdisjoint(groups[test])
        visited.extend(test.tolist())
    assert sorted(visited) == list(range(60))


def test_bootstrap_resamples_questions_not_individual_rows():
    from scripts_permstudy.phase2.smoke_release_gate import question_bootstrap

    # Each complete question has one right and one wrong prediction: every
    # question-level resample must remain exactly .5, unlike row resampling.
    groups = np.repeat([f"synthetic-{i}" for i in range(10)], 2)
    labels = np.tile([0, 1], 10)
    predictions = np.zeros(20, dtype=int)
    assert question_bootstrap(groups, labels, predictions, repetitions=200) == [
        0.5,
        0.5,
    ]
    with pytest.raises(ValueError):
        question_bootstrap(np.array(["one", "two"]), np.array([0, 1]), np.array([0, 1]))


def test_oof_predictions_are_complete_and_reproducible_on_synthetic_pairs():
    from scripts_permstudy.phase2.smoke_release_gate import diagnose_source

    rows = []
    for i in range(20):
        for label in (0, 1):
            rows.append(
                {
                    "original_question_id": f"synthetic-{i:02}",
                    "label": label,
                    "generator": "same-generator",
                    "token_count": 8,
                    "text": "synthetic valid reasoning"
                    if label
                    else "synthetic wrong reasoning",
                }
            )
    report, predictions = diagnose_source(rows, bootstrap_repetitions=30)
    again, second = diagnose_source(rows[::-1], bootstrap_repetitions=30)
    assert report == again and predictions == second
    assert report["format_only"]["balanced_accuracy"] == 0.5
    assert report["generator_only"]["balanced_accuracy"] == 0.5
    assert report["full_text_diagnostic_only"]["balanced_accuracy"] == 1.0
    assert (
        len(predictions) == 40
        and len({(r["original_question_id"], r["label"]) for r in predictions}) == 40
    )


def test_cli_help_and_isolation_fail_closed_without_data_access(tmp_path, capsys):
    from scripts_permstudy.phase2.smoke_release_gate import main

    assert main(["--help"]) == 0
    assert "smoke_release_gate" in capsys.readouterr().out
    assert main(["--data-root", "."]) == 2
    assert "contract_error" in capsys.readouterr().out
