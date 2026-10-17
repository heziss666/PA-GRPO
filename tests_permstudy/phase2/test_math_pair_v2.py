"""Synthetic-only behavior checks for the fixed Pair-v2 intervention."""

from copy import deepcopy
from fractions import Fraction
import importlib

import pytest


def module():
    assert importlib.util.find_spec("scripts_permstudy.phase2.math_pair_v2"), (
        "Pair-v2 entry is not implemented"
    )
    return importlib.import_module("scripts_permstudy.phase2.math_pair_v2")


def item(cid="positive", generator="one", text="Proof.", count=2):
    return {
        "candidate_id": cid,
        "generator": generator,
        "text": text,
        "token_count": count,
        "response_hash": "a" * 64,
    }


def test_component_normalization_and_equal_five_way_weighting():
    m = module()
    assert m.count_distance(0, 0) == 0
    assert m.count_distance(0, 3) == Fraction(3, 4)
    assert m.count_distance(3, 0) == Fraction(3, 4)
    assert m.count_distance(100, 2) == Fraction(98, 103)
    a, b = item(), item("negative", "two")
    assert m.distance(a, b) == Fraction(1, 5)
    b["generator"] = "one"
    assert m.distance(a, b) == 0
    for bad in (-1, float("nan"), 1.5, True):
        with pytest.raises(ValueError):
            m.count_distance(bad, 0)


def test_scoring_ignores_answer_values_and_words_with_identical_structure():
    m = module()
    assert (
        m.distance(
            item(text="Claim $123$!", count=9), item(text="Proof $987$!", count=9)
        )
        == 0
    )
    assert 0 < m.distance(item(text="Proof."), item(text="Proof.\n\n- Next $x$.")) < 1


def test_best_proposal_and_fallback_are_input_order_independent():
    m = module()
    p = item()
    old = item("old-negative", "two")
    new = item("new-negative", "one")
    options = [(p, old), (p, new)]
    assert (
        m.choose_proposal(options, ("positive", "old-negative"))[1]["candidate_id"]
        == "new-negative"
    )
    assert m.choose_proposal(options[::-1], ("positive", "old-negative")) == (p, new)
    # On equal objective score, avoid an unnecessary candidate replacement.
    same = item("another-negative", "two")
    assert m.choose_proposal([(p, same), (p, old)], ("positive", "old-negative")) == (
        p,
        old,
    )


def test_both_verdicts_and_exact_candidate_hash_identity_are_required():
    m = module()
    proposal = {
        "proposal_id": "p",
        "original_question_id": "synthetic-question",
        "generation_run_id": "g",
        "verification_run_id": "v",
        "positive": item(),
        "negative": item("negative"),
    }
    decision = {
        "proposal_id": "p",
        "original_question_id": "synthetic-question",
        "generation_run_id": "g",
        "verification_run_id": "v",
        "positive_candidate_id": "positive",
        "negative_candidate_id": "negative",
        "positive_response_hash": "a" * 64,
        "negative_response_hash": "a" * 64,
        "positive_verdict": "PASS",
        "negative_verdict": "INCORRECT_CONFIRMED",
        "positive_reason": "synthetic valid derivation",
        "negative_reason": "synthetic wrong result",
    }
    assert m.validate_decisions([proposal], [decision]) == {"p": True}
    for field, value in [
        ("positive_verdict", "UNSURE"),
        ("negative_verdict", "UNSURE"),
        ("positive_verdict", "FAIL"),
        ("negative_verdict", "ACTUALLY_CORRECT"),
    ]:
        changed = {**decision, field: value}
        assert m.validate_decisions([proposal], [changed]) == {"p": False}
    for field in (
        "generation_run_id",
        "negative_candidate_id",
        "positive_response_hash",
    ):
        with pytest.raises(ValueError):
            m.validate_decisions([proposal], [{**decision, field: "foreign"}])
    for decisions in (
        [],
        [decision, decision],
        [{**decision, "proposal_id": "foreign"}],
    ):
        with pytest.raises(ValueError):
            m.validate_decisions([proposal], decisions)
    changed = deepcopy(decision)
    changed["positive_reason"] = ""
    with pytest.raises(ValueError):
        m.validate_decisions([proposal], [changed])


def test_preserved_question_folds_and_one_diagnostic_run_guard(tmp_path):
    m = module()
    with m.diagnostic_once(tmp_path):
        assert (tmp_path / "diagnostic.started").exists()
    with pytest.raises(FileExistsError):
        with m.diagnostic_once(tmp_path):
            pytest.fail("second classifier fit must never execute")


def test_structure_readout_exposes_length_and_generator_imbalance_without_text():
    m = module()
    rows = [
        dict(item(generator="qwen", count=10), label=1),
        dict(item(generator="llama", count=20), label=0),
    ]
    result = m.describe_rows(rows)
    assert result["structure"]["token_length"] == {
        "chosen_mean": 10.0,
        "rejected_mean": 20.0,
    }
    assert result["generators"] == {"chosen": {"qwen": 1}, "rejected": {"llama": 1}}
    assert "Proof." not in str(result)


@pytest.mark.parametrize(
    "positive_verdict,expected_negative",
    [("PASS", "b-new"), ("UNSURE", "a-old"), ("FAIL", None)],
)
def test_tiny_synthetic_prepare_review_finalize_keeps_or_replaces_exact_pair(
    tmp_path, monkeypatch, positive_verdict, expected_negative
):
    from dataclasses import replace
    from types import SimpleNamespace
    import json
    from hashlib import sha256
    from tests_permstudy.data_pipeline.test_pairs import (
        candidate,
        question,
        UPSTREAM_BINDINGS,
        TOKENIZER_REVISION,
    )
    from permstudy.data_pipeline.schema import Source, VerificationStatus, PairRecord
    from permstudy.data_pipeline.pairs import select_pair
    from transformers import AutoTokenizer

    m = module()
    q = replace(
        question(),
        source=Source.MATH,
        context=None,
        question=None,
        answers=(),
        gold_label=None,
        original_question_id="math:train:synthetic",
        question_content_hash=sha256(b"Synthetic arithmetic problem").hexdigest(),
        problem="Synthetic arithmetic problem",
        solution=r"Synthetic solution: \boxed{1}",
        category="Algebra",
        level="Level 1",
    )

    class Tokenizer:
        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            return list(range(len(text)))

    tok = Tokenizer()
    cs = [
        candidate("positive", " Proof. ", question_record=q),
        candidate(
            "a-old",
            "Claim.",
            question_record=q,
            generator_id="llama-3.1-8b-instruct",
            status=VerificationStatus.INCORRECT,
        ),
        candidate(
            "b-new", "Error.", question_record=q, status=VerificationStatus.INCORRECT
        ),
    ]
    old = select_pair(
        q, cs, tok, TOKENIZER_REVISION, upstream_bindings=UPSTREAM_BINDINGS
    )
    assert old.negative_candidate_id == "a-old"
    inputs = (
        [old],
        {(old.generation_run_id, q.original_question_id): q},
        {c.plan.key: c for c, v in cs},
        {(v.generation_run_id, v.candidate_id): v for c, v in cs},
        {"synthetic_input": "1" * 64},
        {"source_results": {"math": {"synthetic_baseline": True}}},
    )
    monkeypatch.setattr(m, "load_inputs", lambda root: inputs)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **kw: tok)
    result = m.prepare(SimpleNamespace(), tmp_path)
    folder = tmp_path / f"phase2/math_pair_v2/{result['run_id']}"
    payload = json.loads((folder / "proposals.private.json").read_text())
    assert len(payload["proposals"]) == 1
    p = payload["proposals"][0]
    d = {
        k: p[k]
        for k in (
            "proposal_id",
            "original_question_id",
            "generation_run_id",
            "verification_run_id",
        )
    }
    for side in ("positive", "negative"):
        for key in ("candidate_id", "response_hash"):
            d[f"{side}_{key}"] = p[side][key]
    d.update(
        positive_verdict=positive_verdict,
        negative_verdict="INCORRECT_CONFIRMED",
        positive_reason="Synthetic quality review",
        negative_reason="Synthetic wrong derivation",
    )
    (tmp_path / "decisions.json").write_text(json.dumps([d]))

    # The existing classifier is tested separately; this fixture tests the real
    # record/quality/export joins, not sklearn fitting on a one-question cohort.
    def diagnose(rows):
        assert rows[0]["text"] == " Proof. "  # Baseline uses raw model output, too.
        return {"question_count": 1}, []

    monkeypatch.setattr(m.gate, "diagnose_source", diagnose)
    args = SimpleNamespace(run_id=result["run_id"], decisions="decisions.json")
    if expected_negative is None:
        from scripts_permstudy.data._common import IntegrityError

        with pytest.raises(IntegrityError):
            m.finalize(args, tmp_path)
        assert not (folder / "diagnostic.started").exists()
        assert not (folder / "pairs.jsonl").exists()
        return
    m.finalize(args, tmp_path)
    actual = PairRecord.from_dict(json.loads((folder / "pairs.jsonl").read_text()))
    assert actual.negative_candidate_id == expected_negative
    assert old.negative_candidate_id == "a-old"
    assert actual.original_question_id == q.original_question_id
    assert len((folder / "permutations.jsonl").read_text().splitlines()) == 2
    assert not list(folder.glob("*.parquet"))
    with pytest.raises(FileExistsError):
        m.finalize(args, tmp_path)
