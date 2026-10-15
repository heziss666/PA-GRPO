"""Small synthetic tests for the diagnostic's identity and numerical contracts."""

import importlib
import random

import pytest


def module():
    return importlib.import_module("scripts_permstudy.phase2.diagnose_judge_subgroups")


def fixture_rows():
    return [
        {
            "pair_id": "synthetic-pair",
            "original_question_id": "synthetic-question",
            "source": "math",
            "permutation_id": permutation,
            "rollout_slot": slot,
            "response": "<answer>A</answer>"
            if permutation == 1 or slot < 8
            else "<answer>B</answer>",
            "finish_reason": "stop",
        }
        for permutation in (0, 1)
        for slot in range(16)
    ]


def test_slices_semantics_and_population_advantages_are_not_batch_order():
    # Reversing AB/BA semantics, using sample sigma, or mixing the N8 halves breaks these literals.
    diagnostic = module()
    rows = fixture_rows()
    random.Random(7).shuffle(rows)
    views = diagnostic.analyze_rollouts(rows)["overall"]
    a, b, full = (views[k] for k in ("N8-A", "N8-B", "N16"))
    assert a["ab_accuracy"] == 1 and a["ba_accuracy"] == 0
    assert a["semantic_consistency"] == 0 and a["consistent_accuracy"] == 0
    assert a["mean_delta"] == pytest.approx(2)
    assert a["mean_global_sigma"] == pytest.approx(1)
    assert a["global_nonzero_pair_fraction"] == 1
    assert a["local_nonzero_subgroup_fraction"] == 0
    assert a["mean_local_weight"] == 1
    assert b["semantic_consistency"] == 1 and b["consistent_accuracy"] == 0
    assert b["global_zero_sigma_fraction"] == 1
    assert b["mean_local_weight"] == 0
    assert full["ab_accuracy"] == 0.5 and full["ba_accuracy"] == 0
    assert full["mean_delta"] == pytest.approx(1)
    assert full["mean_global_sigma"] == pytest.approx(3**0.5 / 2)
    assert full["local_nonzero_subgroup_fraction"] == 0.5


def test_invalid_outputs_stay_in_accuracy_and_consistency_denominators():
    rows = fixture_rows()
    rows[16]["response"] = "Answer: A"
    result = module().analyze_rollouts(rows)["overall"]["N8-A"]
    assert result["format_valid_rate"] == 15 / 16
    assert result["ab_accuracy"] == 1 and result["ba_accuracy"] == 0
    assert result["semantic_consistency"] == 0
    assert result["invalid_count"] == 1


@pytest.mark.parametrize(
    "mutation", ["duplicate", "missing", "foreign_slot", "mixed_source"]
)
def test_incomplete_or_corrupt_identity_is_rejected(mutation):
    rows = fixture_rows()
    if mutation == "duplicate":
        rows.append(dict(rows[0]))
    elif mutation == "missing":
        rows.pop()
    elif mutation == "foreign_slot":
        rows[0]["rollout_slot"] = 16
    else:
        rows[0]["source"] = "reclor"
    with pytest.raises(ValueError):
        module().analyze_rollouts(rows)


def test_actual_controlled_reward_components_without_logging_or_tensorboard():
    from my_reward.judge_qwen import controlled_base_reward

    result = controlled_base_reward(
        "<thinking>" + "x" * 500 + "</thinking><answer>A</answer>", "A"
    )
    assert result["answer"] == "A"
    assert {k: v for k, v in result.items() if k != "answer"} == pytest.approx(
        {"correctness": 1.0, "format": 0.3, "length": 0.1, "base_reward": 1.4}
    )
    assert controlled_base_reward("Final Answer: A", "A")[
        "base_reward"
    ] == pytest.approx(-1.4)


def test_qwen25_does_not_receive_qwen3_thinking_template_switch():
    from evaluation.evaluate_models import detect_model_type, render_chat_prompt

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return kwargs

    assert detect_model_type("Qwen/Qwen2.5-7B-Instruct") == "qwen2"
    assert detect_model_type("Qwen/Qwen3-8B") == "qwen3"
    assert render_chat_prompt(Tokenizer(), [], "qwen2") == {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    assert render_chat_prompt(Tokenizer(), [], "qwen3")["enable_thinking"] is False


def test_resume_and_returned_request_identity(tmp_path):
    diagnostic = module()
    prepared = {
        "run_id": "synthetic-run",
        "items": [
            {
                "pair_id": "synthetic-pair",
                "original_question_id": "synthetic-question",
                "source": "math",
                "permutation_id": p,
                "messages": [],
            }
            for p in (0, 1)
        ],
    }

    class Engine:
        def __init__(self):
            self.seen = []

        def generate(self, requests):
            self.seen.extend(r["request_id"] for r in requests)
            return [
                dict(
                    r,
                    response="<answer>A</answer>",
                    finish_reason="stop",
                    output_tokens=8,
                )
                for r in reversed(requests)
            ]

    engine = Engine()
    diagnostic.generate_pending(prepared, tmp_path / "rollouts.jsonl", engine, limit=3)
    assert len(engine.seen) == len(set(engine.seen)) == 3
    diagnostic.generate_pending(prepared, tmp_path / "rollouts.jsonl", engine)
    assert len(engine.seen) == len(set(engine.seen)) == 32
    diagnostic.generate_pending(prepared, tmp_path / "rollouts.jsonl", engine)
    assert len(engine.seen) == 32


def test_duplicate_backend_request_is_not_persisted(tmp_path):
    diagnostic = module()
    prepared = {
        "run_id": "synthetic-run",
        "items": [
            {
                "pair_id": "synthetic-pair",
                "original_question_id": "synthetic-question",
                "source": "math",
                "permutation_id": p,
                "messages": [],
            }
            for p in (0, 1)
        ],
    }

    class Engine:
        def generate(self, requests):
            row = dict(
                requests[0],
                response="<answer>A</answer>",
                finish_reason="stop",
                output_tokens=8,
            )
            return [row, row]

    with pytest.raises(ValueError):
        diagnostic.generate_pending(
            prepared, tmp_path / "rollouts.jsonl", Engine(), batch_size=2
        )
    assert not (tmp_path / "rollouts.jsonl").exists()
