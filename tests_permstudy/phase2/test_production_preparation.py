"""CPU-only preparation; no model loading or inference is permitted here."""

from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import pytest

from permstudy.data_pipeline.schema import CandidatePlan, Split
from tests_permstudy.phase2.test_run_real_smoke import (
    assignment,
    math_question,
    reclor_question,
)


def production_pool():
    questions = []
    assignments = []
    for factory in (math_question, reclor_question):
        for index in range(240):
            question = factory(index + 100)
            questions.append(question)
            assignments.append(
                replace(assignment(question), stratum=f"cell={index % 2}")
            )
    old = (math_question(1), reclor_question(1))
    from scripts_permstudy.phase2 import run_real_smoke as cli

    fresh = replace(
        math_question(2), original_question_id=cli.MATH_COMPLETION_NEW_IDS[0]
    )
    held = (math_question(3), reclor_question(3))
    questions.extend((*old, fresh, *held))
    assignments.extend(assignment(q) for q in (*old, fresh))
    assignments.extend(
        replace(assignment(q), split=Split.INTERNAL_HOLDOUT) for q in held
    )
    return tuple(questions), tuple(assignments), tuple(assignment(q) for q in old)


def test_new_production_selection_is_stratified_train_only_and_excludes_diagnostics(
    monkeypatch,
    tmp_path,
):
    from scripts_permstudy.data import _common as io, plan_generation as phase1
    from scripts_permstudy.phase2 import run_real_smoke as cli

    questions, assignments, old_smoke = production_pool()
    monkeypatch.setattr(io, "records", lambda *args: list(questions))
    monkeypatch.setattr(
        phase1,
        "load_split",
        lambda *args: (SimpleNamespace(assignments=assignments), old_smoke),
    )
    split = {"split_manifest_hash": cli.MATH_COMPLETION_SPLIT_HASH}
    selected = []
    for source in ("math", "reclor"):
        recipe = f"{source}-production-200-v1"
        records, chosen = cli.load_phase2_split(tmp_path, split, recipe=recipe)
        assert len(records) == len(chosen) == 200
        assert Counter(a.stratum for a in chosen) == {"cell=0": 100, "cell=1": 100}
        assert all(a.split is Split.TRAIN and a.source.value == source for a in chosen)
        excluded = {a.original_question_id for a in old_smoke} | set(
            cli.MATH_COMPLETION_NEW_IDS
        )
        assert excluded.isdisjoint(q.original_question_id for q in records)
        selected.extend(a.original_question_id for a in chosen)
        monkeypatch.setattr(
            phase1,
            "load_split",
            lambda *args: (
                SimpleNamespace(assignments=tuple(reversed(assignments))),
                old_smoke,
            ),
        )
        assert cli.load_phase2_split(tmp_path, split, recipe=recipe) == (
            records,
            chosen,
        )
    assert len(set(selected)) == 400


@pytest.mark.parametrize("source,budget", [("math", 1024), ("reclor", 512)])
def test_production_plan_has_800_candidates_two_models_and_new_prompt_namespace(
    source, budget, monkeypatch
):
    from scripts_permstudy.phase2 import run_real_smoke as cli

    factory = math_question if source == "math" else reclor_question
    chosen = tuple(assignment(factory(i + 100)) for i in range(200))
    split = {
        "split_manifest_hash": cli.MATH_COMPLETION_SPLIT_HASH,
        "output_manifest_hash": "3" * 64,
    }
    recipe = f"{source}-production-200-v1"
    plan = cli._generation_plan(split, chosen, recipe=recipe)
    assert len({p.key for p in plan.candidates}) == 800
    assert Counter(p.generator_id for p in plan.candidates) == {
        "qwen2.5-7b-instruct": 400,
        "llama-3.1-8b-instruct": 400,
    }
    for config in plan.generator_configs:
        assert config.max_new_tokens == budget
        cli._validate_phase2_config(config)
        messages = cli.render_messages(
            factory(100), prompt_template_revision=config.prompt_template_revision
        )
        assert "Private solution" not in str(messages)
        assert (
            "keep the reasoning concise"
            if source == "math"
            else "Your final line MUST be exactly:"
        ) in str(messages)
        with pytest.raises(ValueError):
            cli._validate_phase2_config(replace(config, max_new_tokens=budget + 1))
    with pytest.raises(ValueError):
        cli._generation_plan(split, chosen[:-1], recipe=recipe)
    with pytest.raises(ValueError):
        cli._generation_plan(
            split,
            tuple(replace(a, split=Split.INTERNAL_HOLDOUT) for a in chosen),
            recipe=recipe,
        )
    monkeypatch.setattr(cli, "PRODUCTION_SELECTION_VERSION", "changed_selection_v2")
    assert (
        cli._generation_plan(split, chosen, recipe=recipe).generation_run_id
        != plan.generation_run_id
    )


def test_plan_command_persists_formal_plan_without_constructing_engine(
    monkeypatch, tmp_path
):
    from scripts_permstudy.data import _common as io
    from scripts_permstudy.phase2 import run_real_smoke as cli

    questions = tuple(math_question(i + 100) for i in range(200))
    split = {
        "split_manifest_hash": cli.MATH_COMPLETION_SPLIT_HASH,
        "output_manifest_hash": "3" * 64,
    }
    io.persist_bytes(
        tmp_path, "split/fixture/manifest.json", io.canonical_json_bytes(split), 1
    )
    monkeypatch.setattr(io, "load_manifest", lambda *args: split)

    def forbidden_engine(*args):
        pytest.fail("CPU planning must not construct an engine")

    args = cli.build_parser().parse_args(
        [
            "plan",
            "--split-manifest",
            "split/fixture/manifest.json",
            "--recipe",
            "math-production-200-v1",
        ]
    )
    result = cli.dispatch(
        args,
        tmp_path,
        split_loader=lambda *args: (questions, tuple(assignment(q) for q in questions)),
        engine_factory_builder=forbidden_engine,
    )
    assert result["count"] == 800
    manifest = io.read_json(tmp_path / "generation" / result["run_id"] / "planned.json")
    assert manifest["counts"] == {"planned": 800, "successful": 0}
    assert len(io.records(tmp_path, manifest, "plans.jsonl", CandidatePlan)) == 800
    assert not list(tmp_path.rglob("candidates.jsonl"))
    assert (
        cli.dispatch(
            args,
            tmp_path,
            split_loader=lambda *args: (
                questions,
                tuple(assignment(q) for q in questions),
            ),
        )
        == result
    )


def test_training_adapter_preserves_identity_and_matches_strict_reward_parser():
    from copy import deepcopy
    from permstudy.controlled_evaluator import extract_controlled_answer
    from permstudy.data_pipeline.trainer_export import build_trainer_rows
    from scripts_permstudy.phase2 import run_real_smoke as cli
    from tests_permstudy.data_pipeline.test_trainer_export import _records

    rows = build_trainer_rows(*_records())
    original = deepcopy(rows)
    adapted = cli.controlled_training_rows(rows)
    assert rows == original
    for before, after in zip(rows, adapted, strict=True):
        assert after["extra_info"] == before["extra_info"]
        assert after["reward_model"] == before["reward_model"]
        assert after["prompt"][1]["content"].endswith(
            "End with exactly one <answer>A</answer> or <answer>B</answer> tag. Do not write anything after it."
        )
        assert not after["prompt"][1]["content"].endswith("Answer with A or B only.")
        label = after["reward_model"]["ground_truth"]
        assert (
            extract_controlled_answer(f"<answer>{label}</answer>", "thinking", 2)
            == label
        )
        assert extract_controlled_answer(label, "thinking", 2) is None
    rows[0]["prompt"][1]["content"] += "unrecognized template"
    with pytest.raises(ValueError):
        cli.controlled_training_rows(rows)


def test_training_command_writes_separate_hash_bound_parquet(monkeypatch, tmp_path):
    import io as buffers
    import pyarrow as pa
    import pyarrow.parquet as pq
    from permstudy.data_pipeline.trainer_export import build_trainer_rows
    from scripts_permstudy.data import _common as io
    from scripts_permstudy.phase2 import run_real_smoke as cli
    from tests_permstudy.data_pipeline.test_trainer_export import _records

    rows = build_trainer_rows(*_records())
    output = buffers.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows), output)
    ref = io.persist_bytes(
        tmp_path, "trainer_exports/direct/train.parquet", output.getvalue(), 4
    )
    paths = {
        role: f"{role}/fixture/manifest.json"
        for role in ("split", "generation", "pairs", "permutations")
    }
    upstream = {
        role: io.persist_bytes(tmp_path, path, b"{}", 1).to_dict()
        for role, path in paths.items()
    }
    manifest = {
        "output_manifest_hash": "4" * 64,
        "artifacts": [ref.to_dict()],
        "upstream_manifests": upstream,
        "upstream_bindings": {
            role: str(index) * 64 for index, role in enumerate(paths, 1)
        },
    }
    # Phase 1 covers the envelope loader. Exercise real Parquet read/write here.
    monkeypatch.setattr(io, "load_manifest", lambda *args: manifest)
    args = cli.build_parser().parse_args(
        [
            "prepare-training",
            "--trainer-manifest",
            "trainer_export/direct/manifest.json",
        ]
    )
    result = cli.dispatch(args, tmp_path)
    written = io.read_json(
        tmp_path / "trainer_export" / result["run_id"] / "manifest.json"
    )
    adapted_ref = io.select_refs(written, "train.parquet")[0]
    adapted = pq.read_table(tmp_path / adapted_ref.relative_path).to_pylist()
    assert adapted == cli.controlled_training_rows(rows)
    assert result["count"] == 4
    assert written["training_authorized"] is False
    assert written["typed_config"]["parameters"]["input_parquet_sha256"] == ref.sha256
    assert (tmp_path / ref.relative_path).read_bytes() == output.getvalue()
    (tmp_path / ref.relative_path).write_bytes(b"corrupted")
    with pytest.raises(io.IntegrityError):
        cli.dispatch(args, tmp_path)
