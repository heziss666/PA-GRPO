"""Persist a fake-only generation plan bound to a verified split envelope."""

from collections.abc import Sequence
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts_permstudy.data import _common as io
from permstudy.data_pipeline import generation, splitting
from permstudy.data_pipeline.schema import (
    CandidatePlan,
    QuestionRecord,
    SplitAssignment,
)


def build_parser() -> io.argparse.ArgumentParser:
    parser = io.parser("plan_generation")
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--generator-config", action="append", required=True)
    parser.add_argument("--samples-per-question", type=io.positive_int, default=2)
    parser.add_argument("--shard-size", type=io.positive_int, required=True)
    return parser


def load_split(root, manifest):
    questions = io.records(root, manifest, "questions.jsonl", QuestionRecord)
    seed = manifest["typed_config"]["parameters"]["split_seed"]
    result = io.checked(
        splitting.build_internal_split, questions, split_seed=seed, integrity=True
    )
    refs = io.select_refs(manifest, "split.canonical.jsonl")
    if (
        len(refs) != 1
        or refs[0].sha256 != result.split_manifest_hash
        or manifest.get("split_manifest_hash") != result.split_manifest_hash
    ):
        raise io.IntegrityError()
    if io.rooted(
        root, refs[0].relative_path
    ).read_bytes() != splitting.split_manifest_bytes(questions, result, seed):
        raise io.IntegrityError()
    if io.records(root, manifest, "assignments.jsonl", SplitAssignment) != list(
        result.assignments
    ):
        raise io.IntegrityError()
    smoke = io.checked(
        splitting.select_smoke_questions,
        result.assignments,
        split_seed=seed,
        integrity=True,
    )
    if io.records(root, manifest, "smoke.jsonl", SplitAssignment) != smoke:
        raise io.IntegrityError()
    return result, smoke


def load_plan(root, manifest):
    split = io.upstream(root, manifest, "split")
    _, smoke = load_split(root, split)
    parameters = manifest["typed_config"]["parameters"]
    configs = [
        io.checked(generation.GenerationConfig.from_dict, row, integrity=True)
        for row in parameters["generators"]
    ]
    if any(c.backend != "fake" for c in configs):
        raise io.PhaseBoundaryError()
    plan = io.checked(
        generation.plan_generation,
        smoke,
        configs,
        samples_per_question=parameters["samples_per_question"],
        shard_size=parameters["shard_size"],
        split_upstream_manifest_hash=split["output_manifest_hash"],
        integrity=True,
    )
    if (
        plan.generation_run_id != manifest["run_id"]
        or plan.config_hash != manifest["config_hash"]
        or any(c.split_manifest_hash != split["split_manifest_hash"] for c in configs)
        or list(plan.candidates)
        != io.records(root, manifest, "plans.jsonl", CandidatePlan)
    ):
        raise io.IntegrityError()
    return plan


def dispatch(args, root):
    if len(args.generator_config) != 3 or args.samples_per_question != 2:
        raise io.CLIContractError()
    split = io.load_manifest(root, args.split_manifest, "split")
    _, smoke = load_split(root, split)
    configs = [
        io.checked(
            generation.GenerationConfig.from_dict, io.read_json(io.rooted(root, path))
        )
        for path in args.generator_config
    ]
    if any(c.backend != "fake" for c in configs):
        raise io.PhaseBoundaryError()
    if any(c.split_manifest_hash != split["split_manifest_hash"] for c in configs):
        raise io.IntegrityError()
    plan = io.checked(
        generation.plan_generation,
        smoke,
        configs,
        samples_per_question=args.samples_per_question,
        shard_size=args.shard_size,
        split_upstream_manifest_hash=split["output_manifest_hash"],
    )
    config = generation.generation_stage_config(
        configs,
        samples_per_question=args.samples_per_question,
        shard_size=args.shard_size,
        split_upstream_manifest_hash=split["output_manifest_hash"],
    )
    refs = io.select_refs(
        split,
        "questions.jsonl",
        "assignments.jsonl",
        "smoke.jsonl",
        "split.canonical.jsonl",
    )
    refs.append(
        io.persist_records(
            root, f"generation/{plan.generation_run_id}/plans.jsonl", plan.candidates
        )
    )
    return io.output(
        io.persist_manifest(
            root,
            "generation",
            config,
            refs,
            {"planned": len(plan.candidates)},
            {"split": args.split_manifest},
            extra={"split_manifest_hash": split["split_manifest_hash"]},
            filename="plan.json",
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    return io.execute(build_parser, dispatch, argv)


if __name__ == "__main__":
    raise SystemExit(main())
