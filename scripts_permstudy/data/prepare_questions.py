"""Acquire external source snapshots or build their unified split."""

from collections.abc import Sequence
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts_permstudy.data import _common as io
from permstudy.data_pipeline import splitting
from permstudy.data_pipeline.ids import run_id
from permstudy.data_pipeline.schema import QuestionRecord, Source
from permstudy.data_pipeline.sources import SourceSnapshot
from permstudy.data_pipeline.sources import math as math_source, reclor


def build_parser() -> io.argparse.ArgumentParser:
    parser = io.parser("prepare_questions", common=False)
    subs = parser.add_subparsers(dest="command", required=True)
    r = subs.add_parser("acquire-reclor")
    io.add_common(r)
    r.add_argument("--reclor-dir", required=True)
    r.add_argument(
        "--acknowledge-reclor-noncommercial", action="store_true", required=True
    )
    m = subs.add_parser("acquire-math")
    io.add_common(m)
    m.add_argument("--math-revision", required=True)
    s = subs.add_parser("build-split")
    io.add_common(s)
    s.add_argument("--reclor-manifest", required=True)
    s.add_argument("--math-manifest", required=True)
    s.add_argument("--split-seed", type=int, default=42)
    return parser


def dispatch(args, root):
    if args.command == "acquire-reclor":
        snapshot = reclor.load_reclor_train(
            io.rooted(root, args.reclor_dir, relative=False),
            root,
            acknowledge_noncommercial=args.acknowledge_reclor_noncommercial,
        )
        return io.output(snapshot.private_manifest)
    if args.command == "acquire-math":
        snapshot = math_source.load_math_train(
            math_source.MATH_REPO_ID,
            args.math_revision,
            io.rooted(root, "cache/datasets"),
            math_source.hf_dataset_loader,
            data_root=root,
        )
        return io.output(snapshot.private_manifest)
    paths = {
        "math_questions": args.math_manifest,
        "reclor_questions": args.reclor_manifest,
    }
    manifests = {
        role: io.load_manifest(root, path, "sources") for role, path in paths.items()
    }
    snapshots = {
        role: SourceSnapshot(
            Source(m["source"]),
            m["source_revision"],
            m["source_snapshot_id"],
            io.records(root, m, "questions.jsonl", QuestionRecord),
            m,
        )
        for role, m in manifests.items()
    }
    bindings = {role: m["output_manifest_hash"] for role, m in manifests.items()}
    config = io.checked(splitting.split_stage_config, args.split_seed, bindings)
    result = io.checked(
        splitting.build_bound_internal_split,
        snapshots,
        bindings,
        root,
        split_seed=args.split_seed,
        integrity=True,
    )
    questions = sorted(
        (q for s in snapshots.values() for q in s.questions),
        key=lambda q: q.original_question_id,
    )
    smoke = io.checked(
        splitting.select_smoke_questions, result.assignments, split_seed=args.split_seed
    )
    prefix = f"split/{run_id('split', config)}"
    refs = [
        io.persist_records(root, f"{prefix}/questions.jsonl", questions),
        io.persist_records(root, f"{prefix}/assignments.jsonl", result.assignments),
        io.persist_records(root, f"{prefix}/smoke.jsonl", smoke),
        io.persist_bytes(
            root,
            f"{prefix}/split.canonical.jsonl",
            splitting.split_manifest_bytes(questions, result, args.split_seed),
            len(result.assignments) + 1,
        ),
    ]
    manifest = io.persist_manifest(
        root,
        "split",
        config,
        refs,
        {"questions": len(questions), "smoke": len(smoke)},
        paths,
        extra={"split_manifest_hash": result.split_manifest_hash},
    )
    return io.output(manifest)


def main(argv: Sequence[str] | None = None) -> int:
    return io.execute(build_parser, dispatch, argv)


if __name__ == "__main__":
    raise SystemExit(main())
