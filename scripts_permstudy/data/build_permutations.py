"""Validated command boundary for build_permutations."""

from collections.abc import Sequence
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts_permstudy.data import _common as io
from scripts_permstudy.data.build_reasoning_pairs import load_pairs
from permstudy.data_pipeline import permutations
from permstudy.data_pipeline.ids import run_id
from permstudy.data_pipeline.schema import PermutationRecord


def load_permutations(root, manifest):
    pair_manifest = io.upstream(root, manifest, "pairs")
    pairs = load_pairs(root, pair_manifest)
    config = permutations.permutation_stage_config(manifest["upstream_bindings"])
    expected = [
        p
        for pair in pairs
        for p in permutations.build_permutations(
            pair, upstream_bindings=manifest["upstream_bindings"]
        )
    ]
    records = io.records(root, manifest, "permutations.jsonl", PermutationRecord)
    if config != manifest["typed_config"] or records != expected:
        raise io.IntegrityError()
    return records


def build_parser() -> io.argparse.ArgumentParser:
    parser = io.parser("build_permutations")
    parser.add_argument("--pair-manifest", required=True)
    return parser


def dispatch(args, root):
    manifest = io.load_manifest(root, args.pair_manifest, "pairs")
    pairs = load_pairs(root, manifest)
    bindings = {"pairs": manifest["output_manifest_hash"]}
    config = permutations.permutation_stage_config(bindings)
    records = [
        p
        for pair in pairs
        for p in permutations.build_permutations(pair, upstream_bindings=bindings)
    ]
    identity = run_id("permutations", config)
    refs = [
        io.persist_records(root, f"permutations/{identity}/permutations.jsonl", records)
    ]
    return io.output(
        io.persist_manifest(
            root,
            "permutations",
            config,
            refs,
            {"permutations": len(records)},
            {"pairs": args.pair_manifest},
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    return io.execute(build_parser, dispatch, argv)


if __name__ == "__main__":
    raise SystemExit(main())
