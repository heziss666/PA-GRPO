"""Validated command boundary for export_training_dataset."""

from collections.abc import Sequence
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts_permstudy.data import _common as io
from scripts_permstudy.data.plan_generation import load_split
from scripts_permstudy.data.verify_candidates import load_candidates
from scripts_permstudy.data.build_reasoning_pairs import load_pairs
from scripts_permstudy.data.build_permutations import load_permutations
from permstudy.data_pipeline import trainer_export


def load_export_inputs(root, paths):
    manifests = {
        role: io.load_manifest(root, path, role) for role, path in paths.items()
    }
    for downstream, upstream in (
        ("generation", "split"),
        ("pairs", "generation"),
        ("permutations", "pairs"),
    ):
        if (
            manifests[downstream]["upstream_bindings"][upstream]
            != manifests[upstream]["output_manifest_hash"]
        ):
            raise io.IntegrityError()
    load_split(root, manifests["split"])
    load_candidates(root, manifests["generation"])
    load_pairs(root, manifests["pairs"])
    load_permutations(root, manifests["permutations"])
    return manifests


def build_parser() -> io.argparse.ArgumentParser:
    parser = io.parser("export_training_dataset")
    for role in ("split", "generation", "pair", "permutation"):
        parser.add_argument(f"--{role}-manifest", required=True)
    return parser


def dispatch(args, root):
    paths = {
        "split": args.split_manifest,
        "generation": args.generation_manifest,
        "pairs": args.pair_manifest,
        "permutations": args.permutation_manifest,
    }
    manifests = load_export_inputs(root, paths)
    summary = trainer_export.export_trainer_parquet(manifests, root)
    bindings = {
        role: manifest["output_manifest_hash"] for role, manifest in manifests.items()
    }
    config = trainer_export.export_stage_config(bindings)
    return io.output(
        io.persist_manifest(
            root,
            "trainer_export",
            config,
            [summary.dataset_artifact],
            {"rows": summary.row_count},
            paths,
            extra={
                "prompt_template_hash": summary.prompt_template_hash,
                "dataset_sha256": summary.dataset_artifact.sha256,
            },
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    return io.execute(build_parser, dispatch, argv)


if __name__ == "__main__":
    raise SystemExit(main())
