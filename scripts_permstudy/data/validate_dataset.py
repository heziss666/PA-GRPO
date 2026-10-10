"""Validated command boundary for validate_dataset."""

from collections.abc import Sequence
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts_permstudy.data import _common as io


def build_parser() -> io.argparse.ArgumentParser:
    parser = io.parser("validate_dataset")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--manifest")
    for role in (
        "split",
        "generation",
        "verification",
        "pair",
        "permutation",
        "audit",
        "export",
    ):
        group.add_argument(f"--{role}-manifest")
    parser.add_argument("--check-isolation", action="store_true")
    return parser


def dispatch(args, root):
    key, reference = next(
        (key, value)
        for key, value in vars(args).items()
        if (key == "manifest" or key.endswith("_manifest")) and value
    )
    stage = (
        {
            "pair": "pairs",
            "permutation": "permutations",
            "export": "trainer_export",
        }.get(key.removesuffix("_manifest"), key.removesuffix("_manifest"))
        if key != "manifest"
        else None
    )
    manifest = io.load_manifest(root, reference, stage)
    if manifest["stage"] == "split":
        from scripts_permstudy.data.plan_generation import load_split

        load_split(root, manifest)
    elif manifest["stage"] == "generation":
        from scripts_permstudy.data.plan_generation import load_plan

        load_plan(root, manifest)
    elif manifest["stage"] == "verification":
        from scripts_permstudy.data.verify_candidates import load_verification

        load_verification(root, manifest)
    elif manifest["stage"] == "pairs":
        from scripts_permstudy.data.build_reasoning_pairs import load_pairs

        load_pairs(root, manifest)
    elif manifest["stage"] == "permutations":
        from scripts_permstudy.data.build_permutations import load_permutations

        load_permutations(root, manifest)
    elif manifest["stage"] == "audit":
        from scripts_permstudy.data.audit_generator_distribution import load_audit

        load_audit(root, manifest)
    elif manifest["stage"] == "trainer_export":
        from scripts_permstudy.data.export_training_dataset import load_export_inputs
        from permstudy.data_pipeline import trainer_export

        load_export_inputs(
            root,
            {
                role: ref["relative_path"]
                for role, ref in manifest["upstream_manifests"].items()
            },
        )
        config = trainer_export.export_stage_config(manifest["upstream_bindings"])
        if config != manifest["typed_config"]:
            raise io.IntegrityError()
        refs = io.select_refs(manifest, "train.parquet")
        if len(refs) != 1 or refs[0].sha256 != manifest.get("dataset_sha256"):
            raise io.IntegrityError()
        import pyarrow.parquet as parquet

        rows = parquet.read_table(io.rooted(root, refs[0].relative_path)).to_pylist()
        trainer_export.validate_trainer_rows(rows)
        if len(rows) != refs[0].record_count:
            raise io.IntegrityError()
    return io.output(manifest)


def main(argv: Sequence[str] | None = None) -> int:
    return io.execute(build_parser, dispatch, argv)


if __name__ == "__main__":
    raise SystemExit(main())
