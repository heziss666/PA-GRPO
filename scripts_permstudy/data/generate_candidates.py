"""Execute one fake generation shard and register its verified output artifacts."""

from collections.abc import Sequence
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts_permstudy.data import _common as io
from scripts_permstudy.data.plan_generation import load_plan
from permstudy.data_pipeline import generation
from permstudy.data_pipeline.io import ShardLock
from permstudy.data_pipeline.schema import ArtifactRef, CandidateRecord, FailureRecord


def build_parser() -> io.argparse.ArgumentParser:
    parser = io.parser("generate_candidates")
    parser.add_argument("--generation-manifest", required=True)
    parser.add_argument("--generator-id", required=True)
    parser.add_argument("--shard-id", required=True)
    parser.add_argument("--backend", choices=("fake", "vllm"), default="fake")
    parser.add_argument("--recover-stale-lock", action="store_true")
    return parser


def make_fake_backend(shard):
    """Default dry-run payload is deliberately unverifiable; Task 17 supplies fixtures."""
    return generation.FakeGenerationBackend(
        {
            plan.key: generation.GenerationResult(
                plan, "Synthetic Phase 1 dry run.", "stop", 0, None
            )
            for plan in shard.candidates
        }
    )


def dispatch(args, root):
    if args.backend == "vllm":
        raise io.PhaseBoundaryError()
    manifest = io.load_manifest(root, args.generation_manifest, "generation")
    plan = load_plan(root, manifest)
    shard = io.checked(
        generation.build_generation_shard_plan, plan, args.generator_id, args.shard_id
    )
    output = io.rooted(
        root,
        f"generation/{plan.generation_run_id}/shards/{args.generator_id}/{args.shard_id}",
    )
    progress = io.rooted(root, f"generation/{plan.generation_run_id}/progress.json")
    # One run-level registry lock prevents lost updates across independent shard writers.
    with ShardLock.acquire(
        io.rooted(root, f"generation/{plan.generation_run_id}/registry.lock"),
        {"run_id": plan.generation_run_id, "shard_id": "registry"},
        recover_stale=args.recover_stale_lock,
    ):
        previous = (
            io.load_manifest(root, progress.relative_to(root).as_posix(), "generation")
            if progress.exists()
            else manifest
        )
        if previous["run_id"] != plan.generation_run_id:
            raise io.IntegrityError()
        summary = generation.run_generation_shard(
            shard,
            make_fake_backend(shard),
            output,
            recover_stale_lock=args.recover_stale_lock,
        )
        shard_payload = io.read_json(output / "manifest.json")
        # Runner owns its specialized envelope; verify the exact committed artifact bytes.
        refs = [
            ArtifactRef.from_dict(raw)
            for raw in previous["artifacts"]
            if Path(raw["relative_path"]).parent != output.relative_to(root)
        ]
        for raw in shard_payload["artifacts"]:
            ref = ArtifactRef.from_dict(raw)
            io.verify_artifact_ref(output, ref)
            refs.append(
                io.artifact(
                    root,
                    io.rooted(
                        root, (output / ref.relative_path).relative_to(root).as_posix()
                    ),
                    ref.record_count,
                )
            )
        updated = dict(previous, artifacts=[ref.to_dict() for ref in refs])
        candidates = io.records(root, updated, "candidates.jsonl", CandidateRecord)
        failures = io.records(root, updated, "failures.jsonl", FailureRecord)
        expected = {item.key: item for item in plan.candidates}
        if len({item.plan.key for item in candidates}) != len(candidates) or any(
            expected.get(item.plan.key) != item.plan for item in candidates + failures
        ):
            raise io.IntegrityError()
        updated["counts"] = {
            "planned": len(plan.candidates),
            "successful": len(candidates),
            "historical_failures": len(failures),
        }
        io.write_atomic_manifest(progress, updated)
        if len(candidates) == len(plan.candidates):
            io.persist_manifest(
                root,
                "generation",
                manifest["typed_config"],
                refs,
                updated["counts"],
                {"split": manifest["upstream_manifests"]["split"]["relative_path"]},
                extra={"split_manifest_hash": manifest["split_manifest_hash"]},
            )
    return {"run_id": plan.generation_run_id, "count": summary.successful}


def main(argv: Sequence[str] | None = None) -> int:
    return io.execute(build_parser, dispatch, argv)


if __name__ == "__main__":
    raise SystemExit(main())
