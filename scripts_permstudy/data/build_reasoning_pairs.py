"""Validated command boundary for build_reasoning_pairs."""

from collections.abc import Sequence
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts_permstudy.data import _common as io
from scripts_permstudy.data.verify_candidates import load_verification
from permstudy.data_pipeline import pairs
from permstudy.data_pipeline.ids import run_id
from permstudy.data_pipeline.schema import PairRecord


def load_tokenizer(revision, root):
    from transformers import AutoTokenizer
    from transformers.utils.hub import cached_file, extract_commit_hash

    cache_dir = str(io.rooted(root, "cache/tokenizers"))
    config_path = cached_file(
        pairs.TOKENIZER_REPOSITORY,
        "tokenizer_config.json",
        revision=revision,
        cache_dir=cache_dir,
    )
    if extract_commit_hash(config_path, None) != revision:
        raise io.DependencyContractError()
    return AutoTokenizer.from_pretrained(
        pairs.TOKENIZER_REPOSITORY,
        revision=revision,
        cache_dir=cache_dir,
        trust_remote_code=False,
    )


def validate_tokenizer(tokenizer, revision):
    commits = (
        getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
        getattr(tokenizer, "_commit_hash", None),
    )
    if (
        getattr(tokenizer, "name_or_path", None) != pairs.TOKENIZER_REPOSITORY
        or any(commit is not None and commit != revision for commit in commits)
        or not callable(getattr(tokenizer, "encode", None))
    ):
        raise io.DependencyContractError()


def load_pairs(root, manifest):
    verification = io.upstream(root, manifest, "verification")
    generation, questions, candidates, _, verified = load_verification(
        root, verification
    )
    if (
        generation["output_manifest_hash"]
        != manifest["upstream_bindings"]["generation"]
    ):
        raise io.IntegrityError()
    revision = manifest["typed_config"]["parameters"]["tokenizer_revision"]
    config = io.checked(
        pairs.pair_stage_config, revision, manifest["upstream_bindings"], integrity=True
    )
    if config != manifest["typed_config"]:
        raise io.IntegrityError()
    selected = io.records(root, manifest, "pairs.jsonl", PairRecord)
    io.checked(pairs.pair_manifest_bytes, selected, integrity=True)
    by_candidate = {c.plan.candidate_id: c for c in candidates}
    by_verified = {v.candidate_id: v for v in verified}
    by_question = {q.original_question_id: q for q in questions}
    for pair in selected:
        if (
            pair.pair_run_id != manifest["run_id"]
            or pair.generation_run_id != generation["run_id"]
            or pair.verification_run_id != verification["run_id"]
            or pair.tokenizer_revision != revision
            or pair.tokenizer_repo != pairs.TOKENIZER_REPOSITORY
            or pair.original_question_id not in by_question
        ):
            raise io.IntegrityError()
        for candidate_id, status in (
            (pair.positive_candidate_id, "correct"),
            (pair.negative_candidate_id, "incorrect"),
        ):
            if (
                candidate_id not in by_candidate
                or candidate_id not in by_verified
                or by_verified[candidate_id].verification_status.value != status
                or by_candidate[candidate_id].plan.original_question_id
                != pair.original_question_id
            ):
                raise io.IntegrityError()
    return selected


def build_parser() -> io.argparse.ArgumentParser:
    parser = io.parser("build_reasoning_pairs")
    parser.add_argument("--verification-manifest", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    return parser


def dispatch(args, root):
    io.checked(
        pairs.pair_stage_config,
        args.tokenizer_revision,
        {"generation": "0" * 64, "verification": "0" * 64},
    )
    verification = io.load_manifest(root, args.verification_manifest, "verification")
    generation, questions, candidates, _, verified = load_verification(
        root, verification
    )
    bindings = {
        "verification": verification["output_manifest_hash"],
        "generation": generation["output_manifest_hash"],
    }
    config = pairs.pair_stage_config(args.tokenizer_revision, bindings)
    tokenizer = load_tokenizer(args.tokenizer_revision, root)
    validate_tokenizer(tokenizer, args.tokenizer_revision)
    by_candidate = {c.plan.candidate_id: c for c in candidates}
    selected = []
    for question in questions:
        group = [
            (by_candidate[v.candidate_id], v)
            for v in verified
            if v.original_question_id == question.original_question_id
        ]
        pair = pairs.select_pair(
            question,
            group,
            tokenizer,
            args.tokenizer_revision,
            upstream_bindings=bindings,
        )
        if pair is not None:
            selected.append(pair)
    identity = run_id("pairs", config)
    refs = [io.persist_records(root, f"pairs/{identity}/pairs.jsonl", selected)]
    return io.output(
        io.persist_manifest(
            root,
            "pairs",
            config,
            refs,
            {"pairs": len(selected)},
            {
                "verification": args.verification_manifest,
                "generation": verification["upstream_manifests"]["generation"][
                    "relative_path"
                ],
            },
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    return io.execute(build_parser, dispatch, argv)


if __name__ == "__main__":
    raise SystemExit(main())
