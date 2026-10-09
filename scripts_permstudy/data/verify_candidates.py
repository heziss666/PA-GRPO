"""Validated command boundary for verify_candidates."""

from collections.abc import Sequence
from pathlib import Path
import math
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts_permstudy.data import _common as io
from scripts_permstudy.data.plan_generation import load_plan
from permstudy.data_pipeline import verification
from permstudy.data_pipeline.ids import run_id
from permstudy.data_pipeline.lineage import role_bound_stage_config
from permstudy.data_pipeline.schema import (
    CandidateRecord,
    QuestionRecord,
    QuestionVerificationRecord,
    Source,
    VerificationRecord,
)
from permstudy.data_pipeline.verification import (
    verify_math_question,
    verify_reclor_question,
)


def verification_config(generation_hash, gold_timeout, candidate_timeout):
    if any(
        not math.isfinite(value) or value <= 0
        for value in (gold_timeout, candidate_timeout)
    ):
        raise io.CLIContractError()
    return role_bound_stage_config(
        {
            "verification_config_schema": "verification_config_v1",
            "gold_timeout_seconds": gold_timeout,
            "candidate_timeout_seconds": candidate_timeout,
            "math_verifier_version": verification.MATH_VERIFY_VERSION,
            "math_parser_version": verification.MATH_PARSER_VERSION,
            "reclor_verifier_version": verification.RECLOR_VERIFIER_VERSION,
            "reclor_parser_version": verification.RECLOR_PARSER_VERSION,
        },
        {"generation": generation_hash},
        {"generation"},
    )


def load_candidates(root, manifest):
    plan = load_plan(root, manifest)
    candidates = io.records(root, manifest, "candidates.jsonl", CandidateRecord)
    expected = {p.key: p for p in plan.candidates}
    if (
        len(candidates) != len(expected)
        or len({c.plan.key for c in candidates}) != len(expected)
        or any(expected.get(c.plan.key) != c.plan for c in candidates)
    ):
        raise io.IntegrityError()
    questions = io.records(root, manifest, "questions.jsonl", QuestionRecord)
    by_id = {q.original_question_id: q for q in questions}
    return [
        by_id[qid] for qid in sorted({p.original_question_id for p in plan.candidates})
    ], candidates


def load_verification(root, manifest):
    generation = io.upstream(root, manifest, "generation")
    questions, candidates = load_candidates(root, generation)
    gold = io.records(
        root, manifest, "question_verifications.jsonl", QuestionVerificationRecord
    )
    verified = io.records(root, manifest, "verifications.jsonl", VerificationRecord)
    parameters = manifest["typed_config"]["parameters"]
    config = io.checked(
        verification_config,
        generation["output_manifest_hash"],
        parameters["gold_timeout_seconds"],
        parameters["candidate_timeout_seconds"],
        integrity=True,
    )
    if config != manifest["typed_config"]:
        raise io.IntegrityError()
    by_question = {q.original_question_id: q for q in questions}
    by_gold = {q.original_question_id: q for q in gold}
    by_candidate = {c.plan.candidate_id: c for c in candidates}
    if len(by_gold) != len(gold) or set(by_gold) != set(by_question):
        raise io.IntegrityError()
    if any(
        g.verification_run_id != manifest["run_id"]
        or g.source != by_question[g.original_question_id].source
        for g in gold
    ):
        raise io.IntegrityError()
    expected = {
        c.plan.candidate_id
        for c in candidates
        if by_gold[c.plan.original_question_id].gold_parse_status == "ok"
    }
    if (
        len({v.candidate_id for v in verified}) != len(verified)
        or {v.candidate_id for v in verified} != expected
    ):
        raise io.IntegrityError()
    for v in verified:
        p = by_candidate[v.candidate_id].plan
        if (
            v.verification_run_id != manifest["run_id"]
            or v.generation_run_id != generation["run_id"]
            or (v.original_question_id, v.source, v.generator_id)
            != (p.original_question_id, p.source, p.generator_id)
        ):
            raise io.IntegrityError()
    return generation, questions, candidates, gold, verified


def build_parser() -> io.argparse.ArgumentParser:
    parser = io.parser("verify_candidates")
    parser.add_argument("--generation-manifest", required=True)
    parser.add_argument("--gold-timeout-seconds", type=float, required=True)
    parser.add_argument("--candidate-timeout-seconds", type=float, required=True)
    return parser


def dispatch(args, root):
    # Reject invalid limits before reading any source payload.
    verification_config(
        "0" * 64, args.gold_timeout_seconds, args.candidate_timeout_seconds
    )
    generation = io.load_manifest(root, args.generation_manifest, "generation")
    questions, candidates = load_candidates(root, generation)
    config = verification_config(
        generation["output_manifest_hash"],
        args.gold_timeout_seconds,
        args.candidate_timeout_seconds,
    )
    identity = run_id("verification", config)
    gold, verified = [], []
    for question in questions:
        group = [
            c
            for c in candidates
            if c.plan.original_question_id == question.original_question_id
        ]
        if question.source is Source.RECLOR:
            q, records = verify_reclor_question(
                question,
                group,
                verification_run_id=identity,
                verifier_timeout_seconds=args.candidate_timeout_seconds,
            )
        else:
            q, records = verify_math_question(
                question,
                group,
                args.gold_timeout_seconds,
                args.candidate_timeout_seconds,
                verification_run_id=identity,
            )
        gold.append(q)
        verified.extend(records)
    prefix = f"verification/{identity}"
    refs = [
        io.persist_records(root, f"{prefix}/question_verifications.jsonl", gold),
        io.persist_records(root, f"{prefix}/verifications.jsonl", verified),
    ]
    return io.output(
        io.persist_manifest(
            root,
            "verification",
            config,
            refs,
            {"questions": len(gold), "candidates": len(verified)},
            {"generation": args.generation_manifest},
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    return io.execute(build_parser, dispatch, argv)


if __name__ == "__main__":
    raise SystemExit(main())
