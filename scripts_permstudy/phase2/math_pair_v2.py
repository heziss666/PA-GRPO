"""Fixed CPU-only MATH rematching; quality decisions precede one diagnostic."""

from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from statistics import mean
import json
import subprocess
import sys

from permstudy.data_pipeline import pairs as pairing, permutations
from permstudy.data_pipeline.canonical import (
    canonical_json_bytes,
    normalize_text_v1,
    sha256_hex,
)
from permstudy.data_pipeline.ids import pair_id as semantic_pair_id
from permstudy.data_pipeline.isolation import sanitize_public_manifest
from permstudy.data_pipeline.schema import PairRecord, VerificationStatus
from scripts_permstudy.data import _common as io
from scripts_permstudy.phase2 import smoke_release_gate as gate, run_real_smoke as real


GROUPS = {
    "length": ("token_length", "character_length"),
    "latex": ("latex_backslashes", "latex_dollars", "brace_open", "brace_close"),
    "layout": (
        "line_count",
        "blank_lines",
        "paragraph_count",
        "list_items",
        "heading_lines",
        "code_fences",
        "whitespace",
    ),
    "punct": (
        "period",
        "comma",
        "colon",
        "semicolon",
        "exclamation",
        "question_mark",
        "paren_open",
        "paren_close",
        "square_open",
        "square_close",
    ),
}
POLICY = {
    "version": "math_pair_v2_equal_five_factors_v1",
    "count_normalization": "abs(a-b)/(1+a+b)_exact_rational",
    "groups": GROUPS,
    "aggregation": "within_group_mean_then_five_factor_mean",
    "generator_distance": "same_0_different_1",
    "proposal": "one_best_per_question_reject_or_unsure_falls_back_to_original",
    "tie_break": "score_original_pair_first_token_gap_positive_id_negative_id",
    "scope": "146_existing_approved_math_questions_same_generation_and_verification_run",
    "quality": "positive_PASS_and_negative_INCORRECT_CONFIRMED_required",
    "diagnostic": "existing_fixed_5fold_question_groups_bootstrap2000_once_no_tuning",
}
BASELINE_RUN = "7d4c4f9812f89a26133677baf84a33a988df99f23b3c3b7d86da815fd78e3421"


def count_distance(a, b):
    if any(type(x) is not int or x < 0 for x in (a, b)):
        raise ValueError("structure counts must be nonnegative integers")
    return Fraction(abs(a - b), 1 + a + b)


def distance(a, b):
    features = [gate.structure_features(x["text"], x["token_count"]) for x in (a, b)]
    values = [
        sum((count_distance(features[0][k], features[1][k]) for k in keys), Fraction())
        / len(keys)
        for keys in GROUPS.values()
    ]
    values.append(Fraction(a["generator"] != b["generator"]))
    return sum(values, Fraction()) / 5


def choose_proposal(options, original_ids):
    if not options:
        raise ValueError("missing original verified pair")
    return min(
        options,
        key=lambda ab: (
            distance(*ab),
            (ab[0]["candidate_id"], ab[1]["candidate_id"]) != original_ids,
            abs(ab[0]["token_count"] - ab[1]["token_count"]),
            ab[0]["candidate_id"],
            ab[1]["candidate_id"],
        ),
    )


def describe_rows(rows):
    features = {
        label: [
            gate.structure_features(r["text"], r["token_count"])
            for r in rows
            if r["label"] == label
        ]
        for label in (0, 1)
    }
    return {
        "generators": {
            name: dict(
                sorted(
                    Counter(r["generator"] for r in rows if r["label"] == label).items()
                )
            )
            for label, name in ((1, "chosen"), (0, "rejected"))
        },
        "structure": {
            key: {
                "chosen_mean": mean(r[key] for r in features[1]),
                "rejected_mean": mean(r[key] for r in features[0]),
            }
            for key in sorted(gate.structure_features("", 0))
        },
    }


def validate_decisions(proposals, decisions):
    expected = {p["proposal_id"]: p for p in proposals}
    if len(expected) != len(proposals) or len(decisions) != len(expected):
        raise ValueError("decision coverage mismatch")
    result = {}
    for d in decisions:
        pid = d.get("proposal_id")
        if pid not in expected or pid in result:
            raise ValueError("foreign or duplicate decision")
        p = expected[pid]
        for k in ("original_question_id", "generation_run_id", "verification_run_id"):
            if d.get(k) != p[k]:
                raise ValueError("decision lineage mismatch")
        for side in ("positive", "negative"):
            for k in ("candidate_id", "response_hash"):
                if d.get(f"{side}_{k}") != p[side][k]:
                    raise ValueError("decision candidate mismatch")
            if (
                not isinstance(d.get(f"{side}_reason"), str)
                or not d[f"{side}_reason"].strip()
            ):
                raise ValueError("missing quality reason")
        if d.get("positive_verdict") not in {"PASS", "FAIL", "UNSURE"} or d.get(
            "negative_verdict"
        ) not in {
            "INCORRECT_CONFIRMED",
            "ACTUALLY_CORRECT",
            "UNSURE",
        }:
            raise ValueError("unknown quality verdict")
        result[pid] = (
            d["positive_verdict"] == "PASS"
            and d["negative_verdict"] == "INCORRECT_CONFIRMED"
        )
    return result


@contextmanager
def diagnostic_once(folder):
    # Retained after errors, too: never silently refit after inspecting a result.
    with (folder / "diagnostic.started").open("xb") as marker:
        marker.write(b"fixed_real_diagnostic_started\n")
    yield


def load_inputs(root):
    _, _, hashes = gate.load_checked_dataset(root)
    records = gate._read_records(
        root / gate.PREFIX / "combined-provisional-pairs.jsonl", PairRecord
    )
    baseline_path = root / f"phase2/smoke_release/{BASELINE_RUN}/report-private.json"
    baseline = io.read_json(baseline_path)
    if baseline["input_hashes"] != hashes:
        raise io.IntegrityError()
    hashes["baseline_report"] = sha256_hex(baseline_path.read_bytes())
    questions, candidates, verified = {}, {}, {}
    for gen, ver in sorted(
        {
            (p.generation_run_id, p.verification_run_id)
            for p in records
            if p.source.value == "math"
        }
    ):
        manifest = io.load_manifest(
            root, f"verification/{ver}/manifest.json", "verification"
        )
        _, _, _, qs, cs, _, vs = real._load_real_verification(
            root, manifest, real.load_phase2_split
        )
        questions.update({(gen, q.original_question_id): q for q in qs})
        candidates.update({c.plan.key: c for c in cs})
        verified.update({(v.generation_run_id, v.candidate_id): v for v in vs})
    return records, questions, candidates, verified, hashes, baseline


def candidate_item(candidate, tokenizer):
    text = normalize_text_v1(candidate.response)
    return {
        "candidate_id": candidate.plan.candidate_id,
        "generator": candidate.plan.generator_id,
        "text": text,
        "token_count": len(tokenizer.encode(text, add_special_tokens=False)),
        "response_hash": sha256_hex(text.encode()),
    }


def prepare(args, root):
    from transformers import AutoTokenizer

    records, questions, candidates, verified, hashes, _ = load_inputs(root)
    selected = [p for p in records if p.source.value == "math"]
    revision = {p.tokenizer_revision for p in selected}
    if len(revision) != 1:
        raise io.IntegrityError()
    revision = next(iter(revision))
    cache = (
        root
        / f"cache/tokenizers/models--Qwen--Qwen2.5-7B-Instruct/snapshots/{revision}"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        cache, local_files_only=True, trust_remote_code=False
    )
    config = {
        "policy": POLICY,
        "upstream_bindings": hashes,
        "tokenizer_revision": revision,
        "script_sha256": sha256_hex(Path(__file__).read_bytes()),
    }
    identity = sha256_hex(canonical_json_bytes(config))
    by_question = defaultdict(list)
    for key, c in candidates.items():
        if key in verified:
            by_question[(key[0], c.plan.original_question_id)].append(
                (c, verified[key])
            )
    proposals = []
    for pair in selected:
        scope = by_question[(pair.generation_run_id, pair.original_question_id)]
        if any(
            v.verification_run_id != pair.verification_run_id
            or c.plan.split != pair.split
            or c.plan.split_manifest_hash != pair.split_manifest_hash
            for c, v in scope
        ):
            raise io.IntegrityError()
        groups = pairing.build_dedup_groups(scope)
        sides = {
            s: [
                candidate_item(candidates[g.representative_key], tokenizer)
                for g in groups
                if g.verification_status is s
            ]
            for s in (VerificationStatus.CORRECT, VerificationStatus.INCORRECT)
        }
        options = [
            (p, n)
            for p in sides[VerificationStatus.CORRECT]
            for n in sides[VerificationStatus.INCORRECT]
        ]
        old_ids = pair.positive_candidate_id, pair.negative_candidate_id
        old = next(
            (
                x
                for x in options
                if (x[0]["candidate_id"], x[1]["candidate_id"]) == old_ids
            ),
            None,
        )
        if old is None or (old[0]["token_count"], old[1]["token_count"]) != (
            pair.positive_token_count,
            pair.negative_token_count,
        ):
            raise io.IntegrityError()
        best = choose_proposal(options, old_ids)
        if best == old:
            continue
        p = {
            "original_pair_id": pair.pair_id,
            "original_question_id": pair.original_question_id,
            "generation_run_id": pair.generation_run_id,
            "verification_run_id": pair.verification_run_id,
            "positive": best[0],
            "negative": best[1],
            "old_distance": str(distance(*old)),
            "new_distance": str(distance(*best)),
        }
        p["proposal_id"] = sha256_hex(
            canonical_json_bytes({"config_hash": identity, **p})
        )
        p["question"] = questions[
            (pair.generation_run_id, pair.original_question_id)
        ].to_dict()
        proposals.append(p)
    proposals.sort(key=lambda p: p["original_question_id"])
    payload = {
        "run_id": identity,
        "config": config,
        "proposals": proposals,
        "pair_count": len(selected),
        "review_kind": "AI_ASSISTED_PROVISIONAL_NOT_HUMAN_CERTIFIED",
    }
    io.persist_bytes(
        root,
        f"phase2/math_pair_v2/{identity}/proposals.private.json",
        canonical_json_bytes(payload),
        1,
    )
    return {"run_id": identity, "count": len(proposals)}


def make_pair(old, proposal, identity):
    a, b = proposal["positive"], proposal["negative"]
    return replace(
        old,
        pair_run_id=identity,
        pair_id=semantic_pair_id(
            old.generation_run_id,
            old.original_question_id,
            a["candidate_id"],
            b["candidate_id"],
            old.schema_version,
        ),
        positive_candidate_id=a["candidate_id"],
        negative_candidate_id=b["candidate_id"],
        positive_response_hash=a["response_hash"],
        negative_response_hash=b["response_hash"],
        positive_generator_id=a["generator"],
        negative_generator_id=b["generator"],
        positive_token_count=a["token_count"],
        negative_token_count=b["token_count"],
    )


def finalize(args, root):
    if not io.HASH.fullmatch(args.run_id):
        raise io.CLIContractError()
    folder = io.rooted(root, f"phase2/math_pair_v2/{args.run_id}")
    payload = io.read_json(folder / "proposals.private.json")
    config = payload["config"]
    if (
        payload["run_id"] != args.run_id
        or sha256_hex(canonical_json_bytes(config)) != args.run_id
        or config["policy"] != json.loads(canonical_json_bytes(POLICY))
        or config["script_sha256"] != sha256_hex(Path(__file__).read_bytes())
    ):
        raise io.IntegrityError()
    records, questions, candidates, _, hashes, baseline = load_inputs(root)
    if hashes != config["upstream_bindings"]:
        raise io.IntegrityError()
    # Recompute proposals, enforcing candidate hashes, original cohort and scoring.
    prepare(args, root)
    decisions_path = io.rooted(root, args.decisions)
    decisions = io.read_json(decisions_path)
    accepted = validate_decisions(payload["proposals"], decisions)
    decisions_hash = sha256_hex(decisions_path.read_bytes())
    identity = sha256_hex(
        canonical_json_bytes(
            {
                "upstream_bindings": {
                    "selection": args.run_id,
                    "quality_decisions": decisions_hash,
                },
                "policy": POLICY,
            }
        )
    )
    by_old = {p["original_pair_id"]: p for p in payload["proposals"]}
    old_math = [p for p in records if p.source.value == "math"]
    decision_map = {d["proposal_id"]: d for d in decisions}
    new_math = []
    for old in old_math:
        proposal = by_old.get(old.pair_id)
        if proposal:
            d = decision_map[proposal["proposal_id"]]
            if (
                d["positive_candidate_id"] == old.positive_candidate_id
                and d["positive_verdict"] == "FAIL"
            ) or (
                d["negative_candidate_id"] == old.negative_candidate_id
                and d["negative_verdict"] == "ACTUALLY_CORRECT"
            ):
                # Do not silently use a newly disproved original as the fallback.
                raise io.IntegrityError()
        new_math.append(
            make_pair(old, proposal, identity)
            if proposal and accepted[proposal["proposal_id"]]
            else replace(old, pair_run_id=identity)
        )
    pair_bytes = pairing.pair_manifest_bytes(new_math)
    pair_hash = sha256_hex(pair_bytes)
    perms = [
        v
        for p in new_math
        for v in permutations.build_permutations(
            p, upstream_bindings={"pairs": pair_hash}
        )
    ]
    # Validate semantic candidate joins and mirror mappings with the existing exporter.
    from permstudy.data_pipeline import trainer_export

    trainer_export.build_trainer_rows(
        [questions[(p.generation_run_id, p.original_question_id)] for p in old_math],
        list(candidates.values()),
        new_math,
        perms,
    )
    new_rows = []
    for p in new_math:
        for label, cid, count in (
            (1, p.positive_candidate_id, p.positive_token_count),
            (0, p.negative_candidate_id, p.negative_token_count),
        ):
            c = candidates[(p.generation_run_id, cid)]
            new_rows.append(
                {
                    "original_question_id": p.original_question_id,
                    "label": label,
                    "source": "math",
                    "generator": c.plan.generator_id,
                    "text": c.response,
                    "token_count": count,
                }
            )
    io.persist_bytes(
        root,
        f"phase2/math_pair_v2/{args.run_id}/pairs.jsonl",
        pair_bytes,
        len(new_math),
    )
    io.persist_bytes(
        root,
        f"phase2/math_pair_v2/{args.run_id}/permutations.jsonl",
        permutations.permutation_manifest_bytes(perms),
        len(perms),
    )
    with diagnostic_once(folder):
        result, predictions = gate.diagnose_source(new_rows)
    counts = {
        "original_math_pairs": len(old_math),
        "retained_math_pairs": len(new_math),
        "proposed_replacements": len(payload["proposals"]),
        "accepted_replacements": sum(accepted.values()),
        "fallback_to_original": sum(not a for a in accepted.values()),
        "same_generator_before": sum(
            p.positive_generator_id == p.negative_generator_id for p in old_math
        ),
        "same_generator_after": sum(
            p.positive_generator_id == p.negative_generator_id for p in new_math
        ),
        "lost_math_pairs": 0,
        "added_pairs": 0,
        "diagnostic_executions": 1,
        "GPU_runs": 0,
        "restored_unsure": 0,
        "parquet_exports": 0,
    }
    report = {
        "run_id": identity,
        "selection_run_id": args.run_id,
        "config": config,
        "decisions_sha256": decisions_hash,
        "pair_hash": pair_hash,
        "completion_counts": counts,
        "pair_v1": baseline["source_results"]["math"],
        "pair_v2": result,
        "structure_v2": describe_rows(new_rows),
        "structure_v1": describe_rows(
            [
                {
                    "label": label,
                    "generator": c.plan.generator_id,
                    "text": c.response,
                    "token_count": count,
                }
                for p in old_math
                for label, cid, count in (
                    (1, p.positive_candidate_id, p.positive_token_count),
                    (0, p.negative_candidate_id, p.negative_token_count),
                )
                for c in [candidates[(p.generation_run_id, cid)]]
            ]
        ),
        "review_kind": payload["review_kind"],
        "historical_formal_gates_unchanged": True,
        "formal_training_authorized": False,
        "out_of_sample_independent_validation": False,
    }
    io.persist_bytes(
        root,
        f"phase2/math_pair_v2/{args.run_id}/diagnostic.private.json",
        canonical_json_bytes(report),
        1,
    )
    io.persist_bytes(
        root,
        f"phase2/math_pair_v2/{args.run_id}/oof.private.json",
        canonical_json_bytes(predictions),
        len(predictions),
    )
    public = sanitize_public_manifest(
        {
            "schema_version": POLICY["version"],
            "git_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=io.REPO, text=True
            ).strip(),
            "run_ids": {"selection": args.run_id, "pair_v2": identity},
            "artifact_hashes": {
                "pairs": pair_hash,
                "quality_decisions": decisions_hash,
                "diagnostic": sha256_hex(canonical_json_bytes(report)),
            },
            "question_counts": {"math": len(new_math)},
            "pair_count": len(new_math),
            "permutation_count": len(perms),
            "completion_counts": counts,
            "phase_status": "MATH_PAIR_V2_CPU_COMPLETE_REVIEW_PENDING",
            "real_generation_performed": False,
        }
    )
    io.persist_bytes(
        root,
        f"phase2/math_pair_v2/{args.run_id}/summary.json",
        canonical_json_bytes(public),
        1,
    )
    return {"run_id": identity, "count": counts["accepted_replacements"]}


def parser():
    p = io.parser("math_pair_v2")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    f = sub.add_parser("finalize")
    f.add_argument("--run-id", required=True)
    f.add_argument("--decisions", required=True)
    return p


def main(argv=None):
    return io.execute(
        parser,
        lambda a, r: {"prepare": prepare, "finalize": finalize}[a.command](a, r),
        argv,
    )


if __name__ == "__main__":
    sys.exit(main())
