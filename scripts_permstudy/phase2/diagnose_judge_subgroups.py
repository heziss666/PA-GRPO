"""One bounded, inference-only AB/BA diagnosis over the retained 105 pairs."""

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import time

import torch

from my_reward.judge_qwen import controlled_base_reward
from permstudy.advantages import compute_controlled_advantage
from permstudy.data_pipeline.canonical import canonical_json_bytes, sha256_hex
from permstudy.data_pipeline.io import append_record, scan_jsonl
from permstudy.data_pipeline.isolation import validate_external_to_repository

MODEL = "Qwen/Qwen2.5-7B-Instruct"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
EPS = 1e-6
VIEWS = {"N8-A": range(8), "N8-B": range(8, 16), "N16": range(16)}
REPO = Path(__file__).resolve().parents[2]


def _key(row):
    return row["pair_id"], row["permutation_id"], row["rollout_slot"]


def requests_for(prepared):
    requests = []
    for item in prepared["items"]:
        for slot in range(16):
            key = [prepared["run_id"], item["pair_id"], item["permutation_id"], slot]
            digest = sha256_hex(
                canonical_json_bytes(["judge_request_seed_v1", 42, key])
            )
            requests.append(
                {
                    **item,
                    "run_id": prepared["run_id"],
                    "rollout_slot": slot,
                    "request_id": sha256_hex(canonical_json_bytes(key)),
                    "seed": int(digest[:8], 16) % (2**31),
                }
            )
    if len({_key(r) for r in requests}) != len(requests):
        raise ValueError("duplicate planned judge identity")
    return sorted(
        requests,
        key=lambda r: (
            r["rollout_slot"],
            r["original_question_id"],
            r["permutation_id"],
        ),
    )


def generate_pending(prepared, output, engine, *, batch_size=16, limit=None):
    """Append complete responses only; reject reordered/duplicate/foreign identities."""
    output = Path(output)
    planned = {r["request_id"]: r for r in requests_for(prepared)}
    prior = scan_jsonl(output) if output.exists() else None
    done = set()
    for row in prior.records if prior else ():
        request = planned.get(row["request_id"])
        if request is None or row["request_id"] in done:
            raise ValueError("foreign or duplicate persisted request")
        if any(row.get(k) != v for k, v in request.items() if k != "messages"):
            raise ValueError("persisted request config/identity mismatch")
        done.add(row["request_id"])
    pending = [r for key, r in planned.items() if key not in done]
    if limit is not None:
        pending = pending[:limit]
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        outputs = list(engine.generate(batch))
        by_id = {r["request_id"]: r for r in outputs}
        if len(by_id) != len(outputs) or set(by_id) != {r["request_id"] for r in batch}:
            raise ValueError("backend returned duplicate/missing/foreign requests")
        for request in batch:
            result = by_id[request["request_id"]]
            if not isinstance(result.get("response"), str):
                raise ValueError("backend response is not text")
            row = {k: v for k, v in request.items() if k != "messages"}
            row.update(
                {k: result[k] for k in ("response", "finish_reason", "output_tokens")}
            )
            append_record(output, row)
        print(
            json.dumps(
                {
                    "completed": len(done) + min(start + batch_size, len(pending)),
                    "planned": len(planned),
                }
            ),
            flush=True,
        )


def _distribution(values):
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "min": min(values),
        "max": max(values),
        "q25": float(torch.quantile(tensor, 0.25)),
        "median": float(torch.quantile(tensor, 0.5)),
        "q75": float(torch.quantile(tensor, 0.75)),
    }


def _view(rows, slots):
    rows = sorted((r for r in rows if r["rollout_slot"] in slots), key=_key)
    pair_ids = sorted({r["pair_id"] for r in rows})
    scored = [
        {
            **r,
            **controlled_base_reward(
                r["response"], "A" if r["permutation_id"] == 0 else "B"
            ),
        }
        for r in rows
    ]
    advantage = compute_controlled_advantage(
        torch.tensor([r["base_reward"] for r in scored], dtype=torch.float64),
        [r["pair_id"] for r in scored],
        [r["permutation_id"] for r in scored],
        [r["rollout_slot"] - slots.start for r in scored],
        mode="alc",
        eps=EPS,
    )
    paired = {(r["pair_id"], r["rollout_slot"], r["permutation_id"]): r for r in scored}
    details = []
    for pair_id in pair_ids:
        indices = [i for i, r in enumerate(scored) if r["pair_id"] == pair_id]
        groups = [
            [scored[i] for i in indices if scored[i]["permutation_id"] == p]
            for p in (0, 1)
        ]
        rewards = [[r["base_reward"] for r in group] for group in groups]
        means = [statistics.mean(group) for group in rewards]
        sigmas = [statistics.pstdev(group) for group in rewards]
        sigma = statistics.pstdev(rewards[0] + rewards[1])
        counts = [sum(r["correctness"] > 0 for r in group) for group in groups]
        details.append(
            {
                "pair_id": pair_id,
                "source": groups[0][0]["source"],
                "mu_ab": means[0],
                "mu_ba": means[1],
                "delta": abs(means[0] - means[1]),
                "signed_gap": means[0] - means[1],
                "global_sigma": sigma,
                "local_sigma": sigmas,
                "local_weight": float(advantage.local_weight[indices[0]]),
                "global_nonzero": bool(
                    (advantage.global_advantage[indices].abs() > 1e-6).any()
                ),
                "local_nonzero": [value > 0 for value in sigmas],
                "correct_counts": counts,
                "correctness_states": [
                    "all_correct"
                    if c == len(slots)
                    else "zero_correct"
                    if c == 0
                    else "mixed"
                    for c in counts
                ],
            }
        )
    consistent = correct = 0
    for pair_id in pair_ids:
        for slot in slots:
            a, b = paired[(pair_id, slot, 0)], paired[(pair_id, slot, 1)]
            valid = a["answer"] is not None and b["answer"] is not None
            consistent += bool(valid and a["answer"] != b["answer"])
            correct += bool(a["correctness"] > 0 and b["correctness"] > 0)
    subgroup_count = 2 * len(details)
    result = {
        "pair_count": len(pair_ids),
        "rollout_count": len(rows),
        "ab_accuracy": sum(
            r["correctness"] > 0 for r in scored if r["permutation_id"] == 0
        )
        / (len(rows) / 2),
        "ba_accuracy": sum(
            r["correctness"] > 0 for r in scored if r["permutation_id"] == 1
        )
        / (len(rows) / 2),
        "format_valid_rate": sum(r["answer"] is not None for r in scored) / len(rows),
        "invalid_count": sum(r["answer"] is None for r in scored),
        "length_finish_count": sum(r.get("finish_reason") == "length" for r in rows),
        "semantic_consistency": consistent / (len(rows) / 2),
        "consistent_accuracy": correct / (len(rows) / 2),
        "mean_delta": statistics.mean(d["delta"] for d in details),
        "mean_global_sigma": statistics.mean(d["global_sigma"] for d in details),
        "mean_local_weight": statistics.mean(d["local_weight"] for d in details),
        "delta_distribution": _distribution([d["delta"] for d in details]),
        "global_sigma_distribution": _distribution(
            [d["global_sigma"] for d in details]
        ),
        "local_weight_distribution": _distribution(
            [d["local_weight"] for d in details]
        ),
        "global_zero_sigma_fraction": sum(d["global_sigma"] == 0 for d in details)
        / len(details),
        "local_zero_sigma_fraction": sum(
            v == 0 for d in details for v in d["local_sigma"]
        )
        / subgroup_count,
        "global_nonzero_pair_fraction": sum(d["global_nonzero"] for d in details)
        / len(details),
        "local_nonzero_subgroup_fraction": sum(
            v for d in details for v in d["local_nonzero"]
        )
        / subgroup_count,
        "global_mixed_local_constant_pair_fraction": sum(
            d["global_sigma"] > 0 and 0 in d["local_sigma"] for d in details
        )
        / len(details),
        "local_weight_ge_half_fraction": sum(d["local_weight"] >= 0.5 for d in details)
        / len(details),
        "correctness_states": {
            label: dict(Counter(d["correctness_states"][p] for d in details))
            for p, label in enumerate(("AB", "BA"))
        },
    }
    result["accuracy_gap"] = abs(result["ab_accuracy"] - result["ba_accuracy"])
    return result, details


def analyze_rollouts(rows):
    """Require the complete single N16 experiment; invalid answers stay in denominators."""
    identities = set()
    metadata = {}
    for row in rows:
        key = _key(row)
        if (
            type(key[1]) is not int
            or type(key[2]) is not int
            or key[1] not in (0, 1)
            or not 0 <= key[2] < 16
            or key in identities
        ):
            raise ValueError("invalid/duplicate diagnostic identity")
        identities.add(key)
        info = row["original_question_id"], row["source"]
        if metadata.setdefault(key[0], info) != info:
            raise ValueError("mixed pair metadata")
    if not rows or identities != {
        (p, perm, slot) for p in metadata for perm in (0, 1) for slot in range(16)
    }:
        raise ValueError("incomplete N16 experiment")
    output = {}
    for source in ("overall", "math", "reclor"):
        subset = [r for r in rows if source == "overall" or r["source"] == source]
        if not subset:
            continue
        output[source] = {}
        details = {}
        for view, slots in VIEWS.items():
            output[source][view], values = _view(subset, slots)
            details[view] = {d["pair_id"]: d for d in values}
        changes = []
        for pair_id, a in details["N8-A"].items():
            b, full = details["N8-B"][pair_id], details["N16"][pair_id]
            changes.append(
                {
                    "pair_id": pair_id,
                    "views": {k: details[k][pair_id] for k in VIEWS},
                    "attenuated": a["delta"] >= 0.25
                    and full["delta"] <= a["delta"] / 2,
                    "disappeared": a["delta"] >= 0.25 and full["delta"] == 0,
                    "stable_gap": min(a["delta"], b["delta"], full["delta"]) >= 0.25
                    and a["signed_gap"] * b["signed_gap"] > 0,
                }
            )
        output[source]["comparisons"] = {
            "attenuated_count": sum(d["attenuated"] for d in changes),
            "disappeared_count": sum(d["disappeared"] for d in changes),
            "stable_gap_count": sum(d["stable_gap"] for d in changes),
            "pairs": changes,
        }
    return output


def prepare(root, destination):
    from permstudy.data_pipeline import trainer_export
    from permstudy.data_pipeline.schema import PairRecord, PermutationRecord
    from scripts_permstudy.data import _common as io
    from scripts_permstudy.phase2 import run_real_smoke as pipeline

    evidence = io.read_json(
        REPO
        / "artifacts/data_pipeline/phase2/production_200_retained_negative_check_20261015.json"
    )
    items, bindings = [], {}
    for source in ("math", "reclor"):
        gen, ver = (
            evidence["run_ids"][source + suffix]
            for suffix in ("_generation", "_verification")
        )
        prefix = root / "phase2" / gen / ver
        verification = io.load_manifest(
            root, f"verification/{ver}/manifest.json", "verification"
        )
        _, _, _, questions, candidates, _, _ = pipeline._load_real_verification(
            root, verification, pipeline.load_phase2_split
        )
        paths = {
            "pair_view": prefix / "ai_positive_screen_provisional_pairs.jsonl",
            "permutation_view": prefix / "label_checked_permutation_view.jsonl",
        }
        for label, path in paths.items():
            digest = sha256_hex(path.read_bytes())
            if digest != evidence["artifact_hashes"][source + "_" + label]:
                raise ValueError("retained source view differs from approved hash")
            bindings[source + "_" + label] = digest

        def records(path, cls):
            return [
                cls.from_dict({k: v for k, v in r.items() if k != "record_hash"})
                for r in scan_jsonl(path).records
            ]

        pairs = records(paths["pair_view"], PairRecord)
        perms = records(paths["permutation_view"], PermutationRecord)
        rows = trainer_export.build_trainer_rows(questions, candidates, pairs, perms)
        rows = pipeline.controlled_training_rows(rows)
        for row in rows:
            if row["extra_info"]["split"] != "train":
                raise ValueError("Holdout data in diagnostic")
            items.append({**row["extra_info"], "messages": row["prompt"]})
    if len(items) != 210 or len({r["pair_id"] for r in items}) != 105:
        raise ValueError("diagnostic must use exactly 105 retained pairs")
    config = {
        "model": MODEL,
        "revision": REVISION,
        "temperature": 1.0,
        "top_p": 1.0,
        "seed": 42,
        "seed_mapping": "judge_request_seed_v1",
        "n": 16,
        "max_tokens": 256,
        "max_prompt_tokens": 4096,
        "eps": EPS,
        "std_correction": 0,
        "gap_definition": "delta_ge_0.25",
        "local_weight_threshold": 0.5,
        "input_hash": sha256_hex(canonical_json_bytes(items)),
        "upstream_views": bindings,
        "prompt_version": pipeline.CONTROLLED_TRAINING_PROMPT_VERSION,
        "reward_version": "unchanged_PA_base_controlled_v1",
    }
    config["engine"] = {
        "batch_size": 16,
        "dtype": "bfloat16",
        "max_model_len": 4352,
        "max_num_seqs": 32,
        "gpu_memory_utilization": 0.85,
        "enforce_eager": True,
    }
    import subprocess

    config["git_sha"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    prepared = {
        "config": config,
        "run_id": sha256_hex(canonical_json_bytes(config)),
        "items": items,
    }
    io.persist_bytes(
        root,
        destination.relative_to(root).as_posix(),
        canonical_json_bytes(prepared),
        len(items),
    )
    return prepared


class NativeEngine:
    def __init__(self, model_path):
        import vllm
        from transformers import AutoConfig
        from evaluation.evaluate_models import render_chat_prompt

        if (
            Path(model_path).name != REVISION
            or AutoConfig.from_pretrained(model_path, local_files_only=True).model_type
            != "qwen2"
        ):
            raise ValueError("expected the pinned Qwen2.5 snapshot")
        self.module, self.render = vllm, render_chat_prompt
        args = vllm.EngineArgs(
            model=model_path,
            tokenizer=model_path,
            dtype="bfloat16",
            seed=42,
            tensor_parallel_size=1,
            max_model_len=4352,
            gpu_memory_utilization=0.85,
            max_num_seqs=32,
            enforce_eager=True,
            disable_log_stats=True,
        )
        self.engine = vllm.LLMEngine.from_engine_args(args)
        self.tokenizer = self.engine.get_tokenizer_group().get_lora_tokenizer()

    def generate(self, requests):
        expected = {r["request_id"] for r in requests}
        for request in requests:
            prompt = self.render(self.tokenizer, request["messages"], "qwen2")
            if len(self.tokenizer.encode(prompt, add_special_tokens=False)) > 4096:
                raise ValueError(
                    "diagnostic prompt exceeds training context; no truncation allowed"
                )
            sampling = self.module.SamplingParams(
                n=1, seed=request["seed"], temperature=1.0, top_p=1.0, max_tokens=256
            )
            self.engine.add_request(request["request_id"], prompt, sampling)
        outputs = {}
        while self.engine.has_unfinished_requests():
            for output in self.engine.step():
                if output.request_id not in expected:
                    raise ValueError("foreign native engine request")
                if not output.finished:
                    continue
                if output.request_id in outputs or len(output.outputs) != 1:
                    raise ValueError("duplicate or multi-output native request")
                completion = output.outputs[0]
                outputs[output.request_id] = {
                    "request_id": output.request_id,
                    "response": completion.text,
                    "finish_reason": completion.finish_reason,
                    "output_tokens": len(completion.token_ids),
                }
        return list(outputs.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "analyze"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--directory", default="phase2/judge-subgroups-qwen25-v1")
    parser.add_argument("--model-path")
    args = parser.parse_args()
    root = args.data_root.resolve()
    validate_external_to_repository(REPO, root)
    directory = (root / args.directory).resolve()
    if not directory.is_relative_to(root):
        raise ValueError("diagnostic output must stay in the private data root")
    from scripts_permstudy.data import _common as io

    prepared_path = directory / "prepared.json"
    if args.command == "prepare":
        prepared = prepare(root, prepared_path)
        print(json.dumps({"run_id": prepared["run_id"], "pairs": 105, "planned": 3360}))
        return
    prepared = io.read_json(prepared_path)
    if prepared["run_id"] != sha256_hex(
        canonical_json_bytes(prepared["config"])
    ) or prepared["config"]["input_hash"] != sha256_hex(
        canonical_json_bytes(prepared["items"])
    ):
        raise ValueError("prepared diagnostic input was changed")
    raw = directory / "rollouts.jsonl"
    if args.command == "run":
        started = time.monotonic()
        from transformers import AutoTokenizer
        from evaluation.evaluate_models import render_chat_prompt

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, local_files_only=True
        )
        lengths = [
            len(
                tokenizer.encode(
                    render_chat_prompt(tokenizer, r["messages"], "qwen2"),
                    add_special_tokens=False,
                )
            )
            for r in prepared["items"]
        ]
        if max(lengths) > 4096:
            raise ValueError(
                "preflight context too long; no truncation or GPU inference"
            )
        io.persist_bytes(
            root,
            (directory / "template-preflight.json").relative_to(root).as_posix(),
            canonical_json_bytes(
                {
                    "chat_template_sha256": sha256_hex(
                        tokenizer.chat_template.encode()
                    ),
                    "max_input_tokens": max(lengths),
                    "input_count": len(lengths),
                }
            ),
            1,
        )
        engine = NativeEngine(args.model_path)
        # Same engine and stream for tiny preflight and the full experiment.
        generate_pending(prepared, raw, engine, limit=4)
        first = scan_jsonl(raw).records
        if not all(
            controlled_base_reward(
                r["response"], "A" if r["permutation_id"] == 0 else "B"
            )["answer"]
            is not None
            for r in first[:4]
        ):
            raise ValueError(
                "tiny preflight output contract failed; full run not started"
            )
        generate_pending(prepared, raw, engine)
        completed = len(scan_jsonl(raw).records)
        io.persist_bytes(
            root,
            (directory / f"runtime-{completed}.json").relative_to(root).as_posix(),
            canonical_json_bytes(
                {
                    "elapsed_seconds": time.monotonic() - started,
                    "model": MODEL,
                    "revision": REVISION,
                    "completed": len(scan_jsonl(raw).records),
                }
            ),
            1,
        )
    else:
        rows = scan_jsonl(raw).records

        # Resume validation binds every record to the exact prepared identity and run.
        class NoEngine:
            def generate(self, requests):
                raise ValueError("analysis requires complete saved rollout records")

        generate_pending(prepared, raw, NoEngine())
        result = analyze_rollouts(rows)
        io.persist_bytes(
            root,
            (directory / "statistics.private.json").relative_to(root).as_posix(),
            canonical_json_bytes(result),
            1,
        )
        public = {
            source: {
                view: metrics
                for view, metrics in values.items()
                if view != "comparisons"
            }
            for source, values in result.items()
        }
        for source, values in result.items():
            public[source]["comparisons"] = {
                k: v for k, v in values["comparisons"].items() if k != "pairs"
            }
        io.persist_bytes(
            root,
            (directory / "statistics.aggregate.json").relative_to(root).as_posix(),
            canonical_json_bytes(public),
            1,
        )
        print(
            json.dumps(
                {
                    "run_id": prepared["run_id"],
                    "completed": len(rows),
                    "training_steps": 0,
                }
            )
        )


if __name__ == "__main__":
    main()
