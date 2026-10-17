"""CPU preparation and explicitly gated launch of one 32GB plumbing Smoke."""

from collections import defaultdict
from pathlib import Path
import os
import subprocess
import sys

import pyarrow.parquet as pq

from permstudy.data_pipeline import trainer_export
from permstudy.data_pipeline.canonical import canonical_json_bytes, sha256_hex
from scripts_permstudy.data import _common as io
from scripts_permstudy.phase2.run_real_smoke import controlled_training_rows

REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
SOURCE = "phase2/smoke_only_exports/ad8355ce93da7a1050ac95d628f3b905b188761673f2238a56197b72d31031e1/SMOKE_ONLY_NOT_FOR_FORMAL_EXPERIMENT.parquet"
SOURCE_HASH = "15b3b0944fed63d8269109c05df5298e768bde9a87455b8e0e5c60e919ce06fb"
SOURCE_RECEIPT_HASH = "a6c9475e5ffb0b9769d50c54348b7a4a84623608df6d0801a27f80626fdb3c50"


def select_short_pairs(rows, tokenizer):
    adapted = controlled_training_rows(rows)
    groups = defaultdict(list)
    for row in adapted:
        if row["extra_info"]["split"] != "train":
            raise io.IntegrityError()
        groups[row["extra_info"]["original_question_id"]].append(row)
    eligible = defaultdict(list)
    lengths = {}
    for qid, group in groups.items():
        group.sort(key=lambda r: r["extra_info"]["permutation_id"])
        if len(group) != 2 or [r["extra_info"]["permutation_id"] for r in group] != [
            0,
            1,
        ]:
            raise io.IntegrityError()
        lens = [
            len(
                tokenizer.apply_chat_template(
                    r["prompt"], tokenize=True, add_generation_prompt=True
                )
            )
            for r in group
        ]
        lengths[qid] = lens
        if max(lens) <= 1024:
            eligible[group[0]["extra_info"]["source"]].append(qid)
    selected = {}
    for source in ("math", "reclor"):
        selected[source] = sorted(
            eligible[source], key=lambda q: (sha256_hex(("42:" + q).encode()), q)
        )[:2]
        if len(selected[source]) != 2:
            raise io.IntegrityError()
    result, selected_lengths = [], []
    for i in range(2):
        for source in ("math", "reclor"):
            qid = selected[source][i]
            result.extend(groups[qid])
            selected_lengths.extend(lengths[qid])
    return result, selected_lengths


def training_overrides(model, data, run, *, project_root=None):
    """Configuration only; standalone ref BF16 is inactive for shared LoRA ref."""
    return [
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=false",
        "grouping.identity_mode=explicit",
        "permutation_training.mode=global_only",
        "permutation_training.sigma_gate_enabled=false",
        "permutation_training.sigma_gate_threshold=null",
        f"data.train_files={data}",
        f"data.val_files={data}",
        "data.train_batch_size=4",
        "data.val_batch_size=4",
        "data.seed=42",
        "data.max_prompt_length=1024",
        "data.max_response_length=256",
        "data.filter_overlong_prompts=false",
        "data.truncation=error",
        "data.shuffle=false",
        "data.dataloader_num_workers=0",
        f"actor_rollout_ref.model.path={model}",
        f"actor_rollout_ref.model.tokenizer_path={model}",
        "actor_rollout_ref.model.lora_rank=32",
        "actor_rollout_ref.model.lora_alpha=64",
        "actor_rollout_ref.model.target_modules=all-linear",
        "actor_rollout_ref.model.enable_gradient_checkpointing=true",
        "+actor_rollout_ref.model.override_config.attn_implementation=sdpa",
        "actor_rollout_ref.model.use_remove_padding=false",
        "actor_rollout_ref.model.use_fused_kernels=false",
        "actor_rollout_ref.actor.strategy=fsdp2",
        "actor_rollout_ref.ref.strategy=fsdp2",
        "actor_rollout_ref.actor.fsdp_config.model_dtype=fp32",
        "actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16",
        "actor_rollout_ref.actor.optim.lr=1e-5",
        "actor_rollout_ref.actor.ppo_mini_batch_size=4",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.actor.use_kl_loss=true",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.fsdp_config.param_offload=true",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=true",
        "actor_rollout_ref.ref.fsdp_config.param_offload=true",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.mode=sync",
        "actor_rollout_ref.rollout.n=2",
        "actor_rollout_ref.rollout.temperature=1.0",
        "actor_rollout_ref.rollout.top_p=1.0",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.50",
        "actor_rollout_ref.rollout.enforce_eager=true",
        "actor_rollout_ref.rollout.max_num_seqs=8",
        "actor_rollout_ref.rollout.max_model_len=1536",
        "actor_rollout_ref.rollout.max_num_batched_tokens=2048",
        "actor_rollout_ref.rollout.load_format=safetensors",
        "actor_rollout_ref.rollout.layered_summon=true",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
        "trainer.balance_batch=true",
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
        "trainer.total_epochs=2",
        "trainer.total_training_steps=2",
        "trainer.save_freq=1",
        "trainer.test_freq=-1",
        "trainer.val_before_train=false",
        "trainer.resume_mode=disable",
        "trainer.max_actor_ckpt_to_keep=2",
        "trainer.logger=[console,tensorboard]",
        "trainer.project_name=smoke32_only",
        "trainer.experiment_name=global_only",
        f"trainer.default_local_dir={run}/checkpoints",
        f"trainer.rollout_data_dir={run}/rollouts",
        "reward_model.reward_manager=batch",
        f"custom_reward_function.path={Path(project_root or io.REPO).as_posix()}/my_reward/judge_qwen.py",
        "custom_reward_function.name=compute_score",
    ]


def compose_config(overrides):
    from hydra import compose, initialize_config_dir
    from permstudy.trainer_advantage import validate_controlled_training_config

    with initialize_config_dir(
        config_dir=str(io.REPO / "verl/trainer/config"), version_base=None
    ):
        config = compose(config_name="ppo_trainer", overrides=overrides)
    validate_controlled_training_config(config)
    return config


def prepare(args, root):
    source = io.rooted(root, args.source)
    if sha256_hex(source.read_bytes()) != SOURCE_HASH:
        raise io.IntegrityError()
    if (
        sha256_hex((source.parent / "manifest.json").read_bytes())
        != SOURCE_RECEIPT_HASH
    ):
        raise io.IntegrityError()
    tokenizer_path = Path(args.tokenizer_path).resolve()
    if REVISION not in tokenizer_path.parts:
        raise io.CLIContractError()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    original = pq.read_table(source).to_pylist()
    if len(original) != 760:
        raise io.IntegrityError()
    rows, lengths = select_short_pairs(original, tokenizer)
    config = {
        "schema": "smoke32_preparation_v1",
        "usage": ["SMOKE_ONLY", "NOT_FOR_FORMAL_EXPERIMENT"],
        "source_sha256": SOURCE_HASH,
        "source_receipt_sha256": SOURCE_RECEIPT_HASH,
        "model_repo": "Qwen/Qwen2.5-7B-Instruct",
        "revision": REVISION,
        "chat_template_sha256": sha256_hex(tokenizer.chat_template.encode()),
        "source_relative_path": source.relative_to(root).as_posix(),
        "row_semantic_sha256": sha256_hex(canonical_json_bytes(rows)),
        "pair_ids": [r["extra_info"]["pair_id"] for r in rows[::2]],
        "prompt_lengths": lengths,
        "selection": "short_complete_pairs_seed42_2_per_source_interleaved",
        "reference_policy": "adapter_disabled_actor_shared_fp32_no_separate_ref",
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=io.REPO, text=True
        ).strip(),
        "script_sha256": sha256_hex(Path(__file__).read_bytes()),
        "overrides": training_overrides("MODEL", "DATA", "RUN", project_root="REPO"),
        "GPU_authorized": False,
        "historical_gate_status": "PASS_WITH_WARNINGS",
    }
    identity = sha256_hex(canonical_json_bytes(config))
    folder = io.rooted(root, f"phase2/smoke32/{identity}")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "SMOKE_ONLY_NOT_FOR_FORMAL_EXPERIMENT.parquet"
    receipt_path = folder / "manifest.json"
    previous = io.read_json(receipt_path) if receipt_path.exists() else None
    if previous and sha256_hex(path.read_bytes()) != previous["parquet_sha256"]:
        raise io.IntegrityError()
    if not path.exists():
        trainer_export._write_parquet_atomically(path, rows)
    if pq.read_table(path).to_pylist() != rows:
        raise io.IntegrityError()
    receipt = {
        "run_id": identity,
        "config": config,
        "parquet_sha256": sha256_hex(path.read_bytes()),
        "row_count": 8,
        "pair_count": 4,
        "roundtrip_exact": True,
    }
    io.persist_bytes(
        root,
        receipt_path.relative_to(root).as_posix(),
        canonical_json_bytes(receipt),
        1,
    )
    compose_config(training_overrides("MODEL", path.as_posix(), folder.as_posix()))
    return {"run_id": identity, "count": 8}


def load_prepared(args, root):
    if not io.HASH.fullmatch(args.run_id):
        raise io.CLIContractError()
    folder = io.rooted(root, f"phase2/smoke32/{args.run_id}")
    receipt = io.read_json(folder / "manifest.json")
    config = receipt["config"]
    data = folder / "SMOKE_ONLY_NOT_FOR_FORMAL_EXPERIMENT.parquet"
    if (
        receipt["run_id"] != args.run_id
        or sha256_hex(canonical_json_bytes(config)) != args.run_id
        or sha256_hex(data.read_bytes()) != receipt["parquet_sha256"]
        or sha256_hex(canonical_json_bytes(pq.read_table(data).to_pylist()))
        != config["row_semantic_sha256"]
        or config["script_sha256"] != sha256_hex(Path(__file__).read_bytes())
        or config["overrides"]
        != training_overrides("MODEL", "DATA", "RUN", project_root="REPO")
        or sha256_hex(io.rooted(root, config["source_relative_path"]).read_bytes())
        != SOURCE_HASH
    ):
        raise io.IntegrityError()
    return folder, data


def check(args, root):
    folder, data = load_prepared(args, root)
    compose_config(training_overrides("MODEL", data.as_posix(), folder.as_posix()))
    return {"run_id": args.run_id, "count": 8}


def launch(args, root):
    if not args.authorize_gpu:
        raise io.PhaseBoundaryError()
    folder, data = load_prepared(args, root)
    model = Path(args.model_path).resolve()
    if REVISION not in model.parts or not 1 <= args.timeout_seconds <= 1200:
        raise io.CLIContractError()
    run = folder / ("restore" if args.resume_step_one else "training")
    if run.exists():
        raise io.IntegrityError()
    overrides = training_overrides(model.as_posix(), data.as_posix(), run.as_posix())
    if args.resume_step_one:
        checkpoint = folder / "training/checkpoints/global_step_1"
        if not (checkpoint / "data.pt").is_file():
            raise io.IntegrityError()
        overrides += [
            "trainer.resume_mode=resume_path",
            f"trainer.resume_from_path={checkpoint.as_posix()}",
        ]
    config = compose_config(overrides)
    from omegaconf import OmegaConf

    run.mkdir()
    io.persist_bytes(
        root,
        (run / "resolved_config.private.yaml").relative_to(root).as_posix(),
        OmegaConf.to_yaml(config, resolve=True).encode(),
        1,
    )
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "CUDA_VISIBLE_DEVICES": "0",
        "TENSORBOARD_DIR": str(run / "tensorboard"),
    }
    # No automatic dtype/backend retry, and no automatic GPU authorization.
    with (run / "train.private.log").open("xb") as log:
        result = subprocess.run(
            [
                "timeout",
                "--signal=TERM",
                "--kill-after=30s",
                str(args.timeout_seconds),
                sys.executable,
                "-m",
                "verl.trainer.main_ppo",
                *overrides,
            ],
            cwd=io.REPO,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise RuntimeError("training stopped; inspect private log; no automatic retry")
    return {"run_id": args.run_id, "count": 8}


def parser():
    result = io.parser("smoke32")
    sub = result.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--source", default=SOURCE)
    p.add_argument("--tokenizer-path", required=True)
    p = sub.add_parser("check")
    p.add_argument("--run-id", required=True)
    p = sub.add_parser("launch")
    p.add_argument("--run-id", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--authorize-gpu", action="store_true")
    p.add_argument("--resume-step-one", action="store_true")
    p.add_argument("--timeout-seconds", type=int, default=900)
    return result


def main(argv=None):
    return io.execute(
        parser,
        lambda a, r: {"prepare": prepare, "check": check, "launch": launch}[a.command](
            a, r
        ),
        argv,
    )


if __name__ == "__main__":
    sys.exit(main())
