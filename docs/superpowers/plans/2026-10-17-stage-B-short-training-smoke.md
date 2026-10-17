# Stage B short-training Smoke execution plan

**Status: PREPARATION ONLY — GPU, derived-data publication and training NOT AUTHORIZED.**

**Goal:** Check real Qwen2.5-7B rollout, controlled reward/advantage, LoRA
updates and checkpoint restoration, not algorithm effectiveness.

**Architecture:** Reuse `verl.trainer.main_ppo`, the existing controlled reward
and advantage adapters, FSDP2/LoRA and synchronous vLLM. Keep the Stage A
snapshot immutable; propose a separately hashed, balanced answer-tag execution
view derived only from it. No algorithm, parser, Gate or launcher development.

**Tech stack:** Python 3.12, torch 2.6.0+cu124, vLLM 0.8.5,
Transformers 4.57.1, PEFT 0.17.1; vendored local `verl`, not a pip replacement.

**Spec:** The user-approved Stage A/B/C/D plan in this conversation, and
`docs/experiments/2026-10-17-smoke-only-stage-A-release.md`.

## Fixed baseline and authority

- Inspected repository baseline: `97fc3d73835768facdacab84e4cc1702eb3dafd3`.
- Stage A Parquet SHA256:
  `15b3b0944fed63d8269109c05df5298e768bde9a87455b8e0e5c60e919ce06fb`.
- Stage A export: `ad8355ce93da7a1050ac95d628f3b905b188761673f2238a56197b72d31031e1`.
- 380 pairs: 146 MATH + 234 ReClor; 760 rows, Train only; 17 UNSURE excluded.
- Preserve MATH WARN, historical formal FAIL and all frozen source/label hashes.
- GPU authorization must explicitly cover the derived execution view, resource
  price/cap, five proposed optimizer steps below and checkpoint replay.
- Main merge, Pair-v2, evaluation/accuracy claims and formal four-method runs HOLD.

## CPU precheck: measured results and required integration choice

The Stage A Parquet was independently read and its hash/760-row semantic digest
checked. Cached tokenizer only, with `local_files_only=True`; no model weights,
network download, vLLM engine, training process or derived files were created.

| Native Qwen2 chat-template prompt lengths | Frozen direct | Proposed answer-tag view |
|---|---:|---:|
| Rows | 760 | 760 |
| Minimum / median | 350 / 662 | 396 / 708 |
| P95 / maximum | 1477.5 / 1999 | 1523.5 / 2045 |
| Over 4096 tokens | 0 | 0 |

Tokenizer/model revision: `a09a35458c702b33eeacc393d103063234e8bc28`,
`Qwen/Qwen2.5-7B-Instruct`. Native chat-template SHA256:
`cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f`.

**Do not train directly on the frozen direct-prompt file.** Its instructions
request bare A/B. `my_reward/judge_qwen.py:controlled_base_reward` and the
controlled trainer require a unique `<answer>A/B</answer>` tag: measured on
synthetic gold A, bare `A` gives -1.4 while `<answer>A</answer>` gives 1.2.
Controlled modes reject an override back to permissive parsing.

Existing `run_real_smoke.controlled_training_rows()` already supplies
`pairwise_judge_answer_tag_v1`. Its full-760-row in-memory result has canonical
semantic SHA256
`c636de2e0c808e59546bd2c521275fd40f7a0015de687431997f008eb8d85b2d`.
Every `extra_info` and `reward_model` field remains identical. Do not change
reward rules to accommodate the direct prompt.

The existing `prepare-training` CLI expects a formal single-run export
RunManifest, whereas Stage A is a bounded multi-run export receipt. Do not pass
that receipt to the CLI or fabricate a RunManifest. After explicit approval,
call the existing conversion function on the hash-checked Stage A rows and
write an independent execution view using the existing Parquet facilities.

For the minimal GPU test, propose **16 pairs, 8 per source, 32 rows**, selected
before generation: rank unique questions within each source by
`SHA256(UTF8("42:" + original_question_id))`, tie-break by question ID; take
the first 8 of each. Alternate MATH/ReClor pair groups in that rank order and
retain each complete AB/BA group. Each 16-prompt training batch then contains
4 MATH + 4 ReClor pairs. This is a plumbing subset, not a re-screen or a
performance-selected benchmark. Bind its exact pair list privately, source
Parquet hash, prompt version/hash, row semantic hash, Parquet hash and code SHA
in its Smoke-only receipt. Never modify Stage A, restore excluded pairs or
claim this subset is the full 380-pair dataset. Re-read the written view and
compare every field before GPU authorization is exercised.

This proposed selection was also constructed in memory only: 16 pairs / 32
rows, each 16-row batch containing 8 MATH + 8 ReClor rows. Its canonical row
SHA256 is `0bd4383b623b5b2e786d08ae58f4b8f481ed9e40067e67e7fa01f957816bd466`.
No subset Parquet or release receipt has been published. Both this selection
and the answer-tag adaptation require explicit approval as part of Stage B.

## Resource proposal and remote checks

Propose one **80GB Ampere/Hopper GPU**, e.g. A100-80GB / H800 / H100, not a
24/32/48GB substitute or a new Blackwell/software migration in this Smoke.
80GB is a conservative proposal, not a proven minimum or guarantee. Historical
Qwen3 Task 4 reached about 79 GiB reserved even with CPU offload; Qwen2.5 and
this smaller batch/context differ, so remeasure rather than copy that result.
Official [vLLM 0.8.5 installation requirements](https://docs.vllm.ai/en/v0.8.5/getting_started/installation/gpu.html)
cover Linux/CUDA GPU compatibility; the actual torch/vLLM/FlashAttention binary
combination must pass remote imports. NVIDIA documents the
[A100 80GB](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/a100-80gb-datasheet-update-a4-nvidia-1485612-r12-web.pdf)
and [H100 memory configurations](https://www.nvidia.com/en-sg/data-center/h100/).

- Host RAM: prefer at least 128GB for actor/ref/optimizer CPU offload.
- External data disk: **at least 200 GiB free**, not merely total capacity,
  before this proposal. Full FP32 model shards can approach 30GB per saved
  checkpoint, plus optimizer/extra/dataloader state; five checkpoint outputs
  and temporary space must fit. Existing model caches count against free
  space. Do not assume LoRA training produces adapter-only resume checkpoints.
- All data, HF/verl caches, checkpoints, logs and TensorBoard files external to
  the Git checkout. The reward hard-codes `logs/judge_qwen3_8b.log` and
  `tb_judge_qwen3_8b`; relocate these via verified links into the private run
  root on the disposable remote checkout. Preserve any existing targets first;
  do not overwrite, remove or commit them. All console/rollout logs are private.
- Confirm the cached, unquantized exact model revision and tokenizer; do not
  download/re-quantize or replace models under this plan without approval.
- Linux preflight: `nvidia-smi`, Python/package versions, `pip check`, vendored
  `verl.__file__`, `torch.cuda.is_available()`, BF16 support, free disk/RAM,
  complete model shard hashes and identical source/execution Parquet hashes.
  Only run the GPU checks after resource authorization.
- Current Windows CPU environment has Hydra/Ray/PEFT/torch/tensordict but no
  vLLM. CPU composition cannot certify Linux CUDA compatibility.

Budget proposal: **20–45 minutes if the cached training environment works,
maximum 60 billed GPU minutes total**, including initialization/checkpoints.
This is an estimate, not measured Qwen2.5 training throughput. Record actual
provider billing start/end. Current instance inventory and hourly price are
unknown and were not queried. If confirmed price is P per card-hour, the
compute ceiling is approximately P (provider rounding may differ), plus
separately confirmed disk charges. No rental or spend is authorized now.

## Fixed run configuration and commands (future execution only)

Do not invoke the old `scripts/run_judge_qwen.sh`: it defaults to Qwen3,
old datasets and background execution. The Task 4 launcher is also a fixed
historical contract, not this dataset's verifier. Use the actual trainer CLI
in the foreground with resolved Hydra config and captured private logs.

The future operator sets `MODEL_PATH` to the verified pinned local snapshot,
`SMOKE_DATA` to the approved 32-row answer-tag view and `RUN_ROOT` to a new
external private directory. No machine-specific path belongs in Git.

```bash
set -euo pipefail
: "${MODEL_PATH:?verified pinned local snapshot required}"
: "${SMOKE_DATA:?approved hash-checked answer-tag Smoke view required}"
: "${RUN_ROOT:?new external private run directory required}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
COMMON=(
  algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=false
  grouping.identity_mode=explicit
  permutation_training.eps=1e-6
  permutation_training.sigma_gate_enabled=false
  permutation_training.sigma_gate_threshold=null
  data.train_files="$SMOKE_DATA" data.val_files="$SMOKE_DATA"
  data.train_batch_size=16 data.val_batch_size=16 data.seed=42
  data.max_prompt_length=4096 data.max_response_length=256
  data.filter_overlong_prompts=false data.truncation=error data.shuffle=false
  data.dataloader_num_workers=0
  actor_rollout_ref.model.path="$MODEL_PATH"
  actor_rollout_ref.model.tokenizer_path="$MODEL_PATH"
  actor_rollout_ref.model.lora_rank=32 actor_rollout_ref.model.lora_alpha=64
  actor_rollout_ref.model.target_modules=all-linear
  actor_rollout_ref.model.enable_gradient_checkpointing=true
  actor_rollout_ref.actor.strategy=fsdp2 actor_rollout_ref.ref.strategy=fsdp2
  actor_rollout_ref.actor.optim.lr=1e-5
  actor_rollout_ref.actor.ppo_mini_batch_size=16
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.use_kl_loss=true
  actor_rollout_ref.actor.kl_loss_coef=0.001
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.fsdp_config.param_offload=true
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true
  actor_rollout_ref.ref.fsdp_config.param_offload=true
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.mode=sync
  actor_rollout_ref.rollout.n=2 actor_rollout_ref.rollout.temperature=1.0
  actor_rollout_ref.rollout.top_p=1.0
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.gpu_memory_utilization=0.50
  actor_rollout_ref.rollout.enforce_eager=true
  actor_rollout_ref.rollout.max_num_seqs=32
  actor_rollout_ref.rollout.max_model_len=4352
  actor_rollout_ref.rollout.max_num_batched_tokens=8192
  actor_rollout_ref.rollout.load_format=safetensors
  actor_rollout_ref.rollout.layered_summon=true
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
  trainer.balance_batch=true trainer.n_gpus_per_node=1 trainer.nnodes=1
  trainer.total_epochs=2 trainer.save_freq=1 trainer.test_freq=-1
  trainer.val_before_train=false trainer.max_actor_ckpt_to_keep=2
  trainer.logger='["console","tensorboard"]'
  trainer.project_name=stage_B_plumbing_smoke
  reward_model.reward_manager=batch
  custom_reward_function.path="$PROJECT_ROOT/my_reward/judge_qwen.py"
  custom_reward_function.name=compute_score
)
# CPU composition only; inspect before ever invoking main_ppo.
python - "${COMMON[@]}" permutation_training.mode=global_only <<'PY'
import sys
from pathlib import Path
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from permstudy.trainer_advantage import validate_controlled_training_config
with initialize_config_dir(config_dir=str(Path('verl/trainer/config').resolve()), version_base=None):
    config = compose(config_name='ppo_trainer', overrides=sys.argv[1:])
validate_controlled_training_config(config)
print(OmegaConf.to_yaml(config, resolve=True))
PY
```

Keep FSDP actor's existing FP32 initialization and mixed-precision policy;
BF16 model files/vLLM do not imply that all training parameters/optimizer states
are BF16. Do not silently lower rank, quantize or disable KL to fit memory.
`data.seed=42` is the loader seed, not a claim that every existing worker RNG is
42: this code uses rank+1000 for rollout-worker RNG and default vLLM seed 0.
Preserve and record those existing semantics; no seed-mapping development here.

The exact `COMMON` array above was parsed from this document and composed on
CPU for `global_only`, `eis`, `alc` and `pa_controlled`; all four compositions
and `validate_controlled_training_config()` calls passed (32 rollouts per
step). The existing focused runtime-config test also passed, 1/1. These are
configuration checks, not real trainer/CUDA compatibility or gradient tests.
Both Bash command blocks also passed `bash -n` syntax-only checking; their
training commands were not executed. No new test/launcher code was added.

After authorization and successful preflight, use the same `COMMON` array:

```bash
: "${GPU_DEADLINE_EPOCH:?confirmed billing start plus 3600 seconds required}"
# Helper shown in the plan, not an installed new launcher/framework.
run_smoke() {
  local name="$1" mode="$2" steps="$3"; shift 3
  local dir="$RUN_ROOT/$name"
  local remaining=$((GPU_DEADLINE_EPOCH - $(date +%s) - 120))
  test "$remaining" -ge 600                # reserve backup/shutdown time
  local limit=1200
  if [ "$remaining" -lt "$limit" ]; then limit="$remaining"; fi
  test ! -e "$dir"                         # never reuse earlier evidence
  mkdir -p "$dir"
  export TENSORBOARD_DIR="$dir/tensorboard"
  local args=("${COMMON[@]}"
    permutation_training.mode="$mode"
    trainer.total_training_steps="$steps"
    trainer.resume_mode=disable
    trainer.experiment_name="$name"
    trainer.default_local_dir="$dir/checkpoints"
    trainer.rollout_data_dir="$dir/rollouts" "$@")
  python -m verl.trainer.main_ppo "${args[@]}" --cfg job --resolve \
    > "$dir/resolved_config.private.yaml"
  timeout --signal=TERM --kill-after=60s "${limit}s" \
    python -m verl.trainer.main_ppo "${args[@]}" \
    2>&1 | tee "$dir/train.private.log"
}
run_smoke global global_only 2
# Inspect finite metrics, real updates and both checkpoints before continuing.
run_smoke global_restore global_only 2 \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path="$RUN_ROOT/global/checkpoints/global_step_1"
# The original horizon stays 2; this replays exactly one update from step 1.
# Verify actual model/optimizer/scheduler/RNG and data.pt restoration first.
run_smoke alc alc 1
run_smoke pa pa_controlled 1 permutation_training.consistency_coef=1.0
```

Each stage is conditional on the preceding checks, remaining disk and total
budget. Four processes, **five executed optimizer steps total**: two baseline,
one restored/replayed, one ALC and one PA. The fresh ALC/PA runs start from the
base model, not the baseline-trained adapter. EIS's local path is covered by
existing CPU tests; a separate EIS GPU run is not proposed in this minimal
budget. `n=2` is a plumbing setting, not the proposed formal comparison N=8.
At 16 prompts x 2 rollouts, each step has 32 response rows: up to 160 across
these five updates, not a new full-dataset Judge diagnostic.

## Acceptance and hard stops

- Real vLLM initializes; no backend fallback or model/revision substitution.
- Rollouts carry exact pair/permutation/slot identities and complete pairs;
  masked tokens, reward components, advantages, gradients and loss are finite.
  Existing controlled assertions must run, not be disabled. Check
  `base_reward + consistency == total` and zero consistency for Global/ALC;
  PA uses its existing coefficient and no sigma gate.
- Existing `adv/global_std`, `adv/local_std`, `adv/final_std`,
  `adv/final_rms`, `adv/local_weight_mean` metrics must be emitted.
  Confirm the ALC local and PA consistency paths were actually exercised. If
  a small batch is degenerate, report INCONCLUSIVE for that component rather
  than claim it passed or repeatedly sample to manufacture a signal.
- Positive finite gradient/update evidence; checkpoint step-1 vs step-2 LoRA
  tensors must be finite and at least one changed. Nonzero LoRA-B alone is not
  sufficient proof of this run's update. Verify LoRA rank32/alpha64 and no
  unexpected trainable base weights. The existing worker writes optional
  `actor/lora_adapter/adapter_model.safetensors` sidecars but catches errors;
  a "saved" log line is not proof that the adapter exists. Inspect actual
  tensors/files (or verified LoRA entries of the full model shard), not just
  console messages.
- Global restoration must log successful model AND optimizer/extra/RNG and
  dataloader restoration, then produce a newly updated step-2 checkpoint.
  A checkpoint filename or a metadata-only load is not acceptance. Bitwise
  equivalence of re-generated rollouts is not promised by this Smoke.
- Stop on source/execution hash mismatch, identity/label errors, Holdout or
  quarantined rows, NaN/Inf, missing reward/advantage components, no real LoRA
  update, failed restore, CUDA/ABI/dependency errors or OOM. Do not change
  algorithms, model, batch/context or data to continue without approval.
- Stop if the first rollout has not completed within 10 minutes after worker
  initialization, any command exceeds its 20-minute timeout, fewer than 40 GiB
  remain before the next checkpoint, or total billing approaches 60 minutes.
  The total cap overrides per-command timeouts. Leave time for backup/shutdown;
  do not start a new command without sufficient budget. Verify each finished
  process released its GPU memory before starting the next; unexpected live
  workers are a stop condition, not permission to kill unrelated workloads.
- Reusing Train data as `val_files` only satisfies loader construction;
  validation is disabled. It is not held-out evaluation and yields no accuracy
  claim. Do not use the old Task 4 verifier as this Smoke's release gate.
- Keep raw logs/rollouts/checkpoints private; back up and hash-check evidence,
  then shut down only under explicit execution/shutdown authorization. Preserve
  data disks. Public result: steps, finite/update/restore checks, peak VRAM,
  wall/billed time, cost and input/code identities, not raw text or tokens.

Only a successfully executed, reviewed Smoke may be called training-chain
verified. No reward/loss trend, ranking of methods, formal dataset release or
PA-GRPO effectiveness conclusion follows. The next decision is explicit
approval of this scoped proposal and confirmed server price/specification;
until then GPU and all training remain HOLD.
