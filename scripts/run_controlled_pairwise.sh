#!/usr/bin/env bash
# Short controlled wiring run; formal runs use the data pipeline's paired parquet.
set -euo pipefail

: "${MODEL_PATH:?MODEL_PATH is required (the pinned Qwen3-8B snapshot)}"
: "${TRAIN_DATASET:?TRAIN_DATASET is required (complete, contiguous AB/BA rows)}"
: "${RUN_DIR:?RUN_DIR is required (private output directory)}"
METHOD="${METHOD:-global_only}"
case "$METHOD" in
  global_only|eis|alc|pa_controlled) ;;
  *) echo "unsupported METHOD: $METHOD" >&2; exit 2 ;;
esac
EXPERIMENT_NAME="$METHOD"
if [[ "$METHOD" == "pa_controlled" ]]; then
  EXPERIMENT_NAME=pa_controlled_nogate
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
if [[ ! -f "$TRAIN_DATASET" ]]; then
  echo "TRAIN_DATASET does not exist" >&2
  exit 2
fi
if [[ -e "$RUN_DIR" ]]; then
  echo "RUN_DIR must be new to keep checkpoints attributable to this run" >&2
  exit 2
fi

# One final FSDP checkpoint can be large; retain one and check capacity first.
python - "$RUN_DIR" "${MIN_FREE_GB:-45}" <<'PY'
from pathlib import Path
import shutil
import sys
import torch
target = Path(sys.argv[1]).expanduser().resolve()
probe = target.parent
while not probe.exists():
    probe = probe.parent
free_gb = shutil.disk_usage(probe).free / 1024**3
if free_gb < float(sys.argv[2]):
    raise SystemExit(f"insufficient checkpoint storage: {free_gb:.1f} GiB free")
if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU required for the controlled training run")
print(f"GPU={torch.cuda.get_device_name(0)} free_disk_gib={free_gb:.1f}")
PY

mkdir -p "$RUN_DIR" "$PROJECT_ROOT/logs"
export TENSORBOARD_DIR="$RUN_DIR/tensorboard"
STEPS="${STEPS:-2}"
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=false \
  grouping.identity_mode=explicit permutation_training.mode="$METHOD" \
  permutation_training.consistency_coef=1.0 \
  permutation_training.sigma_gate_enabled=false \
  permutation_training.sigma_gate_threshold=null \
  permutation_training.sigma_gate_source=unspecified \
  data.train_files="$TRAIN_DATASET" data.val_files="$TRAIN_DATASET" \
  data.train_batch_size="${TRAIN_BATCH_SIZE:-16}" data.seed=42 data.shuffle=false \
  data.max_prompt_length="${MAX_PROMPT_LENGTH:-4096}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH:-256}" \
  data.filter_overlong_prompts=false data.truncation=error \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.tokenizer_path="$MODEL_PATH" \
  actor_rollout_ref.model.lora_rank=32 actor_rollout_ref.model.lora_alpha=64 \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.actor.strategy=fsdp2 actor_rollout_ref.actor.optim.lr=1e-5 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE:-16}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=true actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.mode=sync \
  actor_rollout_ref.rollout.n="${ROLLOUT_N:-2}" \
  actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.50 \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.max_num_seqs=32 \
  actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN:-4352}" \
  actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=true \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  reward_model.reward_manager=batch \
  custom_reward_function.path="$PROJECT_ROOT/my_reward/judge_qwen.py" \
  custom_reward_function.name=compute_score \
  trainer.balance_batch=true trainer.n_gpus_per_node=1 trainer.nnodes=1 \
  trainer.total_epochs="$STEPS" trainer.total_training_steps="$STEPS" \
  trainer.save_freq="$STEPS" trainer.max_actor_ckpt_to_keep=1 \
  trainer.test_freq=-1 trainer.val_before_train=false trainer.resume_mode=disable \
  trainer.logger='["console"]' trainer.project_name=controlled_pairwise \
  trainer.experiment_name="$EXPERIMENT_NAME" trainer.rollout_data_dir=null \
  trainer.default_local_dir="$RUN_DIR/checkpoints" \
  2>&1 | tee "$RUN_DIR/train.log"
