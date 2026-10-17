# 32GB Smoke: CPU preparation, not GPU execution

## Decision and scope

**SMOKE32_CPU_PREPARED_GPU_UNVERIFIED.** The user approved CPU preparation
and data transfer, not real model loading or training. This is a risky
resource-adaptation attempt, not evidence that 32GB can train the model.
The existing 80GB Stage B plan remains a separate alternative; this narrower
proposal replaces its 16-pair/five-update resource budget only for a future
explicitly approved 32GB attempt.

Code: `d0bc2afa9a59f9bbe8bc1dbe7ddc9e5ab7ca4c72`.
Only a thin preparation/launch entry point and focused synthetic tests were
added. Core FSDP, reward/advantage, parser, Pair and Gate algorithms did not
change. Historical MATH WARN/formal FAIL and the 380-pair Stage A snapshot are
unchanged. No UNSURE/quarantined data were restored; no candidate regeneration,
Pair-v2, formal training, evaluation, main merge or GPU shutdown occurred.

## Checked hardware and environment

- Driver reports RTX 4080 SUPER, 32,760 MiB total, compute capability 8.9.
  This differs from the standard retail model's memory specification; the
  provider's exact modified/virtual hardware configuration is not certified.
- Container limits: 66,571,993,088 bytes RAM (62 GiB), 12 CPU cores.
  Host `free`/`nproc` totals are not the allocation. No swap was reported.
- External disk: 250 GiB total, approximately 147 GiB free at precheck.
- Python 3.12.15; torch 2.6.0+cu124; vLLM 0.8.5; Transformers 4.57.1;
  PEFT 0.17.1; Ray 2.59.0; tensordict 0.10.0; xformers 0.0.29.post2;
  pandas 3.0.6; PyArrow 22.0.0; datasets 4.4.1. `pip check` passed.
- External `flash_attn` is absent. HF SDPA, padding removal disabled and
  fused kernels disabled are proposed; vLLM's own backend/kernel is separate
  and has **not** been executed. Import success is not a GPU-kernel test.
- BF16 is advertised by the reported Ada capability; no GPU BF16 operation
  has been executed to certify the provider hardware.
- Four cached weight shards of the fixed Qwen2.5-7B revision were present.
  Each actual SHA256 matched its cache blob digest. An official online
  metadata request did not succeed; this is a cache-integrity check, not a
  new independent comparison with official online LFS metadata. No weights
  were downloaded.

The safetensors headers describe 7,615,616,512 BF16 parameters. Their FP32
storage alone is 30,462,466,048 bytes (about 28.37 GiB), leaving little room
on this GPU for initialization buffers, LoRA, activations and CUDA overhead.
FSDP2 still moves parameters to the GPU before ordinary offload takes effect.
OOM remains a material risk and must not trigger automatic precision changes.

### Reference precision correction

With LoRA rank >0, `RayPPOTrainer.ref_in_actor` is true. It does not initialize
a separate reference worker; `compute_ref_log_prob()` uses the Actor with its
adapter disabled. Thus `ref.fsdp_config.model_dtype=bfloat16` is a defensive
standalone-ref setting and **does not save a second FP32 model in this path**.
The effective shared reference remains the Actor base. Do not force a second
model or change Actor/LoRA precision to make the suggestion appear effective.

## Frozen source and short-pair execution view

The exact Stage A Parquet and adjacent receipt were transferred over SSH and
SHA256 checked. Stage A remains 380 pairs / 760 rows. The derivative is
explicitly `SMOKE_ONLY / NOT_FOR_FORMAL_EXPERIMENT`:

- Use existing `controlled_training_rows()` to change only Judge instructions
  from direct A/B to the already-established answer-tag contract.
- Tokenize with the native Qwen2 chat template at revision
  `a09a35458c702b33eeacc393d103063234e8bc28`, offline/local files only.
- Eligible only if **both** permutations fit 1024 tokens, without truncation.
  Rank eligible questions per source by SHA256 of `42:` plus question ID,
  tie-break by ID; take exactly 2 MATH + 2 ReClor.
- Interleave complete MATH/ReClor pair groups: each batch of 4 prompts
  contains one pair per source. Same candidates, responses, labels,
  identities and provenance; no cross-run candidate splicing or relabeling.
- Selected prompt lengths: 466, 466, 585, 585, 810, 810, 731, 731.
- 4 pairs / 8 rows; exact Parquet readback equality passed. The manifest binds
  source/receipt hashes, selected pair IDs, token lengths, chat template,
  portable configuration, code SHA and row/Parquet hashes.

Preparation identity:
`01600655484852bda0643e05472eaa9a1579735c17d8586630b4cfe2e071674d`.

Parquet SHA256:
`ca13e6c40464dfb31476a17fe9d4eb14b6d36c15dcba2bf4cba55bc2416444c0`.

Canonical row SHA256:
`fd8f5f867f23346b621d9280d11257107c436b35cf94c1b59614203bb731988e`.

Raw files, identities and logs remain external/private. Only sanitized
evidence is committed under `artifacts/data_pipeline/phase2/smoke32_cpu_preparation/`.

## Proposed runtime configuration

| Field | Value |
|---|---|
| First method / updates | Global-only / 2 optimizer steps |
| Train batch / PPO mini / PPO micro | 4 / 4 / 1 |
| Rollout N / responses per step | 2 / 8 |
| Prompt / response limit | 1024 / 256, no truncation |
| vLLM model length / max sequences / batched tokens | 1536 / 8 / 2048 |
| vLLM memory fraction | 0.50 |
| Actor storage / LoRA | FP32 / rank32, alpha64, all-linear |
| HF attention | SDPA; no remove-padding or fused kernels |
| Reference | Shared Actor without adapter, no second model |
| Offload | Actor parameters/optimizer and nominal reference enabled |
| KL | Existing actor KL loss, 0.001; not in reward |
| Checkpoints | model + optimizer + extra + dataloader; keep 2 |

The narrower two-step run plus optional one-step restore replay is estimated
to fit the current disk if at least **110 GiB are free before starting** and
40 GiB remain before subsequent checkpoint writes. This is a safety estimate,
not measured checkpoint size; full FP32 state is not an adapter-only save.
Do not delete earlier data/cache to meet the threshold without approval.
The 62 GiB RAM cap and initialization/transient peaks remain unproven.

## CPU checks actually executed

- 6 new synthetic tests plus 3 existing export tests: 9 passed locally.
  New tests were observed failing before implementation. Ruff/format/diff
  checks passed. A read-only reviewer checked the new entry point; a later
  portability correction prevents local absolute checkout paths from
  entering the canonical configuration binding and was regression-tested.
- A tiny randomly initialized Qwen2 CPU model (not real 7B weights) completed
  SDPA forward/backward using the actual monkey-patch/PEFT path. Only LoRA
  parameters were trainable and FP32; gradients were finite. This does not
  validate real FSDP2/CUDA/optimizer state.
- The remote dirty checkout stayed at `4fcd664` with its original one-line
  modification untouched. Code was imported through a verified small Git
  bundle into an independent detached worktree at the execution commit.
- On the remote CPU, real `RLHFDataset → DataProto → repeat_for_rollout →
  merge_identity_into_extra_infos` passed: 8 input rows, 16 expanded records,
  16 unique pair/permutation/slot identities. No generated tokens involved.
- The actual trainer entry, FSDP worker and vLLM rollout modules imported.
  Local and remote preparation/check validate the same portable namespace.
- No GPU training command has run; checked CUDA state was uninitialized and
  `nvidia-smi` still reported 0 MiB used. GPU compatibility, LoRA updates,
  checkpoint save/restore and actual peak RAM/VRAM are still **PENDING**.

These are local/SSH CPU checks, not GitHub CI and not a completed GPU Smoke.

## Commands and stopping boundary

Use the isolated remote checkout, its existing `pagrpo-p2` interpreter and
external private root. `MODEL_PATH` must identify the hash-checked snapshot
ending in the exact revision. No secret or machine-specific path is committed.

CPU-only, already exercised:

```bash
python -m scripts_permstudy.phase2.smoke32 --data-root "$PAGRPO_DATA_ROOT" \
  prepare --tokenizer-path "$MODEL_PATH"
python -m scripts_permstudy.phase2.smoke32 --data-root "$PAGRPO_DATA_ROOT" \
  check --run-id 01600655484852bda0643e05472eaa9a1579735c17d8586630b4cfe2e071674d
```

**NOT AUTHORIZED; shown for review only:**

```bash
python -m scripts_permstudy.phase2.smoke32 --data-root "$PAGRPO_DATA_ROOT" \
  --private-log private_logs/smoke32-launch.private.log \
  launch --run-id 01600655484852bda0643e05472eaa9a1579735c17d8586630b4cfe2e071674d \
  --model-path "$MODEL_PATH" --timeout-seconds 900 --authorize-gpu
```

Only after explicit GPU authorization may the future operator invoke it.
The entry rejects launch without the explicit flag; presence of a flag in a
document is not human authorization. It uses the pinned data/config, saves
resolved config and private logs, runs the existing trainer under a timeout,
and refuses to overwrite an earlier training directory. Reward logs are
redirected by a verified link outside the new checkout, not into Git.

Propose at most 15 minutes for the initial attempt; do not start an automated
retry. Watch RAM/VRAM and initialization logs. Stop on OOM, CPU OOM-kill,
CUDA/attention/ABI failures, hash or identity mismatch, NaN/Inf, missing
components, unexpected trainable weights or no nonzero update. If the FP32
Actor cannot initialize reliably, report `BLOCKED_RESOURCE`; changing Actor
precision requires separate approval and optimizer-dtype validation.

Success requires real two-step reward/advantage/gradient checks and changed
finite LoRA tensors between checkpoints, not just process exit zero. A
degenerate reward batch is INCONCLUSIVE, not permission to change algorithms
or select responses post hoc. Restore validation, if separately included in
GPU authorization, uses `launch --resume-step-one` to reload the step-1 model,
optimizer, extra/RNG and dataloader and replay step 2 in a new directory. This
executes **one additional optimizer update**, not zero-cost reload verification.
ALC/PA/EIS GPU checks from the larger proposal are not included in this initial
32GB attempt. No formal four-method training is authorized.
