# H800 Global-only two-step Smoke: training path passed, restore pending

## Scope and outcome

**H800_GLOBAL_ONLY_TWO_STEP_SMOKE_PASS_RESTORE_PENDING.** After the user
approved adding the missing TensorBoard dependency and retrying, the same
prepared two-step configuration completed with launcher exit code 0.
This verifies the narrow Global-only training path, not algorithm efficacy,
the other three methods, full checkpoint recovery or formal data release.

Execution code remains `d0bc2afa9a59f9bbe8bc1dbe7ddc9e5ab7ca4c72`.
Preparation identity remains
`01600655484852bda0643e05472eaa9a1579735c17d8586630b4cfe2e071674d`.
It identifies the same input/config, not a unique execution attempt. A fresh
external private root separates this retry from the previous H800 dependency
failure and the 32GB OOM. Both earlier failure records remain unchanged.

TensorBoard 2.20.0 and its missing dependencies were installed after a dry-run
check rejected any existing-package version changes. SummaryWriter imported
on CPU and `pip check` passed. PyTorch 2.6.0+cu124, vLLM 0.8.5, Transformers
4.57.1 and PEFT 0.17.1 were unchanged. The resolved configurations compare
equal after normalizing only the new output/private-data root.

## Actual checks

The retry started at 2026-10-17 08:42:24 UTC on H800 PCIe. After the user
reported completion and switched the instance to no-GPU mode, the following
read-only CPU checks were performed against the persisted artifacts:

| Check | Result |
|---|---|
| Frozen execution view | 4 pairs, 2 MATH + 2 ReClor, 8 AB/BA rows; hash unchanged |
| Optimizer steps | 2, latest checkpoint iteration is 2 |
| Rollouts | 8 per step, 16 total; all 16 answer formats valid |
| Step-1 reward outcomes | 4 correct, 4 incorrect; range -0.8 to 1.2 |
| Step-2 reward outcomes | 5 correct, 3 incorrect; range -0.8 to 1.2 |
| Gradient norms | 4.9134426 and 2.8033493; finite and nonzero |
| Global advantage standard deviation | 0.9999990 and 0.7071060; nonzero |
| Recorded scalar metrics | All values across 74 tags finite |
| LoRA precision | All 392 saved tensors FP32 and finite in both checkpoints |
| LoRA change step 1 to 2 | 392/392 tensors changed; maximum absolute delta about 1.00136e-5 |
| LoRA B tensors | 196/196 nonzero at step 2 |
| Pair identity | 16 base records; 8 unique pair/slot mirror groups, no unpaired or skipped groups |
| Global-only reward policy | All saved consistency components zero |
| Checkpoint files | Model, optimizer, extra state, dataloader state and LoRA sidecars present at both steps |

The existing advantage path validates dimensions, masks and finiteness at
runtime. Identity checks used the frozen pair IDs and reward logs, not an
assumption based on row count. Token-weighted advantage means need not be
zero when response lengths differ. Near-zero aggregate policy loss does not
by itself demonstrate no learning; actual finite nonzero gradients and the
changed LoRA tensors provide the update evidence here.

Both LoRA sidecars were successfully loaded on CPU for elementwise comparison.
This is **not** a full model/optimizer/RNG/dataloader restore replay. That
additional optimizer update was not run or claimed. Only the Global-only
mode was exercised; local-advantage diagnostic metrics do not mean EIS/ALC
training occurred. No performance ranking is supported by two tiny batches.

## Preservation and authorization

Private logs, resolved config, TensorBoard events, rollouts and reward logs
were copied locally; trainer-log/config hashes match the server. Large full
checkpoints and both complete adapter sidecars remain on the server data disk.
Adapter comparison and hashes were checked there on CPU. A slow adapter
download was stopped, its incomplete local copy marked `.partial` and not
used for verification; adapter/full-checkpoint local backup is deferred.
Public evidence contains only sanitized hashes, counts and this report,
not raw prompts, responses, gold or machine paths.

The user reports the instance is now in no-GPU mode; no training or GPU
allocation was started during this inspection. This does not independently
verify the platform's billing status, and no shutdown was requested here.

The original 380-pair Stage A snapshot, MATH WARN, historical formal FAIL,
quarantine and UNSURE exclusions are unchanged. Formal four-method training,
Pair-v2 and full checkpoint replay remain pending separate decisions.
