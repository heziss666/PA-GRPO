# Linux Fidelity Confirmation Report

## Scope and provenance

- Repository: `heziss666/PA-GRPO`
- Harness commit: `f4f801ebed5056ab6cd72ce8d6fabb7f1743759d`
- Pinned PA upstream commit: `0ee9abd903cb4ac4945f1176e943d20436470096`
- Date checked: 2026-10-05
- Runtime: Ubuntu 24.04.5 LTS on WSL2, Python 3.12.3
- PyTorch: 2.6.0+cpu
- Ray: 2.49.0
- GPU visibility: NVIDIA GeForce RTX 5070 Ti Laptop GPU, driver 616.92, 12227 MiB
- Execution device: CPU tensors; GPU visibility was checked independently
- vLLM: not installed, because this synthetic advantage confirmation does not use rollout inference

This report covers Task 2A only. It does not claim that vLLM, FSDP, LoRA,
full GRPO training, or an 8B model has run.

## Path-execution proof

The tests call the real `verl.trainer.ppo.ray_trainer.compute_advantage()` with
a real `DataProto`. A test-only wrapper around
`apply_group_baseline_from_returns()` records a sentinel without replacing the
function's behavior.

All three required checks passed:

1. The test sentinel confirms `apply_group_baseline_from_returns()` was called.
2. Captured stdout/stderr does not contain `fallback to original GRPO`.
3. `data.meta_info` contains `pair_baseline_metrics`.

The unequal-length result also differs from the original pre-pair-baseline GRPO
advantages, so the final values cannot be explained by the fallback branch.

## Paper states

The paper-form global advantage operates on one scalar reward per response:

```text
(reward - group_mean) / (group_std + epsilon)
```

Changing only response token length should not change this scalar definition.
The paper also specifies a low-standard-deviation threshold gate. No numerical
threshold with reliable provenance has been established, so the controlled
implementation leaves that gate disabled by setting its threshold to `None`.

## Official code does

The pinned official path first calls
`compute_grpo_outcome_advantage(..., norm_adv_by_std_in_grpo=False)`, broadcasting
the group-centered scalar value through `response_mask`. It then collapses
`returns` using an unmasked mean across the padded response width inside
`apply_group_baseline_from_returns()`, applies a population group baseline, and
clips the resulting scalar advantage.

The function has a broad exception handler that can fall back to original GRPO.
The path-execution checks above ensure that fallback was not used in this run.

## Official Linux run observes

### Equal-length case

Setup:

```text
rewards = [1, 1, -1, -1]
lengths = [8, 8, 8, 8]
group_ids = [0, 0, 0, 0]
```

Results:

```text
official-real       = [ 1.0,  1.0, -1.0, -1.0]
official-compatible = [ 1.0,  1.0, -1.0, -1.0]
paper               ≈ [ 1.0,  1.0, -1.0, -1.0]
```

- Official-real vs compatible L-infinity difference: `0.0`
- Official-real vs paper L-infinity difference: `9.536743e-07`
- Sign disagreement rate: `0.0`

### Unequal-length case

Setup:

```text
rewards = [1, 1, -1, -1]
lengths = [2, 8, 4, 10]
group_ids = [0, 0, 0, 0]
```

Results:

```text
official-real       = [ 0.44721359,  1.34164059, -0.44721359, -1.34164059]
official-compatible = [ 0.44721359,  1.34164071, -0.44721359, -1.34164071]
paper               ≈ [ 0.99999905,  0.99999905, -0.99999905, -0.99999905]
original GRPO       = [ 1.0,         1.0,        -1.0,        -1.0       ]
```

- Official-real vs compatible L-infinity difference: `1.192093e-07`
- Official-real vs paper mean absolute difference: `0.44721350`
- Official-real vs original GRPO mean absolute difference: `0.44721350`
- Sign disagreement rate: `0.0`

The real Linux execution therefore confirms implementation-level response-length
dependence in the pinned official PA path. This dependence is absent from the
paper's response-level scalar formulation.

One official diagnostic,
`pair_baseline/std_of_pair_stds`, is undefined for this one-group fixture and is
stored as JSON `null`; it is not used to compute advantages.

## Project controlled implementation does

`permstudy.advantages.global_advantage_paper()` computes statistics directly
from response-level scalar rewards and is independent of response masks and
token counts. It preserves the optional strict sigma gate but does not invent a
default threshold. The official-compatible helper remains available only for
characterization and reproduction comparisons.

## Project decision

1. Treat response-length dependence as confirmed for the pinned official path.
2. Keep PA-Official behavior unchanged for sanity reproduction.
3. Use the response-level paper implementation for controlled PA/EIS/ALC
   comparisons.
4. Do not modify vendored `verl` as part of this audit.
5. Do not begin Task 2B Trainer integration until this report is reviewed and
   confirmed by the user.
6. Do not begin EIS/ALC or large-scale training before Task 2B identity
   propagation and controlled evaluator tests are complete.

## Reproduction

```bash
python -m pytest tests_permstudy/test_linux_official_advantage.py -q
python scripts_permstudy/linux_fidelity/run_official_advantage_fidelity.py \
  --git-commit f4f801ebed5056ab6cd72ce8d6fabb7f1743759d
```

Machine-readable results:

- `advantage_equal_length.json`
- `advantage_unequal_length.json`
- `environment.txt`
