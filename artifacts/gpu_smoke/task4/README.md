# Task 4 real GPU smoke evidence

## Review status

**Task 4 repository evidence: SEALED.**

**Task 4 overall project gate: PASS.**

An independent evidence review found no Critical, Important, or Minor issues and explicitly approved this curated artifact for commit. This status applies to the documented compact evidence and its disclosed retention limits; it does not imply that deleted full-run artifacts were reconstructed or that the strict verifier can be replayed from this directory alone.

The successful execution was `task4_run_003_offload`. It used the training code at commit `22945d2c187d2a748e94221f3e7834fcbca10d61` with a clean working tree. The training did **not** run at the later verifier fix `e38d3a8` or the merge-to-main commit `944e196`.

## Provenance

| Role | Commit | Meaning |
|---|---|---|
| Successful training code | `22945d2c187d2a748e94221f3e7834fcbca10d61` | Code and working tree recorded by the successful run |
| Later verifier runtime-marker fix | `e38d3a8` | Allowed the verifier to recognize the real vLLM worker marker emitted at `VLLM_LOGGING_LEVEL=WARN`; not the training commit |
| Merge to `main` | `944e196` | Integration commit after the run; not the training commit |

The model was `Qwen/Qwen3-8B` at revision `b968826d9c46dd6066d109eabc6255188de91218`. The retained model manifest records the local snapshot path and hashes. A later independent review matched the five model shard hashes plus `config.json`, `tokenizer_config.json`, and `model.safetensors.index.json` against that official revision.

The successful run used actor parameter and optimizer CPU offload:

```text
actor_rollout_ref.actor.fsdp_config.param_offload=true
actor_rollout_ref.actor.fsdp_config.optimizer_offload=true
actor_rollout_ref.rollout.gpu_memory_utilization=0.50
```

## Captured execution evidence

The files in this directory are retained outputs from the actual preflight, successful training run, adapter/evaluation workflow, or the original strict verifier:

| File | Captured fact |
|---|---|
| `preflight.txt` | H800 GPU, driver/CUDA details, CUDA availability, package versions, vendored `verl` path, training Git SHA, disk state, and `pip check` |
| `environment.txt` | Run identifiers, model/data/log paths, Python version, reward-log start offset, and installed packages |
| `git_state.txt` | Training commit `22945d2...` and clean `git status --porcelain` |
| `model_manifest.json` | Model identity, source, revision, and selected local-file hashes |
| `dataset_manifest.json` | Source and fixture hashes plus the exact 8 pairs / 16 rows |
| `resolved_config.yaml` | Hydra-composed configuration used by the successful run |
| `command.sh` | Exact successful training command |
| `train.log` | Actual stdout/stderr from real vLLM/FSDP2/GRPO training |
| `reward_log_window.json` | Original reward-log path and start offset |
| `metrics_summary.json` | Original strict-verifier summary, including `passed=true`, `failures=[]`, two optimization steps, finite metrics, vLLM initialization, and no detected fallback |
| `identity_pairing_summary.json` | Two reward blocks, 32 pair checks, zero unpaired/duplicate keys, and observed balance-batch reorder |
| `checkpoint_manifest.json` | Step 1 and step 2 checkpoint state verified before compact cleanup |
| `lora_update.json` | 252 non-zero finite LoRA-B tensors in the verified merged adapter |
| `controlled_eval_summary.json` | Controlled evaluation with 4 samples, `evaluation_contract=controlled`, and `num_options=2` |
| `adapter_normalization.json` | Captured correction of merged adapter metadata to rank 32, alpha 64, scaling 2.0 |

The successful log records two completed optimization steps. Actor gradient norms were `0.20984750986099243` and `0.18690121173858643`; pair-baseline metrics were present at both steps. The verifier summaries record 32 valid pair checks, observed reorder, step 1/2 checkpoints, non-zero finite LoRA-B tensors, and the controlled evaluation. No accuracy threshold was part of the Task 4 gate.

## Post-hoc evidence review

The following occurred after the original execution and is review activity, not training-time evidence:

- The final transfer input was integrity-checked before curation.
- Model hashes were independently compared with the official Hugging Face revision named above.
- The successful reward-log window was rechecked while the original private log still existed. It covered bytes `105664..210738`, or `105074` bytes and 160 lines. The private suffix SHA256 was `9a9bd08f29f0401002ff45764c9725d3c8169d5517de53ae8d3bfc7c5fb6572b`.
- The raw reward suffix remains outside GitHub because it is private.
- `checksums.txt` was generated during repository curation and excludes itself.

The original strict-verifier stdout/stderr transcript and shell exit code were not retained. No `strict_verifier_output.txt` has been invented. The retained machine-readable outputs listed above are the original verifier outputs; notably, `metrics_summary.json` records `passed=true` and an empty `failures` list.

The complete checkpoint, rollout, reward-log, merged-adapter, and controlled-evaluation trees are no longer available in this compact artifact. Consequently the strict verifier cannot be faithfully replayed from this directory alone. Deleted files were not reconstructed to manufacture a post-hoc PASS.

## Historical execution context

Run 001 occurred before the retained offload runs, but its original artifacts were not retained. It is historical execution context only. No specific Run 001 failure reason is claimed as a captured fact.

Run 002 (`task4_run_002_offload`) has retained review-support evidence outside this repository directory. That evidence shows that actor parameter/optimizer CPU offload allowed real vLLM startup, two rollout steps completed, and the step 1 checkpoint was saved. Saving the step 2 checkpoint failed after `/root/autodl-tmp` exhausted storage; Ray warned about the disk condition and PyTorch ended with `PytorchStreamWriter failed writing file`. This is classified as a storage/environment failure, not a trainer, vLLM, GRPO, reward, advantage, or identity-path failure.

Run 003 used the same effective training configuration as Run 002. The Run 002 to Run 003 remediation addressed storage capacity; it did not change the algorithm or effective training configuration.

## Not-retained artifacts

The following are intentionally absent or were not retained:

- Run 001 command, config, log, and failure artifacts;
- original strict-verifier stdout/stderr transcript and shell exit code;
- full checkpoint trees after compact cleanup;
- raw rollout dumps;
- raw reward-log suffix, retained privately only;
- merged LoRA adapter weights;
- raw controlled-evaluation output tree;
- TensorBoard events;
- base model weights and provider caches.

Their absence must not be filled by reconstructed files or inferred transcripts.

## Integrity

`checksums.txt` covers every committed file in this directory except `checksums.txt` itself. It is a repository-curation manifest, not an original training artifact.
