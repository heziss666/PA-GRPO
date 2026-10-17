# 32GB training Smoke: blocked during Actor initialization

## Result

**BLOCKED_RESOURCE — Actor/FSDP2 CUDA out of memory.** The user explicitly
authorized the prepared initial two-step Global-only attempt. The launcher
ran once with a 900-second limit and exited with failure before rollout or
optimization. No automatic retry, precision change or algorithm change ran.
This is not a training Smoke PASS or evidence about algorithm effectiveness.

Execution code: `d0bc2afa9a59f9bbe8bc1dbe7ddc9e5ab7ca4c72`.
The branch's approved preparation evidence was at `013727a` when launched.
Preparation run:
`01600655484852bda0643e05472eaa9a1579735c17d8586630b4cfe2e071674d`.
The initial command started at 2026-10-17 08:09:38 UTC; post-failure resource
and artifact checks completed at 08:11:37 UTC. These are observation times,
not an independently measured billable-duration report.

## Observed failure

The stack reaches `init_model → _build_model_optimizer → apply_fsdp2 →
fully_shard → _init_sharded_param`. CUDA reported an additional **2.03 GiB**
allocation request with **1.44 GiB** free out of **31.47 GiB**. The process
then occupied 30.03 GiB including non-PyTorch memory; PyTorch reported
28.80 GiB allocated and 970.71 MiB reserved but unallocated.

This occurred in Actor initialization, not vLLM rollout. Shrinking batch or
response length did not remove the initialization bottleneck. Actor storage
remained FP32 with rank-32 LoRA and CPU offload enabled. As documented in CPU
preparation, the reference shares the adapter-disabled Actor; its nominal
standalone BF16 setting does not reduce this shared Actor's storage.

No rollout files, optimizer-step metrics or checkpoint directories were
produced. Reward, advantage, loss, LoRA updates and checkpoint save/restore
remain **unverified**, not failed numerical checks. No restore replay ran.
Container memory counters reported zero CPU OOM events and zero OOM kills.

## Data and evidence

The 4-pair / 8-row subset still has SHA256
`ca13e6c40464dfb31476a17fe9d4eb14b6d36c15dcba2bf4cba55bc2416444c0`.
The frozen 380-pair Stage A data, labels, quarantine decisions, MATH WARN and
historical formal FAIL were not changed.

The private trainer log and resolved configuration were copied to the local
external data root; their SHA256 values matched the server. Only hashes and
sanitized counts are published in the adjacent experiment summary. The
isolated execution checkout remained clean; the original dirty checkout was
not overwritten.

## Stopping boundary

Post-failure checks showed 0 MiB GPU usage and no GPU compute processes.
The server was **not shut down**; an idle GPU instance may still be billed.
No configuration change, second attempt, restore replay or formal training
is authorized by this result. A larger-memory GPU is the straightforward
next option. Retaining 32GB would require separate approval for an Actor
precision/loading change and validation of LoRA/optimizer precision; it is
not an automatic fallback.
