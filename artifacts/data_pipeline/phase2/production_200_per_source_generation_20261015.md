# First 200-per-source production batch: raw generation complete

Execution code: `06025470507842c53fae1d42d2553c57195b7828`.
The user authorized inference after preflight and shutdown after completion.
The approved source-specific recipes ran unchanged on one H800 PCIe 80GB.
The two pinned model snapshots were already cached; no weights were downloaded.
Phase 1 algorithms, parsers, gates and training code were not changed.

| Source | New Train questions | Qwen2.5-7B | Llama3.1-8B | Saved candidates |
| --- | ---: | ---: | ---: | ---: |
| MATH | 200 | 400 | 400 | 800 |
| ReClor | 200 | 400 | 400 | 800 |
| Total | 400 | 800 | 800 | 1,600 |

All four generator processes exited with code 0. The process wall times,
including initialization and model loading, were respectively 537.209 seconds
(Qwen/MATH), 209.347 (Qwen/ReClor), 519.584 (Llama/MATH), and 269.283
(Llama/ReClor). The controller's measured wall time was 1,553.176 seconds,
approximately 25.9 minutes. This is **not provider-billed GPU time**; actual
instance billing and price were not independently verified. Resource logs remain
private. Models ran sequentially, not simultaneously.

The approved split is unchanged. Formal generation manifests, artifact hashes,
the full selected question/candidate joins, 800 unique composite candidate keys
per source and semantic candidate hashes were reloaded and checked locally.
The 49 historical candidate shard hashes still match their pre-execution values.
No candidate text was rewritten, completed or normalized in the raw files.

## Raw output is not verified training data

MATH has 703 `stop` and 97 `length` finishes; ReClor has 798 `stop` and two
`length` finishes. A successful generation record only means the response was
saved. It does not establish `correct`, `incorrect` or parser validity.
Truncated responses remain subject to the existing invalid/exclusion rule.
No verification, audit decisions, pair selection, trainer export, gates or
training were executed for these two new runs in this step. There is no claim
of 400 pairs, a smoke PASS, or authorization for training.

## Backup and shutdown

The raw backup contains all 200 candidate shard files plus formal manifests,
plans, selected questions and private execution receipts. Its local and remote
SHA256 values both equal:

`a696c5649d790b48af15c24a5274ae1454285562c227750115ab313123e84a80`

Full private records are retained in the external data root at
`generation/<source-generation-run>/shards/<generator>/<shard>/candidates.jsonl`.
Both generation run IDs and hashes are in the adjacent sanitized JSON receipt.
The backup is retained locally and on the server data disk; no data disk or
model cache was deleted. Neither raw responses nor private logs enter Git.

After the local backup hash was checked and no GPU compute processes remained,
the provider shutdown mechanism was requested at `2026-10-15T04:01:06Z`.
SSH closed by the remote host, and a subsequent connection was refused.
These observations do not independently establish the provider console's
shutdown/billing state; confirm the instance shows shutdown in AutoDL.

Next work can use local CPU verification and selected-pair quality inspection.
No further GPU generation or training was started. Main merge remains HOLD.
