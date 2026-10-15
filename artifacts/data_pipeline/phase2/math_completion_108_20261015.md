# MATH completion diagnostic: executed 2026-10-15

Approved launcher `c658bc4` generated exactly 108 candidates from 27 MATH training
questions: 17 previously unpaired smoke questions and ten fresh questions.
Qwen2.5-7B and Llama-3.1-8B ran sequentially, using pinned cached weights,
temperature 0.8, top-p 0.95, max-new-tokens 1024, and the concise boxed-answer
instruction. No model weights were downloaded, no 32B model ran, and no training
or pilot expansion occurred. Original candidates and failed smoke evidence are
preserved; generation runs were not mixed.

| Cohort | Candidates | Length finishes | Correct | Incorrect | Invalid | Ambiguous | Auto pairs | AI-screened pair view |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Retest: 17 questions | 68 | 9 | 39 | 12 | 16 | 1 | 6 | 5 |
| Fresh: 10 questions | 40 | 0 | 23 | 8 | 9 | 0 | 4 | 3 |
| Total | 108 | 9 | 62 | 20 | 25 | 1 | 10 | 8 |

All 27 gold records parsed successfully. Comparing only the same 17 questions
and the same two model identities, length finishes fell from 34/68 to 9/68.
This is descriptive, not a controlled length ablation: prompt, output budget,
and run-scoped sampling seeds changed together. Auto pair yields are 6/17 and
4/10, respectively. All ten paired questions are new coverage relative to the
previous pair sets, not extra copies of already-paired questions.

Qwen produced 44 correct, nine incorrect, one ambiguous, and no invalid outputs.
Llama produced 18 correct, 11 incorrect, and 25 invalid outputs (nine truncated).
Thus truncation is reduced, but Llama MATH validity remains a material issue.
No additional parsing exceptions were introduced to improve these numbers.

## Review and training boundary

The assistant read both selected responses, the question, and official solution
for all ten automatic pairs. No final-answer label disagreement was found.
Two selected positives nevertheless have substantially faulty reasoning despite
correct final answers; one alternate positive repeats the same flaw. All three
candidate identities are privately marked training-blocked. Removing the two
affected pairs leaves an eight-pair diagnostic view (five retest, three fresh).

This view is a subset of the unchanged automatic PairRecords, not a new pair
algorithm/run, a completed human audit, or a released training dataset. The
formal 46-record audit selection remains uncompleted. No fabricated audit
decisions, trainer export, new Gate pass, or training authorization were created.
The legacy REAL_SMOKE_FAIL result remains intact.

## Execution and preservation

Qwen ran 10:34:45-10:37:37 and Llama 10:38:22-10:40:45 (Asia/Shanghai): 315 seconds
of combined generator-process wall time, including loading and warmup, not
provider-billed GPU time. Both processes exited zero and released GPU memory.
Raw candidates/manifests/logs were downloaded and hash-matched before shutdown.
At 10:42:06 the provider shutdown stop sequence was requested; a subsequent SSH
attempt was refused. The provider console/billing state was not independently
verified because browser access timed out. Data disks were not deleted.

CPU verification, diagnostics, and selected-pair review ran locally after the GPU
stage. They are locally archived; these later CPU artifacts have not yet been
uploaded back to the now-unreachable instance. Public JSON contains only
allowlisted counts, identities, and hashes. Full text stays in private storage.

This bounded experiment is finished. Further GPU sampling, training-data release,
and a 200-pair pilot require a separate decision. Do not keep resampling the same
smoke solely to erase historical warnings or change verifier/Gate rules.
