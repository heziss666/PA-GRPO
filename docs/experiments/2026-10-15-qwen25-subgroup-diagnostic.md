# Qwen2.5-7B retained-pair subgroup diagnostic

Authorized scope: backbone/config switch and an untrained, inference-only
diagnostic. No optimizer, policy loss, backward pass, adapters, training release,
new candidate generation or parser relaxation. Historical Qwen3 Task 4 evidence
and the original Verification labels remain unchanged. Latest source baseline:
`d20005eb86a65c72470fa7c92db31184ffccd5b2`.

## Fixed experiment

- Qwen/Qwen2.5-7B-Instruct and tokenizer revision
  `a09a35458c702b33eeacc393d103063234e8bc28`, cached unquantized BF16 weights.
- Only retained 36 MATH + 69 ReClor Train pairs; two independently attributed
  source runs and their hash-bound views. Exclude 38 blocked + 9 unsure pairs.
- Existing trainer-row renderer plus `pairwise_judge_answer_tag_v1` prompt.
  Apply Qwen2's actual chat template, without Qwen3's thinking switch. Do not
  force artificial thinking tags or change the existing PA length reward.
- Seed 42, temperature 1, top-p 1, output limit 256, prompt limit 4096, TP=1.
  These match the short controlled rollout's decoding settings; n=16 here is
  the diagnostic's sampling intervention, not a change to short training n=2.
- 105 * 2 * 16 = 3360 requests; one loaded engine, batch size 16, max sequences
  32, context 4352, memory utilization .85. CPU validates all prompt lengths
  before engine initialization. No truncation of the pair's question/responses.
- Deterministic request identity binds diagnostic run, pair, permutation and
  slot. `judge_request_seed_v1` hashes that identity with seed 42 to a 31-bit
  per-request seed. Completion order does not define identity. JSONL records
  are append-only, checked by existing record hashes; resume fills missing IDs.
- First four planned requests test saving and strict output format in the same
  engine. Failure stops the full run; successful requests are not regenerated.

## Metrics fixed before real inference

N8-A = slots 0..7, N8-B = 8..15, N16 = 0..15 from the same saved run. The
views overlap and are not independent experiments. Statistics pair AB/BA by
`(pair_id, rollout_slot)`; an invalid answer counts as inaccurate and cannot
count as semantically consistent. Consistent Accuracy requires both correct.
Each rate retains every planned item in its denominator. Length finishes are
reported separately; Judge scoring uses the unchanged strict answer contract,
not the candidate-verifier's separate finish-reason policy.

Base reward is the actual existing correctness + format + length helper from
`my_reward/judge_qwen.py`, with no Consistency Reward. Use population sigma
(`correction=0`) and existing `compute_controlled_advantage(..., mode="alc",
eps=1e-6)`. N8-B slots are reindexed only inside that pure function; stored
identities remain 8..15. Nonzero numerical advantage uses absolute tolerance
1e-6. Zero reward variance is identified from equal values. Report subgroup
all-correct / zero-correct / mixed counts and invalid counts separately;
zero-correct does not imply every output was a valid wrong answer.

For operational diagnostics only: noticeable mean reward gap = delta >= .25;
attenuation = N8-A noticeable and N16 gap <= half N8-A; disappearance = N16
gap zero; stable = all three gaps >= .25 with the two halves sharing direction.
Also report local-weight >= .5. These are descriptive cutoffs, not new Gates,
significance tests or proof that ALC improves performance. Keep per-pair details
private; public tables show source and overall aggregates/distributions.

## Commands and execution boundary

```bash
export PAGRPO_DATA_ROOT=<external-private-root>
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m scripts_permstudy.phase2.diagnose_judge_subgroups prepare --data-root "$PAGRPO_DATA_ROOT"
python -m scripts_permstudy.phase2.diagnose_judge_subgroups run --data-root "$PAGRPO_DATA_ROOT" --model-path <cached-snapshot-ending-in-the-pinned-revision>
python -m scripts_permstudy.phase2.diagnose_judge_subgroups analyze --data-root "$PAGRPO_DATA_ROOT"
```

Run the small CPU diagnostic tests and reward/advantage regressions before
committing executable code. Transfer exact prepared bytes privately and check
SHA256 on both hosts. Record code commit, native chat-template hash, config,
completion count and elapsed time. User separately authorized SSH/GPU execution
and shutdown after results are saved; do not infer authorization for training.
An external process timeout bounds inference; failures stop rather than silently
changing model, context, sampling or reward rules. Copy raw JSONL and private
statistics locally, check archive/hash equality, then shut down without deleting
the data disk or cache. Public evidence is sanitized; no original text or secrets.
