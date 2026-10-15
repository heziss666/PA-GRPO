# Qwen2.5-7B untrained Judge: AB/BA N8/N16 diagnostic

Status: completed inference and descriptive analysis; **training remains HOLD**.
No policy loss, backward, optimizer step, new source candidate, parser change,
Verification relabeling, restoration of excluded pairs, or main merge occurred.

## Execution and provenance

- Executed code: `5bfb07950d67d4cef9ff3e28e799353ea13aa1f9` on the existing H800 80GB.
- Judge **and tokenizer**: `Qwen/Qwen2.5-7B-Instruct@a09a35458c702b33eeacc393d103063234e8bc28`.
  Cached unquantized BF16 weights; vLLM 0.8.5; one engine reused throughout.
- Input: retained 36 MATH + 69 ReClor Train pairs, 210 mirrored permutations.
  Original source generation/pair namespaces remain separate. The 38 blocked
  and 9 unsure pairs remain excluded. These are diagnostic, not training-released data.
- Exactly **3,360/3,360 unique requests**, AB/BA each slots 0..15. Seed 42,
  temperature 1, top-p 1, per-request `judge_request_seed_v1`, output limit 256.
  Input limit 4096, observed maximum 1881; no input truncation.
- Existing controlled answer-tag prompt, actual Qwen2 native chat template;
  no Qwen3 thinking toggle and no relaxed extraction.
- Four-request format/save preflight passed, then the remaining requests used
  the same engine. Inference plus remote analysis: **144.16 seconds** wall time.
  This is not provider-billed time; actual provider hours/cost were not retrieved.
- Raw Rollouts, preparation, runtime, per-pair metrics and logs were copied to
  private local storage. Backup SHA256 matched before shutdown. Shutdown was
  requested; SSH closed and a subsequent connection was refused. Provider
  console/billing state was not independently inspected; data/cache were preserved.
- Local CPU recomputation reproduced every aggregate exactly. One repeated
  individual lambda differed by 5.56e-17 between platforms; all record identities,
  counts and classifications matched. This is below the 1e-12 check tolerance.
- Focused tests: **103 passed** locally; focused Ruff/format and diff checks passed.
  Read-only code review found no Critical/Important findings. No remote CI claim.

## Results

All accuracy/consistency denominators include invalid outputs. N8-A is slots
0..7, N8-B is 8..15, N16 is their union, **not three independent experiments**.
Semantic consistency means the two valid choices select the same underlying
candidate (opposite surface letters). Consistent accuracy means both are correct.
Gap is absolute AB-minus-BA accuracy; AB was higher in every displayed view.

### Overall (105 pairs)

| Metric | N8-A | N8-B | N16 |
|---|---:|---:|---:|
| AB accuracy | 68.33% | 69.64% | 68.99% |
| BA accuracy | 46.55% | 47.14% | 46.85% |
| Accuracy gap (percentage points) | 21.79 | 22.50 | 22.14 |
| Strict valid format | 99.52% | 99.52% | 99.52% |
| Semantic consistency | 47.98% | 48.57% | 48.27% |
| Consistent accuracy | 31.55% | 32.86% | 32.20% |
| Mean absolute base-reward gap delta | 0.9943 | 0.9643 | 0.9769 |
| Mean global reward sigma | 0.6921 | 0.6999 | 0.7153 |
| Global zero-sigma pairs | 20.95% | 20.00% | 16.19% |
| Local zero-sigma subgroups (pooled AB/BA) | 66.19% | 61.90% | 56.19% |
| Pairs with nonzero global advantage | 79.05% | 80.00% | 83.81% |
| AB subgroups with nonzero local advantage | 33.33% | 37.14% | 40.00% |
| BA subgroups with nonzero local advantage | 34.29% | 39.05% | 47.62% |
| Global variable, >=1 local subgroup constant | 66.67% | 63.81% | 63.81% |
| Mean ALC local weight | 0.6767 | 0.6853 | 0.6977 |
| ALC local weight >= .5 | 73.33% | 75.24% | 73.33% |
| Invalid outputs (count) | 8 | 8 | 16 |
| Length finishes (count) | 2 | 4 | 6 |

### MATH (36 pairs)

| Metric | N8-A | N8-B | N16 |
|---|---:|---:|---:|
| AB accuracy | 70.83% | 69.79% | 70.31% |
| BA accuracy | 63.19% | 64.93% | 64.06% |
| Accuracy gap (percentage points) | 7.64 | 4.86 | 6.25 |
| Strict valid format | 98.78% | 98.78% | 98.78% |
| Semantic consistency | 63.54% | 63.19% | 63.37% |
| Consistent accuracy | 48.96% | 49.31% | 49.13% |
| Mean absolute base-reward gap delta | 0.6847 | 0.6389 | 0.6618 |
| Mean global reward sigma | 0.5755 | 0.5677 | 0.5999 |
| Global zero-sigma pairs | 30.56% | 33.33% | 25.00% |
| Local zero-sigma subgroups (pooled AB/BA) | 59.72% | 54.17% | 47.22% |
| Pairs with nonzero global advantage | 69.44% | 66.67% | 75.00% |
| AB subgroups with nonzero local advantage | 41.67% | 50.00% | 52.78% |
| BA subgroups with nonzero local advantage | 38.89% | 41.67% | 52.78% |
| Global variable, >=1 local subgroup constant | 55.56% | 41.67% | 44.44% |
| Mean ALC local weight | 0.5632 | 0.5454 | 0.5744 |
| ALC local weight >= .5 | 63.89% | 61.11% | 61.11% |
| Invalid outputs (count) | 7 | 7 | 14 |
| Length finishes (count) | 2 | 4 | 6 |

### ReClor (69 pairs)

| Metric | N8-A | N8-B | N16 |
|---|---:|---:|---:|
| AB accuracy | 67.03% | 69.57% | 68.30% |
| BA accuracy | 37.86% | 37.86% | 37.86% |
| Accuracy gap (percentage points) | 29.17 | 31.70 | 30.43 |
| Strict valid format | 99.91% | 99.91% | 99.91% |
| Semantic consistency | 39.86% | 40.94% | 40.40% |
| Consistent accuracy | 22.46% | 24.28% | 23.37% |
| Mean absolute base-reward gap delta | 1.1558 | 1.1341 | 1.1413 |
| Mean global reward sigma | 0.7530 | 0.7688 | 0.7755 |
| Global zero-sigma pairs | 15.94% | 13.04% | 11.59% |
| Local zero-sigma subgroups (pooled AB/BA) | 69.57% | 65.94% | 60.87% |
| Pairs with nonzero global advantage | 84.06% | 86.96% | 88.41% |
| AB subgroups with nonzero local advantage | 28.99% | 30.43% | 33.33% |
| BA subgroups with nonzero local advantage | 31.88% | 37.68% | 44.93% |
| Global variable, >=1 local subgroup constant | 72.46% | 75.36% | 73.91% |
| Mean ALC local weight | 0.7360 | 0.7582 | 0.7620 |
| ALC local weight >= .5 | 78.26% | 82.61% | 79.71% |
| Invalid outputs (count) | 1 | 1 | 2 |
| Length finishes (count) | 0 | 0 | 0 |

### ALC weight distributions

Population standard deviation and eps=1e-6; lambda uses only the unchanged
base reward, never consistency reward. Min/25th/median/75th/max describe
per-pair weights; displayed rounding does not change stored precision.

| Source | View | Min | Q25 | Median | Q75 | Max |
|---|---|---:|---:|---:|---:|---:|
| overall | N8-A | 0.0000 | 0.2520 | 1.0000 | 1.0000 | 1.0000 |
| overall | N8-B | 0.0000 | 0.5164 | 1.0000 | 1.0000 | 1.0000 |
| overall | N16 | 0.0000 | 0.3780 | 1.0000 | 1.0000 | 1.0000 |
| math | N8-A | 0.0000 | 0.0000 | 0.7559 | 1.0000 | 1.0000 |
| math | N8-B | 0.0000 | 0.0000 | 0.5469 | 1.0000 | 1.0000 |
| math | N16 | 0.0000 | 0.1608 | 0.6762 | 1.0000 | 1.0000 |
| reclor | N8-A | 0.0000 | 0.5164 | 1.0000 | 1.0000 | 1.0000 |
| reclor | N8-B | 0.0000 | 0.5164 | 1.0000 | 1.0000 | 1.0000 |
| reclor | N16 | 0.0000 | 0.5164 | 1.0000 | 1.0000 | 1.0000 |

### N16 subgroup correctness states

Zero-correct includes invalid outputs; it does **not** assert all were valid
wrong answers. Full N8 states and reward-gap/sigma quantiles are in the
[aggregate JSON](qwen25_judge_subgroups_20261015.json).

| Source | Permutation | All correct | Zero correct | Mixed |
|---|---|---:|---:|---:|
| overall | AB | 48 | 15 | 42 |
| overall | BA | 27 | 29 | 49 |
| math | AB | 14 | 3 | 19 |
| math | BA | 13 | 5 | 18 |
| reclor | AB | 34 | 12 | 23 |
| reclor | BA | 14 | 24 | 31 |

### Within-run sampling stability

Predeclared descriptive cutoff: noticeable delta >= .25; attenuated means N16
is <= half N8-A's noticeable gap; disappeared means N16 gap is zero; stable
means both N8 halves and N16 have delta >= .25, with matching direction in the
halves. These are not Gates or significance tests. Per-pair identities and
flags are retained in the private statistics, not publicly exposed.

| Source | Noticeable N16 | Attenuated | Disappeared | Stable gap in all views |
|---|---:|---:|---:|---:|
| overall | 80/105 | 4 | 0 | 74/105 |
| math | 23/36 | 2 | 0 | 21/36 |
| reclor | 57/69 | 2 | 0 | 53/69 |

## Interpretation and next research step

1. **These pairs are not trivially separable for this untrained Judge.** Pooled
   N16 accuracy is 1946/3360 = 57.92% (MATH 67.19%, ReClor 53.08%). Only 12/105
   pairs were correct on all 32 judgments. This is performance on a filtered
   Train subset, not a held-out generalization score or a fresh label audit.
2. **A substantial descriptive AB/BA difference remains**, concentrated in
   ReClor: 30.43 percentage points versus MATH's 6.25. The same positive/negative
   responses were mirrored, but finite sampling, chosen prompt, candidate style,
   selection bias and residual data-quality risk limit attribution. This does
   not establish statistically significant, general position bias.
3. **N16 does not erase the difference.** The pooled gaps are 21.79, 22.50 and
   22.14 points; 74 pairs retain a noticeable same-direction gap in both halves
   and their union. Four attenuate by the defined half-gap rule and none
   disappear. More slots expose additional within-group variation (global
   nonzero pairs 83 -> 88, versus 84 in N8-B). A larger nested sample alone
   does not prove estimator convergence or independent replication.
4. **There is usable but uneven reward signal.** N16 global nonzero advantage
   occurs for 88/105 pairs; local nonzero signal occurs in AB 42/105 and BA
   50/105. Seventeen ReClor pairs have different constant rewards between
   permutations but zero local variation in both. For those pairs local
   advantage is exactly zero: neither EIS nor a large ALC weight creates a
   nonzero local correction. A high lambda is not evidence of useful correction.
5. **Do not increase difficulty indiscriminately.** Preserve these pairs as a
   diagnostic/possible wiring subset subject to training release; extend to new
   unique Train questions with reliable labels and source/position diversity.
   Track easy (12 always-correct), stubborn wrong (5 zero-correct in both),
   and mixed-signal cases separately. This run supports studying the existing
   recipe's hard cases rather than another parser/prompt optimization project.
6. **Local heterogeneity motivates a controlled comparison, not an efficacy
   claim.** Global/local variance differs materially, and lambda spans 0..1;
   this supplies an empirical reason to compare Global-only, EIS, ALC and
   PA-Controlled-NoGate under equal data/budgets. It does not show EIS or ALC
   improves reward, accuracy, learning stability, or sample efficiency.

The 16 strict-invalid outputs (14 MATH, 2 ReClor) stay in denominators. Six
are length finishes. The original PA component rules are unchanged: all 3360
length components were -.1; format components were +.3 for 3354 and -.3 for 6.
Tag presence used by that existing format component is **not** the strict
answer parser, hence 16 invalid answers do not imply 16 negative format
components. All 16 have negative correctness. No correction to reward semantics
was made to favor Qwen2.5. The diagnostic has not released trainer data.

A short four-method GPU wiring test can be proposed after the necessary data
release decision, but was **not** run here. Do not infer authorization to train,
expand sampling, or merge main from completion of this diagnostic. Historical
Qwen3 Task 4 evidence remains historical; no Qwen2.5 GRPO success is claimed.

## Evidence hashes

- Diagnostic run: `b6bc5d9a7e258e9bbe1af19d34832df9f70832e43300b802937675e0125cb29d`.
- Exact prepared bytes: `1b46ee2313da6530f576385c994fb420332d7de839ce46e355a05a91620b9a85`.
- Raw Rollout JSONL: `611ae6973b44644a158d148ec7217492901ad3ad28450882c0f69005162c36f9`.
- Original private grouped statistics: `5f20dfd2382175032e64ac76a5d4b27aceff324a1b2961c2abae7e1b9eeddff5`.
- Native chat template: `cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f`.
- Verified private backup: `0cccd4567f2f778b0af055510847907f468bffa63cf6b9889f3c04d2ef2ae7fa`.

The [public aggregate JSON](qwen25_judge_subgroups_20261015.json) includes
only allowlisted metadata, counts and aggregate statistics. Original question,
response and full Judge generation text remain private. The two source input
runs and hash-bound retained views remain traceable through the prior
[negative-label check](production_200_retained_negative_check_20261015.json)
and the pinned diagnostic code; no combined source Generation Manifest exists.
