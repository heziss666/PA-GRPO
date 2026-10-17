# MATH Pair-v2: modest bias reduction, warning remains

**CPU implementation and one diagnostic completed; review pending.**
The fixed rule replaced 36 of 146 MATH pairs without removing questions.
Same-generator pairs increased from 57 to 79. Format-only Balanced Accuracy
fell from 71.92% to 68.15%, still **PASS_WITH_WARNINGS**, not PASS. No further
weight adjustment, diagnostic fit, Parquet export or GPU execution followed.

## Fixed intervention and quality checks

The [fixed policy](2026-10-17-math-pair-v2-fixed-policy.md) and implementation
were committed and pushed as `0db9828d727dbde4341e746a4e23d8d795eed27e`
before real selection and diagnostic results. Each count distance was
`abs(a-b)/(1+a+b)`; four feature-group means and the 0/1 generator distance
were averaged equally using exact rational arithmetic. There was no fitted
normalization or classifier-based score tuning.

Of 47 strictly better matching proposals, both responses received a fresh
AI-assisted contextual check. Thirty-six proposed positives passed; eight
had explicit reasoning defects and three were uncertain. All 47 proposed
negatives were confirmed incorrect in that review. Only the 36 dual-approved
proposals replaced their originals. The other 11 retained the original pair;
no second-best replacements were sought after rejection.

Excluded alternatives included false floor inequalities, invalid complex-sum
claims, contradictory parameter cases, incorrect algebra/transformation
reasoning and an erroneous intermediate value despite a correct final result.
Two uncertain proposals lacked an adequate global derivation or uniqueness
argument; another had an unclear false start. These are AI-supported
judgments, not independent human certification.

One uncertain positive was also the original selected candidate. Under the
predeclared fallback rule its prior provisional pair was retained, with the
new uncertainty preserved in private review notes. No explicit newly proven
defect contradicted an original candidate's approval. This is **not** a claim
that all 146 pairs freshly passed a full quality audit; that one uncertainty
requires attention before any formal release.

## Before/after diagnostic

Same 146 question groups / 292 chosen-rejected records, identical five-fold
assignments, same train-fold-only preprocessing, and 2,000 whole-question
bootstrap resamples. Raw response text was used in both versions; matching
normalization did not silently clean the diagnostic text. Pair-v1 results
were read from the stored baseline, not refitted. The Pair-v2 diagnostic
executed once with a persistent no-refit marker.

| Metric | Pair-v1 | Pair-v2 |
|---|---:|---:|
| MATH pairs | 146 | 146 |
| Same-generator pairs | 57 | 79 |
| Generator-only BA, 95% CI | 77.05% [72.60,81.51] | 70.89% [66.78,75.00] |
| Format-only BA, 95% CI | 71.92% [66.78,77.05] | 68.15% [63.36,72.95] |
| Full-text diagnostic BA, 95% CI | 69.18% [64.38,73.97] | 64.04% [59.59,68.84] |
| Format status, unchanged 65%/75% rules | WARN | WARN |

The format-only point estimate decreased 3.77 percentage points and the
generator-only estimate 6.16 points. These are internal intervention
diagnostics on an already-inspected cohort, not untouched independent
validation or evidence about trained-judge effectiveness. Intervals are
conditional on the fixed OOF predictions; no significance claim is made.
Generator-only remains an association diagnostic, and full-text prediction
can reflect substantive correctness differences, not just style leakage.

## Residual bias and trade-off

| Generator | Chosen v1 | Chosen v2 | Rejected v1 | Rejected v2 |
|---|---:|---:|---:|---:|
| Qwen2.5-7B | 111 | 96 | 32 | 35 |
| Llama-3.1-8B | 35 | 50 | 114 | 111 |

Source/label association remains. Mean dollar counts remain 5.48 chosen vs
21.57 rejected, and mean backslash counts 50.76 vs 32.77. Some layout gaps
decreased, but heading differences were unchanged and the list-count gap
slightly increased. These descriptive feature counts are not causal effects.

The **mean absolute within-pair token gap increased from 106.72 to 121.05**
(median 76.5 to 95.5). Do not confuse that with the difference between chosen
and rejected overall mean lengths, which decreased. Pair-v1 minimized token
gap; the equal-weight multi-factor rule traded some length matching for
other structure/source factors. Pair-v2 is not uniformly better and remains
a separately preserved candidate data-control variant.

## Integrity, evidence and stopping point

The output has 146 unique Train questions and 292 mirrored AB/BA records.
Thirty-six candidate-pair identities changed, with semantic IDs recomputed;
all original question/content/split/generation/verification identities were
inherited. Existing trainer record joins validated in memory; no Trainer
Parquet was written. ReClor's 234 pairs, original candidates, verifier labels,
isolation records, the old 380-pair view and frozen Stage A Parquet were not
modified. The protected Pair/AB-BA/Stage-A hashes matched before and after.
No old UNSURE/quarantined question was admitted.

- Selection run: `1cb559d089b3116d9bbc584ba39c62478805801df590798dc47030b3b6bef393`.
- Pair-v2 run: `4cff87b9b6681965e56b05763bf5b1a4b70f88a42399a457abfb0f4029a71c0b`.
- Pair JSONL SHA256: `408e21a308c9ae9a775472e75d579577f55490a1174b5f986d114dcc167b7db5`.
- AB/BA JSONL SHA256: `680f4e62349db8b87912e6b8157fbbb47fc9b62d2511f1eaf0e4405a16c310a0`.
- Quality decisions SHA256: `85974bb45a289c0d7f61f9406dbb856f17ecaadc154d72fc6f42f7691291c19f`.

Raw artifacts, questions, gold, responses, individual review reasons and OOF
predictions remain in the external private data root. Public evidence is the
allowlisted `artifacts/data_pipeline/phase2/math_pair_v2/summary.json`.
Focused and relevant regression tests: 48 passed locally; Ruff/format/diff
checks passed. A read-only review caught the diagnostic-text representation
issue before real diagnostics; its correction and scoped re-review passed.
These are local results, not GitHub CI.

Historical formal FAIL/MATH WARN are unchanged. Formal Audit/Gates, formal
training and main merge remain HOLD. Stop here for review; do not retune this
cohort or automatically promote Pair-v2 to the training dataset.
