# Retained-pair negative checks: 105 candidate pairs remain

CPU review baseline: `ea9e58b60ce5720047200e725bdd1c3561b28657`.
Scope was limited to the 105 provisionally retained pairs from the preceding
positive AI screen. No GPU, regeneration, parser modification, training or
new pipeline framework was used.

| Source | Check | Checked pairs | Newly detected false incorrect | Remaining candidate pairs |
| --- | --- | ---: | ---: | ---: |
| MATH | Negative answer versus problem context and official reference | 36 | 0 | 36 |
| ReClor | Question/ordered-option identity and final-option label | 69 | 0 | 69 |
| Total | Source-appropriate bounded checks | 105 | 0 | 105 |

All 36 MATH negatives were read with their problems and references. Independent
arithmetic, symbolic expressions, substitutions or counterexamples supported
their non-equivalence. Ordered coordinate tuples and symbolic expressions were
not treated as wrong merely because their strings differed. The already
quarantined matrix-alias question was not restored. No new false-incorrect case
was found in this retained subset; this is an AI-supported assessment, not a
claim that the verifier is generally free of context-dependent errors.

For ReClor, the existing official-label verifier was reused on selected
positives and negatives. The source/split records, content hashes, all four
ordered options, composite candidate identities and persisted final labels
matched. Selected positives matched the official label and negatives selected
a different option. Full negative reasoning was not rereviewed, as requested.

## Shared-data structure, not a training release

Existing PairRecords were retained unchanged. The existing AB/BA builder
reconstructed 72 MATH and 138 ReClor records, each matching its original
canonical permutation record. Each pair has exactly `0=AB / 1=BA`, mirrored
candidate identities and correct surfaces A/B. All 105 distinct questions are
in Train, with no Holdout overlap or duplicate question identity across sources.
Response hashes and question/candidate joins were rechecked.

The private pair views and filtered permutation views identify the same data
for Global-only, EIS, ALC and PA-Controlled. They are selection views, not new
run manifests, Parquet exports or method-specific resamples. No Trainer Export
or model execution was performed.

The prior 38 blocked pairs and 9 unsure pairs remain excluded, with no recovery
attempt or relabeling. Positive-screen caveats remain applicable. All 857
pre-existing files under generation, verification, pair, permutation and audit
directories had unchanged byte hashes after this check. Historical public
receipts and private review artifacts were not overwritten.

Actual per-pair notes, bounded check code and structural evidence remain in the
external private data root. A separate local backup was checked member-by-member;
its SHA256 and the reviewed-view hashes are in the adjacent allowlisted JSON.
Only sanitized evidence is committed. No new full Phase 1 test run is claimed.

## Decision boundary

**105 pairs (36 MATH + 69 ReClor) remain candidates after the requested negative
checks; they are not formally training-approved.** Formal Audit/Gates and human
signoff remain incomplete, and no waiver is inferred from this bounded check.
Main merge and training remain HOLD. The next decision can define the minimum
research release conditions and separately authorize a short training wiring
test, rather than automatically launching another sampling run. Against a
200-pair target, the candidate count is still short by 95; that arithmetic is
not a forecast of future yield or a claim of 105 fully certified pairs.
