# Expansion quality screening and CPU Smoke Release Conditions v1

## Decision

**CPU_SMOKE_RELEASE_PASS_WITH_WARNINGS.** No Smoke data release, Trainer Parquet export, GPU execution or training was performed. Historical formal Audit/Gate results are unchanged. Formal four-method training and main merge remain HOLD.

The user approved these conditions before execution. The implementation was committed as `2d907ceb5912e1308e16c874d07601c4bf565e3a` before reading the real classifier results. No threshold, feature family or hyperparameter was tuned after observing them.

## Quality screening of the new batch

500 new Train questions per source produced 4,000 candidates and 393 automatic pairs. This is not 1,000 reliable pairs.

| Source | Automatic pairs | AI-provisional retain | Quarantine | Needs review |
|---|---:|---:|---:|---:|
| MATH | 148 | 110 | 34 | 4 |
| ReClor | 245 | 165 | 67 | 13 |
| Total | 393 | 275 | 101 | 17 |

All 393 selected positives were read against the question and official solution/options. Only positive-stage retained pairs proceeded to negative checks: 111 MATH negatives received contextual equivalence review; 165 ReClor negatives received official-label, ordered-option, content/identity and terminal-choice checks, not a second full negative-reasoning review. One of the 111 MATH negatives was actually correct after resolving its option-letter meaning, so that pair was quarantined. An additional explicitly defined symbol-alias false incorrect was found in an already-positive-quarantined pair; it did not reduce retained count again. Raw verification labels and parsers were not changed.

Main positive defects were false algebra/geometry, coincidentally correct final answers, conditional inversions, necessary/sufficient-condition confusion, misidentified argument roles and contradictory option explanations. Main-review boundary adjudications were retained separately from first-stage notes. All review conclusions are AI-supported and provisional, not independent human certification. Initial schema probes incidentally displayed the first negative; this review is not claimed to be blinded.

Combining the 275 new provisional pairs with the previously screened 105 gives **380 unique Train questions: 146 MATH + 234 ReClor**, represented by 760 AB/BA rows. The 17 new UNSURE pairs remain excluded. Earlier quarantined and unsure pairs were not restored. Combined records retain their original run identities; candidates were not spliced across generation runs.

## Fixed CPU protocol

- `seed=42`; five `StratifiedGroupKFold` folds, separately per source, grouping by `original_question_id`.
- Exactly one chosen and one rejected response per question: 292 MATH records and 468 ReClor records. No duplicate AB/BA appearances are classifier observations.
- The same folds are used for all three models. One-hot encoding, standardization and TF-IDF are fitted only inside each training fold. No tuning or model selection.
- Logistic regression: `C=1`, `solver=liblinear`, `max_iter=1000`, `random_state=42`.
- Generator-only model: generator identity only; predicts source/label association, not proof of text-style exploitation.
- Format-only model: aggregate token/character lengths, whitespace, line/paragraph counts, list/heading counts, code fences, LaTeX backslash/dollar counts and punctuation counts. No lexical tokens, answer identities, numeric values, choice letters, embeddings, bag-of-words or question text enter this model.
- Full-text model: response-only word TF-IDF, n-grams 1–2, `min_df=2`, `max_features=5000`, sublinear TF. Diagnostic only: real reasoning-quality/content differences can also predict labels.
- Primary metric: pooled out-of-fold Balanced Accuracy. The 95% percentile intervals use 2,000 **whole-question** bootstrap resamples, preserving both labels of each sampled question. They quantify uncertainty conditional on these fixed OOF predictions, not full model-refitting or split-selection uncertainty.
- Fixed format-only warning lines: **BA <65% PASS; 65% <= BA <75% PASS_WITH_WARNINGS; BA >=75% FAIL**. These are project preregistered engineering guardrails, not universal statistical standards. Status uses the unrounded point estimate; confidence intervals do not change the rule. Overall status is the worst format-only status if integrity checks pass; integrity failure always blocks release. Generator-only and full-text results do not set Gate status.
- Only overall PASS permits freezing a Smoke-only selection and Parquet export. Warnings require a separate explicit decision. GPU Smoke always needs separate authorization.

## Results

| Source | Questions | Generator-only BA (95% CI) | Format-only BA (95% CI) | Full-text diagnostic BA (95% CI) | Format status |
|---|---:|---:|---:|---:|---|
| MATH | 146 | 77.05% (72.60–81.51%) | **71.92% (66.78–77.05%)** | 69.18% (64.38–73.97%) | PASS_WITH_WARNINGS |
| ReClor | 234 | 66.88% (62.39–71.58%) | **53.85% (50.00–57.69%)** | 54.70% (50.64–58.76%) | PASS |

MATH's point estimate is below 75%, so its interval crossing 75% does not retroactively change the predefined warning decision. Its structural features nevertheless carry a measurable predictive signal in this diagnostic. Mean standardized coefficients identify LaTeX dollar/backslash and layout counts among influential correlated features; these coefficients do not establish causation or unique feature importance. Generator/label association is stronger in MATH, but the generator-only score alone cannot show that a judge reads that association from text. ReClor's weaker format result is not proof that every possible style shortcut is absent.

## Integrity and release state

The checked selection exactly matches prior positive/negative screening approvals. Historical negative checks must explicitly confirm the source-specific approval status, cover the selection exactly once, match question/run/candidate identities and report zero new holds or false-incorrect findings. Formal generation/verification/pair/permutation artifacts are hash-validated; selected records are joined by composite identity. Existing trainer row builders validate question content, response hashes, same-question chosen/rejected records, Train identity and AB/BA mirror mappings. All 760 rows were constructed **in memory only** for structural validation.

- Duplicate original question IDs: 0.
- Holdout overlap: 0.
- Known quarantined/unsure pairs admitted: 0.
- Formal Audit completed: no.
- Smoke data frozen/exported: **no**, because overall status is a warning.
- GPU runs/training steps: 0.
- No parser, verifier, frozen Phase 1 algorithm, ALC formula or historical label changes.

## Reproduction and evidence

Optional CPU dependencies are pinned in `requirements-phase2-cpu-diagnostic.txt`, separately from the Phase 1 lock. Set `PAGRPO_DATA_ROOT` to the existing external private data root and run:

```bash
python -m scripts_permstudy.phase2.smoke_release_gate \
  --private-log private_logs/smoke-release-gate.private.log
```

Run identity: `7d4c4f9812f89a26133677baf84a33a988df99f23b3c3b7d86da815fd78e3421`.

Read-only review identified a missing fail-closed check on explicit historical negative-approval statuses. Commit `22387b970fe9e461e22fecfe1880d04e3ca8e78a` strengthened only that integrity boundary; the actual approvals already satisfied it. Rerunning produced exactly identical source metrics and byte-identical private OOF predictions. The original run `c6bfad2c1de958e9c86170e9c08563a0218a6379f54b7862daa2e3753ebbacaf` remains preserved. No classifier features, hyperparameters, thresholds or labels changed.

The allowlisted public summary is `artifacts/data_pipeline/phase2/smoke_release_v1/summary.json`. It records code/dependency versions, source/run/artifact hashes and integer completion counts. Float metrics are reported here and in the private report, not inserted into integer-only public count fields. Private questions, gold, responses, per-question OOF predictions, feature vectors, individual review notes and local paths are not committed. Package data and original source files remain outside Git.

Combined provisional Pair JSONL SHA256: `db0f294450144c9553a848f3ce1d436551824b0ee18ee97c387adec7bbf2988b`.

Combined provisional AB/BA JSONL SHA256: `5505e619af505e7f8581ae839f8494d2f5cc69ccf3a3aad07c3a8f4acba74c20`.

Focused CPU checks: 14 passed; with related production-preparation tests: 24 passed (local execution, not GitHub CI). Ruff and format checks passed. The tests include grouped folds, question-level bootstrap, threshold boundaries, equal-structure semantic substitutions, diagnostic-only full-text success, CLI isolation and rejected historical approval/identity mutations. A separate read-only code review found the integrity issue closed and no further findings. The Phase 1 full suite was not repeated.

The next decision is whether to explicitly accept the MATH warning for a **plumbing-only** training Smoke, or request a separately specified CPU mitigation. Neither is automatically authorized by these results. No new sampling, threshold adjustment or formal four-method training is proposed as part of this task.

Method references: [scikit-learn grouped cross-validation](https://scikit-learn.org/1.5/modules/cross_validation.html), [training-fold-only preprocessing](https://scikit-learn.org/1.5/common_pitfalls.html).
