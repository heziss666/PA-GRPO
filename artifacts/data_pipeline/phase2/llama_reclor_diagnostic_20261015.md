# Llama ReClor Prompt-v2 diagnostic

Generation ran on 2026-10-14; CPU finalization and local backup completed on
2026-10-15. This is a single-model, single-source diagnostic, not a replacement
for the original three-model smoke or its preserved REAL_SMOKE_FAIL result.

- 20 ReClor training questions, two samples each: 40/40 generated and verified.
- All 40 outputs use the strict terminal `Final Answer: X` format; no relaxed
  parser fallback was needed. Exact-match labels: 27 correct, 13 incorrect.
- Ten selected records were individually reviewed by the AI assistant against
  the question, options, official label, and response: ten AGREE. Reviewer
  provenance and rationale are private. This is not human sign-off and does not
  certify every reasoning statement. One correct-final-answer response contains
  an overbroad reasoning claim, noted in the private review.
- Existing minimum-token-gap selection produces seven pairs and 14 AB/BA rows.
  Parquet read-back confirms seven pair identities, each with permutations 0/1.
- Functional Gate has no failures. The unchanged Statistical Gate returns FAIL:
  no MATH questions were planned, and all positives/negatives necessarily come
  from Llama. These checks describe the deliberately restricted diagnostic;
  thresholds were not changed or waived to declare a full smoke PASS.
- The old 12-pair set and new seven-pair set cover only 13 unique questions,
  not 19. Six ReClor questions overlap. The sets remain separate, run-scoped
  artifacts; no cross-run pool or combined training dataset was constructed.
- The original 240 candidates and historical failed artifacts are preserved.
  No model inference, weight download, or training occurred during this CPU work.

The allowlisted JSON receipt contains run IDs, counts, and hashes. Full texts,
review decisions, logs, and Parquet stay outside Git in private storage.

## Next research action (not an execution authorization)

Keep the improved ReClor instruction. The next GPU proposal should target MATH
completion and reliable negative-answer yield, with a separately versioned
generation configuration if length or prompts change. Do not infer a guarantee
of improvement from the ReClor-only result, merge overlapping samples as new
questions, start a 200-pair pilot, or train from this diagnostic automatically.

The existing public label-check receipt is historical and intentionally remains
unchanged; this receipt records the later diagnostic separately.
