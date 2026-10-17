# Fresh-pair production batch: 500 MATH + 500 ReClor

User authorized execution on the cloned RTX 4090 48GB instance. This batch
advances the approximately 1000 reliable-pair objective; 1000 selected questions
are not 1000 verified pairs, and no training is authorized.

- Fixed existing Train/Holdout split; select 500 fresh Train questions per source
  with the existing proportional-stratum selector and seed 42.
- Exclude original smoke, the earlier fresh MATH diagnostic cohort, and the
  exact previous 200-question production cohort of each source. Never resplit.
- Reuse original immutable model revisions: Qwen2.5-7B-Instruct and
  Llama-3.1-8B-Instruct, two samples per model/question: 4000 candidates total.
- Keep previous production prompt text, temperature .8, top-p .95, seed 42,
  batch size 8, TP 1, context 4096, memory utilization .85; MATH output 1024,
  ReClor output 512. No quantization, 32B, parser, reward or Gate changes.
- New recipes `math-production-500-v2` and `reclor-production-500-v2` bind the
  selection size/version and previous-cohort exclusion to new run namespaces.
  Original 200-v1 recipe/config/output identities must remain unchanged.
- Use the existing CLI, append-only shard/resume runner and formal manifests.
  Each source/model process reuses one engine across its shards; models run
  sequentially. No new generation framework or candidate cross-run merging.
- CPU preflight validates hashes, disjoint question IDs and native input lengths
  before GPU execution. Stop on context/config failure rather than truncate.
- Record actual completion, timing and failures privately; copy and hash-check
  raw evidence locally, then shut down without deleting the data disk/cache.
- Perform verification and existing minimum-gap pair construction on local CPU.
  Report automatic pairs separately from quality-screened/released pairs;
  ambiguous/gold-error/false-label/obviously flawed positive cases stay isolated.
- Preserve provenance/model-label statistics. Do not preferentially choose a
  generator for the positive or negative side, or change correctness criteria
  to increase pair yield. No new AB/BA Judge sampling or training in this batch.

Implementation scope: only extend the Phase 2 recipe/selection entry point and
four focused CPU cases. Existing Phase 1 algorithms remain frozen.
