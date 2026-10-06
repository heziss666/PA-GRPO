# Controlled Training Data Pipeline Design

**Status:** Design approved subject to final minor-diff confirmation; implementation not started

**Branch:** `codex/data-pipeline-plan`

**Scope:** Phase 1 system design only; implementation is intentionally deferred

**Phase 1 terminal status:** `DATA_PIPELINE_SYSTEM_READY`

## 1. Purpose and non-goals

This document defines a controlled, reproducible data-production system for constructing preference pairs from MATH and ReClor. The system separates data acquisition, generation, verification, auditing, pair selection, permutation construction, and statistical gating so that each stage has an immutable, hash-pinned input contract.

Phase 1 proves that the production system is internally correct using synthetic fixtures, fake generation, and private source acquisition/splitting. It does **not** prove that the real three-model generation recipe works or that the resulting candidate distribution is healthy.

Phase 1 must not:

- start real vLLM generation;
- download or run the three generator weights for inference;
- produce the 240-candidate real smoke;
- claim `FUNCTIONAL_GATE_PASS`, `STATISTICAL_GATE_PASS`, or `REAL_SMOKE_PASS`;
- expand to a 200-pair pilot;
- commit real source text, real model responses, real gold answers, private manifests, secrets, or machine-specific absolute paths.

After this design is independently reviewed, a separate detailed implementation plan will be written. No implementation work is authorized by this document alone.

## 2. Stage boundary

### 2.1 Phase 1: system readiness

Phase 1 covers:

- this design and subsequent implementation plan;
- schemas, canonicalization, identifiers, and validation;
- MATH and ReClor source acquisition;
- question identity, source-local deduplication, and deterministic internal split;
- fake generation backend and backend-neutral adapter contract;
- generation planning, sharding, append-only writes, retry, and resume;
- ReClor and MATH verification;
- deterministic correct-versus-incorrect pair selection;
- AB/BA permutation construction;
- verifier audit protocol;
- Functional and Statistical Gate logic;
- runtime data-isolation checks;
- Windows/WSL unit tests and fake end-to-end tests;
- private question and split manifests stored outside Git.

The only successful Phase 1 status is:

```text
DATA_PIPELINE_SYSTEM_READY
```

### 2.2 Phase 2: real smoke

Phase 2 starts only after an independent Phase 1 code review explicitly passes. It runs on Linux/AutoDL with an external data root, external Hugging Face caches, and an environment-provided `HF_TOKEN`:

```text
Qwen/Qwen2.5-7B-Instruct
Qwen/Qwen2.5-32B-Instruct
meta-llama/Llama-3.1-8B-Instruct
        -> 240 real candidates
        -> machine verification
        -> human verifier audit
        -> Functional Gate
        -> Statistical Gate
```

Only a Phase 2 `PASS` permits the 200-pair pilot. `PASS_WITH_WARNINGS` requires explicit human approval; `FAIL` requires changing the recipe, verifier, or generator configuration and rerunning the smoke.

## 3. Data sources and legal boundary

### 3.1 MATH

- Dataset repository: `EleutherAI/hendrycks_math`.
- Use all train configurations.
- Resolve and record the immutable dataset revision/commit actually used.
- Record the source snapshot metadata in a private manifest.

### 3.2 ReClor

- Use the official locally acquired ReClor training data.
- ReClor is restricted to the current non-commercial research use.
- Official ReClor validation and test data do not participate in the internal split.
- Raw archives, extracted records, derived full-text manifests, and verified pools remain outside Git.
- Local acquisition accepts a runtime directory or archive path, but the path exists only in environment configuration and private manifests.

### 3.3 Runtime configuration

Public documentation and committed configuration use placeholders only:

```ini
PAGRPO_DATA_ROOT=<external-data-root>
PAGRPO_RECLOR_DIR=<external-reclor-directory>
PAGRPO_RECLOR_ARCHIVE=<external-reclor-archive>
```

Real local absolute paths are permitted as runtime inputs, but they must never be written to committed configuration, public artifacts, sanitized manifests, or public logs.

## 4. Data isolation and safe logging

Production data must live outside the Git repository. At startup, the system resolves the repository and data roots to real absolute paths and rejects any containment relationship. The check must handle `..`, symlinks, Windows junctions, and case-insensitive Windows path comparison.

The same production check applies to the effective Hugging Face cache locations, including:

```text
HF_HOME
TRANSFORMERS_CACHE
HUGGINGFACE_HUB_CACHE
```

An effective cache inside the repository is a production error, not merely a warning.

The `HF_TOKEN` is read only from the environment. Code may report `token_present=true/false` and access success/failure, but never the token value. Llama gated-access failure must fail fast before generation writes begin.

Public logging uses an allowlist. It may contain identifiers, generator and shard identifiers, status, sanitized error type, retry counts, aggregate counts, and timing. It must not contain question text, candidate responses, gold answers, tokens, private absolute paths, or unfiltered third-party exception payloads. Optional full private diagnostics, if later enabled, must be written below the external private data root and remain disabled by default.

Sanitized public manifests are allowlist-based and may contain only fields such as:

- run and schema identifiers;
- source and immutable source revision;
- split manifest hash;
- question, candidate, and pair counts;
- generator repository and immutable revision;
- status counts and aggregate statistics;
- artifact hashes.

Git-tracked content must not contain real MATH/ReClor question text, real candidate responses, real source gold answers, secrets, private manifests, or user-specific absolute paths. Explicitly marked **synthetic** fixtures may contain synthetic questions, responses, and gold values because they are required to test verifiers and pair construction.

## 5. Question identity, deduplication, and split

### 5.1 Question normalization v1

Question text is canonicalized using:

1. Unicode NFC;
2. CRLF and CR converted to LF;
3. leading and trailing whitespace stripped;
4. internal text and line structure preserved.

Normalization is versioned as `question_normalization_v1`.

### 5.2 Identity

For ReClor, construct this exact structured payload:

```json
{
  "answers": ["A...", "B...", "C...", "D..."],
  "context": "...",
  "question": "...",
  "schema": "reclor_question_content_v1"
}
```

Each text value is normalized with `question_normalization_v1`; answer array order is preserved. The object is serialized as canonical JSON with UTF-8 encoding, lexicographically sorted keys, `,` and `:` separators without optional whitespace, `ensure_ascii=false`, and no trailing newline. `question_content_hash` is SHA256 over those exact bytes. Including the structured field names and array boundaries prevents ambiguous concatenation. The identity remains:

```text
original_question_id = reclor:train:{official id_string}
```

The official ID is stable provenance, while the structured full content hash proves that the content associated with that ID has not changed.

For MATH:

```text
full_hash = SHA256(canonical problem text)
original_question_id = math:train:{first 20 bytes of full_hash}
```

Twenty bytes means exactly 40 hexadecimal characters. The full 64-hex SHA256 is always retained as `question_content_hash`; the shortened prefix is only the readable ID component.

### 5.3 Source-local deduplication

Deduplication occurs independently within each source snapshot, never across MATH and ReClor.

- MATH deduplicates by canonical problem hash and checks solution, category, and level for conflicts.
- ReClor deduplicates by canonical context, question, and ordered answers and checks the official label and all identity fields for conflicts.
- Equal content with consistent metadata keeps one deterministic provenance representative.
- Equal content with conflicting metadata fails fast.

### 5.4 Deterministic split

All internal splits operate at `original_question_id` granularity. A candidate, pair, or permutation can never cross a split, and downstream stages are forbidden from resplitting.

```text
train/internal-holdout = 90/10
split_seed = 42
```

- MATH first stratifies by category and level. If a stratum is too small, it falls back to category, then source-level stratification. Every fallback is recorded.
- ReClor stratifies by the official gold label.
- Official ReClor validation and test sets are excluded.

The split manifest uses canonical UTF-8 JSONL, LF line endings, sorted keys, and records sorted by question ID. It records source revisions or source file hashes, normalization version, split algorithm, seed, and any stratification fallback. Its SHA256 is the `split_manifest_hash`.

Every downstream run binds all three values:

```text
original_question_id
question_content_hash
split_manifest_hash
```

Resume is permitted only when the split manifest hash matches exactly. A changed source revision, file hash, or question content requires a new generation run.

## 6. Smoke selection and generation plan

The real Phase 2 smoke is selected from the internal training split:

```text
20 MATH questions
20 ReClor questions
3 generators x 2 samples per question
= 240 planned candidates
```

Question selection is deterministic and stratified according to the source split metadata.

Each model is run in a separate vLLM process, one model at a time. Generation is sharded, append-only, and resumable:

```text
question manifest
  -> generator A shards -> completion
  -> generator B shards -> completion
  -> generator C shards -> completion
  -> merge candidate pool
  -> verify
  -> build pairs
```

The backend-neutral interface is:

```python
generate(batch, generation_config) -> list[GenerationResult]
```

Phase 1 implements a fake backend and a vLLM adapter contract. The vLLM import must be genuinely lazy: importing `permstudy.data_pipeline.generation` on Windows or WSL without vLLM installed must succeed. vLLM is imported only when that backend is selected, at which point absence produces a clear configuration error. Phase 1 never starts a real model.

### 6.1 Candidate identity

`candidate_id` is deterministic and backend-neutral. It is derived from:

```text
original_question_id
generator_id
sampling_index
```

It is independent of backend return order. Decoding parameters are deliberately not part of `candidate_id`; they are pinned by the generation run manifest. Any generation configuration change creates a new `generation_run_id` and cannot be resumed into an old run.

`candidate_id` is a reusable logical ID, not a globally unique record key. The unique key for every candidate record is:

```text
(generation_run_id, candidate_id)
```

Resume, shard merge, verification lookup, deduplication provenance, and artifact references must always use the composite key. No stage may use a bare `candidate_id` to match records across generation runs.

### 6.2 Shards and resume

Each generator shard has:

```text
candidates.jsonl
failures.jsonl
manifest.json
```

Successful candidate records are append-only. Failures are appended separately. Resume computes the planned composite `(generation_run_id, candidate_id)` keys minus the successful composite keys for that same run.

Historical failures do not lower final generation completion after a later retry succeeds.

The shard manifest pins at least the model repository and immutable revision, backend, prompt template revision/hash, full sampling configuration, question/split manifest hash, shard ID, and planned count.

OOM is a recoverable failure record and never rolls back completed candidates. However, if resolving OOM changes batch size, tensor parallelism, maximum model length, memory utilization, or any other generation configuration, the old run is preserved and a new `generation_run_id` is mandatory. Resume of the old run is allowed only when the complete pinned configuration is unchanged.

## 7. Storage integrity and locking

Manifests are written to a temporary file in the same directory, flushed and fsynced, then atomically replaced. Large JSONL artifacts are not rewritten atomically; they are append-only, with one complete record per line and a `record_hash` per record.

On recovery:

- an incomplete final JSONL line is quarantined and resume may continue;
- a malformed middle line is an integrity failure;
- all referenced upstream artifacts must match the hashes pinned by their manifest.

Each shard uses an exclusive lock containing:

```text
run_id
shard_id
host
pid
created_at
```

Stale locks are never removed automatically. Recovery requires an explicit `--recover-stale-lock` operation and verification that the recorded process no longer exists before a writer can continue.

Configuration, isolation, and upstream integrity errors fail before output writes. Generation failures are recoverable records. Verifier outcomes such as `invalid`, `ambiguous`, and `error` are data states, not pipeline crashes, but they affect audits and gates.

## 8. Verification contract

All verification outputs use:

```text
correct
incorrect
invalid
ambiguous
error
```

A question-level `gold_verification_error` removes the complete question from pair construction. Only `correct` and `incorrect` candidates can enter the pair pool.

### 8.1 ReClor

ReClor gold is never inferred from prompt text. The official dataset label is converted directly to canonical `A/B/C/D`.

Candidate parsing accepts exactly one valid terminal:

```text
Final Answer: X
```

where `X` is a valid option. Multiple identical final markers are `invalid`; multiple conflicting markers are `ambiguous`.

### 8.2 MATH

MATH candidates must expose one terminal boxed answer in the prescribed final-answer region. Gold extraction, candidate extraction, and equivalence use a pinned `math-verify==0.9.0` contract.

`incorrect` is assigned only if gold parsing succeeds, candidate parsing succeeds, and the verifier explicitly establishes non-equivalence. Multiple parsed expressions or uncertain symbolic comparison are `ambiguous`; timeout or library exception is `error`.

Gold is parsed once per question and cached for all candidates. Gold failure yields `gold_verification_error` for the question. Each candidate comparison has an independent recorded timeout.

Verification records contain run and candidate identifiers, verifier name/version, parser version, gold and prediction parse status, canonical representations where privately allowed, verification status, timeout, and sanitized error type. ReClor uses `reclor_exact_match`; it must not be labeled as `math-verify`.

## 9. Response canonicalization and deterministic pair selection

Response normalization v1 applies:

1. Unicode NFC;
2. CRLF and CR converted to LF;
3. leading and trailing whitespace stripped;
4. internal text and line structure preserved.

The normalized response hash is SHA256 over canonical UTF-8 bytes. Raw candidate records are retained. Duplicate normalized responses form a run-scoped deduplication group whose representative is the lexicographically smallest `candidate_id`. The group preserves the normalized response hash, representative composite candidate key, all member composite candidate keys, and member generators. If members with one normalized response hash have conflicting verification statuses, pair construction fails with a verifier/integrity error rather than silently selecting a representative.

Responses with `finish_reason=length`, empty content, or no verifiable terminal answer are excluded from the pair pool even if an incidental answer can be extracted.

All response lengths are measured using the tokenizer for `Qwen/Qwen2.5-7B-Instruct` at a recorded immutable revision:

```python
len(tokenizer.encode(canonical_response, add_special_tokens=False))
```

For each eligible question, enumerate every representative `correct x incorrect` combination and select the pair with minimum absolute token-length gap. There is no same-generator or cross-generator preference. Exact ties are resolved lexicographically by candidate IDs.

The pair identifier is run-scoped:

```text
pair_id = H(
  generation_run_id,
  original_question_id,
  response_pos_id,
  response_neg_id,
  pair_schema_version
)
```

Selected response hashes are also stored. Thus rebuilding within one generation run is stable, while a new generation run necessarily enters a new pair namespace.

The pair manifest hash is computed from canonical UTF-8 JSONL with sorted keys, LF line endings, records sorted by `original_question_id`, and no timestamps or other volatile fields in the hashed payload.

## 10. AB/BA permutations

Each selected semantic pair produces exactly two surface forms:

```text
AB: A=positive, B=negative
BA: A=negative, B=positive
```

Permutation records inherit the question ID, split, split manifest hash, pair ID, generation run ID, and pinned upstream hashes. They do not independently resplit or recover source files.

## 11. Human verifier audit

The smoke audit uses the exact verified manifest that feeds pair construction. It runs before pair construction or is cryptographically bound to the same verification output.

Audit coverage:

1. inspect every `ambiguous`, `error`, and `gold_verification_error`;
2. inspect every `invalid`;
3. sample `correct` and `incorrect` by `source x generator x status`, using `audit_seed=42`, up to five per cell and all records when a cell has fewer than five.

Human verdicts are:

```text
AGREE
DISAGREE
UNSURE
```

An apparent false `correct` or false `incorrect` receives a second review of the source item, gold, candidate, parser output, and verifier result. A confirmed `DISAGREE` causes the Statistical Gate to fail. `UNSURE` enters a separate review queue and is not counted as a confirmed verifier error.

Audit records include `generation_run_id`, `verification_run_id`, candidate ID, source, generator, machine status, audit seed, verdict, and reason code. Reason codes include:

```text
missing_final_marker
duplicate_same_marker
conflicting_markers
math_parse_failure
gold_parse_failure
timeout
dependency_error
prompt_contract_mismatch
other
```

Systematic prompt, parser, or verifier problems block expansion even if the affected records would otherwise be excluded from pairs. Fixing a verifier reruns verification, audit, pair, and permutation stages against the immutable candidate pool; it does not regenerate candidates.

## 12. Gates

### 12.1 Functional Gate

The Functional Gate checks that planned candidates are accounted for after retry/resume, manifests and hashes are valid, verification completed, audit artifacts bind to the correct run, and pair/permutation records satisfy all contracts. Generation completion of 100% means that every planned candidate eventually has a successful record after allowed retry/resume; a first-attempt failure alone does not fail the gate.

### 12.2 Statistical Gate

The Statistical Gate emits exactly one of:

```text
PASS
PASS_WITH_WARNINGS
FAIL
```

All Gate metrics use the following fixed denominators.

For each source:

```text
pair_yield(source)
= number of planned source questions that produce one final selected pair
  / number of planned smoke questions for that source
```

The real smoke denominator is 20 for MATH and 20 for ReClor. Each question contributes at most one selected pair to the numerator.

For each generator/source cell:

```text
invalid_error_rate(generator, source)
= (# verification_status=invalid + # verification_status=error)
  / # successfully generated candidates in that generator/source cell
```

At 100% real-smoke generation completion, the cell denominator is `20 questions x 2 samples = 40`. Historical failed attempts and missing candidates are not included in this denominator; unresolved missing candidates independently fail the 100% completion requirement.

Generator dominance uses only representatives in final selected pairs:

```text
positive_share(generator, scope)
= # selected pairs whose positive representative comes from generator in scope
  / # selected pairs in scope

negative_share(generator, scope)
= # selected pairs whose negative representative comes from generator in scope
  / # selected pairs in scope
```

`scope=pooled` uses all selected MATH and ReClor pairs; `scope=source` uses only selected pairs from that source. A zero selected-pair denominator leaves dominance undefined, while the corresponding pair-yield rule already produces a hard failure.

Correct rate excludes non-binary verifier states:

```text
correct_rate(generator, source)
= # correct
  / (# correct + # incorrect)
```

If `correct + incorrect = 0`, correct rate is undefined and reported diagnostically; it is not treated as either 0% or 100%.

Hard failure conditions:

- final generation completion after retry/resume is below 100%;
- either source has valid-pair yield below 20%;
- any generator/source cell has `invalid + error` above 20%;
- in pooled valid pairs, a single generator supplies at least 90% of all positive or all negative responses;
- a confirmed human-audit `DISAGREE` exists for a `correct` or `incorrect` label;
- audit reveals a systematic prompt/parser/verifier defect.

Warnings:

- either source has pair yield below 30%;
- pooled or per-source generator share of positives or negatives exceeds 70%;
- any generator/source correct rate is 0% or 100%;
- per-source generator dominance reaches 90% or more, recorded as a strong warning;

`ambiguous` is always reported separately and never merged into `incorrect`.

Diagnostic-only outputs include ambiguous rate, Cramer's V, generator-by-label contingency tables, same-generator versus cross-generator pair proportions, source-wise and generator-wise correct rates, response length distributions, and positive-negative token-length gaps. Phase 2 v1 does not derive a warning from ambiguous rate because no deterministic threshold has been approved. The 40-question smoke is too small for significance claims from Cramer's V.

Gate actions:

```text
FAIL               -> fix recipe/verifier/config and rerun smoke
PASS_WITH_WARNINGS -> mandatory human review before any pilot
PASS               -> may proceed to the 200-pair pilot
```

## 13. Layering and run lineage

The approved package structure is:

```text
permstudy/data_pipeline/
  schema.py
  canonical.py
  ids.py
  isolation.py
  io.py
  sources/math.py
  sources/reclor.py
  splitting.py
  generation/base.py
  generation/fake.py
  generation/vllm.py
  generation/runner.py
  verification/reclor.py
  verification/math.py
  pairs.py
  permutations.py
  audit.py
  gates.py
```

Command-line entry points are planned as:

```text
scripts_permstudy/data/prepare_questions.py
scripts_permstudy/data/plan_generation.py
scripts_permstudy/data/generate_candidates.py
scripts_permstudy/data/verify_candidates.py
scripts_permstudy/data/build_reasoning_pairs.py
scripts_permstudy/data/build_permutations.py
scripts_permstudy/data/audit_generator_distribution.py
scripts_permstudy/data/validate_dataset.py
scripts_permstudy/data/run_fake_e2e.py
```

Schemas use frozen dataclasses and explicit validators; Phase 1 does not add a schema framework.

Each layer records `run_id`, `schema_version`, direct upstream manifest hash, its own configuration hash, and output manifest hash. A layer may read only:

1. its direct immutable upstream manifest; and
2. upstream artifacts explicitly referenced and hash-pinned by that manifest.

No layer may read untracked or freely discovered data files. A changed verifier configuration creates a new verification run and therefore new downstream audit, pair, and permutation namespaces without changing the generation run.

## 14. Test matrix

Phase 1 tests must cover:

- canonical normalization and stable cross-platform hashes;
- ReClor and MATH identity construction, collision checks, and source-local deduplication;
- metadata conflict failures;
- deterministic stratified splitting and both MATH fallback levels;
- split inheritance and prevention of cross-split candidates, pairs, and permutations;
- resolved-path repository containment, symlink/junction, case, cache, and redaction checks;
- generation planning, backend-neutral candidate IDs, fake backend behavior, and lazy vLLM import;
- append-only resume, historical failure then success, incomplete-tail recovery, malformed-middle failure, record hashes, and stale-lock recovery rules;
- ReClor terminal marker parsing and official-label handling;
- MATH gold caching, strict terminal extraction, equivalence, ambiguity, timeout, and errors with `math-verify==0.9.0`;
- response normalization, deduplication, fixed-tokenizer length measurement, tie-breaking, run-scoped pair IDs, and canonical pair hashes;
- exact AB/BA permutation semantics and lineage;
- deterministic audit sampling, verdicts, and reason codes;
- all Gate thresholds and status transitions;
- equivalent canonical hashes on Windows and Linux/WSL.

Two end-to-end paths are required:

1. a committed, explicitly marked synthetic fixture that runs acquisition-like loading through fake generation, verification, audit, pairing, permutation, and gates on Windows/WSL;
2. a private real-source acquisition and split followed by the fake full pipeline, with all full-text artifacts outside Git.

## 15. Phase 1 acceptance criteria

Phase 1 may report `DATA_PIPELINE_SYSTEM_READY` only when all of the following are true:

- the design and detailed implementation plan have been independently reviewed before implementation;
- source acquisition records immutable MATH revision and ReClor file hashes privately;
- the production data root and effective model caches are external and pass resolved-path isolation checks;
- question identities, source-local deduplication, deterministic 90/10 split, and split manifest are reproducible;
- the fake 40-question/240-candidate plan completes end to end;
- interruption and resume yield the same successful candidate set and canonical manifest hashes as an uninterrupted run;
- importing the generation package succeeds without vLLM installed;
- MATH verification pins `math-verify==0.9.0`;
- pair and permutation outputs are deterministic and hash-stable;
- sanitized public artifacts pass the field allowlist;
- Git scanning finds no real source questions, real responses, real source gold, tokens/secrets, private manifests, or user-specific absolute paths;
- explicitly marked synthetic fixture questions, responses, and gold are the only full-text fixture exception;
- Windows/WSL unit and fake end-to-end tests pass;
- an independent Phase 1 code review passes.

Passing these criteria proves that the data-production system is ready for real smoke execution. It does not prove real generator quality or authorize the 200-pair pilot.

## 16. Review checkpoint

This commit is intentionally design-only. After review approval, the next deliverable is a detailed, stepwise implementation plan with test-first checkpoints. No production pipeline code or real generation should begin before that approval.
