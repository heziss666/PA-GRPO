# Phase 2 Real Candidate Smoke Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to execute this plan task-by-task, but only after the corresponding authorization gate below is explicitly opened. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate and validate the fixed 240-candidate real smoke on one Linux 80 GB GPU, then make an evidence-based decision about whether a separately approved 200-pair pilot may be proposed.

**Architecture:** Keep the approved Phase 1 data pipeline immutable and use it as the source of truth for question selection, candidate identity, verification, audit, pairing, permutations, trainer export, and gates. Add at most one thin Phase 2 execution launcher around the existing backend-neutral generation API because the Phase 1 CLIs intentionally reject real vLLM execution; run the three generators sequentially into the external private data root, then reuse the existing canonical records and gates without adding a new framework.

**Tech Stack:** Linux, Python 3.12, CUDA 12.4-compatible provider driver/runtime, PyTorch 2.6.0, vLLM 0.8.5, `transformers==4.57.1`, `datasets==4.4.1`, `huggingface-hub==0.36.0`, `pyarrow==22.0.0`, `math-verify[antlr4_9_3]==0.9.0`, one NVIDIA A800/H800/H100-class 80 GB GPU, and the repository at `main@34ef6f4f0825d042d8ae8f1038afc7daa364a75f`.

**Spec:** `docs/superpowers/specs/2026-10-06-controlled-training-data-pipeline-design.md`, `docs/superpowers/plans/2026-10-06-controlled-training-data-pipeline-phase1.md`, `artifacts/data_pipeline/phase1/summary.json`, and `PLAN.md` sections 7.3–7.5.

## Global Constraints

- Phase 1 is complete and frozen at `main@34ef6f4`; do not change `permstudy/data_pipeline/`, `scripts_permstudy/data/`, or their Phase 1 tests, manifests, audit rules, gates, parsers, verifiers, pair selection, and trainer export to make the real smoke pass.
- This document does not authorize AutoDL rental, model downloads, vLLM startup, candidate generation, or the 200-pair pilot.
- The smoke contains exactly 20 MATH and 20 ReClor questions selected by the approved Phase 1 split, three immutable generators, and two samples per question: `40 × 3 × 2 = 240` planned candidate identities.
- The approved generators and revisions are:
  - `Qwen/Qwen2.5-7B-Instruct@a09a35458c702b33eeacc393d103063234e8bc28`
  - `Qwen/Qwen2.5-32B-Instruct@5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd`
  - `meta-llama/Llama-3.1-8B-Instruct@0e9e39f249a16976918f6564b8830bc894c89659`
- `PAGRPO_DATA_ROOT`, Hugging Face caches, model weights, prompts, responses, verification records, audit records, pairs, permutations, and trainer data remain outside the Git repository.
- `HF_TOKEN` is read only from the environment. It may be reported only as `token_present=true/false`; its value, private paths, prompt text, responses, source gold, and raw third-party errors must never enter Git or public logs.
- Each generator runs in a separate process, one model at a time. No process may hold two models concurrently.
- The Phase 2 smoke performs inference and CPU verification only. It performs no GRPO training, no optimizer step, and no 200-pair expansion.
- Each question can produce at most one selected pair, so this 40-question smoke can produce at most 40 pairs. It cannot satisfy a 200-pair pilot requirement.
- Changing any prompt, model revision, sampling value, batch size, tensor-parallel setting, maximum length, or memory-utilization value creates a new `generation_run_id`; outputs from different generation runs must not be merged.
- Existing gate thresholds and denominators are authoritative. Phase 2 may report them but must not tune them after seeing the smoke.

## Authorization Gates

| Gate | Permitted work | Current state |
| --- | --- | --- |
| P0 — plan review | Edit and review this document only | Open |
| P1 — activation review | Implement and CPU-test the single thin Phase 2 launcher; no GPU or model download | Hold |
| P2 — real-smoke execution | Rent the approved GPU, download pinned weights, and execute the 240-candidate smoke within the resource cap | Hold |
| P3 — pilot proposal | Draft a separate 200-pair pilot plan from the completed smoke evidence | Hold |
| P4 — pilot execution | Acquire more questions/candidates and run training | Hold |

Approving this plan closes P0 only. P1, P2, P3, and P4 require separate explicit authorization.

## Known Execution Boundary

The existing Phase 1 interfaces are intentionally safe by default:

- `scripts_permstudy/data/plan_generation.py` rejects non-fake generator configs;
- `scripts_permstudy/data/generate_candidates.py` rejects `--backend vllm`;
- `verify_candidates.py` calls the fake-only `load_plan()`, and the later formal CLI loading chain is consequently not a valid consumer of a `backend="vllm"` generation manifest;
- `VLLMGenerationBackend` requires an injected `engine_factory` and never starts a model by itself.

Before any GPU use, P1 must therefore add exactly one Phase 2-specific entry point, preferably `scripts_permstudy/phase2/run_real_smoke.py`, plus focused CPU tests using a fake injected vLLM module. The entry point owns the real-run orchestration from planning through export and exposes small subcommands such as `plan`, `generate`, `verify`, `audit`, and `finalize`. It consumes existing package APIs directly; it must not call the fake-only Phase 1 loaders for a real run.

The entry point may call `GenerationConfig`, `plan_generation`, `VLLMGenerationBackend`, `run_generation_shard`, `verify_math_question`, `verify_reclor_question`, `build_audit_selection`, `validate_audit_decisions`, `select_pair`, `build_permutations`, `evaluate_functional_gate`, `evaluate_statistical_gate`, `export_trainer_parquet`, and the existing artifact writers and integrity validators. It must not modify or bypass the frozen Phase 1 CLI guards, create a second manifest format, reimplement those algorithms, or write real content to Git. Its only orchestration responsibilities are:

1. load the approved split and question artifacts from `PAGRPO_DATA_ROOT`;
2. render and hash the exact source-specific candidate prompt templates below;
3. construct the three `backend="vllm"` generation configs;
4. inject a real vLLM engine factory and retain one engine for the lifetime of one generator process;
5. loop over all pending shards for one requested generator through the existing append-only runner;
6. load real candidate records and invoke the existing package verification, audit, pair, permutation, gate, and trainer-export functions;
7. emit only the existing artifact schemas and sanitized progress.

`VLLMGenerationBackend.generate()` calls `engine_factory` for every batch. The injected factory must therefore be a process-local caching closure that returns the same already-loaded engine for every batch and shard of that generator. The launcher must process all pending shards for that generator before destroying the engine. A process crash may construct one replacement engine and resume missing composite keys; normal shard progression must not reload weights. P1 tests belong under a new Phase 2-specific test path, such as `tests_permstudy/phase2/`, so the protected Phase 1 test tree remains unchanged.

P1 review must prove, without a GPU, that request IDs map back to the correct `(generation_run_id, candidate_id)`, prompt selection is source-correct, return order is irrelevant, foreign or missing vLLM results fail closed, resume requests only missing composite candidate keys, and one fake engine instance is reused across multiple batches and shards. It must also run one synthetic package-API integration path from vLLM-shaped candidate records through verification, audit decisions, pair selection, AB/BA, Functional/Statistical Gates, trainer Parquet export, and the existing trainer identity round trip. This test proves the real entry point never falls back to the fake-only Phase 1 CLI loading chain.

## Fixed Real-Smoke Recipe

### Prompt contract

The Phase 2 launcher must define these templates as immutable UTF-8 text and bind their exact canonical SHA256 to every generator config.

MATH system text:

```text
Solve the mathematics problem carefully. Show your reasoning, then end the response with exactly one terminal \boxed{...} answer. Do not write any text after the terminal boxed answer.
```

MATH user text:

```text
{problem}
```

ReClor system text:

```text
Solve the logical reasoning multiple-choice problem carefully. Choose exactly one option A, B, C, or D. End the response with exactly one terminal line in the form Final Answer: X, where X is the chosen option. Do not write any text after that line.
```

ReClor user text:

```text
Context:
{context}

Question:
{question}

A. {answer_a}
B. {answer_b}
C. {answer_c}
D. {answer_d}
```

Source gold and official solution text must never be rendered into a candidate prompt.

### Sampling and runtime configuration

The first approved run uses the following values. A review change edits this document before P2; an execution-time change creates a new generation run and requires a new approval decision.

| Parameter | 7B | Llama 8B | 32B |
| --- | ---: | ---: | ---: |
| `temperature` | 0.8 | 0.8 | 0.8 |
| `top_p` | 0.95 | 0.95 | 0.95 |
| `max_new_tokens` | 512 | 512 | 512 |
| `samples_per_question` | 2 | 2 | 2 |
| base `seed` | 42 | 42 | 42 |
| `batch_size` | 8 | 8 | 2 |
| `tensor_parallel_size` | 1 | 1 | 1 |
| `max_model_len` | 4096 | 4096 | 4096 |
| `gpu_memory_utilization` | 0.85 | 0.85 | 0.90 |
| `shard_size` | 8 | 8 | 8 |

Each request uses a deterministic request seed derived from the base seed and the canonical composite candidate key. The derivation algorithm and version are included in the prompt/runtime config hash so the two sampling indices are reproducible without depending on backend return order.

The execution order is Qwen2.5-7B, Llama-3.1-8B, then Qwen2.5-32B. This validates access, formatting, append-only writes, and resume behavior on the smaller models before loading the most expensive model.

## Resource Budget

| Resource | Requirement or cap | Stop condition |
| --- | --- | --- |
| GPU | One Linux NVIDIA GPU with 80 GB VRAM | Reported VRAM below 80 GB or CUDA unavailable |
| Host RAM | 64 GB minimum; 128 GB preferred | Below 64 GB |
| External free disk | 200 GB minimum before downloads | Below 200 GB or any effective cache resolves inside the repo |
| Network transfer | Approximately 100–120 GB for the three immutable model snapshots | Repository/revision differs or gated access fails |
| Private pipeline artifacts | Reserve 10 GB; expected use is substantially below 1 GB excluding caches | Any write resolves inside the repo |
| GPU time | Hard cap: 6 billable GPU-hours for preflight, downloads while rented, model startup, generation, and one unchanged-config resume | Six hours reached; stop without extending |
| Per-model allowance | 7B ≤ 1 hour, Llama 8B ≤ 1 hour, 32B ≤ 3 hours, shared contingency ≤ 1 hour | Allowance exceeded without a completed model |
| Training compute | Zero | Any optimizer/training job is about to start |

The monetary ceiling is `6 × the provider's displayed hourly price` and must be recorded and approved immediately before P2. No stale price from `PLAN.md` is treated as a current quote.

Actual floating-point GPU hours, peak VRAM, disk/cache use, hourly price, and billed amount are written only to a private resource report below `PAGRPO_DATA_ROOT/private_reports/`. The public summary contains only allowlisted integer counts and lineage/status evidence; it does not coerce resource measurements into misleading count fields.

## Task 1: AutoDL Environment Preflight

**Purpose:** Prove that the exact environment can load all three approved model snapshots and write only to approved external locations before candidate generation begins.

**Consumes:** frozen Phase 1 baseline `main@34ef6f4`, the separately approved P1 launcher commit, the Phase 1 private split manifest, the three immutable model revisions, external `PAGRPO_DATA_ROOT`, external Hugging Face caches, and an environment-only `HF_TOKEN`.

**Produces:** One private preflight record containing package versions, GPU facts, effective cache paths as redacted path classes, model-access booleans, free-space counts, the approved monetary ceiling, and PASS/FAIL. It contains no source text, prompt, response, gold, token value, or private absolute path.

- [ ] Verify `git rev-parse HEAD` equals the separately approved P1 launcher commit, the checkout is clean, and `34ef6f4f0825d042d8ae8f1038afc7daa364a75f` is an ancestor of that commit.
- [ ] Verify the protected Phase 1 paths are byte-identical to the frozen baseline: `git diff --exit-code 34ef6f4f0825d042d8ae8f1038afc7daa364a75f -- permstudy/data_pipeline scripts_permstudy/data tests_permstudy/data_pipeline`. Record both the Phase 1 baseline SHA and the approved launcher SHA in the private preflight record.
- [ ] Verify the P1 activation patch has passed review and its CPU-only focused tests; otherwise stop before renting a GPU.
- [ ] On the rented instance, record `nvidia-smi`, GPU model, VRAM, driver, CUDA visibility, host RAM, CPU count, and external free disk.
- [ ] Use Python 3.12 with the previously validated Task 4 base (`torch==2.6.0`, `vllm==0.8.5`) and install the Phase 1 data dependencies so that `datasets==4.4.1` and `math-verify==0.9.0` win. Do not rely on the older `math-verify==0.8.0` line in `requirements-lock.txt`.
- [ ] Execute import/version checks for `torch`, `vllm`, `transformers`, `datasets`, `pyarrow`, and `math_verify`; require CUDA availability and the exact pinned versions above.
- [ ] Run the existing isolation check against `PAGRPO_DATA_ROOT`, `HF_HOME`, `HF_HUB_CACHE`/`HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE`, and the datasets caches. Any effective path inside the repo is a hard FAIL.
- [ ] Confirm `HF_TOKEN` presence without printing it, and check access to all three exact model revisions without silently substituting branches, mirrors, quantized checkpoints, or different revisions.
- [ ] Place the Phase 1 private split on AutoDL by exactly one approved route: securely transfer the already accepted external data-root subset, or deterministically rebuild it from the pinned MATH revision and ReClor four-file snapshot. Never transfer it through Git.
- [ ] After transfer or rebuild, validate every referenced private artifact against its original manifest SHA256, then require identical canonical split bytes, `split_manifest_hash`, 20+20 smoke composite identities, and 240 planned candidate identities. Any mismatch stops execution; it must not be repaired by resplitting or selecting replacements.
- [ ] Perform a load-only preflight in the fixed execution order, destroying each engine before loading the next. No candidate response is persisted during load-only preflight.

**PASS:** Every check succeeds, the 32B model loads under its approved config, and the projected run remains inside the six-hour cap.

**FAIL:** Stop and release the instance. In particular, do not install a Linux NVIDIA kernel driver, switch model revisions, enable quantization, change tensor parallelism, or change memory settings inside the same generation run. Any proposed configuration change returns to plan review and produces a new generation namespace.

## Task 2: Real Candidate Generation

**Purpose:** Materialize all 240 planned candidates without changing the approved question set or generation recipe.

**Consumes:** The approved preflight, Phase 1 smoke assignments, three immutable configs, and the P1 thin launcher.

**Produces:** One generation run containing 240 successful composite candidate keys, append-only per-shard candidate/failure histories, verified manifests, and the canonical semantic candidate-set hash.

- [ ] Recompute the plan from the immutable split and assert exactly 40 unique question IDs, 80 planned candidates per generator, and 240 unique `(generation_run_id, candidate_id)` keys.
- [ ] Start Qwen2.5-7B once in a fresh process, retain that engine across every batch, and execute its ten eight-candidate shards. Release the engine and CUDA memory only after the final shard.
- [ ] Start Llama-3.1-8B once in a fresh process, retain that engine across every batch, and execute its ten shards. Release the engine and CUDA memory only after the final shard.
- [ ] Start Qwen2.5-32B once in a fresh process, retain that engine across every batch, and execute its ten shards. Release the engine and CUDA memory only after the final shard.
- [ ] After every shard, validate the manifest-confirmed JSONL prefix, record count, artifact hash, config hash, split lineage, model revision, and completed composite keys.
- [ ] On a transient process/network failure, restart with the identical config and resume only `planned_keys - successful_keys`. Historical failure records remain append-only.
- [ ] When all shards finish, require exactly 240 successful composite keys and compute the canonical semantic candidate-set hash.

**Failure handling:**

- A first-attempt transient failure may be retried under the identical config.
- A missing or duplicate backend result, foreign request ID, malformed completion, manifest mutation, middle JSONL corruption, or lineage mismatch fails closed.
- `finish_reason=length` is a successful generation record but is later ineligible for pairing.
- OOM is recorded as a recoverable failure. If resolving it requires changing any config value, preserve the old run, stop P2, and request approval for a new run; never continue the old namespace.
- Generation completion below 100% after allowed unchanged-config retries is a Functional Gate FAIL.

## Task 3: Verification and Human Audit

**Purpose:** Establish that machine labels are trustworthy before any pair or trainer artifact is accepted.

**Consumes:** The immutable 240-candidate pool and the exact question manifest used for generation.

**Produces:** Candidate/question verification records, a deterministic audit selection, complete human decisions, and an audit summary bound to the same verification snapshot.

- [ ] Use the Phase 2 entry point to load the real generation artifacts directly, validate their existing schemas and lineage, and pass the resulting records to package APIs. Do not invoke `verify_candidates.py` or any other fake-only downstream CLI loader for the real run.
- [ ] Call the existing ReClor strict terminal parser and official-label exact match; never infer gold from prompt or response text.
- [ ] Call the existing MATH verifier with `math-verify==0.9.0`, parent-enforced `gold_timeout_seconds=30`, and `candidate_timeout_seconds=15`.
- [ ] Require each successful candidate to resolve to exactly one of `correct`, `incorrect`, `invalid`, `ambiguous`, or `error`; keep `gold_verification_error` question-level.
- [ ] Build the existing `audit_seed=42`, `max_per_cell=5` selection from the exact verification snapshot.
- [ ] Human reviewer 1 inspects every `invalid`, `ambiguous`, `error`, and `gold_verification_error`, plus the selected `correct/incorrect` samples stratified by `source × generator × status`.
- [ ] Record only `AGREE`, `DISAGREE`, or `UNSURE` with an approved reason code. Raw review material stays private.
- [ ] Every apparent false `correct/incorrect` or every `UNSURE` receives a second review by the research owner. Only a second-review-confirmed label error becomes `confirmed_disagree=true`; unresolved `UNSURE` remains separate.
- [ ] Validate exact one-to-one coverage between audit selection and decisions before calculating any gate.

**Responsibility:** Automation may prepare identities and statistics, but a human research owner owns the audit verdicts and signs the final audit summary. The agent must not invent AGREE decisions or infer them from verifier output.

**Failure handling:** A confirmed false `correct/incorrect` or systematic prompt/parser/verifier defect blocks expansion. A verifier-only correction reuses the immutable candidates and reruns verification, audit, pair, permutation, export, and gates. A prompt or generation-config correction requires a new generation run.

## Task 4: Functional and Statistical Gates

**Purpose:** Reuse the approved Phase 1 gates without changing thresholds after observing real outputs.

**Consumes:** Canonical plans, candidates, question verification, candidate verification, audit decisions, selected pairs, and AB/BA permutations from one consistent lineage.

**Produces:** An exact Functional Gate result, an exact Statistical Gate result, diagnostic tables, a trainer-ready Parquet when pairs exist, and one allowlisted sanitized Phase 2 summary.

- [ ] Through the same Phase 2 entry point, load `Qwen/Qwen2.5-7B-Instruct` tokenizer revision `a09a35458c702b33eeacc393d103063234e8bc28` and call the existing deterministic response deduplication and minimum-token-gap pair selector.
- [ ] Create at most one pair per question, then exactly AB and BA permutations for every selected pair.
- [ ] Run the existing Functional Gate and require 100% final generation completion, complete verification/audit coverage, valid manifests, and valid pair/permutation integrity.
- [ ] Run the existing Statistical Gate with the frozen denominators and thresholds.
- [ ] Export trainer Parquet only as an independent consumer of canonical pair/permutation records and perform the existing `RLHFDataset → DataProto → explicit identity` round trip.
- [ ] Produce diagnostics for source/generator status counts, pair yield, pooled/per-source generator dominance, correct rates, ambiguous rate, Cramér's V, same/cross-generator pairs, response lengths, and positive-negative token gaps.
- [ ] Write one allowlisted public file at `artifacts/data_pipeline/phase2/summary.json`. It may use only the existing `PUBLIC_EVIDENCE_KEYS`: encode the overall gate outcome in `phase_status`, keep `completion_counts` and other count fields strictly nonnegative integers, and use the existing run/revision/hash fields for lineage. `git_sha` records the approved P1 launcher commit, while `run_ids.phase1_baseline_git_sha` records `34ef6f4f0825d042d8ae8f1038afc7daa364a75f`. Do not add a new top-level evidence key. It must set `real_generation_performed=true` and contain no real text, gold, prompt, response, secret, private path, floating-point GPU hours, peak-VRAM values, or monetary amounts.

The frozen hard failures are:

- final generation completion below 100%;
- either source pair yield below 20%;
- any generator/source `invalid + error` rate above 20%;
- one generator supplies at least 90% of pooled positive or pooled negative representatives;
- any confirmed binary-label disagreement;
- any systematic prompt/parser/verifier defect.

The frozen warnings are:

- either source pair yield below 30%;
- pooled or per-source positive/negative generator share above 70%;
- any defined generator/source correct rate equals 0% or 100%;
- per-source generator dominance at least 90% is a strong warning.

`ambiguous` remains diagnostic-only in v1 and is never merged into `incorrect`.

## Task 5: Pilot Decision

**Purpose:** Turn the real smoke evidence into a bounded recommendation without starting the pilot.

**Consumes:** The signed audit summary, Functional Gate result, Statistical Gate result, sanitized Phase 2 summary, private diagnostic tables, and a human inspection of representative positive/negative pairs.

**Produces:** One of the decisions below. None authorizes pilot execution.

### If `PASS`

- [ ] The research owner inspects representative pairs from each source and generator combination, including shortest/longest responses and smallest/largest token gaps.
- [ ] Draft a separate 200-pair pilot proposal. Estimate the number of required unique questions from the observed source-wise pair yields, add a stated reserve, and preserve source stratification and `original_question_id` isolation.
- [ ] Keep the smoke and pilot namespaces separate. Do not relabel the at-most-40 smoke pairs as a 200-pair dataset.
- [ ] Request explicit P3 approval for the proposal; P4 remains closed.

### If `PASS_WITH_WARNINGS`

- [ ] Do not expand automatically.
- [ ] The research owner reviews every warning, the affected private samples, and generator/source contingency tables.
- [ ] Record an explicit accept/reject rationale. Only an explicit human approval may authorize drafting the pilot proposal.

### If `FAIL`

- [ ] Do not expand or train.
- [ ] Classify the root cause as environment, generation config/prompt, verifier, audit, or data distribution.
- [ ] Reuse candidates only for verifier/audit/pair fixes. Any prompt, sampling, model, or runtime-config change creates a new generation run and requires renewed resource authorization.

## Required Review Package

The Phase 2 smoke review package is deliberately small:

1. the allowlisted `artifacts/data_pipeline/phase2/summary.json`;
2. the signed private audit decisions and private diagnostic tables under `PAGRPO_DATA_ROOT`;
3. the frozen Phase 1 baseline SHA, approved Phase 2 launcher SHA, and immutable model/split/tokenizer revisions;
4. the Functional and Statistical Gate outputs;
5. the private resource report containing actual GPU-hours, peak VRAM, disk/cache use, hourly price, and billed amount;
6. a short research-owner note describing observed error types and pair quality.

No additional manifest layer, audit framework, gate implementation, dashboard, or repeated E2E suite is permitted unless a concrete failure proves the existing system insufficient and a separate design change is approved.

## Completion Criteria

Phase 2 real smoke is complete only when:

- P1 and P2 were separately approved before their actions began;
- the approved launcher commit is a descendant of `main@34ef6f4`, and all protected Phase 1 paths are byte-identical to that baseline;
- all three pinned models ran under one reviewed recipe;
- all 240 planned candidate identities have successful records;
- verification and human audit are complete and lineage-consistent;
- pair/permutation construction, gates, and trainer round trip finish against the same immutable candidate pool;
- resource use remains within the approved cap;
- the sanitized public summary passes the existing isolation allowlist;
- an independent review confirms the final gate result.

Completion of this plan may authorize only a proposal for the 200-pair pilot. It never authorizes pilot execution or PA-GRPO training by itself.
