# Controlled Training Data Pipeline Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify the Phase 1 controlled training-data production system with real MATH/ReClor acquisition and deterministic splits, fake generation, strict verification, deterministic pairs/permutations, audit/gates, and no real vLLM inference.

**Architecture:** Implement a new `permstudy.data_pipeline` package whose stages exchange frozen dataclasses and immutable hash-pinned manifests. All full-text production artifacts live below an external `PAGRPO_DATA_ROOT`; Git contains only code, explicitly synthetic fixtures, sanitized summaries, hashes, and test evidence. CLI wrappers under `scripts_permstudy/data/` call the package layer and never bypass lineage or isolation validation.

**Tech Stack:** Python 3.12, standard-library dataclasses/enum/hashlib/json/pathlib/multiprocessing, NumPy 1.26, Hugging Face `datasets`/`huggingface-hub`, Transformers tokenizer APIs, `math-verify[antlr4_9_3]==0.9.0`, psutil 7.x, pytest 8.4.

**Spec:** `docs/superpowers/specs/2026-10-06-controlled-training-data-pipeline-design.md`

## Global Constraints

- Phase 1 may emit only `DATA_PIPELINE_SYSTEM_READY`; it must never emit `REAL_SMOKE_PASS`, `FUNCTIONAL_GATE_PASS`, or `STATISTICAL_GATE_PASS` for real generators.
- Do not start real vLLM, download generator weights, create the real 240-candidate smoke, or expand to 200 pairs.
- Python version is 3.12 for Windows/WSL Phase 1 and Linux Phase 2.
- `math-verify[antlr4_9_3]` is pinned exactly to `0.9.0`; runtime code checks the installed version before MATH verification.
- MATH uses every `EleutherAI/hendrycks_math` train config at a resolved immutable dataset SHA.
- ReClor uses official `train.json` only; official val/test never enter the internal split.
- `split_seed=42`; all split/sampling units are `original_question_id`.
- MATH split fallback is whole-source and deterministic: try `category+level`, then `category`, then `source`; never mix fallback levels within one split run.
- The real-smoke recipe remains 20 MATH + 20 ReClor questions and two samples from each of three pinned generators.
- Candidate records are keyed by `(generation_run_id, candidate_id)` everywhere.
- Raw questions, gold, candidate responses, verified pools, pairs, permutations, and private manifests remain outside Git.
- Production data/cache roots are rejected if either resolved root contains the other or any effective Hugging Face cache is inside the repository.
- Explicitly synthetic fixtures are the only committed full-text question/response/gold exception.
- Every task follows red-green-refactor: add a focused failing test, observe the intended failure, write the smallest implementation, run focused tests, run the Phase 1 suite, then commit.
- Activate the repository's Python 3.12 Phase 1 environment before running Python module commands; never commit the environment's machine-specific absolute path.

## Approved External Data Layout

All paths below are relative to the resolved `PAGRPO_DATA_ROOT`; manifests store relative paths only.

```text
sources/{source}/{snapshot_id}/questions.jsonl
sources/{source}/{snapshot_id}/manifest.json
splits/{split_run_id}/assignments.jsonl
splits/{split_run_id}/smoke_questions.jsonl
splits/{split_run_id}/manifest.json
generation/{generation_run_id}/{generator_id}/{shard_id}/candidates.jsonl
generation/{generation_run_id}/{generator_id}/{shard_id}/failures.jsonl
generation/{generation_run_id}/{generator_id}/{shard_id}/manifest.json
verification/{verification_run_id}/records.jsonl
verification/{verification_run_id}/manifest.json
audit/{audit_run_id}/selection.jsonl
audit/{audit_run_id}/decisions.jsonl
audit/{audit_run_id}/manifest.json
pairs/{pair_run_id}/pairs.jsonl
pairs/{pair_run_id}/manifest.json
permutations/{permutation_run_id}/permutations.jsonl
permutations/{permutation_run_id}/manifest.json
gates/{gate_run_id}/report.json
private_logs/
```

## File and Responsibility Map

| File | Responsibility |
|---|---|
| `permstudy/data_pipeline/schema.py` | Enums, frozen records, explicit validation, stable dict conversion |
| `permstudy/data_pipeline/canonical.py` | NFC/newline normalization, canonical JSON/JSONL, SHA256 |
| `permstudy/data_pipeline/ids.py` | Question/candidate/run/pair IDs and config hashes |
| `permstudy/data_pipeline/isolation.py` | Resolved-root checks, HF cache checks, redaction, public allowlists |
| `permstudy/data_pipeline/io.py` | Atomic manifests, append-only JSONL, recovery, record hashes, locks |
| `permstudy/data_pipeline/sources/reclor.py` | Official train loading, hashing, validation, source-local dedup |
| `permstudy/data_pipeline/sources/math.py` | Immutable HF revision resolution, all-config loading, dedup |
| `permstudy/data_pipeline/splitting.py` | Deterministic 90/10 split, whole-source fallback, smoke selection |
| `permstudy/data_pipeline/generation/base.py` | Backend protocol and generation config/result types |
| `permstudy/data_pipeline/generation/fake.py` | Deterministic fixture-backed backend |
| `permstudy/data_pipeline/generation/vllm.py` | Lazy-import adapter contract; no Phase 1 real invocation |
| `permstudy/data_pipeline/generation/runner.py` | Planning, sharding, locks, append, failure history, resume |
| `permstudy/data_pipeline/verification/reclor.py` | Official-label gold and strict terminal marker parser |
| `permstudy/data_pipeline/verification/math.py` | Strict boxed extraction and process-isolated math-verify worker |
| `permstudy/data_pipeline/pairs.py` | Run-scoped dedup groups, fixed-tokenizer lengths, deterministic pair |
| `permstudy/data_pipeline/permutations.py` | Exact AB/BA records and lineage |
| `permstudy/data_pipeline/audit.py` | Deterministic audit selection/decision validation and reason codes |
| `permstudy/data_pipeline/gates.py` | Functional/Statistical metrics, thresholds, diagnostics, status |
| `scripts_permstudy/data/*.py` | Thin CLI argument parsing and package-function dispatch |
| `tests_permstudy/data_pipeline/` | Focused unit, CLI, cross-platform, and end-to-end tests |
| `tests_permstudy/fixtures/data_pipeline/` | Explicitly synthetic source and response fixtures |
| `artifacts/data_pipeline/phase1/` | Sanitized Phase 1 evidence only |

## Exact Record and Result Contracts

The implementation may add private helper methods, but the following public fields and names are fixed across tasks.

```python
@dataclass(frozen=True)
class ArtifactRef:
    relative_path: str
    sha256: str
    record_count: int


@dataclass(frozen=True)
class RunManifest:
    schema_version: str
    stage: str
    run_id: str
    config_hash: str
    upstream_manifest_hashes: Sequence[str]
    artifacts: Sequence[ArtifactRef]
    counts: Mapping[str, int]
    created_at_utc: str
    output_manifest_hash: str


@dataclass(frozen=True)
class QuestionRecord:
    schema_version: str
    source: Source
    original_question_id: str
    question_content_hash: str
    source_snapshot_id: str
    source_revision: str
    source_row_id: str
    context: str | None
    question: str | None
    answers: Sequence[str]
    gold_label: str | None
    problem: str | None
    solution: str | None
    category: str | None
    level: str | None
    synthetic: bool


@dataclass(frozen=True)
class SplitAssignment:
    original_question_id: str
    question_content_hash: str
    source: Source
    split: Split
    stratum: str
```

`created_at_utc` and `output_manifest_hash` are retained in the private envelope but excluded from the payload used to calculate `output_manifest_hash`, avoiding a self-reference while preserving the stored digest.

`QuestionRecord.validate()` enforces exactly one source shape: ReClor requires context, question, four answers, and canonical gold label; MATH requires problem, solution, category, and level.

```python
@dataclass(frozen=True)
class CandidateRecord:
    plan: CandidatePlan
    response: str
    finish_reason: str
    generated_token_count: int
    model_revision: str


@dataclass(frozen=True)
class FailureRecord:
    plan: CandidatePlan
    error_type: str
    retry_count: int


@dataclass(frozen=True)
class VerificationRecord:
    generation_run_id: str
    verification_run_id: str
    candidate_id: str
    original_question_id: str
    source: Source
    generator_id: str
    verification_status: VerificationStatus
    gold_parse_status: str
    prediction_parse_status: str
    canonical_gold: str | None
    canonical_prediction: str | None
    verifier_name: str
    verifier_version: str
    parser_version: str
    verifier_timeout_seconds: float
    error_type: str | None


@dataclass(frozen=True)
class PairRecord:
    schema_version: str
    pair_run_id: str
    pair_id: str
    generation_run_id: str
    verification_run_id: str
    original_question_id: str
    question_content_hash: str
    source: Source
    split: Split
    split_manifest_hash: str
    positive_candidate_id: str
    negative_candidate_id: str
    positive_response_hash: str
    negative_response_hash: str
    positive_generator_id: str
    negative_generator_id: str
    positive_token_count: int
    negative_token_count: int
    tokenizer_repo: str
    tokenizer_revision: str


@dataclass(frozen=True)
class PermutationRecord:
    permutation_run_id: str
    pair_id: str
    generation_run_id: str
    original_question_id: str
    split: Split
    split_manifest_hash: str
    permutation_id: int
    permutation_label: str
    surface_a_candidate_id: str
    surface_b_candidate_id: str
    correct_surface: str


@dataclass(frozen=True)
class AuditSelectionRecord:
    audit_run_id: str
    generation_run_id: str
    verification_run_id: str
    candidate_id: str
    source: Source
    generator_id: str
    verification_status: VerificationStatus
    reason_code: str


@dataclass(frozen=True)
class AuditDecision:
    audit_run_id: str
    generation_run_id: str
    verification_run_id: str
    candidate_id: str
    verdict: AuditVerdict
    reason_code: str
    confirmed_disagree: bool
```

`record_hash` is added and validated by the I/O layer over every serialized record; it is not an input to the frozen business record constructors.

Module-local immutable result types are also fixed:

- `IsolationReport(repo_root_hash, data_root_hash, effective_cache_kinds)` contains hashes/kinds only, never raw paths.
- `JsonlScan(records, quarantined_tail_path, file_sha256)` returns validated dictionaries and a private path object.
- `SourceSnapshot(source, source_revision, source_snapshot_id, questions, private_manifest)` carries `QuestionRecord` values.
- `SplitBuildResult(assignments, stratification_level_by_source, fallback_reasons, split_manifest_hash)`.
- `GenerationConfig` contains backend, generator/model IDs and revisions, prompt revision/hash, sampling values, batch size, tensor parallel size, max model length, GPU memory utilization, samples per question, and split manifest hash.
- `GenerationResult(plan, response, finish_reason, generated_token_count, error_type)`.
- `GenerationPlan(generation_run_id, config_hash, candidates, shard_ids)` and `ShardRunSummary(planned, successful, historical_failures, missing, manifest_hash)`.
- `ParsedAnswer(status, answer, error_type)` and `BoxedAnswer(status, boxed_text, error_type)`.
- `ResponseGroup(response_hash, representative_key, member_keys, member_generators, verification_status)`.
- `AuditSummary(required_count, completed_count, verdict_counts, confirmed_disagreements, systematic_issue)`.
- `GateInputs(plans, candidates, verifications, pairs, audit_summary)` plus `FunctionalGateReport(passed, failures, metrics)` and `StatisticalGateReport(status, hard_failures, warnings, diagnostics)`.
- `FakeE2ESummary(run_ids, manifest_hashes, counts, phase_status, real_generation_performed)`.

---

### Task 1: Dependency Contract and Core Schemas

**Files:**
- Create: `permstudy/data_pipeline/__init__.py`
- Create: `permstudy/data_pipeline/schema.py`
- Create: `tests_permstudy/data_pipeline/__init__.py`
- Create: `tests_permstudy/data_pipeline/test_schema.py`
- Modify: `requirements.txt`
- Modify: `requirements-lock.txt`
- Modify: `requirements-windows-smoke.txt`
- Modify: `setup.py`

**Interfaces:**
- Produces: `Source`, `Split`, `VerificationStatus`, `AuditVerdict`, `GateStatus`, `ArtifactRef`, `RunManifest`, `QuestionRecord`, `SplitAssignment`, `CandidatePlan`, `CandidateRecord`, `FailureRecord`, `VerificationRecord`, `PairRecord`, `PermutationRecord`, `AuditSelectionRecord`, `AuditDecision`.
- Produces: every record's `validate() -> None`, `to_dict() -> dict[str, object]`, and matching `from_dict()` classmethod.

- [ ] **Step 1: Write failing schema and dependency tests**

```python
from dataclasses import FrozenInstanceError
from importlib.metadata import version

import pytest

from permstudy.data_pipeline.schema import CandidatePlan, Source, Split


def test_candidate_plan_is_frozen_and_uses_composite_key():
    plan = CandidatePlan(
        schema_version="candidate_plan_v1",
        generation_run_id="gen_abc",
        candidate_id="cand_def",
        original_question_id="reclor:train:synthetic_1",
        question_content_hash="a" * 64,
        source=Source.RECLOR,
        split=Split.TRAIN,
        split_manifest_hash="b" * 64,
        generator_id="qwen_7b",
        sampling_index=0,
        shard_id="00000",
        prompt_hash="c" * 64,
    )
    assert plan.key == ("gen_abc", "cand_def")
    with pytest.raises(FrozenInstanceError):
        plan.candidate_id = "changed"


def test_math_verify_is_exactly_pinned():
    assert version("math-verify") == "0.9.0"
```

- [ ] **Step 2: Run tests and confirm the intended failures**

Run:

```powershell
python -m pytest tests_permstudy/data_pipeline/test_schema.py -v
```

Expected: collection fails because `permstudy.data_pipeline.schema` does not exist; after the package exists but before dependency installation, the version assertion reports the non-0.9.0 version.

- [ ] **Step 3: Pin dependencies and install the Windows Phase 1 set**

Use exact declarations:

```text
requirements.txt: math-verify[antlr4_9_3]==0.9.0 and psutil>=7.1,<8
requirements-lock.txt: math-verify==0.9.0 and psutil==7.1.3
requirements-windows-smoke.txt: math-verify[antlr4_9_3]==0.9.0, datasets==4.4.1, huggingface-hub==0.36.0, psutil==7.1.3
setup.py MATH_REQUIRES: math-verify[antlr4_9_3]==0.9.0
```

Run:

```powershell
python -m pip install "math-verify[antlr4_9_3]==0.9.0" "datasets==4.4.1" "huggingface-hub==0.36.0" "psutil==7.1.3"
python -m pip check
```

Expected: both commands exit 0 and `python -c "import importlib.metadata as m; print(m.version('math-verify'))"` prints `0.9.0`.

- [ ] **Step 4: Implement frozen enums and dataclasses**

Use string enums and explicit validation, including this composite-key property:

```python
class Source(str, Enum):
    MATH = "math"
    RECLOR = "reclor"


class VerificationStatus(str, Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    INVALID = "invalid"
    AMBIGUOUS = "ambiguous"
    ERROR = "error"


@dataclass(frozen=True)
class CandidatePlan:
    schema_version: str
    generation_run_id: str
    candidate_id: str
    original_question_id: str
    question_content_hash: str
    source: Source
    split: Split
    split_manifest_hash: str
    generator_id: str
    sampling_index: int
    shard_id: str
    prompt_hash: str

    @property
    def key(self) -> tuple[str, str]:
        return self.generation_run_id, self.candidate_id
```

Validators must reject empty IDs, non-64-hex hashes, negative sample indices, illegal enum values, wrong ReClor answer count, and source-incompatible question fields. `RunManifest.artifacts` is a tuple of `ArtifactRef`; artifact paths must be relative POSIX paths.

- [ ] **Step 5: Run schema tests and the existing suite**

Run:

```powershell
python -m pytest tests_permstudy/data_pipeline/test_schema.py -v
python -m pytest tests_permstudy -q
```

Expected: focused tests pass and the pre-existing 59-pass baseline does not regress.

- [ ] **Step 6: Commit**

```bash
git add permstudy/data_pipeline tests_permstudy/data_pipeline requirements.txt requirements-lock.txt requirements-windows-smoke.txt setup.py
git commit -m "feat(data): add pipeline schemas and dependency contract"
```

---

### Task 2: Canonical Serialization and Deterministic IDs

**Files:**
- Create: `permstudy/data_pipeline/canonical.py`
- Create: `permstudy/data_pipeline/ids.py`
- Create: `tests_permstudy/data_pipeline/test_canonical_ids.py`

**Interfaces:**
- Produces: `normalize_text_v1(value: str) -> str`.
- Produces: `canonical_json_bytes(value: object) -> bytes`, `canonical_jsonl_bytes(records: Iterable[Mapping[str, object]], sort_key: str) -> bytes`, `sha256_hex(data: bytes) -> str`.
- Produces: `reclor_content_hash(context: str, question: str, answers: Sequence[str]) -> str`, `math_question_identity(problem: str) -> tuple[str, str]`, `candidate_id(question_id: str, generator_id: str, sampling_index: int) -> str`, `run_id(stage: str, config: Mapping[str, object]) -> str`, and `pair_id(generation_run_id: str, original_question_id: str, response_pos_id: str, response_neg_id: str, pair_schema_version: str) -> str`.

- [ ] **Step 1: Write failing canonicalization and boundary tests**

```python
from permstudy.data_pipeline.ids import reclor_content_hash


def test_reclor_hash_preserves_field_boundaries():
    first = reclor_content_hash("ab", "c", ("d", "e", "f", "g"))
    second = reclor_content_hash("a", "bc", ("d", "e", "f", "g"))
    assert first != second


def test_normalization_v1_preserves_internal_whitespace():
    from permstudy.data_pipeline.canonical import normalize_text_v1

    assert normalize_text_v1("  A\r\n  B  ") == "A\n  B"
```

Also assert canonical JSON is UTF-8, sorted-key, compact, no trailing newline; canonical JSONL uses LF and sorts by the requested key; MATH IDs contain 40 hash hex characters while retaining the 64-hex content hash; generation config changes alter `generation_run_id` but not `candidate_id`; pair IDs change with generation run IDs.

- [ ] **Step 2: Run focused tests and observe missing-module failures**

Run: `python -m pytest tests_permstudy/data_pipeline/test_canonical_ids.py -v`

Expected: FAIL because canonical and ID functions do not exist.

- [ ] **Step 3: Implement canonical bytes and IDs**

Use these exact serialization settings:

```python
def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
```

Use length-unambiguous canonical objects for all IDs. `candidate_id` hashes `{"generator_id": generator_id, "original_question_id": question_id, "sampling_index": sampling_index, "schema": "candidate_id_v1"}`. `pair_id` hashes generation run ID, question ID, selected candidate IDs, and `pair_schema_version`.

- [ ] **Step 4: Verify focused and full tests**

Run:

```powershell
python -m pytest tests_permstudy/data_pipeline/test_canonical_ids.py -v
python -m pytest tests_permstudy -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add permstudy/data_pipeline/canonical.py permstudy/data_pipeline/ids.py tests_permstudy/data_pipeline/test_canonical_ids.py
git commit -m "feat(data): add canonical hashes and run-scoped identifiers"
```

---

### Task 3: Runtime Isolation, Cache Validation, and Redaction

**Files:**
- Create: `permstudy/data_pipeline/isolation.py`
- Create: `tests_permstudy/data_pipeline/test_isolation.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `resolve_path(path: Path) -> Path`, `validate_external_roots(repo_root, data_root, env) -> IsolationReport`.
- Produces: `sanitize_exception(error: BaseException) -> str`, `safe_log_fields(fields: Mapping[str, object]) -> dict[str, object]`, `sanitize_public_manifest(payload) -> dict[str, object]`.

- [ ] **Step 1: Write failing containment and redaction tests**

```python
def test_data_root_inside_repo_is_rejected(tmp_path):
    from permstudy.data_pipeline.isolation import IsolationError, validate_external_roots

    repo = tmp_path / "repo"
    repo.mkdir()
    with pytest.raises(IsolationError, match="must not contain one another"):
        validate_external_roots(repo, repo / "private", {})


def test_public_log_rejects_response_and_absolute_path():
    from permstudy.data_pipeline.isolation import SafeLoggingError, safe_log_fields

    with pytest.raises(SafeLoggingError):
        safe_log_fields({"candidate_id": "cand", "response": "synthetic secret"})
```

Add Windows normcase tests, `..` resolution, repo-inside-data-root rejection, symlink/junction tests where the OS permits them, default/effective HF cache checks, HF token redaction, third-party exception sanitization, and public-field allowlist tests.

- [ ] **Step 2: Run tests and observe failures**

Run: `python -m pytest tests_permstudy/data_pipeline/test_isolation.py -v`

Expected: FAIL because `isolation.py` is absent.

- [ ] **Step 3: Implement resolved containment and allowlists**

Normalize with `Path.resolve(strict=False)`, `os.path.normcase`, and `os.path.commonpath`. Reject either root containing the other. Resolve effective caches from explicit environment variables or the Hugging Face defaults before comparing.

Use exact public log keys:

```python
PUBLIC_LOG_KEYS = frozenset({
    "candidate_id", "original_question_id", "generator_id", "shard_id",
    "status", "error_type", "retry_count", "count", "elapsed_seconds",
})
```

Use exact sanitized public-manifest keys:

```python
PUBLIC_MANIFEST_KEYS = frozenset({
    "run_id", "source", "source_revision", "split_manifest_hash",
    "question_count", "candidate_count", "pair_count", "generator_model",
    "generator_revision", "status_counts", "aggregate_statistics",
    "artifact_sha256", "real_generation_performed", "phase_status",
})
```

Map exceptions to stable types such as `timeout`, `oom`, `dependency_error`, `access_denied`, and `unexpected_error`; never return `str(error)` to the public logger.

- [ ] **Step 4: Extend `.gitignore` for private production shapes**

Add repository-local defense-in-depth patterns without naming a user path:

```gitignore
# Controlled data pipeline production artifacts must remain external
data_permstudy/
private_data/
private_logs/
*.private-manifest.json
```

- [ ] **Step 5: Run focused/full tests and commit**

Run `python -m pytest tests_permstudy/data_pipeline/test_isolation.py -v` then `python -m pytest tests_permstudy -q`.

```bash
git add .gitignore permstudy/data_pipeline/isolation.py tests_permstudy/data_pipeline/test_isolation.py
git commit -m "feat(data): enforce external storage and safe logging"
```

---

### Task 4: Immutable Manifests, Append-Only JSONL, Recovery, and Locks

**Files:**
- Create: `permstudy/data_pipeline/io.py`
- Create: `tests_permstudy/data_pipeline/test_io.py`

**Interfaces:**
- Produces: `write_atomic_manifest(path, payload) -> str` returning SHA256.
- Produces: `append_record(path, payload) -> str` returning `record_hash`.
- Produces: `scan_jsonl(path, recover_incomplete_tail=False) -> JsonlScan`.
- Produces: `ShardLock.acquire(lock_path, metadata, recover_stale=False) -> ShardLock`.
- Produces: `verify_artifact_ref(data_root, ArtifactRef) -> None`.

- [ ] **Step 1: Write crash and corruption tests**

```python
def test_incomplete_tail_is_quarantined_but_middle_corruption_fails(tmp_path):
    from permstudy.data_pipeline.io import IntegrityError, scan_jsonl

    path = tmp_path / "records.jsonl"
    path.write_bytes(b'{"id":"a","record_hash":"x"}\n{"id":')
    scan = scan_jsonl(path, recover_incomplete_tail=True)
    assert scan.quarantined_tail_path is not None

    path.write_bytes(b'{"id":\n{"id":"b"}\n')
    with pytest.raises(IntegrityError, match="middle line"):
        scan_jsonl(path, recover_incomplete_tail=True)
```

Also test temp-file + fsync + replace ordering via monkeypatch, stable record hashes, hash mismatch rejection, exclusive lock collision, lock metadata fields, refusal to auto-delete stale locks, explicit recovery only after `psutil.pid_exists(pid)` is false, and relative artifact paths.

- [ ] **Step 2: Run focused tests and observe missing functions**

Run: `python -m pytest tests_permstudy/data_pipeline/test_io.py -v`

Expected: FAIL during import.

- [ ] **Step 3: Implement durable I/O**

`append_record` computes the record hash over the payload without `record_hash`, writes one compact canonical JSON object plus `\n`, flushes, and calls `os.fsync`. `scan_jsonl` validates each record hash. Tail quarantine renames only the incomplete byte suffix to `filename.tail-corrupt`; it never rewrites valid lines.

Acquire locks atomically with `os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)`. Recovery requires `recover_stale=True`, matching run/shard metadata, and a dead PID confirmed by psutil.

The lock JSON contains exactly `run_id`, `shard_id`, `host`, `pid`, and `created_at`; recovery rejects missing identity fields and never removes a live-process lock.

- [ ] **Step 4: Run focused/full tests and commit**

```bash
python -m pytest tests_permstudy/data_pipeline/test_io.py -v
python -m pytest tests_permstudy -q
git add permstudy/data_pipeline/io.py tests_permstudy/data_pipeline/test_io.py
git commit -m "feat(data): add crash-safe manifests and shard locks"
```

---

### Task 5: ReClor Acquisition and Source-Local Deduplication

**Files:**
- Create: `permstudy/data_pipeline/sources/__init__.py`
- Create: `permstudy/data_pipeline/sources/reclor.py`
- Create: `tests_permstudy/data_pipeline/test_source_reclor.py`
- Create: `tests_permstudy/fixtures/data_pipeline/reclor/train.json`

**Interfaces:**
- Produces: `load_reclor_train(source_path: Path, data_root: Path) -> SourceSnapshot`.
- Produces: `canonical_reclor_label(label: int | str) -> str`.
- Consumes: canonical/ID/isolation/I/O APIs from Tasks 2–4.

- [ ] **Step 1: Add an explicitly synthetic fixture and failing tests**

Fixture records include `"synthetic": true`, unique `id_string`, four answers, integer labels, and repeated content cases. Tests assert val/test are ignored, labels map `0..3 -> A..D`, structured hashes preserve answer order, same-content consistent metadata keeps the smallest provenance ID, conflicting labels fail, the private manifest records `license_scope="non_commercial_research"`, and no source text appears in a sanitized manifest.

```python
def test_reclor_label_comes_from_official_field():
    snapshot = load_reclor_train(FIXTURE_DIR, tmp_path / "external")
    assert {record.gold_label for record in snapshot.questions} == {"A", "B", "C", "D"}
```

- [ ] **Step 2: Run focused tests and observe import failure**

Run: `python -m pytest tests_permstudy/data_pipeline/test_source_reclor.py -v`

- [ ] **Step 3: Implement directory and ZIP acquisition**

Accept an extracted directory containing `train.json` or a `.zip` archive only after the CLI supplies `--acknowledge-reclor-noncommercial`. For ZIP input, reject absolute paths and `..` members, extract only below an external staging directory, and record the archive SHA256. For directory input, record `train.json` SHA256. Validate required fields and exactly four ordered answers; never read val/test into the snapshot.

- [ ] **Step 4: Verify outputs remain external and deterministic**

Run the focused test twice and assert identical question records and manifest payload hashes after removing the non-hashed processing timestamp.

- [ ] **Step 5: Run full suite and commit**

```bash
python -m pytest tests_permstudy/data_pipeline/test_source_reclor.py -v
python -m pytest tests_permstudy -q
git add permstudy/data_pipeline/sources tests_permstudy/data_pipeline/test_source_reclor.py tests_permstudy/fixtures/data_pipeline/reclor/train.json
git commit -m "feat(data): acquire and validate ReClor train data"
```

---

### Task 6: MATH Immutable Acquisition and Deduplication

**Files:**
- Create: `permstudy/data_pipeline/sources/math.py`
- Create: `tests_permstudy/data_pipeline/test_source_math.py`
- Create: `tests_permstudy/fixtures/data_pipeline/math_rows.jsonl`

**Interfaces:**
- Produces: `resolve_math_revision(repo_id, requested_revision, api) -> str`.
- Produces: `load_math_train(repo_id, revision, cache_dir, dataset_loader) -> SourceSnapshot`.
- The loader injection point has signature `(repo_id: str, config: str, revision: str, cache_dir: Path) -> Iterable[Mapping[str, object]]`.

- [ ] **Step 1: Write failing immutable-revision and all-config tests**

Tests use a fake `HfApi` and fake dataset loader, assert `main` is resolved to a 40-hex commit SHA before loading, config names are sorted, every config's train rows are consumed, category comes from config, level/solution conflicts fail, IDs use 40 hash hex characters, and manifests store the full 64-hex question hash.

```python
def test_math_loader_pins_revision_before_reading(monkeypatch, tmp_path):
    calls = []
    snapshot = load_math_train(
        repo_id="EleutherAI/hendrycks_math",
        revision="1" * 40,
        cache_dir=tmp_path / "hf",
        dataset_loader=lambda repo, config, revision, cache: calls.append(revision) or SYNTHETIC_ROWS,
    )
    assert calls and set(calls) == {"1" * 40}
    assert snapshot.source_revision == "1" * 40
```

- [ ] **Step 2: Run focused tests and observe failures**

Run: `python -m pytest tests_permstudy/data_pipeline/test_source_math.py -v`

- [ ] **Step 3: Implement official HF acquisition**

Use `HfApi().dataset_info(repo_id, revision=requested_revision).sha`, `get_dataset_config_names(repo_id, revision=resolved_sha)`, and `load_dataset(repo_id, config, split="train", revision=resolved_sha, cache_dir=cache_dir)`. Require the repo ID exactly `EleutherAI/hendrycks_math`, sort configs and rows deterministically, and record config/category, level, problem, solution, and upstream row position privately.

- [ ] **Step 4: Run focused/full tests and commit**

```bash
python -m pytest tests_permstudy/data_pipeline/test_source_math.py -v
python -m pytest tests_permstudy -q
git add permstudy/data_pipeline/sources/math.py tests_permstudy/data_pipeline/test_source_math.py tests_permstudy/fixtures/data_pipeline/math_rows.jsonl
git commit -m "feat(data): acquire immutable MATH train snapshot"
```

---

### Task 7: Deterministic Split, Fallback, and Smoke Selection

**Files:**
- Create: `permstudy/data_pipeline/splitting.py`
- Create: `tests_permstudy/data_pipeline/test_splitting.py`

**Interfaces:**
- Produces: `build_internal_split(questions, split_seed=42) -> SplitBuildResult`.
- Produces: `select_smoke_questions(assignments, per_source=20, split_seed=42) -> list[SplitAssignment]`.

- [ ] **Step 1: Write failing determinism and leakage tests**

Tests assert input reorder does not alter assignments/hash; ReClor uses gold label; MATH chooses `category+level` when feasible, falls back the entire source to `category` when any primary stratum is infeasible, then to `source` when category is infeasible; fallback reasons are recorded; every question occurs once; the exact 90/10 count is stable; smoke uses train only and returns 20 per source; candidate/pair/permutation split mismatches are rejected.

- [ ] **Step 2: Run focused tests and observe failures**

Run: `python -m pytest tests_permstudy/data_pipeline/test_splitting.py -v`

- [ ] **Step 3: Implement the exact whole-source fallback algorithm**

For `N` records, set holdout count to `min(N-1, max(1, (N + 5) // 10))`. A stratification level is feasible only if every group has at least two records and `group_count <= holdout_count <= N - group_count`. Choose the first feasible level from the approved fallback chain.

Initialize every group's holdout quota to one. Allocate remaining holdout slots one at a time to the eligible group with the smallest exact fraction `quota/group_size`, comparing fractions by integer cross multiplication and breaking ties by canonical group key. Inside each group, order questions by `SHA256("42\0" + original_question_id)` and assign the first quota to holdout.

Smoke selection uses a separate proportional allocator because 20 samples may be fewer than the number of strata. For each stratum compute `floor(group_size * 20 / source_train_size)`, then distribute remaining slots by descending integer remainder `(group_size * 20) % source_train_size`, canonical stratum key, and available capacity. Within each stratum choose the lowest `SHA256("42\0smoke\0" + original_question_id)` scores. Use the selected MATH split stratification level and ReClor gold labels; fail if a source has fewer than 20 train questions.

- [ ] **Step 4: Add a golden cross-platform hash test**

For the committed synthetic fixture, assert a literal expected `split_manifest_hash`. Run the same test on Windows and WSL; do not update the literal independently per platform.

- [ ] **Step 5: Run focused/full tests and commit**

```bash
python -m pytest tests_permstudy/data_pipeline/test_splitting.py -v
python -m pytest tests_permstudy -q
git add permstudy/data_pipeline/splitting.py tests_permstudy/data_pipeline/test_splitting.py
git commit -m "feat(data): add deterministic question-level splits"
```

---

### Task 8: Generation Planning, Fake Backend, and Lazy vLLM Contract

**Files:**
- Create: `permstudy/data_pipeline/generation/__init__.py`
- Create: `permstudy/data_pipeline/generation/base.py`
- Create: `permstudy/data_pipeline/generation/fake.py`
- Create: `permstudy/data_pipeline/generation/vllm.py`
- Create: `tests_permstudy/data_pipeline/test_generation_backends.py`

**Interfaces:**
- Produces: `GenerationBackend` protocol with `generate(batch: Sequence[CandidatePlan], config: GenerationConfig) -> list[GenerationResult]`.
- Produces: `plan_generation(smoke_assignments, generators, samples_per_question=2, shard_size=8) -> GenerationPlan`.
- Produces: `FakeGenerationBackend(response_map)` and `VLLMGenerationBackend(engine_factory=None)`.

- [ ] **Step 1: Write failing backend and planning tests**

Assert the 40-question recipe creates 240 unique composite keys; each logical candidate ID is stable across two generation runs; response return order does not change IDs; every permutation of input produces the same shard plan; fake backend can emit success, retryable failure, and `finish_reason=length`; importing `permstudy.data_pipeline.generation` succeeds after blocking `vllm` imports.

```python
def test_vllm_is_lazy(monkeypatch):
    real_import = importlib.import_module
    monkeypatch.setattr(importlib, "import_module", lambda name: (_ for _ in ()).throw(ImportError()) if name == "vllm" else real_import(name))
    import permstudy.data_pipeline.generation
```

- [ ] **Step 2: Run focused tests and observe failures**

Run: `python -m pytest tests_permstudy/data_pipeline/test_generation_backends.py -v`

- [ ] **Step 3: Implement backend protocol and deterministic planning**

Define the three generator aliases and repositories in a frozen config table. Require immutable revisions to be supplied; reject branch names. Assign shards after sorting by `(generator_id, original_question_id, sampling_index)`.

```python
GENERATOR_REPOSITORIES = {
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-32b-instruct": "Qwen/Qwen2.5-32B-Instruct",
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
}
```

`VLLMGenerationBackend.__init__` must not import vLLM. Its `generate` method imports with `importlib.import_module("vllm")`, constructs adapter requests through an injected engine factory, and maps only request ID, text, finish reason, and token count into `GenerationResult`. Tests use a fake vLLM module; Phase 1 CLI never selects the real backend.

- [ ] **Step 4: Run focused/full tests and commit**

```bash
python -m pytest tests_permstudy/data_pipeline/test_generation_backends.py -v
python -m pytest tests_permstudy -q
git add permstudy/data_pipeline/generation tests_permstudy/data_pipeline/test_generation_backends.py
git commit -m "feat(data): plan candidates and define generation backends"
```

---

### Task 9: Append-Only Generation Runner and Resume

**Files:**
- Create: `permstudy/data_pipeline/generation/runner.py`
- Create: `tests_permstudy/data_pipeline/test_generation_runner.py`

**Interfaces:**
- Produces: `run_generation_shard(plan, backend, output_dir, recover_stale_lock=False) -> ShardRunSummary`.
- Produces: `successful_candidate_keys(shard_dir) -> frozenset[tuple[str, str]]`.
- Consumes: Task 4 durable I/O and locks.

- [ ] **Step 1: Write failing interruption/resume tests**

Simulate a backend that succeeds twice, fails once, then raises process interruption. Assert successful lines remain, failure history is separate, resume calls only missing composite keys, retry success raises final completion to 100%, old failures remain auditable, and a second run with a changed config/run ID never treats old logical IDs as complete.

- [ ] **Step 2: Run focused tests and observe failures**

Run: `python -m pytest tests_permstudy/data_pipeline/test_generation_runner.py -v`

- [ ] **Step 3: Implement one-writer append/resume flow**

Order operations exactly: validate isolation/config/upstream hashes; acquire lock; scan/quarantine incomplete tail; calculate missing composite keys; call backend in deterministic batches; append each success or failure immediately; atomically update manifest counts; release lock in `finally`.

Map OOM to `error_type="oom"`. Compare the complete generation config hash before resume; any changed batch size, tensor parallel size, max model length, memory utilization, prompt hash, sampling config, model revision, or split hash raises `RunMismatchError`.

- [ ] **Step 4: Verify interrupted and uninterrupted hashes match**

Run the focused test that executes both routes and asserts equal successful composite-key sets and canonical output-manifest hashes.

- [ ] **Step 5: Run full suite and commit**

```bash
python -m pytest tests_permstudy/data_pipeline/test_generation_runner.py -v
python -m pytest tests_permstudy -q
git add permstudy/data_pipeline/generation/runner.py tests_permstudy/data_pipeline/test_generation_runner.py
git commit -m "feat(data): add append-only generation resume"
```

---

### Task 10: Strict ReClor Verification

**Files:**
- Create: `permstudy/data_pipeline/verification/__init__.py`
- Create: `permstudy/data_pipeline/verification/reclor.py`
- Create: `tests_permstudy/data_pipeline/test_verification_reclor.py`

**Interfaces:**
- Produces: `parse_reclor_candidate(response: str) -> ParsedAnswer`.
- Produces: `verify_reclor(question, candidate) -> VerificationRecord`.

- [ ] **Step 1: Write the terminal-marker truth table as failing tests**

```python
@pytest.mark.parametrize(
    ("response", "status", "answer"),
    [
        ("reason\nFinal Answer: A", "parsed", "A"),
        ("Final Answer: A\nextra", "invalid", None),
        ("Final Answer: A\nFinal Answer: A", "invalid", None),
        ("Final Answer: A\nFinal Answer: B", "ambiguous", None),
        ("answer A", "invalid", None),
        ("Final Answer: E", "invalid", None),
    ],
)
def test_reclor_terminal_contract(response, status, answer):
    parsed = parse_reclor_candidate(response)
    assert (parsed.status, parsed.answer) == (status, answer)
```

Also assert official integer/string labels canonicalize directly, candidate text never changes gold, exact match yields correct/incorrect, and composite keys are preserved.

- [ ] **Step 2: Run focused tests and observe failures**

Run: `python -m pytest tests_permstudy/data_pipeline/test_verification_reclor.py -v`

- [ ] **Step 3: Implement strict parsing and exact-match records**

Only whitespace may follow the unique marker. Count all case-sensitive `Final Answer:` markers; identical duplicates are invalid and conflicting valid answers are ambiguous. Store `verifier_name="reclor_exact_match"`, parser version, parse statuses, and no math-verify fields.

- [ ] **Step 4: Run focused/full tests and commit**

```bash
python -m pytest tests_permstudy/data_pipeline/test_verification_reclor.py -v
python -m pytest tests_permstudy -q
git add permstudy/data_pipeline/verification tests_permstudy/data_pipeline/test_verification_reclor.py
git commit -m "feat(data): add strict ReClor verification"
```

---

### Task 11: Strict MATH Verification with Cross-Platform Timeouts

**Files:**
- Create: `permstudy/data_pipeline/verification/math.py`
- Create: `tests_permstudy/data_pipeline/test_verification_math.py`

**Interfaces:**
- Produces: `extract_math_gold(solution: str) -> BoxedAnswer`.
- Produces: `extract_math_candidate(response: str) -> BoxedAnswer`.
- Produces: `verify_math_question(question, candidates, timeout_seconds) -> list[VerificationRecord]`.

API references are the official [Math-Verify README](https://github.com/huggingface/Math-Verify/blob/main/README.md), the published [0.9.0 package](https://pypi.org/project/math-verify/0.9.0/), and the documented [Windows timeout/pickling failure](https://github.com/huggingface/Math-Verify/issues/79). The parent-enforced spawn-process timeout is intentional and must not be replaced with the library's signal-based timeout on Windows.

- [ ] **Step 1: Write failing extraction/status/cache tests**

Test a terminal nested `\boxed{\frac{1}{2}}`, missing marker, malformed braces, duplicate same candidate markers -> invalid, conflicting markers -> ambiguous, gold parse failure -> question-level `gold_verification_error`, explicit equivalent/non-equivalent cases, child exception -> error, and one gold parse for six normal candidates.

- [ ] **Step 2: Write a Windows timeout regression test**

Inject a top-level worker function that blocks. Assert one candidate becomes `error` with `error_type="timeout"`, the child is terminated, and the next candidate is processed in a fresh worker. The test must pass under Windows `spawn` and Linux/WSL.

- [ ] **Step 3: Run focused tests and observe failures**

Run: `python -m pytest tests_permstudy/data_pipeline/test_verification_math.py -v`

- [ ] **Step 4: Implement strict extraction and a per-question worker process**

The parent starts a spawn-context worker for one question. The worker imports `parse`/`verify`, parses gold once with:

```python
gold = parse(
    boxed_gold,
    fallback_mode="no_fallback",
    extraction_mode="first_match",
    parsing_timeout=None,
    raise_on_error=True,
)
```

For each candidate it parses the strict extracted boxed region and calls:

```python
equivalent = verify(
    gold,
    prediction,
    strict=True,
    timeout_seconds=None,
    raise_on_error=True,
)
```

The parent enforces `timeout_seconds` with a Pipe poll. The child returns only strings, booleans, status codes, and sanitized error types—never SymPy objects—avoiding cross-process pickling. On timeout, terminate/join the worker, mark only that candidate as error, restart for remaining candidates, and record the worker restart. Empty parse results are ambiguous, not incorrect; only explicit `equivalent is False` after successful parses is incorrect.

- [ ] **Step 5: Add runtime version enforcement**

Before starting a worker, require `importlib.metadata.version("math-verify") == "0.9.0"`; otherwise raise `DependencyContractError` before writing verification output.

- [ ] **Step 6: Run focused/full tests on Windows and WSL**

Run:

```powershell
python -m pytest tests_permstudy/data_pipeline/test_verification_math.py -v
$repoWindows = git rev-parse --show-toplevel
$repoWsl = (wsl.exe -d Ubuntu-24.04 -- wslpath -a $repoWindows).Trim()
wsl.exe -d Ubuntu-24.04 -- bash -lc "cd '$repoWsl' && python3.12 -m pytest tests_permstudy/data_pipeline/test_verification_math.py -v"
python -m pytest tests_permstudy -q
```

Expected: all available platform runs pass; if WSL lacks the approved Python environment, record that environment failure without changing verifier code and complete WSL verification in the configured Phase 1 environment before acceptance.

- [ ] **Step 7: Commit**

```bash
git add permstudy/data_pipeline/verification/math.py tests_permstudy/data_pipeline/test_verification_math.py
git commit -m "feat(data): add strict process-isolated MATH verification"
```

---

### Task 12: Deterministic Response Deduplication and Pair Selection

**Files:**
- Create: `permstudy/data_pipeline/pairs.py`
- Create: `tests_permstudy/data_pipeline/test_pairs.py`

**Interfaces:**
- Produces: `build_dedup_groups(records) -> list[ResponseGroup]`.
- Produces: `select_pair(question, verified_candidates, tokenizer, tokenizer_revision) -> PairRecord | None`.

- [ ] **Step 1: Write failing filtering/dedup/tie tests**

Assert only correct/incorrect enter; `finish_reason=length`, empty, and missing final answers are excluded; CRLF/NFC duplicates group together; representative is the smallest composite key; groups retain all member IDs/generators; status conflict for one response hash fails; token count calls `encode(canonical_response, add_special_tokens=False)`; minimum absolute gap wins; exact gap ties use candidate IDs; pair ID changes across generation runs and is stable within one run.

- [ ] **Step 2: Run focused tests and observe failures**

Run: `python -m pytest tests_permstudy/data_pipeline/test_pairs.py -v`

- [ ] **Step 3: Implement deterministic selection**

Require tokenizer repository `Qwen/Qwen2.5-7B-Instruct` and a 40-hex immutable revision in the pair-run config. Sort candidate groups by `(response_hash, generation_run_id, candidate_id)`, enumerate correct × incorrect, and minimize `(abs(pos_tokens-neg_tokens), pos_candidate_id, neg_candidate_id)`.

- [ ] **Step 4: Add canonical pair-manifest golden hash test**

Serialize sorted by `original_question_id`, UTF-8, sorted keys, compact separators, LF, and no timestamps in the hashed payload. Assert a literal hash from the synthetic fixture.

- [ ] **Step 5: Run focused/full tests and commit**

```bash
python -m pytest tests_permstudy/data_pipeline/test_pairs.py -v
python -m pytest tests_permstudy -q
git add permstudy/data_pipeline/pairs.py tests_permstudy/data_pipeline/test_pairs.py
git commit -m "feat(data): select deterministic verified response pairs"
```

---

### Task 13: AB/BA Permutation Construction and Lineage

**Files:**
- Create: `permstudy/data_pipeline/permutations.py`
- Create: `tests_permstudy/data_pipeline/test_permutations.py`

**Interfaces:**
- Produces: `build_permutations(pair: PairRecord) -> tuple[PermutationRecord, PermutationRecord]`.

- [ ] **Step 1: Write failing exact-mapping tests**

Assert exactly two outputs; AB sets A=positive/B=negative/correct_surface=A; BA sets A=negative/B=positive/correct_surface=B; both inherit question/split/split hash/pair/generation lineage; cross-split or hash mismatch raises before output; output order and canonical hash are stable.

- [ ] **Step 2: Run focused tests and observe failures**

Run: `python -m pytest tests_permstudy/data_pipeline/test_permutations.py -v`

- [ ] **Step 3: Implement the two-record constructor**

Use integer permutation IDs compatible with existing trainer identity code (`0` for AB, `1` for BA) plus explicit string labels and semantic candidate IDs.

- [ ] **Step 4: Run focused/full tests and commit**

```bash
python -m pytest tests_permstudy/data_pipeline/test_permutations.py -v
python -m pytest tests_permstudy -q
git add permstudy/data_pipeline/permutations.py tests_permstudy/data_pipeline/test_permutations.py
git commit -m "feat(data): build lineage-safe AB BA permutations"
```

---

### Task 14: Human Audit Selection and Decision Validation

**Files:**
- Create: `permstudy/data_pipeline/audit.py`
- Create: `tests_permstudy/data_pipeline/test_audit.py`

**Interfaces:**
- Produces: `build_audit_selection(records, audit_seed=42, max_per_cell=5) -> list[AuditSelectionRecord]`.
- Produces: `validate_audit_decisions(selection, decisions) -> AuditSummary`.

- [ ] **Step 1: Write failing coverage and deterministic-sampling tests**

Assert all invalid/ambiguous/error/gold-error records are selected; correct/incorrect are grouped by source × generator × status; at most five per cell selected by SHA256 seed score; small cells are exhaustive; input order does not matter; decisions allow only AGREE/DISAGREE/UNSURE; missing/duplicate/foreign decision IDs fail; reason codes use the approved enum.

- [ ] **Step 2: Run focused tests and observe failures**

Run: `python -m pytest tests_permstudy/data_pipeline/test_audit.py -v`

- [ ] **Step 3: Implement selection and confirmed disagreement handling**

Selection records bind generation and verification run IDs. Decision validation distinguishes first-review disagreement from `confirmed_disagree=True`; only confirmed disagreement contributes a hard-fail signal. `UNSURE` remains a separate count and queue. The only reason codes are `missing_final_marker`, `duplicate_same_marker`, `conflicting_markers`, `math_parse_failure`, `gold_parse_failure`, `timeout`, `dependency_error`, `prompt_contract_mismatch`, and `other`.

- [ ] **Step 4: Run focused/full tests and commit**

```bash
python -m pytest tests_permstudy/data_pipeline/test_audit.py -v
python -m pytest tests_permstudy -q
git add permstudy/data_pipeline/audit.py tests_permstudy/data_pipeline/test_audit.py
git commit -m "feat(data): add deterministic verifier audit protocol"
```

---

### Task 15: Functional and Statistical Gates

**Files:**
- Create: `permstudy/data_pipeline/gates.py`
- Create: `tests_permstudy/data_pipeline/test_gates.py`

**Interfaces:**
- Produces: `evaluate_functional_gate(inputs: GateInputs) -> FunctionalGateReport`.
- Produces: `evaluate_statistical_gate(inputs: GateInputs) -> StatisticalGateReport`.
- Produces: `cramers_v(table: Sequence[Sequence[int]]) -> float | None` without SciPy.

- [ ] **Step 1: Write table-driven threshold tests**

Cover completion 99%/100%; pair yield 19%/20%/29%/30%; invalid+error 20%/just above 20%; pooled dominance 89%/90%; per-source dominance 70%/above 70%/90%; correct rate 0%/100% with denominator `correct+incorrect`; undefined zero denominator; confirmed disagreement; systematic prompt/parser issue; and ambiguous rate remaining diagnostic-only.

```python
def test_ambiguous_rate_never_changes_v1_gate_status():
    low = evaluate_statistical_gate(make_inputs(ambiguous=0))
    high = evaluate_statistical_gate(make_inputs(ambiguous=200))
    assert low.status == high.status
    assert high.diagnostics["ambiguous_rate"] > low.diagnostics["ambiguous_rate"]
```

- [ ] **Step 2: Run focused tests and observe failures**

Run: `python -m pytest tests_permstudy/data_pipeline/test_gates.py -v`

- [ ] **Step 3: Implement exact denominators and precedence**

Compute completion from planned versus successful composite keys; pair yield from selected-pair questions/planned source questions; invalid+error from successful generated candidates in each cell; dominance from selected representatives/selected pairs; correct rate from correct/(correct+incorrect). Apply status precedence `FAIL`, then `PASS_WITH_WARNINGS`, then `PASS`.

Hard failures are completion below 100%, any source pair yield below 20%, any generator/source invalid+error rate above 20%, pooled positive or negative generator share at least 90%, confirmed binary-label disagreement, or systematic prompt/parser/verifier issue. Warnings are source pair yield below 30%, pooled or per-source generator share above 70%, binary correct rate exactly 0% or 100%, and per-source share at least 90% as a strong warning. Ambiguous rate never changes status in v1.

Compute Cramer's V as `sqrt(chi2 / (n * min(rows-1, columns-1)))`, returning `None` for empty/degenerate tables.

- [ ] **Step 4: Run focused/full tests and commit**

```bash
python -m pytest tests_permstudy/data_pipeline/test_gates.py -v
python -m pytest tests_permstudy -q
git add permstudy/data_pipeline/gates.py tests_permstudy/data_pipeline/test_gates.py
git commit -m "feat(data): implement deterministic pipeline gates"
```

---

### Task 16: Thin Data-Pipeline CLIs

**Files:**
- Create: `scripts_permstudy/data/prepare_questions.py`
- Create: `scripts_permstudy/data/plan_generation.py`
- Create: `scripts_permstudy/data/generate_candidates.py`
- Create: `scripts_permstudy/data/verify_candidates.py`
- Create: `scripts_permstudy/data/build_reasoning_pairs.py`
- Create: `scripts_permstudy/data/build_permutations.py`
- Create: `scripts_permstudy/data/audit_generator_distribution.py`
- Create: `scripts_permstudy/data/validate_dataset.py`
- Create: `scripts_permstudy/data/run_fake_e2e.py`
- Create: `tests_permstudy/data_pipeline/test_cli.py`

**Interfaces:**
- Every script exposes `build_parser() -> argparse.ArgumentParser` and `main(argv: Sequence[str] | None = None) -> int`.
- CLIs accept relative manifest references below `PAGRPO_DATA_ROOT`; package functions perform all business logic.

The fixed CLI argument contract is:

| Script | Required inputs | Stage output |
|---|---|---|
| `prepare_questions.py` | `--data-root`, `--source`, ReClor dir/archive or `--math-revision`, `--split-seed 42`; ReClor also requires `--acknowledge-reclor-noncommercial` | source and split manifests |
| `plan_generation.py` | `--data-root`, `--split-manifest`, three `--generator-config` JSON files, `--samples-per-question 2`, `--shard-size` | generation plan manifest |
| `generate_candidates.py` | `--data-root`, `--generation-manifest`, `--generator-id`, `--shard-id`, `--backend fake`; optional `--recover-stale-lock` | candidate/failure shard |
| `verify_candidates.py` | `--data-root`, `--generation-manifest`, `--verifier-timeout-seconds` | verification run |
| `build_reasoning_pairs.py` | `--data-root`, `--verification-manifest`, `--tokenizer-revision` | pair run |
| `build_permutations.py` | `--data-root`, `--pair-manifest` | permutation run |
| `audit_generator_distribution.py` | `--data-root`, `--verification-manifest`, optional `--decisions`, `--audit-seed 42` | audit selection/summary |
| `validate_dataset.py` | `--data-root` plus one stage manifest; optional `--check-isolation` | sanitized validation report |
| `run_fake_e2e.py` | `--data-root`, `--fixture-root` or `--split-manifest` | fake E2E summary |

- [ ] **Step 1: Write failing parser/default/boundary tests**

Assert every CLI imports without vLLM; `prepare_questions` requires explicit data root and source input; split seed defaults to 42; generation defaults to fake; `--backend vllm` exits with `PhaseBoundaryError` in Phase 1 before importing vLLM; stale-lock recovery requires the explicit flag; no CLI prints absolute paths or payload text; exit codes are 0 success, 2 contract/config error, 3 integrity error, 4 dependency/access error.

- [ ] **Step 2: Run focused tests and observe missing-script failures**

Run: `python -m pytest tests_permstudy/data_pipeline/test_cli.py -v`

- [ ] **Step 3: Implement parsers and package dispatch**

Use repository-root injection consistent with current `scripts_permstudy` scripts. Each `main()` catches only pipeline-defined errors, logs sanitized IDs/counts, and returns the fixed code. Unexpected errors are sanitized publicly and written with traceback only when private logging is explicitly enabled below data root.

- [ ] **Step 4: Verify CLI help and no-vLLM import**

Run each script with `--help`, then run `python -c "import permstudy.data_pipeline.generation"` in the Windows environment without vLLM.

- [ ] **Step 5: Run focused/full tests and commit**

```bash
python -m pytest tests_permstudy/data_pipeline/test_cli.py -v
python -m pytest tests_permstudy -q
git add scripts_permstudy/data tests_permstudy/data_pipeline/test_cli.py
git commit -m "feat(data): add controlled pipeline command line interfaces"
```

---

### Task 17: Committed Synthetic Fake End-to-End

**Files:**
- Create: `tests_permstudy/fixtures/data_pipeline/synthetic_questions.jsonl`
- Create: `tests_permstudy/fixtures/data_pipeline/synthetic_fake_responses.json`
- Create: `tests_permstudy/data_pipeline/test_fake_e2e.py`
- Modify: `scripts_permstudy/data/run_fake_e2e.py`

**Interfaces:**
- Produces: `run_fake_e2e(data_root, fixture_root) -> FakeE2ESummary`.

- [ ] **Step 1: Create the 40-question explicitly synthetic fixture**

Provide 20 synthetic MATH and 20 synthetic ReClor questions, each marked `synthetic=true`. Provide six fake responses per question covering correct/incorrect, duplicate, invalid, ambiguous, error, and historical-failure-then-success behavior while guaranteeing at least one selected pair per source.

- [ ] **Step 2: Write a failing full-flow test**

The test runs source loading -> split/smoke -> 240 candidate planning -> interrupted fake generation -> resume -> verification -> audit selection with synthetic AGREE decisions -> pairs -> permutations -> gates. Assert 240 final successful composite keys, one pair maximum per question, two permutations per pair, stable manifest hashes across a clean rerun, and no production-only phase status.

- [ ] **Step 3: Run the test and observe missing orchestration behavior**

Run: `python -m pytest tests_permstudy/data_pipeline/test_fake_e2e.py -v`

- [ ] **Step 4: Implement the orchestration using only package APIs**

`run_fake_e2e.py` must not duplicate stage logic. It creates an external temporary root when `--data-root` is supplied by tests, writes private artifacts there, and prints only the sanitized summary.

- [ ] **Step 5: Verify Windows and WSL hashes**

Run the E2E in both environments and compare the emitted canonical manifest hashes byte-for-byte.

- [ ] **Step 6: Run full suite and commit**

```bash
python -m pytest tests_permstudy/data_pipeline/test_fake_e2e.py -v
python -m pytest tests_permstudy -q
git add tests_permstudy/fixtures/data_pipeline tests_permstudy/data_pipeline/test_fake_e2e.py scripts_permstudy/data/run_fake_e2e.py
git commit -m "test(data): add synthetic fake pipeline end to end"
```

---

### Task 18: Private Real-Source Phase 1 Run and Sanitized Acceptance Evidence

**Files:**
- Create: `artifacts/data_pipeline/phase1/README.md`
- Create: `artifacts/data_pipeline/phase1/environment.txt`
- Create: `artifacts/data_pipeline/phase1/pytest_output.txt`
- Create: `artifacts/data_pipeline/phase1/fake_e2e_summary.json`
- Create: `artifacts/data_pipeline/phase1/private_source_summary.json`
- Create: `artifacts/data_pipeline/phase1/git_data_scan.txt`
- Modify: `docs/superpowers/specs/2026-10-06-controlled-training-data-pipeline-design.md`

**Interfaces:**
- Consumes all Phase 1 CLIs and approved local environment variables.
- Produces only sanitized evidence and the terminal status `DATA_PIPELINE_SYSTEM_READY` after independent code review; before review the report status is `AWAITING_PHASE1_CODE_REVIEW`.

- [ ] **Step 1: Configure external roots without committing values**

Require runtime-only environment variables to have been set outside Git, then derive external caches without printing the paths:

```powershell
if (-not $env:PAGRPO_DATA_ROOT) { throw 'PAGRPO_DATA_ROOT is required' }
if (-not $env:PAGRPO_RECLOR_DIR -and -not $env:PAGRPO_RECLOR_ARCHIVE) { throw 'A ReClor source is required' }
$env:HF_HOME = Join-Path $env:PAGRPO_DATA_ROOT 'hf_home'
$env:HUGGINGFACE_HUB_CACHE = Join-Path $env:HF_HOME 'hub'
$env:TRANSFORMERS_CACHE = Join-Path $env:HF_HOME 'transformers'
```

Do not echo or persist the resolved values. Confirm isolation with `validate_dataset.py --check-isolation`.

- [ ] **Step 2: Acquire real ReClor/MATH privately and build the split**

Run `prepare_questions.py` with official ReClor train and `EleutherAI/hendrycks_math`. Resolve the MATH revision before loading. Confirm private manifests contain file/revision hashes, source counts, dedup counts, fallback level, `split_seed=42`, and split hash; confirm public output contains counts/hashes only.

- [ ] **Step 3: Run the private 40-question fake pipeline**

Use the real private smoke question manifest with the fake backend; do not run vLLM. Complete verification, synthetic audit decisions clearly labeled as Phase 1 plumbing-only, pair/permutation build, and gate calculation. Do not interpret fake-response Statistical Gate results as a real smoke result.

- [ ] **Step 4: Run the complete Windows and WSL test matrix**

Capture actual output, not predicted counts:

```powershell
python -m pytest tests_permstudy -q | Tee-Object artifacts/data_pipeline/phase1/pytest_output.txt
```

Run the same suite in WSL Python 3.12 and append the labeled output. Record platform, Python, package versions, Git SHA, and math-verify version without user paths.

- [ ] **Step 5: Scan every Git-tracked file for prohibited data**

Build a scanner that checks tracked text files for the known private root, ReClor/MATH source snippets selected only in memory, token patterns, `private-manifest`, and non-synthetic fixture gold/response fields. Save only rule names, scanned file count, match count, and PASS/FAIL to `git_data_scan.txt`; never save the searched private snippets.

- [ ] **Step 6: Write sanitized evidence**

`private_source_summary.json` includes source revisions/file hashes, question counts, duplicate counts, selected stratification level, split hash, and fake-run counts. `fake_e2e_summary.json` includes run IDs, artifact hashes, completion counts, verification status counts, pair/permutation counts, and explicit `real_generation_performed=false`.

`README.md` states:

```text
Design: DESIGN_SPEC_APPROVED
Implementation: AWAITING_PHASE1_CODE_REVIEW
Real vLLM generation: NOT RUN
Real smoke gates: NOT EVALUATED
200-pair pilot: NOT AUTHORIZED
```

- [ ] **Step 7: Request independent code review**

Use the requesting-code-review skill against the complete branch diff from `main`. Resolve findings with new focused tests. Do not mark `DATA_PIPELINE_SYSTEM_READY` until review explicitly passes.

- [ ] **Step 8: After review passes, seal Phase 1 evidence and commit**

Update evidence status to `DATA_PIPELINE_SYSTEM_READY`, preserving the prohibitions on real-smoke claims. Update the design review checkpoint to link this implementation plan and Phase 1 evidence.

```bash
git add artifacts/data_pipeline/phase1 docs/superpowers/specs/2026-10-06-controlled-training-data-pipeline-design.md
git commit -m "docs(data): record Phase 1 pipeline readiness evidence"
```

Run `git status --short`, `git diff --check main..HEAD`, and the full test suite once more before any merge request.

---

## Implementation Review Checkpoints

Request a reviewer gate after Tasks 4, 7, 11, 15, 17, and 18. These boundaries correspond to durable storage, immutable source/split lineage, verifier correctness, gate correctness, complete synthetic behavior, and real-source Phase 1 acceptance.

No Task 18 success authorizes Phase 2 automatically. Phase 2 requires a separate approved execution plan for AutoDL, external caches, gated-model access, real vLLM generation, human audit, and real Functional/Statistical Gates.
