# Controlled Training Data Pipeline Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify the Phase 1 controlled training-data production system with real MATH/ReClor acquisition and deterministic splits, fake generation, strict verification, deterministic pairs/permutations, audit/gates, trainer-ready Parquet export, and no real vLLM inference.

**Architecture:** Implement a new `permstudy.data_pipeline` package whose stages exchange frozen dataclasses and immutable hash-pinned manifests. Task 7 and later stage configs role-bind every direct upstream manifest into their existing `run_id(stage, config)` input. Pair/permutation records branch independently to audit/gates and to a trainer-ready Parquet exporter whose output is exercised through the current verl explicit-identity path. All full-text production artifacts live below an external `PAGRPO_DATA_ROOT`; Git contains only code, explicitly synthetic fixtures, sanitized summaries, hashes, and test evidence.

**Tech Stack:** Python 3.12, standard-library dataclasses/enum/hashlib/json/pathlib/multiprocessing, NumPy 1.26, Hugging Face `datasets`/`huggingface-hub`, Transformers tokenizer APIs, `math-verify[antlr4_9_3]==0.9.0`, psutil 7.x, pytest 8.4.

**Spec:** `docs/superpowers/specs/2026-10-06-controlled-training-data-pipeline-design.md`

**Implementation status:** `IMPLEMENTATION_IN_PROGRESS`; Tasks 1-6 complete and reviewed; `fork/main@944e196` merged; Task 7 next. Do not claim `DATA_PIPELINE_SYSTEM_READY` yet.

## Global Constraints

- Phase 1 may emit only `DATA_PIPELINE_SYSTEM_READY`; it must never emit `REAL_SMOKE_PASS`, `FUNCTIONAL_GATE_PASS`, or `STATISTICAL_GATE_PASS` for real generators.
- Do not start real vLLM, download generator weights, create the real 240-candidate smoke, or expand to 200 pairs.
- Python version is 3.12 for Windows/WSL Phase 1 and Linux Phase 2.
- `math-verify[antlr4_9_3]` is pinned exactly to `0.9.0`; runtime code checks the installed version before MATH verification.
- MATH uses every `EleutherAI/hendrycks_math` train config at a resolved immutable dataset SHA.
- Task 18 must preflight configuration discovery and one train load at that exact SHA. A script-only failure stops for an explicit source/dependency protocol decision; it must not silently change revision, parquet ref, `data_files`, or `datasets` version.
- ReClor uses official `train.json` only; official val/test never enter the internal split.
- `split_seed=42`; all split/sampling units are `original_question_id`.
- MATH split fallback is whole-source and deterministic: try `category+level`, then `category`, then `source`; never mix fallback levels within one split run.
- The real-smoke recipe remains 20 MATH + 20 ReClor questions and two samples from each of three pinned generators.
- Candidate records are keyed by `(generation_run_id, candidate_id)` everywhere.
- Task 7 and later typed configs contain role-tagged `upstream_bindings`; canonical role keys and hashes participate in `run_id`, while existing Task 1-6 IDs and golden hashes remain unchanged.
- Qwen3-8B remains only the Task 4 smoke model. Candidate generators, the Qwen2.5-7B pair-length tokenizer, and the controlled experiment backbone retain their separately approved roles.
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
verification/{verification_run_id}/candidate_records.jsonl
verification/{verification_run_id}/question_records.jsonl
verification/{verification_run_id}/manifest.json
audit/{audit_run_id}/selection.jsonl
audit/{audit_run_id}/decisions.jsonl
audit/{audit_run_id}/manifest.json
pairs/{pair_run_id}/pairs.jsonl
pairs/{pair_run_id}/manifest.json
permutations/{permutation_run_id}/permutations.jsonl
permutations/{permutation_run_id}/manifest.json
trainer_exports/{export_run_id}/train.parquet
trainer_exports/{export_run_id}/manifest.json
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
| `permstudy/data_pipeline/trainer_export.py` | Deterministic Judge prompts, trainer rows, Parquet artifact and export lineage |
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
    prediction_parse_status: str
    canonical_prediction: str | None
    verifier_name: str
    verifier_version: str
    parser_version: str
    verifier_timeout_seconds: float
    error_type: str | None


@dataclass(frozen=True)
class QuestionVerificationRecord:
    verification_run_id: str
    original_question_id: str
    source: Source
    gold_parse_status: str
    canonical_gold: str | None
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
    record_kind: str
    original_question_id: str
    candidate_id: str | None
    source: Source
    generator_id: str | None
    verification_status: VerificationStatus | None
    gold_parse_status: str | None
    reason_code: str


@dataclass(frozen=True)
class AuditDecision:
    audit_run_id: str
    generation_run_id: str
    verification_run_id: str
    record_kind: str
    original_question_id: str
    candidate_id: str | None
    verdict: AuditVerdict
    reason_code: str
    confirmed_disagree: bool
```

`QuestionVerificationRecord.gold_parse_status` accepts only `ok` or `gold_verification_error`. Candidate-level `VerificationRecord` never duplicates a gold failure: if gold fails, emit one question record, emit no candidate verification records for that question, and exclude it from pair construction. `AuditSelectionRecord.record_kind` is exactly `candidate` or `question_gold`; the latter has `candidate_id=None`, `generator_id=None`, `verification_status=None`, and carries the question gold status. `record_hash` is added and validated by the I/O layer over every serialized record; it is not an input to the frozen business record constructors.

Module-local immutable result types are also fixed:

- `IsolationReport(repo_root_hash, data_root_hash, effective_cache_kinds)` contains hashes/kinds only, never raw paths.
- `JsonlScan(records, quarantined_tail_path, file_sha256)` returns validated dictionaries and a private path object after any incomplete tail has been durably quarantined and removed from the source JSONL.
- `SourceSnapshot(source, source_revision, source_snapshot_id, questions, private_manifest)` carries `QuestionRecord` values.
- `SplitBuildResult(assignments, stratification_level_by_source, fallback_reasons, split_manifest_hash)`.
- `GenerationConfig` contains backend, generator/model IDs and revisions, prompt revision/hash, sampling values, batch size, tensor parallel size, max model length, GPU memory utilization, samples per question, and split manifest hash.
- `GenerationResult(plan, response, finish_reason, generated_token_count, error_type)`.
- `GenerationPlan(generation_run_id, config_hash, candidates, shard_ids)` and `ShardRunSummary(planned, successful, historical_failures, missing, manifest_hash, semantic_candidate_set_hash)`.
- `ParsedAnswer(status, answer, error_type)` and `BoxedAnswer(status, boxed_text, error_type)`.
- `ResponseGroup(response_hash, representative_key, member_keys, member_generators, verification_status)`.
- `TrainerExportSummary(export_run_id, dataset_artifact, row_count, prompt_template_hash, upstream_bindings, output_manifest_hash)` where `dataset_artifact` is the Parquet `ArtifactRef` and `upstream_bindings` retains role names.
- `AuditSummary(required_count, completed_count, verdict_counts, confirmed_disagreements, systematic_issue)`.
- `GateInputs(plans, candidates, verifications, question_verifications, pairs, permutations, audit_summary)` plus `FunctionalGateReport(passed, failures, metrics)` and `StatisticalGateReport(status, hard_failures, warnings, diagnostics)`.
- `FakeE2ESummary(run_ids, semantic_candidate_set_hash, downstream_manifest_hashes, counts, phase_status, real_generation_performed)`.

Beginning with Task 7, every stage uses the following exact role-tagged upstream binding set in its typed config. These role names are part of the lineage schema and cannot be renamed without a schema-version change:

| Stage | Required `upstream_bindings` roles |
|---|---|
| split | `math_questions`, `reclor_questions` |
| generation | `split` |
| verification | `split`, `generation` |
| pairs | `generation`, `verification` |
| permutations | `pairs` |
| audit | `verification` |
| gates | `generation`, `verification`, `audit`, `pairs`, `permutations` |
| trainer export | `split`, `generation`, `pairs`, `permutations` |

Every later task must include a test proving that changing one required binding changes its stage run ID. `RunManifest.upstream_manifest_hashes` is the lexicographically sorted list of the mapping values; the mapping itself stays in the private config/config hash so role identity is not lost.

---

### Task 1: Dependency Contract and Core Schemas

**Files:**
- Create: `permstudy/data_pipeline/__init__.py`
- Create: `permstudy/data_pipeline/schema.py`
- Create: `tests_permstudy/data_pipeline/__init__.py`
- Create: `tests_permstudy/data_pipeline/test_schema.py`
- Create: `requirements-data-pipeline-phase1.txt`

**Interfaces:**
- Produces: `Source`, `Split`, `VerificationStatus`, `AuditVerdict`, `GateStatus`, `ArtifactRef`, `RunManifest`, `QuestionRecord`, `SplitAssignment`, `CandidatePlan`, `CandidateRecord`, `FailureRecord`, `VerificationRecord`, `QuestionVerificationRecord`, `PairRecord`, `PermutationRecord`, `AuditSelectionRecord`, `AuditDecision`.
- Produces: every record's `validate() -> None`, `to_dict() -> dict[str, object]`, and matching `from_dict()` classmethod.

- [ ] **Step 1: Write failing schema and dependency tests**

```python
from dataclasses import FrozenInstanceError
from importlib.metadata import version

import pytest

from permstudy.data_pipeline.schema import (
    CandidatePlan,
    QuestionVerificationRecord,
    Source,
    Split,
)


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


def test_gold_error_is_a_single_question_level_state():
    record = QuestionVerificationRecord(
        verification_run_id="verify_abc",
        original_question_id="math:train:" + "a" * 40,
        source=Source.MATH,
        gold_parse_status="gold_verification_error",
        canonical_gold=None,
        error_type="math_parse_failure",
    )
    record.validate()
```

- [ ] **Step 2: Run tests and confirm the intended failures**

Run:

```powershell
python -m pytest tests_permstudy/data_pipeline/test_schema.py -v
```

Expected: collection fails because `permstudy.data_pipeline.schema` does not exist; after the package exists but before dependency installation, the version assertion reports the non-0.9.0 version.

- [ ] **Step 3: Pin dependencies and install the Windows Phase 1 set**

Create a dedicated, exact Phase 1 dependency set and leave `requirements.txt`, `requirements-lock.txt`, `requirements-windows-smoke.txt`, and `setup.py` unchanged. The paper-reproduction lock and the existing Windows evaluator smoke environment remain separate contracts.

```text
datasets==4.4.1
huggingface-hub==0.36.0
math-verify[antlr4_9_3]==0.9.0
numpy==1.26.4
pandas==2.3.3
psutil==7.1.3
pyarrow==22.0.0
pytest==8.4.2
tokenizers==0.22.1
transformers==4.57.1
```

Run:

```powershell
python -m pip install -r requirements-data-pipeline-phase1.txt
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

Validators must reject empty IDs, non-64-hex hashes, negative sample indices, illegal enum values, wrong ReClor answer count, and source-incompatible question fields. `QuestionVerificationRecord.gold_parse_status` accepts only `ok` and `gold_verification_error`; candidate records cannot carry question-level gold status. `RunManifest.artifacts` is a tuple of `ArtifactRef`; artifact paths must be relative POSIX paths.

- [ ] **Step 5: Run schema tests and the existing suite**

Run:

```powershell
python -m pytest tests_permstudy/data_pipeline/test_schema.py -v
python -m pytest tests_permstudy -q
```

Expected: focused tests pass and the pre-existing 59-pass baseline does not regress.

- [ ] **Step 6: Commit**

```bash
git add permstudy/data_pipeline tests_permstudy/data_pipeline requirements-data-pipeline-phase1.txt
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

PUBLIC_EVIDENCE_KEYS = frozenset({
    "schema_version", "git_sha", "platform", "python_version",
    "package_versions", "run_ids", "source_revisions", "source_file_hashes",
    "tree_manifest_hash", "question_counts", "duplicate_counts",
    "stratification_level", "fallback_reasons", "split_manifest_hash",
    "semantic_candidate_set_hash", "artifact_hashes", "completion_counts",
    "verification_status_counts", "question_verification_status_counts",
    "pair_count", "permutation_count", "real_generation_performed",
    "phase_status",
})
```

Evidence sanitization validates nested values as scalar/count/hash/version structures and rejects question, response, gold, prompt, solution, context, answer, and absolute-path payload fields at any depth.

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
    from permstudy.data_pipeline.io import IntegrityError, append_record, scan_jsonl

    path = tmp_path / "records.jsonl"
    append_record(path, {"id": "a"})
    with path.open("ab") as stream:
        stream.write(b'{"id":')
    scan = scan_jsonl(path, recover_incomplete_tail=True)
    assert scan.quarantined_tail_path is not None
    assert scan.quarantined_tail_path.read_bytes() == b'{"id":'
    assert path.read_bytes().endswith(b"\n")
    assert scan_jsonl(path).records == scan.records

    path.write_bytes(b'{"id":\n{"id":"b"}\n')
    with pytest.raises(IntegrityError, match="middle line"):
        scan_jsonl(path, recover_incomplete_tail=True)
```

Also test temp-file + fsync + replace ordering via monkeypatch, stable record hashes, hash mismatch rejection, exclusive lock collision, lock metadata fields, refusal to auto-delete stale locks, explicit recovery only after `psutil.pid_exists(pid)` is false, and relative artifact paths.

- [ ] **Step 2: Run focused tests and observe missing functions**

Run: `python -m pytest tests_permstudy/data_pipeline/test_io.py -v`

Expected: FAIL during import.

- [ ] **Step 3: Implement durable I/O**

`append_record` computes the record hash over the payload without `record_hash`, writes one compact canonical JSON object plus `\n`, flushes, and calls `os.fsync`. `scan_jsonl` validates each record hash. Recovery finds the byte offset immediately after the last complete LF, writes the exact incomplete suffix to a new private quarantine file, flushes and fsyncs that quarantine file, truncates the original JSONL to the complete-LF offset, then flushes and fsyncs the repaired original before returning or allowing any append. A malformed non-final line remains an integrity failure and is never repaired automatically. Tests assert that the original file is valid JSONL before resume and that quarantine contains the exact removed bytes.

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
- Create: `tests_permstudy/fixtures/data_pipeline/reclor/val.json`
- Create: `tests_permstudy/fixtures/data_pipeline/reclor/test.json`
- Create: `tests_permstudy/fixtures/data_pipeline/reclor/use_items.txt`

**Interfaces:**
- Produces: `load_reclor_train(source_path: Path, data_root: Path) -> SourceSnapshot`.
- Produces: `canonical_reclor_label(label: int | str) -> str`.
- Consumes: canonical/ID/isolation/I/O APIs from Tasks 2–4.

- [ ] **Step 1: Add an explicitly synthetic fixture and failing tests**

Fixture records include `"synthetic": true`, unique `id_string`, four answers, integer labels, and repeated content cases. Tests assert val/test are never converted to `QuestionRecord`, labels map `0..3 -> A..D`, structured hashes preserve answer order, same-content consistent metadata keeps the smallest provenance ID, conflicting labels fail, and no source text appears in a sanitized manifest. The private source snapshot manifest must contain SHA256 for exactly `train.json`, `val.json`, `test.json`, and `use_items.txt`, a canonical relative-file tree manifest sorted by POSIX relative path, its `tree_manifest_hash`, and `license_scope="non_commercial_research"`; changing any one file changes the tree hash.

```python
def test_reclor_label_comes_from_official_field():
    snapshot = load_reclor_train(FIXTURE_DIR, tmp_path / "external")
    assert {record.gold_label for record in snapshot.questions} == {"A", "B", "C", "D"}
```

- [ ] **Step 2: Run focused tests and observe import failure**

Run: `python -m pytest tests_permstudy/data_pipeline/test_source_reclor.py -v`

- [ ] **Step 3: Implement directory-only Phase 1 acquisition**

Accept only an extracted external directory after the CLI supplies `--acknowledge-reclor-noncommercial`; archive/password handling is explicitly deferred beyond Phase 1. Require all four official snapshot files, hash each one, and build the canonical tree manifest before reading training rows. Validate `train.json` required fields and exactly four ordered answers. `val.json` and `test.json` participate only in snapshot integrity and never produce training records; `use_items.txt` is likewise hashed but not parsed into questions.

- [ ] **Step 4: Verify outputs remain external and deterministic**

Run the focused test twice and assert identical question records and manifest payload hashes after removing the non-hashed processing timestamp.

- [ ] **Step 5: Run full suite and commit**

```bash
python -m pytest tests_permstudy/data_pipeline/test_source_reclor.py -v
python -m pytest tests_permstudy -q
git add permstudy/data_pipeline/sources tests_permstudy/data_pipeline/test_source_reclor.py tests_permstudy/fixtures/data_pipeline/reclor
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
- Create: `permstudy/data_pipeline/lineage.py`
- Create: `permstudy/data_pipeline/splitting.py`
- Create: `tests_permstudy/data_pipeline/test_lineage.py`
- Create: `tests_permstudy/data_pipeline/test_splitting.py`

**Interfaces:**
- Produces: `role_bound_stage_config(parameters: Mapping[str, object], upstream_bindings: Mapping[str, str], allowed_roles: Collection[str]) -> dict[str, object]`.
- Produces: `build_internal_split(questions, split_seed=42) -> SplitBuildResult`.
- Produces: `split_manifest_bytes(questions, result, split_seed=42) -> bytes`, the only renderer Task 16 may persist for the split manifest.
- Produces: `select_smoke_questions(assignments, per_source=20, split_seed=42) -> list[SplitAssignment]`.

- [ ] **Step 1: Write failing role-bound lineage tests**

Test the helper with literal 64-hex hashes. Require canonical output independent of mapping insertion order; identical parameters/bindings produce the same existing `ids.run_id`; changing one hash changes the ID; swapping the same two hashes between `math_questions` and `reclor_questions` changes the ID; unknown/missing roles and non-64-hex values fail. Assert `sorted(config["upstream_bindings"].values())` is exactly the unchanged `RunManifest.upstream_manifest_hashes` representation. Do not alter `ids.run_id` or any Task 1-6 golden hash.

- [ ] **Step 2: Run lineage tests and observe failures**

Run: `python -m pytest tests_permstudy/data_pipeline/test_lineage.py -v`

- [ ] **Step 3: Implement the typed role binding helper**

Return exactly:

```python
{
    "lineage_schema": "role_tagged_upstream_v1",
    "parameters": deepcopy(dict(parameters)),
    "upstream_bindings": {role: upstream_bindings[role] for role in sorted(upstream_bindings)},
}
```

Validate exact role-set equality against `allowed_roles`, identifier-like nonempty role names, and lowercase 64-hex manifest hashes. Deeply snapshot parameters so later caller mutation cannot change a completed typed config. The split stage calls it with roles `math_questions` and `reclor_questions`; all later tasks use their own exact allowed role set.

- [ ] **Step 4: Write failing determinism and leakage tests**

Tests assert input reorder does not alter assignments/hash; ReClor uses gold label; MATH chooses `category+level` when feasible, falls back the entire source to `category` when any primary stratum is infeasible, then to `source` when category is infeasible; fallback reasons are recorded; every question occurs once; the exact 90/10 count is stable; smoke uses train only and returns 20 per source; candidate/pair/permutation split mismatches are rejected.

- [ ] **Step 5: Run focused tests and observe failures**

Run: `python -m pytest tests_permstudy/data_pipeline/test_splitting.py -v`

- [ ] **Step 6: Implement the exact whole-source fallback algorithm**

For `N` records, set holdout count to `min(N-1, max(1, (N + 5) // 10))`. A stratification level is feasible only if every group has at least two records and `group_count <= holdout_count <= N - group_count`. Choose the first feasible level from the approved fallback chain.

Initialize every group's holdout quota to one. Allocate remaining holdout slots one at a time to the eligible group with the smallest exact fraction `quota/group_size`, comparing fractions by integer cross multiplication and breaking ties by canonical group key. Inside each group, order questions by `SHA256("42\0" + original_question_id)` and assign the first quota to holdout.

Smoke selection uses a separate proportional allocator because 20 samples may be fewer than the number of strata. For each stratum compute `floor(group_size * 20 / source_train_size)`, then distribute remaining slots by descending integer remainder `(group_size * 20) % source_train_size`, canonical stratum key, and available capacity. Within each stratum choose the lowest `SHA256("42\0smoke\0" + original_question_id)` scores. Use the selected MATH split stratification level and ReClor gold labels; fail if a source has fewer than 20 train questions.

The split run config is built with `role_bound_stage_config` and the exact source-manifest bindings. Its parameters are exactly `split_seed`, `split_algorithm=SPLIT_ALGORITHM`, and `split_schema_version=SPLIT_SCHEMA`, so changing either version changes the split `run_id` and `config_hash`. Its manifest stores the sorted binding values in `upstream_manifest_hashes`. The split stage must reject source artifacts whose verified manifest hashes do not equal their named bindings before writing assignments.

`split_manifest_bytes()` is the authoritative canonical JSONL renderer. Its first LF-terminated line is a `split_metadata` record containing `split_manifest_v1`, `deterministic_stratified_question_split_v1`, explicit per-source `question_normalization_v1` identifiers, source provenance, seed, stratification levels, and fallback reasons. Each following LF-terminated line is one `split_assignment` record, ordered by `original_question_id`. `split_manifest_hash` is SHA256 over those exact bytes. Task 16 must persist the bytes returned by this function and must not independently rebuild equivalent-looking JSON.

- [ ] **Step 7: Add a golden cross-platform hash test**

For the programmatically constructed synthetic Task 7 unit baseline, assert a literal expected `split_manifest_hash` and run the same test on Windows and WSL; do not update the literal independently per platform. This is a serializer/algorithm unit golden, not the committed-source-fixture acquisition-to-split proof. Task 17 owns the latter cross-platform fake-E2E assertion.

- [ ] **Step 8: Run focused/full tests and commit**

```bash
python -m pytest tests_permstudy/data_pipeline/test_lineage.py tests_permstudy/data_pipeline/test_splitting.py -v
python -m pytest tests_permstudy -q
git add permstudy/data_pipeline/lineage.py permstudy/data_pipeline/splitting.py tests_permstudy/data_pipeline/test_lineage.py tests_permstudy/data_pipeline/test_splitting.py
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
- Produces: `semantic_candidate_set_hash(records: Iterable[CandidateRecord]) -> str`.
- Consumes: Task 4 durable I/O and locks.

- [ ] **Step 1: Write failing interruption/resume tests**

Simulate a backend that succeeds twice, fails once, then raises process interruption. Assert successful lines remain, failure history is separate, resume calls only missing composite keys, retry success raises final completion to 100%, old failures remain auditable, and a second run with a changed config/run ID never treats old logical IDs as complete. Compare with an uninterrupted run: the final successful composite-key sets and `semantic_candidate_set_hash` values must match, while physical execution-history manifest hashes are allowed to differ.

- [ ] **Step 2: Run focused tests and observe failures**

Run: `python -m pytest tests_permstudy/data_pipeline/test_generation_runner.py -v`

- [ ] **Step 3: Implement one-writer append/resume flow**

Order operations exactly: validate isolation/config/upstream hashes; acquire lock; scan/quarantine incomplete tail; calculate missing composite keys; call backend in deterministic batches; append each success or failure immediately; atomically update manifest counts; release lock in `finally`.

Map OOM to `error_type="oom"`. Compare the complete generation config hash before resume; any changed batch size, tensor parallel size, max model length, memory utilization, prompt hash, sampling config, model revision, or split hash raises `RunMismatchError`.

`semantic_candidate_set_hash` sorts successful records by `(generation_run_id, candidate_id)` and hashes canonical semantic payloads containing the composite key, response normalization/hash, finish reason, generated token count, and immutable model revision. It excludes append order, timestamps, retry counts, failure history, record hashes, and physical artifact paths.

- [ ] **Step 4: Verify interrupted and uninterrupted semantic results match**

Run the focused test that executes both routes and asserts equal successful composite-key sets and equal canonical semantic candidate-set hashes. Assert separately that differing failure history is permitted to produce different physical output-manifest hashes and is not a correctness failure.

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
- Produces: `verify_reclor_question(question, candidates) -> tuple[QuestionVerificationRecord, list[VerificationRecord]]`.

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

Only whitespace may follow the unique marker. Count all case-sensitive `Final Answer:` markers; identical duplicates are invalid and conflicting valid answers are ambiguous. The question-level function emits exactly one `QuestionVerificationRecord(gold_parse_status="ok")` for the official canonical label plus one candidate record per input candidate. Store `verifier_name="reclor_exact_match"`, parser version, prediction parse status, and no math-verify fields in candidate records.

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
- Produces: `verify_math_question(question, candidates, gold_timeout_seconds, candidate_timeout_seconds) -> tuple[QuestionVerificationRecord, list[VerificationRecord]]`.

API references are the official [Math-Verify README](https://github.com/huggingface/Math-Verify/blob/main/README.md), the published [0.9.0 package](https://pypi.org/project/math-verify/0.9.0/), and the documented [Windows timeout/pickling failure](https://github.com/huggingface/Math-Verify/issues/79). The parent-enforced spawn-process timeout is intentional and must not be replaced with the library's signal-based timeout on Windows.

- [ ] **Step 1: Write failing extraction/status/cache tests**

Test a terminal nested `\boxed{\frac{1}{2}}`, missing marker, malformed braces, duplicate same candidate markers -> invalid, conflicting markers -> ambiguous, gold parse failure -> exactly one `QuestionVerificationRecord(gold_parse_status="gold_verification_error")` and zero candidate records, explicit equivalent/non-equivalent cases, child exception -> error, and one gold parse for six normal candidates. Assert no gold failure is duplicated into candidate `error` records.

- [ ] **Step 2: Write a Windows timeout regression test**

Inject separate top-level worker hooks that block during gold parsing and candidate verification. A gold timeout must terminate/join the worker, emit one question-level gold error, and stop without sending any candidate. A candidate timeout after `GOLD_READY` must emit one candidate `error` with `error_type="timeout"`, terminate/join the worker, start a fresh worker, complete the gold handshake again, and then process the next candidate. The test must pass under Windows `spawn` and Linux/WSL.

- [ ] **Step 3: Run focused tests and observe failures**

Run: `python -m pytest tests_permstudy/data_pipeline/test_verification_math.py -v`

- [ ] **Step 4: Implement strict extraction and a per-question worker process**

The parent starts a spawn-context worker for one question. The worker imports `parse`/`verify`, and the protocol has two explicit phases. In Phase A, the worker parses gold once with:

```python
gold = parse(
    boxed_gold,
    fallback_mode="no_fallback",
    extraction_mode="first_match",
    parsing_timeout=None,
    raise_on_error=True,
)
```

After parsing, the worker sends exactly one handshake message: `GOLD_READY` with a private canonical representation, or `GOLD_ERROR` with a sanitized error type. The parent waits no longer than `gold_timeout_seconds`; timeout terminates and joins the worker and has the same question-level result as `GOLD_ERROR`. Either failure emits one `QuestionVerificationRecord(gold_parse_status="gold_verification_error")`, emits no candidate records for that question, and sends no candidate messages.

Only after `GOLD_READY`, Phase B sends candidates one at a time. For each candidate the worker parses the strict extracted boxed region and calls:

```python
equivalent = verify(
    gold,
    prediction,
    strict=True,
    timeout_seconds=None,
    raise_on_error=True,
)
```

The parent enforces `candidate_timeout_seconds` separately for every Phase B request with a Pipe poll. The child returns only strings, booleans, status codes, and sanitized error types—never SymPy objects—avoiding cross-process pickling. On candidate timeout, terminate/join the worker, mark only that candidate as error, restart for remaining candidates, repeat Phase A, and record the worker restart. A normal six-candidate question therefore parses gold once; a worker restart may reparse it. Empty parse results are ambiguous, not incorrect; only explicit `equivalent is False` after successful parses is incorrect. Both `parse(..., parsing_timeout=None)` and `verify(..., timeout_seconds=None)` keep the library's internal timeout disabled; the parent process owns both timeouts.

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

### Task 13A: Trainer-Ready Parquet Export and Explicit-Identity Round Trip

**Files:**
- Create: `permstudy/data_pipeline/trainer_export.py`
- Create: `tests_permstudy/data_pipeline/test_trainer_export.py`

**Interfaces:**
- Consumes: verified immutable split, generation, pair, and permutation manifests plus their hash-pinned question/candidate/pair/permutation artifacts.
- Produces: `render_pairwise_judge_prompt(question: QuestionRecord, response_a: str, response_b: str) -> list[dict[str, str]]`.
- Produces: `build_trainer_rows(questions, candidates, pairs, permutations) -> list[dict[str, object]]`.
- Produces: `export_trainer_parquet(upstream_manifests, output_dir: Path) -> TrainerExportSummary`.

Audit and gates continue to consume canonical verification/pair/permutation records directly. Neither module may import `trainer_export`, read its Parquet file, or use export status as a quality signal.

- [ ] **Step 1: Write failing row-contract and lineage tests**

Use one synthetic MATH pair and one synthetic ReClor pair. Assert two rows per pair sorted by `(original_question_id, permutation_id)`. Assert the exact fields `data_source`, `prompt`, `ability`, `reward_model`, and `extra_info`; AB is positive/negative with ground truth A, BA is negative/positive with ground truth B. Require `extra_info.pair_id == PairRecord.pair_id`, preserve `original_question_id`, and write canonical integer `permutation_id` without relying on the legacy `permutation` alias.

Assert MATH ability is `math`, ReClor ability is `logical_reasoning`, and `data_source` is `permstudy_pairwise_judge_v1`. Prompt template `pairwise_judge_direct_v1` is exactly:

```python
[
    {"role": "system", "content": "Reply with only A or B."},
    {
        "role": "user",
        "content": (
            f"Question:\n{rendered_question}\n\n"
            f"Response A:\n{response_a}\n\n"
            f"Response B:\n{response_b}\n\n"
            "Which response is more correct?\n"
            "Answer with A or B only."
        ),
    },
]
```

MATH `rendered_question` is its canonical problem. ReClor renders `Context:`, `Question:`, then ordered `A.` through `D.` options. Assert a changed selected response hash, question content hash, split hash, missing candidate, duplicate permutation, or non-mirrored surface mapping fails before output. Test role-bound export run IDs for identical bindings, one changed binding, and a role swap.

- [ ] **Step 2: Run contract tests and observe failures**

Run: `python -m pytest tests_permstudy/data_pipeline/test_trainer_export.py -v`

- [ ] **Step 3: Implement deterministic row construction and Parquet writing**

Resolve records only through artifacts referenced by the four required role bindings: `split`, `generation`, `pairs`, and `permutations`. Recompute question/response hashes, validate every pair/permutation join, sort rows, and write `train.parquet` below `trainer_exports/{export_run_id}/`. Use pandas/PyArrow without embedding the index. Build the typed export config with `role_bound_stage_config`, template version/hash, row schema `trainer_pairwise_parquet_v1`, and no runtime path or timestamp in the config hash.

The `reward_model` value is `{"ground_truth": correct_surface, "style": "rule"}`. `extra_info` contains `pair_id`, `original_question_id`, `permutation_id`, `permutation_label`, `source`, `split`, `split_manifest_hash`, `generation_run_id`, `pair_run_id`, and `permutation_run_id`. The export manifest uses the standard `RunManifest` envelope and includes the Parquet `ArtifactRef`, prompt-template hash, row count, and dataset SHA256.

- [ ] **Step 4: Add the actual Parquet-to-trainer integration test**

The test must read the written file through the production loader, not `pandas.read_parquet` alone:

Define a test-local `FakeTrainerTokenizer` with nonempty `chat_template`, `pad_token_id=0`, deterministic `apply_chat_template(..., tokenize=False)`, `encode(..., add_special_tokens=False)`, and `__call__(..., return_tensors="pt", add_special_tokens=False)` returning one-row `input_ids` and `attention_mask` tensors. It performs no network or model download.

```python
dataset = RLHFDataset(
    data_files=str(export_path),
    tokenizer=FakeTrainerTokenizer(),
    config=OmegaConf.create({
        "prompt_key": "prompt",
        "max_prompt_length": 4096,
        "filter_overlong_prompts": False,
        "truncation": "error",
        "return_raw_chat": True,
        "cache_dir": str(external_cache),
    }),
)
loaded = collate_fn([dataset[index] for index in range(len(dataset))])
batch = DataProto.from_single_dict(loaded)
attach_permutation_identity(batch, identity_mode="explicit")
repeated = repeat_for_rollout(batch, repeat_times=2, identity_mode="explicit")
merged = merge_identity_into_extra_infos(repeated)
```

Assert PyArrow/Hugging Face preserve nested `prompt`, `reward_model`, and `extra_info`; pair IDs are the deterministic pair hashes rather than question IDs; permutation IDs are `[0, 1]` before repeat; repeated slots are `[0, 1, 0, 1]` per pair in interleaved order; merged reward extras contain pair/permutation/slot identity. Remove `original_question_id` from a copied row and prove explicit identity still succeeds from `pair_id`; remove `pair_id` and prove the formal exporter validation fails even though the trainer offers a legacy-compatible fallback.

- [ ] **Step 5: Prove audit/gates are export-independent**

Import `permstudy.data_pipeline.audit` and `permstudy.data_pipeline.gates` after blocking imports of `permstudy.data_pipeline.trainer_export`, pandas, and pyarrow. Their focused tests must still pass, demonstrating the dependency branch is `pair/permutation -> {audit/gates, export}` rather than `pair -> export -> audit`.

- [ ] **Step 6: Run focused/full tests and commit**

```bash
python -m pytest tests_permstudy/data_pipeline/test_trainer_export.py tests_permstudy/test_trainer_identity_integration.py -v
python -m pytest tests_permstudy -q
git add permstudy/data_pipeline/trainer_export.py tests_permstudy/data_pipeline/test_trainer_export.py
git commit -m "feat(data): export trainer-ready pairwise parquet"
```

---

### Task 14: Human Audit Selection and Decision Validation

**Files:**
- Create: `permstudy/data_pipeline/audit.py`
- Create: `tests_permstudy/data_pipeline/test_audit.py`

**Interfaces:**
- Produces: `build_audit_selection(candidate_records, question_records, generation_run_id, audit_seed=42, max_per_cell=5) -> list[AuditSelectionRecord]`.
- Produces: `validate_audit_decisions(selection, decisions) -> AuditSummary`.

- [ ] **Step 1: Write failing coverage and deterministic-sampling tests**

Assert all candidate `invalid`/`ambiguous`/`error` records and every question-level `gold_verification_error` are selected; question gold errors use `record_kind="question_gold"` with no candidate/generator/status and appear exactly once per question. Candidate `correct`/`incorrect` records are grouped by source × generator × status; at most five per cell are selected by SHA256 seed score; small cells are exhaustive; input order does not matter; decisions allow only AGREE/DISAGREE/UNSURE; missing/duplicate/foreign decision identities fail; reason codes use the approved enum.

- [ ] **Step 2: Run focused tests and observe failures**

Run: `python -m pytest tests_permstudy/data_pipeline/test_audit.py -v`

- [ ] **Step 3: Implement selection and confirmed disagreement handling**

Selection records bind generation and verification run IDs. Candidate audit identity is `(record_kind, generation_run_id, verification_run_id, candidate_id)`; question-gold identity substitutes `original_question_id` for the absent candidate ID. Decision validation distinguishes first-review disagreement from `confirmed_disagree=True`; only confirmed disagreement contributes a hard-fail signal. `UNSURE` remains a separate count and queue. The only reason codes are `missing_final_marker`, `duplicate_same_marker`, `conflicting_markers`, `math_parse_failure`, `gold_parse_failure`, `timeout`, `dependency_error`, `prompt_contract_mismatch`, and `other`.

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

Cover completion 99%/100%; pair yield 19%/20%/29%/30%; candidate invalid+error 20%/just above 20%; pooled dominance 89%/90%; per-source dominance 70%/above 70%/90%; correct rate 0%/100% with denominator `correct+incorrect`; undefined zero denominator; confirmed disagreement; systematic prompt/parser issue; and ambiguous rate remaining diagnostic-only. Add a question-level gold error and assert it is reported/audited but does not enter any generator/source `invalid_error_rate` numerator or denominator.

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

Compute completion from planned versus successful composite keys; pair yield from selected-pair questions/planned source questions; invalid+error from candidate verification records over successful generated candidates in each cell; dominance from selected representatives/selected pairs; correct rate from correct/(correct+incorrect). Functional completeness requires exactly one question verification record per verified question; questions with `ok` require their candidate records, while `gold_verification_error` requires zero candidate verification records. `QuestionVerificationRecord` values are a separate audit/completeness input and never multiply or otherwise enter candidate-level generator rates. Apply status precedence `FAIL`, then `PASS_WITH_WARNINGS`, then `PASS`.

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
- Create: `scripts_permstudy/data/export_training_dataset.py`
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
| `prepare_questions.py acquire-reclor` | `--data-root`, `--reclor-dir`, `--acknowledge-reclor-noncommercial` | immutable ReClor source manifest |
| `prepare_questions.py acquire-math` | `--data-root`, `--math-revision` | immutable MATH source manifest |
| `prepare_questions.py build-split` | `--data-root`, `--reclor-manifest`, `--math-manifest`, `--split-seed 42` | one unified two-source split manifest and smoke manifest |
| `plan_generation.py` | `--data-root`, `--split-manifest`, three `--generator-config` JSON files, `--samples-per-question 2`, `--shard-size` | generation plan manifest |
| `generate_candidates.py` | `--data-root`, `--generation-manifest`, `--generator-id`, `--shard-id`, `--backend fake`; optional `--recover-stale-lock` | candidate/failure shard |
| `verify_candidates.py` | `--data-root`, `--generation-manifest`, `--gold-timeout-seconds`, `--candidate-timeout-seconds` | verification run with separate question/candidate records |
| `build_reasoning_pairs.py` | `--data-root`, `--verification-manifest`, `--tokenizer-revision` | pair run |
| `build_permutations.py` | `--data-root`, `--pair-manifest` | permutation run |
| `export_training_dataset.py` | `--data-root`, `--split-manifest`, `--generation-manifest`, `--pair-manifest`, `--permutation-manifest` | trainer Parquet export run |
| `audit_generator_distribution.py` | `--data-root`, `--verification-manifest`, optional `--decisions`, `--audit-seed 42` | audit selection/summary |
| `validate_dataset.py` | `--data-root` plus one stage manifest; optional `--check-isolation` | sanitized validation report |
| `run_fake_e2e.py` | `--data-root`, `--fixture-root` or `--split-manifest` | fake E2E summary |

- [ ] **Step 1: Write failing parser/default/boundary tests**

Assert every CLI imports without vLLM. `prepare_questions` requires one of the three subcommands: the two acquisition calls each create one immutable source manifest, and `build-split` refuses to run unless both referenced source manifests and artifact hashes validate. Split seed defaults to 42; ReClor has no archive argument in Phase 1; generation defaults to fake; `--backend vllm` exits with `PhaseBoundaryError` in Phase 1 before importing vLLM; stale-lock recovery requires the explicit flag; trainer export requires all four named upstream manifests and refuses mismatched role bindings; no CLI prints absolute paths or payload text; exit codes are 0 success, 2 contract/config error, 3 integrity error, 4 dependency/access error.

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

- [ ] **Step 1: Create the 80-question explicitly synthetic source fixture**

Provide 40 synthetic MATH and 40 synthetic ReClor source questions, each marked `synthetic=true`. ReClor has exactly ten A, ten B, ten C, and ten D official labels so the 90/10 label-stratified split is exercised. The deterministic split leaves 36 train questions per source; smoke selection then takes exactly 20 MATH and 20 ReClor train questions. Provide fake response templates for the selected questions so `40 smoke questions × 3 generators × 2 samples = 240 candidates`, covering correct/incorrect, duplicate, invalid, ambiguous, error, and historical-failure-then-success behavior while guaranteeing at least one selected pair per source.

- [ ] **Step 2: Write a failing full-flow test**

The test runs 80-row source loading -> 90/10 split -> deterministic 20+20 train smoke selection -> 240 candidate planning -> interrupted fake generation -> resume -> verification -> audit selection with synthetic AGREE decisions -> pairs -> permutations -> gates, while the independent consumer branch exports trainer Parquet from the same pair/permutation records. Assert 72 train and 8 held-out source questions overall, 40 selected smoke questions, 240 final successful composite keys, one pair maximum per question, two permutations per pair, equal semantic candidate-set hashes across interrupted and clean runs, stable downstream canonical pair/permutation/export hashes, and no production-only phase status. Read the exported Parquet with `RLHFDataset` and pass one complete pair through the explicit identity/repeat/reward-extra flow. Physical generation-history manifest hashes may differ when retry history differs.

- [ ] **Step 3: Run the test and observe missing orchestration behavior**

Run: `python -m pytest tests_permstudy/data_pipeline/test_fake_e2e.py -v`

- [ ] **Step 4: Implement the orchestration using only package APIs**

`run_fake_e2e.py` must not duplicate stage logic. It creates an external temporary root when `--data-root` is supplied by tests, writes private artifacts there, exports trainer Parquet through the Task 13A package API, and prints only the sanitized summary. Audit/gates use canonical records, not the Parquet output.

- [ ] **Step 5: Verify Windows and WSL semantic hashes**

Run the E2E in both environments and compare successful composite candidate keys, semantic candidate-set hashes, split hashes, canonical pair/permutation hashes, and the canonical trainer-row payload hash byte-for-byte. The Parquet artifact hash is also expected to match under the exact pinned dependency set; if platform metadata differs, fail and diagnose rather than weakening the manifest contract. Do not require equality of physical execution-history manifests that contain different append/failure histories.

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
- Create: `artifacts/data_pipeline/phase1/trainer_export_summary.json`
- Create: `artifacts/data_pipeline/phase1/private_source_summary.json`
- Create: `artifacts/data_pipeline/phase1/git_data_scan.txt`
- Modify: `scripts_permstudy/data/validate_dataset.py`
- Modify: `tests_permstudy/data_pipeline/test_cli.py`
- Modify: `docs/superpowers/specs/2026-10-06-controlled-training-data-pipeline-design.md`

**Interfaces:**
- Consumes all Phase 1 CLIs and approved local environment variables.
- Produces only sanitized evidence and the terminal status `DATA_PIPELINE_SYSTEM_READY` after independent code review; before review the report status is `AWAITING_PHASE1_CODE_REVIEW`.

- [ ] **Step 1: Configure external roots without committing values**

Require runtime-only environment variables to have been set outside Git, then derive external caches without printing the paths:

```powershell
if (-not $env:PAGRPO_DATA_ROOT) { throw 'PAGRPO_DATA_ROOT is required' }
if (-not $env:PAGRPO_RECLOR_DIR) { throw 'PAGRPO_RECLOR_DIR is required in Phase 1' }
$env:HF_HOME = Join-Path $env:PAGRPO_DATA_ROOT 'hf_home'
$env:HUGGINGFACE_HUB_CACHE = Join-Path $env:HF_HOME 'hub'
$env:TRANSFORMERS_CACHE = Join-Path $env:HF_HOME 'transformers'
```

Do not echo or persist the resolved values. Confirm isolation with `validate_dataset.py --check-isolation`.

- [ ] **Step 2: Acquire real ReClor/MATH privately and build the split**

Run `prepare_questions.py acquire-reclor` against the extracted official directory and confirm the private tree manifest hashes `train.json`, `val.json`, `test.json`, and `use_items.txt`, while only train rows become questions.

For MATH, first resolve the requested ref once to an immutable 40-hex SHA without writing an acquisition artifact. Against that exact SHA and external cache, execute `datasets.get_dataset_config_names(repo_id, revision=sha)` and load one train row from one returned config with `datasets.load_dataset(repo_id, config, split="train", revision=sha, cache_dir=...)`. If either operation reports that the pinned revision is script-only/unsupported, stop Task 18 and propose an explicit source/dependency protocol revision. Do not switch to another commit, `refs/convert/parquet`, `data_files=`, or another `datasets` version within the same run.

Only after the exact-SHA preflight passes, run `prepare_questions.py acquire-math` with the same SHA. Then run `prepare_questions.py build-split` with both immutable source manifests and `split_seed=42`. Confirm private manifests contain file/revision hashes, source counts, dedup counts, fallback level, and split hash; confirm public output contains counts/hashes only.

- [ ] **Step 3: Run the private 40-question fake pipeline**

Use the real private smoke question manifest with the fake backend; do not run vLLM. Complete an interrupted/resumed route and a clean route, requiring equal final successful composite keys and semantic candidate-set hashes but not equal physical failure-history manifests. Complete verification, synthetic audit decisions clearly labeled as Phase 1 plumbing-only, pair/permutation build, gate calculation, and trainer Parquet export. Load the export through `RLHFDataset` and the explicit identity/repeat/reward-extra path. Do not interpret fake-response Statistical Gate results as a real smoke result, and do not make audit/gate status depend on export success.

- [ ] **Step 4: Run the complete Windows and WSL test matrix**

Capture actual output, not predicted counts:

```powershell
python -m pytest tests_permstudy -q | Tee-Object artifacts/data_pipeline/phase1/pytest_output.txt
```

Run the same suite in WSL Python 3.12 and append the labeled output. Record platform, Python, package versions, Git SHA, and math-verify version without user paths.

- [ ] **Step 5: Scan every Git-tracked file for prohibited data**

Build a path/schema-aware scanner over `git ls-files -z`; do not search for generic words such as `gold`, `response`, or `private-manifest`. It applies these deterministic rules:

1. reject tracked paths below `data_permstudy/`, `private_data/`, or `private_logs/`, and reject files ending `.private-manifest.json`;
2. scan all tracked text for the exact resolved private-root bytes and HF/secret token patterns; apply generic Windows/POSIX user-home absolute-path detection only to the new `permstudy/data_pipeline/`, `scripts_permstudy/data/`, `tests_permstudy/fixtures/data_pipeline/`, and `artifacts/data_pipeline/phase1/` surfaces so pre-existing upstream examples/evidence are not reclassified by this feature;
3. allow full-text question/response/gold fields only below `tests_permstudy/fixtures/data_pipeline/`, and parse every JSON/JSONL record there to require `synthetic=true`;
4. parse JSON/JSONL below `artifacts/data_pipeline/phase1/` and require keys to be within the public manifest/evidence allowlists, rejecting payload-text fields and absolute paths;
5. inspect newly added pipeline artifacts by schema rather than flagging terminology in design docs, source code, or schemas.

Tests seed one violation for each rule plus design/schema files containing the words `gold` and `response`, proving violations fail and documentation does not self-trigger. Save only rule names, scanned file count, match count, and PASS/FAIL to `git_data_scan.txt`; never save private snippets or paths.

- [ ] **Step 6: Write sanitized evidence**

`private_source_summary.json` includes source revisions/file hashes, the ReClor tree-manifest hash, question counts, duplicate counts, selected stratification level, split hash, and fake-run counts. `fake_e2e_summary.json` includes run IDs, semantic candidate-set hash, sanitized artifact hashes, completion counts, candidate and question-level verification status counts, pair/permutation counts, and explicit `real_generation_performed=false`. `trainer_export_summary.json` stays within the existing public-evidence allowlist: `run_ids.export` holds the export run ID; `artifact_hashes` holds role-tagged `upstream_split`, `upstream_generation`, `upstream_pairs`, `upstream_permutations`, `prompt_template`, and `parquet` hashes; `completion_counts.rows` holds row count; and `phase_status` records the explicit-identity round-trip result. It contains no prompt or response text.

`README.md` states:

```text
Design: DESIGN_SPEC_APPROVED
Implementation: AWAITING_PHASE1_CODE_REVIEW
Real vLLM generation: NOT RUN
Real smoke gates: NOT EVALUATED
200-pair pilot: NOT AUTHORIZED
```

Commit the implemented scanner, its violation/non-self-trigger tests, and the pre-review evidence while the status remains `AWAITING_PHASE1_CODE_REVIEW`:

```bash
git add scripts_permstudy/data/validate_dataset.py tests_permstudy/data_pipeline/test_cli.py artifacts/data_pipeline/phase1
git commit -m "test(data): record Phase 1 pre-review acceptance evidence"
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

Request a reviewer gate after Tasks 4, 7, 11, 13A, 15, 17, and 18. These boundaries correspond to durable storage, immutable source/split lineage, verifier correctness, trainer compatibility, gate correctness, complete synthetic behavior, and real-source Phase 1 acceptance.

No Task 18 success authorizes Phase 2 automatically. Phase 2 requires a separate approved execution plan for AutoDL, external caches, gated-model access, real vLLM generation, human audit, and real Functional/Statistical Gates.
