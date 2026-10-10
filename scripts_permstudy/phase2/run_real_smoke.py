"""Thin Phase 2 orchestration over the frozen Phase 1 data-pipeline APIs.

The module deliberately has no eager vLLM import.  P1 exercises it with an
injected CPU fake; P2 may construct the same adapter in the approved Linux GPU
process without changing any Phase 1 implementation.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from permstudy.data_pipeline import pairs, permutations
from permstudy.data_pipeline.canonical import canonical_json_bytes, sha256_hex
from permstudy.data_pipeline.generation import (
    GenerationConfig,
    GenerationConfigurationError,
    GenerationPlan,
    GenerationRequest,
    VLLMGenerationBackend,
    build_generation_shard_plan,
    generation_stage_config,
    plan_generation,
    run_generation_shard,
    semantic_candidate_set_hash,
)
from permstudy.data_pipeline.ids import run_id
from permstudy.data_pipeline.schema import (
    CandidateRecord,
    AuditDecision,
    AuditSelectionRecord,
    FailureRecord,
    PairRecord,
    PermutationRecord,
    QuestionRecord,
    QuestionVerificationRecord,
    Source,
    VerificationRecord,
)
from permstudy.data_pipeline.verification import (
    verify_math_question,
    verify_reclor_question,
)


REQUEST_SEED_VERSION = "candidate_request_seed_sha256_v1"
PROMPT_TEMPLATE_REVISION = f"phase2_real_candidate_prompt_v1+{REQUEST_SEED_VERSION}"

MATH_SYSTEM_PROMPT = (
    "Solve the mathematics problem carefully. Show your reasoning, then end "
    "the response with exactly one terminal \\boxed{...} answer. Do not write "
    "any text after the terminal boxed answer."
)
RECLOR_SYSTEM_PROMPT = (
    "Solve the logical reasoning multiple-choice problem carefully. Choose "
    "exactly one option A, B, C, or D. End the response with exactly one "
    "terminal line in the form Final Answer: X, where X is the chosen option. "
    "Do not write any text after that line."
)

_REAL_GENERATOR_RECIPE = (
    (
        "qwen2.5-7b-instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "a09a35458c702b33eeacc393d103063234e8bc28",
        8,
        0.85,
    ),
    (
        "llama-3.1-8b-instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "0e9e39f249a16976918f6564b8830bc894c89659",
        8,
        0.85,
    ),
    (
        "qwen2.5-32b-instruct",
        "Qwen/Qwen2.5-32B-Instruct",
        "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd",
        2,
        0.90,
    ),
)
PAIR_TOKENIZER_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"


def prompt_contract_hash(
    request_seed_version: str = REQUEST_SEED_VERSION,
) -> str:
    """Hash the exact prompt text together with request-seed semantics."""
    if not isinstance(request_seed_version, str) or not request_seed_version.strip():
        raise ValueError("request_seed_version must be nonempty text")
    return sha256_hex(
        canonical_json_bytes(
            {
                "math_system": MATH_SYSTEM_PROMPT,
                "math_user": "{problem}",
                "reclor_system": RECLOR_SYSTEM_PROMPT,
                "reclor_user": (
                    "Context:\n{context}\n\nQuestion:\n{question}\n\n"
                    "A. {answer_a}\nB. {answer_b}\nC. {answer_c}\nD. {answer_d}"
                ),
                "request_seed_version": request_seed_version,
            }
        )
    )


def build_real_generation_configs(
    split_manifest_hash: str,
) -> tuple[GenerationConfig, ...]:
    """Build the single approved three-generator smoke recipe."""
    return tuple(
        GenerationConfig(
            backend="vllm",
            generator_id=generator_id,
            model_repository=repository,
            model_revision=revision,
            prompt_template_revision=PROMPT_TEMPLATE_REVISION,
            prompt_template_hash=prompt_contract_hash(),
            temperature=0.8,
            top_p=0.95,
            max_new_tokens=512,
            seed=42,
            batch_size=batch_size,
            tensor_parallel_size=1,
            max_model_len=4096,
            gpu_memory_utilization=gpu_memory_utilization,
            samples_per_question=2,
            split_manifest_hash=split_manifest_hash,
        )
        for (
            generator_id,
            repository,
            revision,
            batch_size,
            gpu_memory_utilization,
        ) in _REAL_GENERATOR_RECIPE
    )


def request_seed(
    base_seed: int,
    generation_run_id: str,
    candidate_id: str,
    *,
    version: str = REQUEST_SEED_VERSION,
) -> int:
    """Derive a backend-order-independent nonnegative signed-64-bit seed."""
    if type(base_seed) is not int or base_seed < 0:
        raise ValueError("base_seed must be a nonnegative integer")
    if not all(
        isinstance(value, str) and value.strip()
        for value in (generation_run_id, candidate_id, version)
    ):
        raise ValueError("request seed identity must be nonempty text")
    digest = sha256_hex(
        canonical_json_bytes(
            {
                "base_seed": base_seed,
                "candidate_id": candidate_id,
                "generation_run_id": generation_run_id,
                "version": version,
            }
        )
    )
    return int(digest[:16], 16) & ((1 << 63) - 1)


def render_messages(question: QuestionRecord) -> tuple[dict[str, str], dict[str, str]]:
    """Render the approved source prompt without source gold or solution text."""
    if not isinstance(question, QuestionRecord):
        raise TypeError("question must be a QuestionRecord")
    question.validate()
    if question.source is Source.MATH:
        user = question.problem
        system = MATH_SYSTEM_PROMPT
    else:
        answer_a, answer_b, answer_c, answer_d = question.answers
        user = (
            f"Context:\n{question.context}\n\nQuestion:\n{question.question}\n\n"
            f"A. {answer_a}\nB. {answer_b}\nC. {answer_c}\nD. {answer_d}"
        )
        system = RECLOR_SYSTEM_PROMPT
    return ({"role": "system", "content": system}, {"role": "user", "content": user})


class NativeVLLMEngine:
    """Adapt vLLM 0.8.5 ``LLMEngine`` to the frozen Phase 1 engine protocol."""

    def __init__(
        self,
        vllm_module,
        config: GenerationConfig,
        questions: Mapping[str, QuestionRecord],
    ) -> None:
        _validate_phase2_config(config)
        self._module = vllm_module
        self._config = config
        self._questions = _validated_questions(questions)
        args = vllm_module.EngineArgs(
            model=config.model_repository,
            tokenizer=config.model_repository,
            revision=config.model_revision,
            tokenizer_revision=config.model_revision,
            trust_remote_code=False,
            tensor_parallel_size=config.tensor_parallel_size,
            max_model_len=config.max_model_len,
            gpu_memory_utilization=config.gpu_memory_utilization,
            seed=config.seed,
            disable_log_stats=True,
        )
        self._engine = vllm_module.LLMEngine.from_engine_args(args)
        self._tokenizer = self._engine.get_tokenizer()

    def generate(
        self,
        requests: Sequence[GenerationRequest],
        config: GenerationConfig,
    ) -> list[object]:
        if config != self._config:
            raise GenerationConfigurationError("cached vLLM engine config changed")
        if self._engine.has_unfinished_requests():
            raise GenerationConfigurationError(
                "vLLM engine contains unfinished requests"
            )
        expected: set[str] = set()
        for request in requests:
            plan = request.plan
            question = self._questions.get(plan.original_question_id)
            if (
                question is None
                or question.question_content_hash != plan.question_content_hash
                or question.source is not plan.source
            ):
                raise GenerationConfigurationError(
                    "candidate question identity is unavailable"
                )
            messages = list(render_messages(question))
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            if not isinstance(prompt, str) or not prompt:
                raise GenerationConfigurationError(
                    "tokenizer returned an invalid prompt"
                )
            if request.request_id in expected:
                raise GenerationConfigurationError(
                    "duplicate generation request identity"
                )
            expected.add(request.request_id)
            sampling = self._module.SamplingParams(
                n=1,
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.max_new_tokens,
                seed=request_seed(
                    config.seed,
                    plan.generation_run_id,
                    plan.candidate_id,
                ),
            )
            self._engine.add_request(request.request_id, prompt, sampling)

        completed: dict[str, object] = {}
        while self._engine.has_unfinished_requests():
            for output in self._engine.step():
                request_id = getattr(output, "request_id", None)
                if request_id not in expected:
                    raise GenerationConfigurationError(
                        "vLLM returned a foreign request identity"
                    )
                if not getattr(output, "finished", False):
                    continue
                if request_id in completed:
                    raise GenerationConfigurationError(
                        "vLLM returned a duplicate request identity"
                    )
                completed[request_id] = output
        if set(completed) != expected:
            raise GenerationConfigurationError(
                "vLLM did not return every planned request"
            )
        return list(completed.values())


class CachedEngineFactory:
    """Create one native engine and reuse it for every batch in one process."""

    def __init__(self, questions: Sequence[QuestionRecord], builder=NativeVLLMEngine):
        self._questions = _validated_questions(
            {question.original_question_id: question for question in questions}
        )
        self._builder = builder
        self._engine = None
        self._config = None

    def __call__(self, vllm_module, config: GenerationConfig):
        if self._engine is None:
            self._config = config
            self._engine = self._builder(vllm_module, config, self._questions)
        elif config != self._config:
            raise GenerationConfigurationError(
                "one generator process cannot reuse an engine for another config"
            )
        return self._engine


def run_generator_shards(
    plan: GenerationPlan,
    generator_id: str,
    output_root,
    questions: Sequence[QuestionRecord],
    *,
    engine_factory: CachedEngineFactory | None = None,
    recover_stale_lock: bool = False,
):
    """Run every shard for one generator through the existing resume runner."""
    factory = engine_factory or CachedEngineFactory(questions)
    backend = VLLMGenerationBackend(engine_factory=factory)
    summaries = []
    prefix = f"{generator_id}/"
    for composite_shard_id in plan.shard_ids:
        if not composite_shard_id.startswith(prefix):
            continue
        shard_id = composite_shard_id[len(prefix) :]
        shard = build_generation_shard_plan(plan, generator_id, shard_id)
        summaries.append(
            run_generation_shard(
                shard,
                backend,
                Path(output_root) / generator_id / shard_id,
                recover_stale_lock=recover_stale_lock,
            )
        )
    if not summaries:
        raise GenerationConfigurationError("generator has no planned shards")
    return tuple(summaries)


def verify_candidate_records(
    questions: Sequence[QuestionRecord],
    candidates: Sequence[CandidateRecord],
    *,
    generation_manifest_hash: str,
    gold_timeout_seconds: float,
    candidate_timeout_seconds: float,
) -> tuple[
    str,
    tuple[QuestionVerificationRecord, ...],
    tuple[VerificationRecord, ...],
]:
    """Connect real candidate records to the approved source verifiers."""
    from scripts_permstudy.data.verify_candidates import verification_config

    config = verification_config(
        generation_manifest_hash,
        gold_timeout_seconds,
        candidate_timeout_seconds,
    )
    verification_run_id = run_id("verification", config)
    materialized = tuple(candidates)
    gold_records: list[QuestionVerificationRecord] = []
    verified_records: list[VerificationRecord] = []
    for question in sorted(questions, key=lambda item: item.original_question_id):
        group = tuple(
            candidate
            for candidate in materialized
            if candidate.plan.original_question_id == question.original_question_id
        )
        if question.source is Source.RECLOR:
            gold, verified = verify_reclor_question(
                question,
                group,
                verification_run_id=verification_run_id,
                verifier_timeout_seconds=candidate_timeout_seconds,
            )
        else:
            gold, verified = verify_math_question(
                question,
                group,
                gold_timeout_seconds,
                candidate_timeout_seconds,
                verification_run_id=verification_run_id,
            )
        gold_records.append(gold)
        verified_records.extend(verified)
    return verification_run_id, tuple(gold_records), tuple(verified_records)


def build_pairs_and_permutations(
    questions: Sequence[QuestionRecord],
    candidates: Sequence[CandidateRecord],
    verifications: Sequence[VerificationRecord],
    tokenizer,
    tokenizer_revision: str,
    *,
    generation_manifest_hash: str,
    verification_manifest_hash: str,
    pair_manifest_hash: str,
) -> tuple[tuple[PairRecord, ...], tuple[PermutationRecord, ...]]:
    """Connect verified real records to existing pair and AB/BA builders."""
    by_candidate = {candidate.plan.key: candidate for candidate in candidates}
    joined_by_question: dict[str, list[tuple[CandidateRecord, VerificationRecord]]] = {}
    for verification in verifications:
        key = (verification.generation_run_id, verification.candidate_id)
        candidate = by_candidate.get(key)
        if candidate is None:
            raise ValueError("verification has no matching composite candidate")
        joined_by_question.setdefault(verification.original_question_id, []).append(
            (candidate, verification)
        )
    pair_bindings = {
        "generation": generation_manifest_hash,
        "verification": verification_manifest_hash,
    }
    selected: list[PairRecord] = []
    for question in sorted(questions, key=lambda item: item.original_question_id):
        pair = pairs.select_pair(
            question,
            tuple(joined_by_question.get(question.original_question_id, ())),
            tokenizer,
            tokenizer_revision,
            upstream_bindings=pair_bindings,
        )
        if pair is not None:
            selected.append(pair)
    permutation_records = tuple(
        record
        for pair in selected
        for record in permutations.build_permutations(
            pair,
            upstream_bindings={"pairs": pair_manifest_hash},
        )
    )
    return tuple(selected), permutation_records


def _validated_questions(
    questions: Mapping[str, QuestionRecord],
) -> dict[str, QuestionRecord]:
    materialized = dict(questions)
    for identity, question in materialized.items():
        if not isinstance(question, QuestionRecord):
            raise TypeError("questions must contain QuestionRecord values")
        question.validate()
        if identity != question.original_question_id:
            raise ValueError("question mapping identity differs from record")
    return materialized


def _validate_phase2_config(config: GenerationConfig) -> None:
    if not isinstance(config, GenerationConfig):
        raise TypeError("config must be a GenerationConfig")
    config.validate()
    if config.backend != "vllm":
        raise GenerationConfigurationError("Phase 2 requires backend='vllm'")
    if (
        config.prompt_template_revision != PROMPT_TEMPLATE_REVISION
        or config.prompt_template_hash != prompt_contract_hash()
    ):
        raise GenerationConfigurationError("Phase 2 prompt contract does not match")


def load_phase2_split(root: Path, manifest: Mapping[str, object]):
    """Load the approved 20+20 private smoke through the frozen split validator."""
    from scripts_permstudy.data import _common as io
    from scripts_permstudy.data.plan_generation import load_split

    _, smoke = load_split(root, manifest)
    questions = io.records(root, manifest, "questions.jsonl", QuestionRecord)
    by_id = {question.original_question_id: question for question in questions}
    selected = tuple(by_id[item.original_question_id] for item in smoke)
    if len(selected) != 40 or {
        source: sum(question.source is source for question in selected)
        for source in Source
    } != {Source.MATH: 20, Source.RECLOR: 20}:
        raise io.IntegrityError()
    return selected, tuple(smoke)


def _generation_plan(
    split_manifest: Mapping[str, object],
    smoke: Sequence,
) -> GenerationPlan:
    configs = build_real_generation_configs(split_manifest["split_manifest_hash"])
    return plan_generation(
        smoke,
        configs,
        samples_per_question=2,
        shard_size=8,
        split_upstream_manifest_hash=split_manifest["output_manifest_hash"],
    )


def _generation_artifacts(root: Path, plan: GenerationPlan, questions):
    from scripts_permstudy.data import _common as io

    prefix = f"generation/{plan.generation_run_id}"
    refs = [
        io.persist_records(root, f"{prefix}/plans.jsonl", plan.candidates),
        io.persist_records(root, f"{prefix}/questions.jsonl", questions),
    ]
    candidates: list[CandidateRecord] = []
    failures: list[FailureRecord] = []
    for composite in plan.shard_ids:
        generator_id, shard_id = composite.split("/", 1)
        directory = root / prefix / "shards" / generator_id / shard_id
        if not (directory / "manifest.json").exists():
            continue
        shard = build_generation_shard_plan(plan, generator_id, shard_id)
        run_generation_shard(shard, _CompletedShardOnlyBackend(), directory)
        manifest = io.read_json(directory / "manifest.json")
        if (
            manifest.get("generation_run_id") != plan.generation_run_id
            or manifest.get("generator_id") != generator_id
            or manifest.get("shard_id") != shard_id
        ):
            raise io.IntegrityError()
        for basename, record_type, destination in (
            ("candidates.jsonl", CandidateRecord, candidates),
            ("failures.jsonl", FailureRecord, failures),
        ):
            path = directory / basename
            scan = io.scan_jsonl(path)
            decoded = [
                io.checked(
                    record_type.from_dict,
                    {key: value for key, value in row.items() if key != "record_hash"},
                    integrity=True,
                )
                for row in scan.records
            ]
            destination.extend(decoded)
            refs.append(io.artifact(root, path, len(decoded)))
    expected = {candidate.key: candidate for candidate in plan.candidates}
    if len({record.plan.key for record in candidates}) != len(candidates) or any(
        expected.get(record.plan.key) != record.plan
        for record in (*candidates, *failures)
    ):
        raise io.IntegrityError()
    return refs, tuple(candidates), tuple(failures)


class _CompletedShardOnlyBackend:
    def generate(self, batch, config):
        raise GenerationConfigurationError("generation shard is incomplete")


def _load_real_generation(
    root: Path,
    manifest: Mapping[str, object],
    split_loader,
):
    from scripts_permstudy.data import _common as io

    split = io.upstream(root, manifest, "split")
    questions, smoke = split_loader(root, split)
    plan = _generation_plan(split, smoke)
    if (
        manifest.get("typed_config")
        != generation_stage_config(
            plan.generator_configs,
            samples_per_question=2,
            shard_size=8,
            split_upstream_manifest_hash=split["output_manifest_hash"],
        )
        or manifest.get("run_id") != plan.generation_run_id
        or io.records(root, manifest, "plans.jsonl", type(plan.candidates[0]))
        != list(plan.candidates)
        or io.records(root, manifest, "questions.jsonl", QuestionRecord)
        != list(questions)
    ):
        raise io.IntegrityError()
    candidates = tuple(io.records(root, manifest, "candidates.jsonl", CandidateRecord))
    expected = {candidate.key: candidate for candidate in plan.candidates}
    if (
        len(candidates) != len(expected)
        or len({candidate.plan.key for candidate in candidates}) != len(candidates)
        or any(
            expected.get(candidate.plan.key) != candidate.plan
            for candidate in candidates
        )
        or manifest.get("candidate_set_hash") != semantic_candidate_set_hash(candidates)
    ):
        raise io.IntegrityError()
    return split, plan, tuple(questions), candidates


def _load_real_verification(root, manifest, split_loader):
    from scripts_permstudy.data import _common as io
    from scripts_permstudy.data.verify_candidates import verification_config

    generation = io.upstream(root, manifest, "generation")
    split, plan, questions, candidates = _load_real_generation(
        root, generation, split_loader
    )
    parameters = manifest["typed_config"]["parameters"]
    config = verification_config(
        generation["output_manifest_hash"],
        parameters["gold_timeout_seconds"],
        parameters["candidate_timeout_seconds"],
    )
    gold = tuple(
        io.records(
            root,
            manifest,
            "question_verifications.jsonl",
            QuestionVerificationRecord,
        )
    )
    verified = tuple(
        io.records(root, manifest, "verifications.jsonl", VerificationRecord)
    )
    if config != manifest["typed_config"] or manifest["run_id"] != run_id(
        "verification", config
    ):
        raise io.IntegrityError()
    by_question = {question.original_question_id: question for question in questions}
    by_gold = {record.original_question_id: record for record in gold}
    by_candidate = {candidate.plan.key: candidate for candidate in candidates}
    if len(by_gold) != len(gold) or set(by_gold) != set(by_question):
        raise io.IntegrityError()
    expected_verified = {
        key
        for key, candidate in by_candidate.items()
        if by_gold[candidate.plan.original_question_id].gold_parse_status == "ok"
    }
    actual_verified = {
        (record.generation_run_id, record.candidate_id) for record in verified
    }
    if len(actual_verified) != len(verified) or actual_verified != expected_verified:
        raise io.IntegrityError()
    return generation, split, plan, questions, candidates, gold, verified


def build_parser():
    from scripts_permstudy.data import _common as io

    parser = io.parser("run_real_smoke", common=False)
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate")
    io.add_common(generate)
    generate.add_argument("--split-manifest", required=True)
    generate.add_argument(
        "--generator-id",
        choices=tuple(item[0] for item in _REAL_GENERATOR_RECIPE),
        required=True,
    )
    generate.add_argument("--recover-stale-lock", action="store_true")

    verify = commands.add_parser("verify")
    io.add_common(verify)
    verify.add_argument("--generation-manifest", required=True)
    verify.add_argument("--gold-timeout-seconds", type=float, default=30.0)
    verify.add_argument("--candidate-timeout-seconds", type=float, default=15.0)

    finalize = commands.add_parser("finalize")
    io.add_common(finalize)
    finalize.add_argument("--verification-manifest", required=True)
    finalize.add_argument("--audit-manifest", required=True)
    finalize.add_argument("--decisions", required=True)
    finalize.add_argument(
        "--tokenizer-revision", choices=(PAIR_TOKENIZER_REVISION,), required=True
    )
    return parser


def _dispatch_generate(
    args,
    root: Path,
    *,
    split_loader,
    engine_factory_builder,
):
    from scripts_permstudy.data import _common as io

    split = io.load_manifest(root, args.split_manifest, "split")
    questions, smoke = split_loader(root, split)
    plan = _generation_plan(split, smoke)
    config = generation_stage_config(
        plan.generator_configs,
        samples_per_question=2,
        shard_size=8,
        split_upstream_manifest_hash=split["output_manifest_hash"],
    )
    factory = (
        engine_factory_builder(questions)
        if engine_factory_builder is not None
        else CachedEngineFactory(questions)
    )
    run_generator_shards(
        plan,
        args.generator_id,
        root / "generation" / plan.generation_run_id / "shards",
        questions,
        engine_factory=factory,
        recover_stale_lock=args.recover_stale_lock,
    )
    refs, candidates, failures = _generation_artifacts(root, plan, questions)
    if len(candidates) == len(plan.candidates):
        manifest = io.persist_manifest(
            root,
            "generation",
            config,
            refs,
            {
                "planned": len(plan.candidates),
                "successful": len(candidates),
                "historical_failures": len(failures),
            },
            {"split": args.split_manifest},
            extra={
                "split_manifest_hash": split["split_manifest_hash"],
                "candidate_set_hash": semantic_candidate_set_hash(candidates),
            },
        )
        return {"run_id": manifest["run_id"], "count": len(candidates)}
    return {"run_id": plan.generation_run_id, "count": len(candidates)}


def _dispatch_verify(args, root: Path, *, split_loader):
    from permstudy.data_pipeline import audit
    from scripts_permstudy.data import _common as io
    from scripts_permstudy.data.verify_candidates import verification_config

    generation = io.load_manifest(root, args.generation_manifest, "generation")
    _, plan, questions, candidates = _load_real_generation(
        root, generation, split_loader
    )
    verification_run_id, gold, verified = verify_candidate_records(
        questions,
        candidates,
        generation_manifest_hash=generation["output_manifest_hash"],
        gold_timeout_seconds=args.gold_timeout_seconds,
        candidate_timeout_seconds=args.candidate_timeout_seconds,
    )
    config = verification_config(
        generation["output_manifest_hash"],
        args.gold_timeout_seconds,
        args.candidate_timeout_seconds,
    )
    prefix = f"verification/{verification_run_id}"
    refs = [
        io.persist_records(root, f"{prefix}/question_verifications.jsonl", gold),
        io.persist_records(root, f"{prefix}/verifications.jsonl", verified),
    ]
    verification = io.persist_manifest(
        root,
        "verification",
        config,
        refs,
        {"questions": len(gold), "candidates": len(verified)},
        {"generation": args.generation_manifest},
    )
    selection = audit.build_audit_selection(
        verified,
        gold,
        plan.generation_run_id,
        audit_seed=42,
        max_per_cell=5,
        verification_manifest_hash=verification["output_manifest_hash"],
    )
    snapshot = audit.verification_snapshot_hash(verified, gold)
    audit_config = audit.audit_stage_config(
        plan.generation_run_id,
        verification_run_id,
        snapshot,
        audit_seed=42,
        max_per_cell=5,
        verification_manifest_hash=verification["output_manifest_hash"],
    )
    audit_identity = run_id("audit", audit_config)
    if any(record.audit_run_id != audit_identity for record in selection):
        raise io.IntegrityError()
    audit_ref = io.persist_records(
        root, f"audit/{audit_identity}/selection.jsonl", selection
    )
    audit_manifest = io.persist_manifest(
        root,
        "audit",
        audit_config,
        [audit_ref],
        {"selected": len(selection)},
        {"verification": f"verification/{verification_run_id}/manifest.json"},
    )
    return io.output(audit_manifest)


def _load_decisions(root: Path, reference: str):
    from scripts_permstudy.data import _common as io

    scan = io.scan_jsonl(io.rooted(root, reference))
    return tuple(
        io.checked(
            AuditDecision.from_dict,
            {key: value for key, value in row.items() if key != "record_hash"},
            integrity=True,
        )
        for row in scan.records
    )


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _dispatch_finalize(args, root: Path, *, split_loader, tokenizer_loader):
    from permstudy.data_pipeline import audit, gates, trainer_export
    from scripts_permstudy.data import _common as io
    from scripts_permstudy.data.audit_generator_distribution import summary_payload
    from scripts_permstudy.data.build_reasoning_pairs import (
        load_tokenizer,
        validate_tokenizer,
    )

    verification = io.load_manifest(root, args.verification_manifest, "verification")
    (
        generation,
        split,
        plan,
        questions,
        candidates,
        gold,
        verified,
    ) = _load_real_verification(root, verification, split_loader)
    audit_manifest = io.load_manifest(root, args.audit_manifest, "audit")
    if (
        audit_manifest["upstream_bindings"]["verification"]
        != verification["output_manifest_hash"]
    ):
        raise io.IntegrityError()
    parameters = audit_manifest["typed_config"]["parameters"]
    selection = audit.build_audit_selection(
        verified,
        gold,
        plan.generation_run_id,
        audit_seed=parameters["audit_seed"],
        max_per_cell=parameters["max_per_cell"],
        verification_manifest_hash=verification["output_manifest_hash"],
    )
    if (
        io.records(root, audit_manifest, "selection.jsonl", AuditSelectionRecord)
        != selection
    ):
        raise io.IntegrityError()
    decisions = _load_decisions(root, args.decisions)
    summary = audit.validate_audit_decisions(selection, decisions)
    audit_prefix = f"audit/{audit_manifest['run_id']}"
    completed_refs = [
        io.persist_records(root, f"{audit_prefix}/selection.jsonl", selection),
        io.persist_records(root, f"{audit_prefix}/decisions.jsonl", decisions),
        io.persist_bytes(
            root,
            f"{audit_prefix}/summary.json",
            canonical_json_bytes(summary_payload(summary)),
            1,
        ),
    ]
    completed_audit = io.persist_manifest(
        root,
        "audit",
        audit_manifest["typed_config"],
        completed_refs,
        {"selected": len(selection), "completed": summary.completed_count},
        {"verification": args.verification_manifest},
        filename="completed.json",
    )

    tokenizer = (
        tokenizer_loader(args.tokenizer_revision, root)
        if tokenizer_loader is not None
        else load_tokenizer(args.tokenizer_revision, root)
    )
    if tokenizer_loader is None:
        validate_tokenizer(tokenizer, args.tokenizer_revision)
    pair_bindings = {
        "generation": generation["output_manifest_hash"],
        "verification": verification["output_manifest_hash"],
    }
    by_candidate = {candidate.plan.key: candidate for candidate in candidates}
    selected_pairs = []
    for question in questions:
        joined = [
            (by_candidate[(record.generation_run_id, record.candidate_id)], record)
            for record in verified
            if record.original_question_id == question.original_question_id
        ]
        pair = pairs.select_pair(
            question,
            joined,
            tokenizer,
            args.tokenizer_revision,
            upstream_bindings=pair_bindings,
        )
        if pair is not None:
            selected_pairs.append(pair)
    pair_config = pairs.pair_stage_config(args.tokenizer_revision, pair_bindings)
    pair_identity = run_id("pairs", pair_config)
    pair_ref = io.persist_records(
        root, f"pairs/{pair_identity}/pairs.jsonl", selected_pairs
    )
    pair_manifest = io.persist_manifest(
        root,
        "pairs",
        pair_config,
        [pair_ref],
        {"pairs": len(selected_pairs)},
        {
            "generation": verification["upstream_manifests"]["generation"][
                "relative_path"
            ],
            "verification": args.verification_manifest,
        },
    )

    permutation_bindings = {"pairs": pair_manifest["output_manifest_hash"]}
    permutation_config = permutations.permutation_stage_config(permutation_bindings)
    permutation_identity = run_id("permutations", permutation_config)
    permutation_records = tuple(
        record
        for pair in selected_pairs
        for record in permutations.build_permutations(
            pair, upstream_bindings=permutation_bindings
        )
    )
    permutation_ref = io.persist_records(
        root,
        f"permutations/{permutation_identity}/permutations.jsonl",
        permutation_records,
    )
    permutation_manifest = io.persist_manifest(
        root,
        "permutations",
        permutation_config,
        [permutation_ref],
        {"permutations": len(permutation_records)},
        {"pairs": f"pairs/{pair_identity}/manifest.json"},
    )

    gate_inputs = gates.GateInputs(
        plans=plan.candidates,
        candidates=candidates,
        verifications=verified,
        question_verifications=gold,
        pairs=tuple(selected_pairs),
        permutations=permutation_records,
        audit_summary=summary,
    )
    functional = gates.evaluate_functional_gate(gate_inputs)
    statistical = gates.evaluate_statistical_gate(gate_inputs)
    status = statistical.status.value if functional.passed else "FAIL"
    report = {
        "functional_failures": list(functional.failures),
        "functional_metrics": _plain(functional.metrics),
        "statistical_diagnostics": _plain(statistical.diagnostics),
        "statistical_hard_failures": list(statistical.hard_failures),
        "statistical_warnings": list(statistical.warnings),
        "status": status,
    }
    io.persist_bytes(
        root,
        f"phase2/{plan.generation_run_id}/gate-report.json",
        canonical_json_bytes(report),
        1,
    )

    manifests = {
        "split": split,
        "generation": generation,
        "pairs": pair_manifest,
        "permutations": permutation_manifest,
    }
    export = trainer_export.export_trainer_parquet(manifests, root)
    export_config = trainer_export.export_stage_config(
        {role: manifest["output_manifest_hash"] for role, manifest in manifests.items()}
    )
    export_manifest = io.persist_manifest(
        root,
        "trainer_export",
        export_config,
        [export.dataset_artifact],
        {"rows": export.row_count},
        {
            "split": generation["upstream_manifests"]["split"]["relative_path"],
            "generation": verification["upstream_manifests"]["generation"][
                "relative_path"
            ],
            "pairs": f"pairs/{pair_identity}/manifest.json",
            "permutations": f"permutations/{permutation_identity}/manifest.json",
        },
        extra={
            "prompt_template_hash": export.prompt_template_hash,
            "dataset_sha256": export.dataset_artifact.sha256,
            "audit_manifest_hash": completed_audit["output_manifest_hash"],
        },
    )
    return io.output(export_manifest)


def dispatch(
    args,
    root: Path,
    *,
    split_loader=load_phase2_split,
    engine_factory_builder=None,
    tokenizer_loader=None,
):
    if args.command == "generate":
        return _dispatch_generate(
            args,
            root,
            split_loader=split_loader,
            engine_factory_builder=engine_factory_builder,
        )
    if args.command == "verify":
        return _dispatch_verify(args, root, split_loader=split_loader)
    if args.command == "finalize":
        return _dispatch_finalize(
            args,
            root,
            split_loader=split_loader,
            tokenizer_loader=tokenizer_loader,
        )
    raise ValueError("unknown Phase 2 command")


def main(
    argv: Sequence[str] | None = None,
    *,
    split_loader=load_phase2_split,
    engine_factory_builder=None,
    tokenizer_loader=None,
) -> int:
    from scripts_permstudy.data import _common as io

    return io.execute(
        build_parser,
        lambda args, root: dispatch(
            args,
            root,
            split_loader=split_loader,
            engine_factory_builder=engine_factory_builder,
            tokenizer_loader=tokenizer_loader,
        ),
        argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
