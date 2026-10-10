"""Thin Phase 2 orchestration over the frozen Phase 1 data-pipeline APIs.

The module deliberately has no eager vLLM import.  P1 exercises it with an
injected CPU fake; P2 may construct the same adapter in the approved Linux GPU
process without changing any Phase 1 implementation.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

from permstudy.data_pipeline import pairs, permutations
from permstudy.data_pipeline.canonical import canonical_json_bytes, sha256_hex
from permstudy.data_pipeline.generation import (
    GenerationConfig,
    GenerationConfigurationError,
    GenerationPlan,
    GenerationRequest,
    VLLMGenerationBackend,
    build_generation_shard_plan,
    run_generation_shard,
)
from permstudy.data_pipeline.ids import run_id
from permstudy.data_pipeline.schema import (
    CandidateRecord,
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
