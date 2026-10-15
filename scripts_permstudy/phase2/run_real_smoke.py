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
LLAMA_RECLOR_PROMPT_REVISION = f"phase2_llama_reclor_prompt_v2+{REQUEST_SEED_VERSION}"
MATH_COMPLETION_PROMPT_REVISION = f"phase2_math_completion_v2+{REQUEST_SEED_VERSION}"
PRODUCTION_SELECTION_VERSION = "fresh_train_stratified_200_seed42_v1"
_PRODUCTION_SOURCES = {
    "math-production-200-v1": Source.MATH,
    "reclor-production-200-v1": Source.RECLOR,
}
_RECIPE_PROMPT_REVISIONS = {
    "smoke-v1": PROMPT_TEMPLATE_REVISION,
    "llama-reclor-prompt-v2": LLAMA_RECLOR_PROMPT_REVISION,
    "math-completion-v2": MATH_COMPLETION_PROMPT_REVISION,
    **{
        recipe: f"phase2_{source.value}_production_200_v1+{REQUEST_SEED_VERSION}"
        for recipe, source in _PRODUCTION_SOURCES.items()
    },
}
MATH_COMPLETION_SYSTEM_PROMPT = (
    "Solve the problem carefully and keep the reasoning concise. End with "
    "exactly one final answer in \\boxed{...}. Do not write anything after "
    "the boxed answer."
)
# Fixed diagnostic selection: approved smoke minus the two already-paired
# questions and the quarantined option-label gold. IDs, not source text.
MATH_COMPLETION_SPLIT_HASH = (
    "94f6dabb281daf22723c3797244c17a9c8fd520a1688c0f509e5d2c5fd1a3bde"
)
MATH_COMPLETION_RETEST_IDS = (
    "math:train:1bf2b23575c2e412cf0149305f47a340eb55d155",
    "math:train:3266eb2e025c6866e8c4e30070a6a26f367ee025",
    "math:train:5cf253158c853c357b77f208519c159106940b5c",
    "math:train:6742b76a3290968746809c479bd3ca8357c05989",
    "math:train:70f3eba20520d155758ed14abf4448f78468391b",
    "math:train:760176fd785b4ee03b6bf68fbbcd6c58aa511e56",
    "math:train:7c95aa4e07619f79c586f2fa56d621d52ea3375b",
    "math:train:8ae4eddfc07d1116f10f59e785249fcb478e64a9",
    "math:train:be6d471e15e916a14db87d19e335f5a635955d08",
    "math:train:c8929edfef69d84efe198556b890c5b6e2aedb88",
    "math:train:c9335349e6cf21dff658dc76461b713041f20007",
    "math:train:cf83cb51d66ec13ff259db96efc5a2dec37089d5",
    "math:train:d02975c0642f250ff98d0f9e407ff1dba7ff575b",
    "math:train:d2d74c73d9cc0d08e95db26e791ff8bbcb1a1c1e",
    "math:train:d88cd1711c78a6d6965246533914cd6ffdfd6abf",
    "math:train:f706f439c308876702e1871c4ee566718739510b",
    "math:train:ff91a3878a8b7f2af8c626f7a5b8ecc13c76c5d1",
)
# Seed-42 selection outside the original smoke: one per category, then
# additional distinct category/level cells by stable SHA256 rank.
MATH_COMPLETION_NEW_IDS = (
    "math:train:127f3a8c152fb51fc2a2dc2eb7df360a78a64742",
    "math:train:239b6cb99dc7a40ab9b018d0482269a707a848a7",
    "math:train:2c02d04046ce7e74a9b291b79295cfb4584092d7",
    "math:train:493954c39f42acda9726cac2338021679550cedb",
    "math:train:6646fe2099094014c0b1b1c25fb3f53495fc9d46",
    "math:train:69a6f35ed6074d78ee273b21a4c63a8585910dee",
    "math:train:96257624c9a0d569e4ec491f2b1327be839c6613",
    "math:train:be5c3dd5a19992cfb6c9831f6cbce9d95c247812",
    "math:train:d6b790dfa830b84dff43c823e3afbc831d490e43",
    "math:train:fe9dbacfd264d177bf3e1e3b9305c26db0425b50",
)
MATH_COMPLETION_TARGET_IDS = MATH_COMPLETION_RETEST_IDS + MATH_COMPLETION_NEW_IDS
RECLOR_OUTPUT_RULES = (
    "\n\nYou may provide reasoning before the final answer.\n"
    "Your final line MUST be exactly:\nFinal Answer: X\n"
    "Replace X with one uppercase letter: A, B, C, or D.\n"
    "Do not output anything after the final answer line."
)

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
    *,
    recipe: str = "smoke-v1",
) -> str:
    """Hash the exact prompt text together with request-seed semantics."""
    if not isinstance(request_seed_version, str) or not request_seed_version.strip():
        raise ValueError("request_seed_version must be nonempty text")
    if recipe not in _RECIPE_PROMPT_REVISIONS:
        raise ValueError("unknown generation recipe")
    selection = (
        {
            "math_target_question_ids": sorted(MATH_COMPLETION_TARGET_IDS),
            "math_retest_ids": sorted(MATH_COMPLETION_RETEST_IDS),
            "math_fresh_ids": sorted(MATH_COMPLETION_NEW_IDS),
            "fresh_selection_seed": 42,
        }
        if recipe == "math-completion-v2"
        else {}
    )
    if recipe in _PRODUCTION_SOURCES:
        selection = {
            "selection_algorithm": PRODUCTION_SELECTION_VERSION,
            "selection_seed": 42,
            "per_source": 200,
            "source": _PRODUCTION_SOURCES[recipe].value,
            "exclude_original_smoke": True,
            "exclude_math_fresh_ids": sorted(MATH_COMPLETION_NEW_IDS),
        }
    return sha256_hex(
        canonical_json_bytes(
            {
                "math_system": (
                    MATH_COMPLETION_SYSTEM_PROMPT
                    if recipe in ("math-completion-v2", "math-production-200-v1")
                    else MATH_SYSTEM_PROMPT
                ),
                "math_user": "{problem}",
                "reclor_system": RECLOR_SYSTEM_PROMPT,
                "reclor_user": (
                    "Context:\n{context}\n\nQuestion:\n{question}\n\n"
                    "A. {answer_a}\nB. {answer_b}\nC. {answer_c}\nD. {answer_d}"
                )
                + (
                    RECLOR_OUTPUT_RULES
                    if recipe in ("llama-reclor-prompt-v2", "reclor-production-200-v1")
                    else ""
                ),
                "request_seed_version": request_seed_version,
                **selection,
            }
        )
    )


def build_real_generation_configs(
    split_manifest_hash: str,
    *,
    recipe: str = "smoke-v1",
) -> tuple[GenerationConfig, ...]:
    """Build one fixed smoke/diagnostic recipe without changing Phase 1 schema."""
    if (
        recipe in ("math-completion-v2", *_PRODUCTION_SOURCES)
        and split_manifest_hash != MATH_COMPLETION_SPLIT_HASH
    ):
        raise GenerationConfigurationError(
            "MATH diagnostic requires the approved split"
        )
    prompt_hash = prompt_contract_hash(recipe=recipe)
    diagnostic = recipe == "llama-reclor-prompt-v2"
    return tuple(
        GenerationConfig(
            backend="vllm",
            generator_id=generator_id,
            model_repository=repository,
            model_revision=revision,
            prompt_template_revision=_RECIPE_PROMPT_REVISIONS[recipe],
            prompt_template_hash=prompt_hash,
            temperature=0.8,
            top_p=0.95,
            max_new_tokens=1024
            if recipe in ("math-completion-v2", "math-production-200-v1")
            else 512,
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
        if not diagnostic or generator_id == "llama-3.1-8b-instruct"
        if recipe not in ("math-completion-v2", *_PRODUCTION_SOURCES)
        or generator_id != "qwen2.5-32b-instruct"
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


def render_messages(
    question: QuestionRecord,
    *,
    prompt_template_revision: str = PROMPT_TEMPLATE_REVISION,
) -> tuple[dict[str, str], dict[str, str]]:
    """Render the approved source prompt without source gold or solution text."""
    if not isinstance(question, QuestionRecord):
        raise TypeError("question must be a QuestionRecord")
    question.validate()
    if prompt_template_revision not in _RECIPE_PROMPT_REVISIONS.values():
        raise GenerationConfigurationError("unknown prompt revision")
    diagnostic = prompt_template_revision == LLAMA_RECLOR_PROMPT_REVISION
    reclor_production = (
        prompt_template_revision == _RECIPE_PROMPT_REVISIONS["reclor-production-200-v1"]
    )
    if (diagnostic or reclor_production) and question.source is not Source.RECLOR:
        raise GenerationConfigurationError("Llama prompt diagnostic requires ReClor")
    math_completion = prompt_template_revision in (
        MATH_COMPLETION_PROMPT_REVISION,
        _RECIPE_PROMPT_REVISIONS["math-production-200-v1"],
    )
    if math_completion and question.source is not Source.MATH:
        raise GenerationConfigurationError("MATH completion diagnostic requires MATH")
    if question.source is Source.MATH:
        user = question.problem
        system = (
            MATH_COMPLETION_SYSTEM_PROMPT if math_completion else MATH_SYSTEM_PROMPT
        )
    else:
        answer_a, answer_b, answer_c, answer_d = question.answers
        user = (
            f"Context:\n{question.context}\n\nQuestion:\n{question.question}\n\n"
            f"A. {answer_a}\nB. {answer_b}\nC. {answer_c}\nD. {answer_d}"
        )
        system = RECLOR_SYSTEM_PROMPT
        if diagnostic or reclor_production:
            user += RECLOR_OUTPUT_RULES
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
        self._tokenizer = self._engine.get_tokenizer_group().get_lora_tokenizer()

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
            messages = list(
                render_messages(
                    question, prompt_template_revision=config.prompt_template_revision
                )
            )
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
    recipe = next(
        (
            name
            for name, revision in _RECIPE_PROMPT_REVISIONS.items()
            if revision == config.prompt_template_revision
        ),
        None,
    )
    if (
        recipe is None
        or (
            recipe == "llama-reclor-prompt-v2"
            and config.generator_id != "llama-3.1-8b-instruct"
        )
        or (
            recipe == "math-completion-v2"
            and (
                config.generator_id
                not in ("qwen2.5-7b-instruct", "llama-3.1-8b-instruct")
                or config.max_new_tokens != 1024
                or config.split_manifest_hash != MATH_COMPLETION_SPLIT_HASH
            )
        )
        or config.prompt_template_hash != prompt_contract_hash(recipe=recipe)
    ):
        raise GenerationConfigurationError("Phase 2 prompt contract does not match")
    if recipe in _PRODUCTION_SOURCES and config not in build_real_generation_configs(
        config.split_manifest_hash, recipe=recipe
    ):
        raise GenerationConfigurationError(
            "production config differs from approved recipe"
        )


def load_phase2_split(root: Path, manifest: Mapping[str, object], *, recipe="smoke-v1"):
    """Load smoke or fixed MATH cohorts through the frozen split validator."""
    from scripts_permstudy.data import _common as io
    from scripts_permstudy.data.plan_generation import load_split

    result, smoke = load_split(root, manifest)
    questions = io.records(root, manifest, "questions.jsonl", QuestionRecord)
    by_id = {question.original_question_id: question for question in questions}
    if recipe in _PRODUCTION_SOURCES:
        from permstudy.data_pipeline.splitting import select_smoke_questions

        if manifest["split_manifest_hash"] != MATH_COMPLETION_SPLIT_HASH:
            raise io.IntegrityError()
        excluded = {a.original_question_id for a in smoke} | set(
            MATH_COMPLETION_NEW_IDS
        )
        # Reuse the existing proportional-stratum allocation. This selects a
        # subset of the frozen training split; it never creates another split.
        fresh = select_smoke_questions(
            tuple(
                a for a in result.assignments if a.original_question_id not in excluded
            ),
            per_source=200,
            split_seed=42,
        )
        chosen = tuple(a for a in fresh if a.source is _PRODUCTION_SOURCES[recipe])
        return tuple(by_id[a.original_question_id] for a in chosen), chosen
    if recipe == "math-completion-v2":
        smoke_ids = {a.original_question_id for a in smoke}
        chosen = tuple(
            a
            for a in result.assignments
            if a.original_question_id in MATH_COMPLETION_TARGET_IDS
        )
        if (
            len(chosen) != 27
            or {a.original_question_id for a in chosen}
            != set(MATH_COMPLETION_TARGET_IDS)
            or any(
                a.source is not Source.MATH or a.split.value != "train" for a in chosen
            )
            or not set(MATH_COMPLETION_RETEST_IDS) <= smoke_ids
            or not set(MATH_COMPLETION_NEW_IDS).isdisjoint(smoke_ids)
        ):
            raise io.IntegrityError()
        return tuple(by_id[a.original_question_id] for a in chosen), chosen
    selected = tuple(by_id[item.original_question_id] for item in smoke)
    if len(selected) != 40 or {
        source: sum(question.source is source for question in selected)
        for source in Source
    } != {Source.MATH: 20, Source.RECLOR: 20}:
        raise io.IntegrityError()
    return selected, tuple(smoke)


def _load_recipe_split(root, manifest, recipe, split_loader):
    if split_loader is load_phase2_split:
        return split_loader(root, manifest, recipe=recipe)
    return split_loader(root, manifest)


def _generation_plan(
    split_manifest: Mapping[str, object],
    smoke: Sequence,
    *,
    recipe: str = "smoke-v1",
) -> GenerationPlan:
    configs = build_real_generation_configs(
        split_manifest["split_manifest_hash"], recipe=recipe
    )
    if recipe == "llama-reclor-prompt-v2":
        smoke = tuple(item for item in smoke if item.source is Source.RECLOR)
    elif recipe == "math-completion-v2":
        smoke = tuple(
            item
            for item in smoke
            if item.original_question_id in MATH_COMPLETION_TARGET_IDS
        )
        if (
            len(smoke) != 27
            or {item.original_question_id for item in smoke}
            != set(MATH_COMPLETION_TARGET_IDS)
            or any(
                item.source is not Source.MATH or item.split.value != "train"
                for item in smoke
            )
        ):
            raise GenerationConfigurationError(
                "MATH diagnostic question selection differs"
            )
    elif recipe in _PRODUCTION_SOURCES:
        if (
            len(smoke) != 200
            or len({a.original_question_id for a in smoke}) != 200
            or any(
                a.source is not _PRODUCTION_SOURCES[recipe]
                or a.split.value != "train"
                or a.original_question_id in MATH_COMPLETION_NEW_IDS
                for a in smoke
            )
        ):
            raise GenerationConfigurationError("production question selection differs")
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
    configs = manifest["typed_config"]["parameters"]["generators"]
    recipe = next(
        (
            name
            for name, revision in _RECIPE_PROMPT_REVISIONS.items()
            if any(item["prompt_template_revision"] == revision for item in configs)
        ),
        "smoke-v1",
    )
    questions, smoke = _load_recipe_split(root, split, recipe, split_loader)
    plan = _generation_plan(split, smoke, recipe=recipe)
    planned_ids = {item.original_question_id for item in plan.candidates}
    questions = tuple(
        item for item in questions if item.original_question_id in planned_ids
    )
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

    plan = commands.add_parser("plan")
    io.add_common(plan)
    plan.add_argument("--split-manifest", required=True)
    plan.add_argument("--recipe", choices=tuple(_PRODUCTION_SOURCES), required=True)

    training = commands.add_parser("prepare-training")
    io.add_common(training)
    training.add_argument("--trainer-manifest", required=True)

    generate = commands.add_parser("generate")
    io.add_common(generate)
    generate.add_argument("--split-manifest", required=True)
    generate.add_argument(
        "--generator-id",
        choices=tuple(item[0] for item in _REAL_GENERATOR_RECIPE),
        required=True,
    )
    generate.add_argument("--recover-stale-lock", action="store_true")
    generate.add_argument(
        "--recipe", choices=tuple(_RECIPE_PROMPT_REVISIONS), default="smoke-v1"
    )

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


def _dispatch_plan(args, root, *, split_loader):
    """Persist a CPU-only plan using the existing generation manifest schema."""
    from scripts_permstudy.data import _common as io

    split = io.load_manifest(root, args.split_manifest, "split")
    questions, assignments = _load_recipe_split(root, split, args.recipe, split_loader)
    plan = _generation_plan(split, assignments, recipe=args.recipe)
    prefix = f"generation/{plan.generation_run_id}"
    refs = [
        io.persist_records(root, f"{prefix}/plans.jsonl", plan.candidates),
        io.persist_records(root, f"{prefix}/questions.jsonl", questions),
    ]
    config = generation_stage_config(
        plan.generator_configs,
        samples_per_question=2,
        shard_size=8,
        split_upstream_manifest_hash=split["output_manifest_hash"],
    )
    io.persist_manifest(
        root,
        "generation",
        config,
        refs,
        {"planned": len(plan.candidates), "successful": 0},
        {"split": args.split_manifest},
        extra={"split_manifest_hash": split["split_manifest_hash"]},
        filename="planned.json",
    )
    return {"run_id": plan.generation_run_id, "count": len(plan.candidates)}


CONTROLLED_TRAINING_PROMPT_VERSION = "pairwise_judge_answer_tag_v1"
CONTROLLED_TRAINING_SYSTEM = (
    "Compare the two responses for correctness. You may reason before the final answer. "
    "End with exactly one <answer>A</answer> or <answer>B</answer> tag."
)
CONTROLLED_TRAINING_FINAL_RULE = (
    "End with exactly one <answer>A</answer> or <answer>B</answer> tag. "
    "Do not write anything after it."
)


def controlled_training_rows(rows):
    """Adapt only the frozen direct prompt, retaining all semantic identities."""
    from copy import deepcopy
    from permstudy.data_pipeline import trainer_export

    trainer_export.validate_trainer_rows(rows)
    adapted = deepcopy(rows)
    direct_rule = "Answer with A or B only."
    for row in adapted:
        content = row["prompt"][1]["content"]
        if not content.endswith(direct_rule):
            raise ValueError("unrecognized direct training prompt")
        row["prompt"] = [
            {"role": "system", "content": CONTROLLED_TRAINING_SYSTEM},
            {
                "role": "user",
                "content": content[: -len(direct_rule)]
                + CONTROLLED_TRAINING_FINAL_RULE,
            },
        ]
    return adapted


def _dispatch_prepare_training(args, root):
    """Create a separate prompt-adapted copy, never a training release."""
    from io import BytesIO
    import pyarrow as pa
    import pyarrow.parquet as pq
    from permstudy.data_pipeline.lineage import role_bound_stage_config
    from scripts_permstudy.data import _common as io

    original = io.load_manifest(root, args.trainer_manifest, "trainer_export")
    inputs = io.select_refs(original, "train.parquet")
    if len(inputs) != 1 or inputs[0].record_count == 0:
        raise io.IntegrityError()
    source = inputs[0]
    io.verify_artifact_ref(root, source)
    rows = pq.read_table(io.rooted(root, source.relative_path)).to_pylist()
    if len(rows) != source.record_count:
        raise io.IntegrityError()
    rows = io.checked(controlled_training_rows, rows, integrity=True)
    config = role_bound_stage_config(
        {
            "input_export_manifest_hash": original["output_manifest_hash"],
            "input_parquet_sha256": source.sha256,
            "prompt_template_version": CONTROLLED_TRAINING_PROMPT_VERSION,
            "prompt_template_hash": sha256_hex(
                canonical_json_bytes(
                    {
                        "system": CONTROLLED_TRAINING_SYSTEM,
                        "final_rule": CONTROLLED_TRAINING_FINAL_RULE,
                    }
                )
            ),
        },
        original["upstream_bindings"],
        {"split", "generation", "pairs", "permutations"},
    )
    identity = run_id("trainer_export", config)
    buffer = BytesIO()
    pq.write_table(pa.Table.from_pylist(rows), buffer)
    ref = io.persist_bytes(
        root, f"trainer_export/{identity}/train.parquet", buffer.getvalue(), len(rows)
    )
    manifest = io.persist_manifest(
        root,
        "trainer_export",
        config,
        [ref],
        {"rows": len(rows)},
        {
            role: item["relative_path"]
            for role, item in original["upstream_manifests"].items()
        },
        extra={"training_authorized": False},
    )
    return io.output(manifest)


def _dispatch_generate(
    args,
    root: Path,
    *,
    split_loader,
    engine_factory_builder,
):
    from scripts_permstudy.data import _common as io

    split = io.load_manifest(root, args.split_manifest, "split")
    questions, smoke = _load_recipe_split(root, split, args.recipe, split_loader)
    plan = _generation_plan(split, smoke, recipe=args.recipe)
    if args.generator_id not in {
        config.generator_id for config in plan.generator_configs
    }:
        raise io.CLIContractError()
    planned_ids = {item.original_question_id for item in plan.candidates}
    questions = tuple(
        item for item in questions if item.original_question_id in planned_ids
    )
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
        (f"phase2/{plan.generation_run_id}/{verification['run_id']}/gate-report.json"),
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
    if args.command == "plan":
        return _dispatch_plan(args, root, split_loader=split_loader)
    if args.command == "prepare-training":
        return _dispatch_prepare_training(args, root)
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
