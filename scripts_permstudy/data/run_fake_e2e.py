"""One offline, explicitly synthetic Phase 1 research pipeline exercise."""

from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts_permstudy.data import _common as io
from scripts_permstudy.data.verify_candidates import verification_config
from permstudy.data_pipeline import audit, gates, pairs, permutations, splitting
from permstudy.data_pipeline.canonical import canonical_json_bytes, sha256_hex
from permstudy.data_pipeline.generation import (
    FakeGenerationBackend,
    GenerationConfig,
    GenerationResult,
    build_generation_shard_plan,
    plan_generation,
    run_generation_shard,
    semantic_candidate_set_hash,
)
from permstudy.data_pipeline.generation.base import (
    GENERATOR_REPOSITORIES,
    generation_stage_config,
)
from permstudy.data_pipeline.ids import run_id
from permstudy.data_pipeline.isolation import validate_external_roots
from permstudy.data_pipeline.schema import (
    AuditDecision,
    AuditVerdict,
    CandidateRecord,
    RunManifest,
    Split,
    Source,
)
from permstudy.data_pipeline.sources import math as math_source, reclor
from permstudy.data_pipeline.trainer_export import (
    build_trainer_rows,
    export_trainer_parquet,
)
from permstudy.data_pipeline.verification import (
    verify_math_question,
    verify_reclor_question,
)


@dataclass(frozen=True)
class FakeE2ESummary:
    private_root: Path
    question_count: int
    train_count: int
    holdout_count: int
    smoke_count: int
    planned_count: int
    successful_count: int
    historical_failure_count: int
    interrupted: bool
    split_hash: str
    successful_keys: tuple[tuple[str, str], ...]
    resumed_successful_keys: tuple[tuple[str, str], ...]
    semantic_candidate_set_hash: str
    resumed_semantic_candidate_set_hash: str
    pair_count: int
    permutation_count: int
    trainer_row_count: int
    pair_hash: str
    resumed_pair_hash: str
    permutation_hash: str
    resumed_permutation_hash: str
    trainer_row_hash: str
    resumed_trainer_row_hash: str
    trainer_relative_path: str
    audit_required_count: int
    audit_completed_count: int
    functional_passed: bool
    statistical_status: str

    def public_summary(self):
        """Expose only synthetic mode, counts and research-semantic digests."""
        return {
            "mode": "synthetic_fake",
            "count": self.successful_count,
            "questions": self.question_count,
            "train": self.train_count,
            "holdout": self.holdout_count,
            "smoke": self.smoke_count,
            "pairs": self.pair_count,
            "permutations": self.permutation_count,
            "trainer_rows": self.trainer_row_count,
            "split_hash": self.split_hash,
            "candidate_set_hash": self.semantic_candidate_set_hash,
            "pair_hash": self.pair_hash,
            "permutation_hash": self.permutation_hash,
            "trainer_row_hash": self.trainer_row_hash,
        }


def _synthetic_gold(boxed, _state):
    return str(int(boxed.removeprefix("\\boxed{").removesuffix("}")))


def _synthetic_verify(gold, boxed, _state):
    if boxed == r"\boxed{synthetic_error}":
        raise ImportError("synthetic dependency failure")
    prediction = _synthetic_gold(boxed, None)
    return prediction == gold, str(prediction)


class _SyntheticTokenizer:
    def encode(self, text, *, add_special_tokens):
        return text.split()


class _InterruptAfterFirstBatch:
    """Exercise the runner's real interruption boundary after durable commits."""

    def __init__(self, backend):
        self.backend = backend
        self.calls = 0

    def generate(self, batch, config):
        self.calls += 1
        if self.calls == 2:
            raise KeyboardInterrupt()
        return self.backend.generate(batch, config)


def _read_candidates(path):
    return tuple(
        CandidateRecord.from_dict({k: v for k, v in row.items() if k != "record_hash"})
        for row in io.scan_jsonl(path).records
    )


def _manifest(root, stage, config, records, *, extra=None):
    # Use the existing package envelope: raw hashes of timestamped upstream
    # manifest files belong to execution history, not research-semantic lineage.
    refs = [
        io.persist_records(root, f"canonical/{name}.jsonl", values)
        for name, values in records.items()
    ]
    for ref in refs:
        io.verify_artifact_ref(root, ref)
    payload = dict(
        schema_version="run_manifest_v1",
        stage=stage,
        run_id=run_id(stage, config),
        config_hash=sha256_hex(canonical_json_bytes(config)),
        typed_config=config,
        upstream_bindings=config["upstream_bindings"],
        upstream_manifest_hashes=sorted(config["upstream_bindings"].values()),
        artifacts=[ref.to_dict() for ref in refs],
        counts={name: len(values) for name, values in records.items()},
    )
    payload.update(extra or {})
    path = io.rooted(root, f"{stage}/{payload['run_id']}/manifest.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = io.write_atomic_manifest(path, payload)
    manifest = io.read_json(path)
    core = RunManifest.from_dict(
        {name: manifest[name] for name in RunManifest.__annotations__}
    )
    if core.output_manifest_hash != digest:
        raise io.IntegrityError()
    return manifest


def _selected_pairs(questions, candidates, verified, bindings):
    by_key = {(v.generation_run_id, v.candidate_id): v for v in verified}
    return tuple(
        pair
        for question in questions
        if (
            pair := pairs.select_pair(
                question,
                [
                    (c, by_key[c.plan.key])
                    for c in candidates
                    if c.plan.original_question_id == question.original_question_id
                ],
                _SyntheticTokenizer(),
                "2" * 40,
                upstream_bindings=bindings,
            )
        )
        is not None
    )


def run_fake_e2e(data_root, fixture_root) -> FakeE2ESummary:
    """Compose approved stages below a fresh external temporary directory."""
    data_root = Path(data_root).resolve()
    validate_external_roots(io.REPO, data_root, io.os.environ)
    fixture_root = Path(fixture_root)
    if not all(
        (fixture_root / name).is_file()
        for name in ("synthetic_questions.jsonl", "synthetic_fake_responses.json")
    ):
        raise io.CLIContractError()
    try:
        rows = [
            json.loads(line)
            for line in (fixture_root / "synthetic_questions.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        templates = io.read_json(fixture_root / "synthetic_fake_responses.json")
    except OSError as error:
        raise io.CLIContractError() from error
    if (
        len(rows) != 80
        or any(row.get("synthetic") is not True for row in rows)
        or templates.get("synthetic") is not True
    ):
        raise io.CLIContractError()
    data_root.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="synthetic-fake-e2e-", dir=data_root))
    math_rows = [
        {k: v for k, v in row.items() if k != "source"}
        for row in rows
        if row["source"] == "math"
    ]
    reclor_rows = [
        {k: v for k, v in row.items() if k != "source"}
        for row in rows
        if row["source"] == "reclor"
    ]
    if len(math_rows) != 40 or len(reclor_rows) != 40:
        raise io.CLIContractError()
    io.persist_bytes(
        root, "synthetic_reclor/train.json", canonical_json_bytes(reclor_rows), 40
    )
    for name in ("test.json", "val.json", "use_items.txt"):
        io.persist_bytes(root, f"synthetic_reclor/{name}", b"[]", 0)
    snapshots = {
        "math_questions": math_source.load_math_train(
            math_source.MATH_REPO_ID,
            "1" * 40,
            root / "cache/datasets",
            lambda *_args: math_rows,
            data_root=root,
            api=SimpleNamespace(
                get_dataset_config_names=lambda *_args, **_kwargs: ["algebra"]
            ),
        ),
        "reclor_questions": reclor.load_reclor_train(
            root / "synthetic_reclor", root, acknowledge_noncommercial=True
        ),
    }
    questions = tuple(
        sorted(
            (q for snapshot in snapshots.values() for q in snapshot.questions),
            key=lambda q: q.original_question_id,
        )
    )
    bindings = {
        role: snapshot.private_manifest["output_manifest_hash"]
        for role, snapshot in snapshots.items()
    }
    split = splitting.build_bound_internal_split(snapshots, bindings, root)
    smoke = splitting.select_smoke_questions(split.assignments)
    split_manifest = _manifest(
        root,
        "split",
        splitting.split_stage_config(42, bindings),
        {"questions": questions, "assignments": split.assignments, "smoke": smoke},
        extra={"split_manifest_hash": split.split_manifest_hash},
    )
    io.persist_bytes(
        root,
        "canonical/split.canonical.jsonl",
        splitting.split_manifest_bytes(questions, split),
        81,
    )
    configs = tuple(
        GenerationConfig(
            backend="fake",
            generator_id=generator,
            model_repository=repository,
            model_revision="3" * 40,
            prompt_template_revision="synthetic_fake_v1",
            prompt_template_hash=sha256_hex(canonical_json_bytes(templates)),
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=256,
            seed=42,
            batch_size=2,
            tensor_parallel_size=1,
            max_model_len=1024,
            gpu_memory_utilization=0.5,
            samples_per_question=2,
            split_manifest_hash=split.split_manifest_hash,
        )
        for generator, repository in sorted(GENERATOR_REPOSITORIES.items())
    )
    plan = plan_generation(
        smoke,
        configs,
        shard_size=8,
        split_upstream_manifest_hash=split_manifest["output_manifest_hash"],
    )
    by_question = {q.original_question_id: q for q in questions}
    generator_positions = {config.generator_id: i for i, config in enumerate(configs)}
    responses = {}
    for candidate in plan.candidates:
        question = by_question[candidate.original_question_id]
        gold = (
            str(
                _synthetic_gold(
                    question.solution[question.solution.index(r"\boxed{") :], None
                )
            )
            if question.source is Source.MATH
            else question.gold_label
        )
        wrong = (
            str(int(gold) + 1)
            if question.source is Source.MATH
            else "ABCD"[("ABCD".index(gold) + 1) % 4]
        )
        template = templates[question.source.value][
            generator_positions[candidate.generator_id] * 2 + candidate.sampling_index
        ]
        response = template.format(gold=gold, wrong=wrong)
        responses[candidate.key] = GenerationResult(
            candidate, response, "stop", len(response.split()), None
        )
    backend = FakeGenerationBackend(responses)
    clean = []
    for composite in plan.shard_ids:
        generator, shard_id = composite.split("/")
        shard = build_generation_shard_plan(plan, generator, shard_id)
        output = root / "clean_generation" / composite
        run_generation_shard(shard, backend, output)
        clean.extend(_read_candidates(output / "candidates.jsonl"))
    representative = build_generation_shard_plan(plan, *plan.shard_ids[0].split("/"))
    failure_map = dict(responses)
    failed = representative.candidates[0]
    failure_map[failed.key] = GenerationResult(
        failed, None, None, 0, templates["historical_failure"]
    )
    resumed_output = root / "resumed_generation"
    interrupted = False
    try:
        run_generation_shard(
            representative,
            _InterruptAfterFirstBatch(FakeGenerationBackend(failure_map)),
            resumed_output,
        )
    except KeyboardInterrupt:
        interrupted = True
    resumed_summary = run_generation_shard(representative, backend, resumed_output)
    representative_keys = {c.key for c in representative.candidates}
    resumed = [c for c in clean if c.plan.key not in representative_keys] + list(
        _read_candidates(resumed_output / "candidates.jsonl")
    )
    clean = tuple(sorted(clean, key=lambda c: c.plan.key))
    resumed = tuple(sorted(resumed, key=lambda c: c.plan.key))
    if clean != resumed:
        raise io.IntegrityError()
    generation = _manifest(
        root,
        "generation",
        generation_stage_config(
            configs,
            samples_per_question=2,
            shard_size=8,
            split_upstream_manifest_hash=split_manifest["output_manifest_hash"],
        ),
        {"candidates": clean},
    )
    verify_config = verification_config(generation["output_manifest_hash"], 10.0, 10.0)
    verification_id = run_id("verification", verify_config)
    smoke_ids = {a.original_question_id for a in smoke}
    selected_questions = tuple(
        q for q in questions if q.original_question_id in smoke_ids
    )
    gold_records, verified = [], []
    for question in selected_questions:
        group = [
            c
            for c in clean
            if c.plan.original_question_id == question.original_question_id
        ]
        if question.source is Source.MATH:
            gold, records = verify_math_question(
                question,
                group,
                10.0,
                10.0,
                verification_run_id=verification_id,
                gold_parser_hook=_synthetic_gold,
                candidate_verifier_hook=_synthetic_verify,
            )
        else:
            gold, records = verify_reclor_question(
                question,
                group,
                verification_run_id=verification_id,
                verifier_timeout_seconds=10.0,
            )
        gold_records.append(gold)
        verified.extend(records)
    verification = _manifest(
        root,
        "verification",
        verify_config,
        {"question_verifications": gold_records, "verifications": verified},
    )
    selection = audit.build_audit_selection(
        verified,
        gold_records,
        plan.generation_run_id,
        verification_manifest_hash=verification["output_manifest_hash"],
    )
    decisions = tuple(
        AuditDecision(
            s.audit_run_id,
            s.generation_run_id,
            s.verification_run_id,
            s.record_kind,
            s.original_question_id,
            s.candidate_id,
            AuditVerdict.AGREE,
            s.reason_code,
            False,
        )
        for s in selection
    )
    audit_summary = audit.validate_audit_decisions(selection, decisions)
    io.persist_records(root, "canonical/audit_selection.jsonl", selection)
    io.persist_records(root, "canonical/synthetic_agree_decisions.jsonl", decisions)
    pair_bindings = {
        "generation": generation["output_manifest_hash"],
        "verification": verification["output_manifest_hash"],
    }
    selected_pairs = _selected_pairs(selected_questions, clean, verified, pair_bindings)
    pair_manifest = _manifest(
        root,
        "pairs",
        pairs.pair_stage_config("2" * 40, pair_bindings),
        {"pairs": selected_pairs},
    )
    perm_bindings = {"pairs": pair_manifest["output_manifest_hash"]}
    surface_records = tuple(
        p
        for pair in selected_pairs
        for p in permutations.build_permutations(pair, upstream_bindings=perm_bindings)
    )
    permutation_manifest = _manifest(
        root,
        "permutations",
        permutations.permutation_stage_config(perm_bindings),
        {"permutations": surface_records},
    )
    inputs = gates.GateInputs(
        plan.candidates,
        clean,
        verified,
        gold_records,
        selected_pairs,
        surface_records,
        audit_summary,
    )
    functional = gates.evaluate_functional_gate(inputs)
    statistical = gates.evaluate_statistical_gate(inputs)
    # Trainer export is an independent consumer of the canonical branch.
    trainer_rows = build_trainer_rows(questions, clean, selected_pairs, surface_records)
    export = export_trainer_parquet(
        {
            "split": split_manifest,
            "generation": generation,
            "pairs": pair_manifest,
            "permutations": permutation_manifest,
        },
        root,
    )
    resumed_pairs = _selected_pairs(
        selected_questions, resumed, verified, pair_bindings
    )
    resumed_permutations = tuple(
        p
        for pair in resumed_pairs
        for p in permutations.build_permutations(pair, upstream_bindings=perm_bindings)
    )
    resumed_rows = build_trainer_rows(
        questions, resumed, resumed_pairs, resumed_permutations
    )
    return FakeE2ESummary(
        root,
        len(questions),
        sum(a.split is Split.TRAIN for a in split.assignments),
        sum(a.split is Split.INTERNAL_HOLDOUT for a in split.assignments),
        len(smoke),
        len(plan.candidates),
        len(clean),
        resumed_summary.historical_failures,
        interrupted,
        split.split_manifest_hash,
        tuple(c.plan.key for c in clean),
        tuple(c.plan.key for c in resumed),
        semantic_candidate_set_hash(clean),
        semantic_candidate_set_hash(resumed),
        len(selected_pairs),
        len(surface_records),
        export.row_count,
        pairs.pair_manifest_hash(selected_pairs),
        pairs.pair_manifest_hash(resumed_pairs),
        permutations.permutation_manifest_hash(surface_records),
        permutations.permutation_manifest_hash(resumed_permutations),
        sha256_hex(canonical_json_bytes(trainer_rows)),
        sha256_hex(canonical_json_bytes(resumed_rows)),
        export.dataset_artifact.relative_path,
        audit_summary.required_count,
        audit_summary.completed_count,
        functional.passed,
        statistical.status.value,
    )


def build_parser() -> io.argparse.ArgumentParser:
    parser = io.parser("run_fake_e2e")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fixture-root")
    group.add_argument("--split-manifest")
    return parser


def dispatch(args, root):
    if args.split_manifest:
        raise io.PhaseBoundaryError()
    return run_fake_e2e(root, args.fixture_root).public_summary()


def main(argv: Sequence[str] | None = None) -> int:
    return io.execute(build_parser, dispatch, argv)


if __name__ == "__main__":
    raise SystemExit(main())
