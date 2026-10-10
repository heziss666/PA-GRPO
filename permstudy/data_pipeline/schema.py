"""Frozen business records with explicit validation and JSON-safe conversion.

Record hashes belong to the I/O envelope, never to these constructors.
"""

from collections.abc import Mapping
from dataclasses import dataclass, fields
from enum import Enum
import math
import re
from types import MappingProxyType, UnionType
from typing import get_args, get_origin, get_type_hints


class Source(str, Enum):
    MATH = "math"
    RECLOR = "reclor"


class Split(str, Enum):
    TRAIN = "train"
    INTERNAL_HOLDOUT = "internal_holdout"


class VerificationStatus(str, Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    INVALID = "invalid"
    AMBIGUOUS = "ambiguous"
    ERROR = "error"


class AuditVerdict(str, Enum):
    AGREE = "AGREE"
    DISAGREE = "DISAGREE"
    UNSURE = "UNSURE"


class GateStatus(str, Enum):
    FAIL = "FAIL"
    PASS_WITH_WARNINGS = "PASS_WITH_WARNINGS"
    PASS = "PASS"


_HASH = re.compile(r"[0-9a-fA-F]{64}\Z")
_REASONS = frozenset({"missing_final_marker", "duplicate_same_marker", "conflicting_markers",
                      "math_parse_failure", "gold_parse_failure", "timeout", "dependency_error",
                      "prompt_contract_mismatch", "other"})
_GOLD_STATUSES = frozenset({"ok", "gold_verification_error"})


def _require(condition, field):
    if not condition:
        raise ValueError(f"Invalid {field}")


def _matches(value, annotation):
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is UnionType:
        return any(_matches(value, arg) for arg in args)
    if origin is tuple:
        return isinstance(value, tuple) and all(_matches(item, args[0]) for item in value)
    if origin is Mapping:
        return isinstance(value, Mapping) and all(_matches(k, args[0]) and _matches(v, args[1]) for k, v in value.items())
    if annotation is float:
        return type(value) in (int, float) and math.isfinite(value)
    if annotation in (int, bool, str):
        return type(value) is annotation
    return isinstance(value, annotation)


def _encode(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, _Record):
        return value.to_dict()
    if isinstance(value, tuple):
        return [_encode(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _encode(item) for key, item in value.items()}
    return value


def _decode(value, annotation):
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is UnionType:
        if value is None and type(None) in args:
            return None
        return _decode(value, next(arg for arg in args if arg is not type(None)))
    if origin is tuple:
        _require(isinstance(value, (list, tuple)), "sequence")
        return tuple(_decode(item, args[0]) for item in value)
    if origin is Mapping:
        _require(isinstance(value, Mapping), "mapping")
        return dict(value)
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        try:
            return annotation(value)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid {annotation.__name__}") from exc
    if isinstance(annotation, type) and issubclass(annotation, _Record):
        return annotation.from_dict(value)
    return value


class _Record:
    def __post_init__(self):
        # Copy collection inputs so callers cannot mutate frozen records indirectly.
        for name, annotation in get_type_hints(type(self)).items():
            value = getattr(self, name)
            if get_origin(annotation) is tuple and isinstance(value, (list, tuple)):
                object.__setattr__(self, name, tuple(value))
            elif get_origin(annotation) is Mapping and isinstance(value, Mapping):
                object.__setattr__(self, name, MappingProxyType(dict(value)))

    def validate(self) -> None:
        for name, annotation in get_type_hints(type(self)).items():
            value = getattr(self, name)
            _require(_matches(value, annotation), name)
            if value is None:
                continue
            if isinstance(value, _Record):
                value.validate()
            if isinstance(value, str) and (name.endswith("_id") or name.endswith("_revision") or name.endswith("_version")):
                _require(bool(value.strip()), name)
            if name.endswith("_hash") or name == "sha256":
                _require(isinstance(value, str) and bool(_HASH.fullmatch(value)), name)
            if type(value) is int:
                _require(value >= 0, name)
        self._validate_fields()

    def _validate_fields(self) -> None:
        pass

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {field.name: _encode(getattr(self, field.name)) for field in fields(self)}

    @classmethod
    def from_dict(cls, payload):
        _require(isinstance(payload, Mapping), cls.__name__)
        annotations = get_type_hints(cls)
        _require(set(payload) == set(annotations), f"{cls.__name__} fields")
        record = cls(**{name: _decode(payload[name], annotation) for name, annotation in annotations.items()})
        record.validate()
        return record


@dataclass(frozen=True)
class ArtifactRef(_Record):
    relative_path: str
    sha256: str
    record_count: int

    def _validate_fields(self):
        path = self.relative_path
        _require(bool(path) and "\\" not in path and ":" not in path and "\x00" not in path
                 and all(part not in ("", ".", "..") for part in path.split("/")), "relative_path")


@dataclass(frozen=True)
class RunManifest(_Record):
    schema_version: str
    stage: str
    run_id: str
    config_hash: str
    upstream_manifest_hashes: tuple[str, ...]
    artifacts: tuple[ArtifactRef, ...]
    counts: Mapping[str, int]
    created_at_utc: str
    output_manifest_hash: str

    def _validate_fields(self):
        _require(bool(self.stage.strip()) and bool(self.created_at_utc.strip()), "manifest metadata")
        for digest in self.upstream_manifest_hashes:
            _require(bool(_HASH.fullmatch(digest)), "upstream_manifest_hashes")
        for artifact in self.artifacts:
            artifact.validate()
        _require(all(key.strip() and value >= 0 for key, value in self.counts.items()), "counts")


@dataclass(frozen=True)
class QuestionRecord(_Record):
    schema_version: str
    source: Source
    original_question_id: str
    question_content_hash: str
    source_snapshot_id: str
    source_revision: str
    source_row_id: str
    context: str | None
    question: str | None
    answers: tuple[str, ...]
    gold_label: str | None
    problem: str | None
    solution: str | None
    category: str | None
    level: str | None
    synthetic: bool

    def _validate_fields(self):
        if self.source is Source.RECLOR:
            _require(all(isinstance(value, str) and value.strip() for value in (self.context, self.question)), "ReClor text")
            _require(len(self.answers) == 4 and all(answer.strip() for answer in self.answers), "ReClor answers")
            _require(self.gold_label in ("A", "B", "C", "D"), "gold_label")
            _require(all(value is None for value in (self.problem, self.solution, self.category, self.level)), "ReClor source shape")
        else:
            _require(all(isinstance(value, str) and value.strip() for value in (self.problem, self.solution, self.category, self.level)), "MATH text")
            _require(self.context is None and self.question is None and not self.answers and self.gold_label is None, "MATH source shape")


@dataclass(frozen=True)
class SplitAssignment(_Record):
    original_question_id: str
    question_content_hash: str
    source: Source
    split: Split
    stratum: str

    def _validate_fields(self):
        _require(bool(self.stratum.strip()), "stratum")


@dataclass(frozen=True)
class CandidatePlan(_Record):
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


@dataclass(frozen=True)
class CandidateRecord(_Record):
    plan: CandidatePlan
    response: str
    finish_reason: str
    generated_token_count: int
    model_revision: str

    def _validate_fields(self):
        _require(bool(self.finish_reason.strip()), "finish_reason")


@dataclass(frozen=True)
class FailureRecord(_Record):
    plan: CandidatePlan
    error_type: str
    retry_count: int

    def _validate_fields(self):
        _require(bool(self.error_type.strip()), "error_type")


@dataclass(frozen=True)
class VerificationRecord(_Record):
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

    def _validate_fields(self):
        _require(bool(self.prediction_parse_status.strip()) and bool(self.verifier_name.strip()), "verifier metadata")
        _require(self.verifier_timeout_seconds > 0, "verifier_timeout_seconds")


@dataclass(frozen=True)
class QuestionVerificationRecord(_Record):
    verification_run_id: str
    original_question_id: str
    source: Source
    gold_parse_status: str
    canonical_gold: str | None
    error_type: str | None

    def _validate_fields(self):
        _require(self.gold_parse_status in _GOLD_STATUSES, "gold_parse_status")
        if self.gold_parse_status == "ok":
            _require(isinstance(self.canonical_gold, str) and bool(self.canonical_gold.strip()) and self.error_type is None, "canonical_gold")
        else:
            _require(self.canonical_gold is None and isinstance(self.error_type, str) and bool(self.error_type.strip()), "gold error")


@dataclass(frozen=True)
class PairRecord(_Record):
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

    def _validate_fields(self):
        _require(bool(self.tokenizer_repo.strip()), "tokenizer_repo")
        _require(self.positive_candidate_id != self.negative_candidate_id, "pair candidates")


@dataclass(frozen=True)
class PermutationRecord(_Record):
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

    def _validate_fields(self):
        _require((self.permutation_id, self.permutation_label, self.correct_surface)
                 in ((0, "AB", "A"), (1, "BA", "B")), "permutation")
        _require(self.surface_a_candidate_id != self.surface_b_candidate_id, "surface candidates")


@dataclass(frozen=True)
class AuditSelectionRecord(_Record):
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

    def _validate_fields(self):
        _audit_identity(self)
        _require(self.reason_code in _REASONS, "reason_code")
        if self.record_kind == "candidate":
            _require(self.generator_id is not None and self.verification_status is not None and self.gold_parse_status is None, "candidate audit")
        else:
            _require(self.generator_id is None and self.verification_status is None and self.gold_parse_status in _GOLD_STATUSES, "question gold audit")


@dataclass(frozen=True)
class AuditDecision(_Record):
    audit_run_id: str
    generation_run_id: str
    verification_run_id: str
    record_kind: str
    original_question_id: str
    candidate_id: str | None
    verdict: AuditVerdict
    reason_code: str
    confirmed_disagree: bool

    def _validate_fields(self):
        _audit_identity(self)
        _require(self.reason_code in _REASONS, "reason_code")
        _require(not self.confirmed_disagree or self.verdict is AuditVerdict.DISAGREE, "confirmed_disagree")


def _audit_identity(record):
    _require(record.record_kind in ("candidate", "question_gold"), "record_kind")
    _require((record.record_kind == "question_gold") == (record.candidate_id is None), "audit identity")
