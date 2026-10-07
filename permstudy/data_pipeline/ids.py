"""Deterministic content identities and versioned run-scoped identifiers.

Candidate IDs identify logical samples; candidate records must use the composite
key (generation_run_id, candidate_id). Pair IDs include the generation run.
"""

from collections.abc import Mapping, Sequence

from .canonical import canonical_json_bytes, normalize_text_v1, sha256_hex


def _require_identifier(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")


def reclor_content_hash(context: str, question: str, answers: Sequence[str]) -> str:
    """Hash normalized ReClor fields while preserving ordered answer boundaries."""
    return sha256_hex(canonical_json_bytes({
        "answers": [normalize_text_v1(answer) for answer in answers],
        "context": normalize_text_v1(context),
        "question": normalize_text_v1(question),
        "schema": "reclor_question_content_v1",
    }))


def math_question_identity(problem: str) -> tuple[str, str]:
    """Return the 20-byte readable MATH ID and retained full content digest."""
    content_hash = sha256_hex(normalize_text_v1(problem).encode("utf-8"))
    return f"math:train:{content_hash[:40]}", content_hash


def candidate_id(question_id: str, generator_id: str, sampling_index: int) -> str:
    """Identify a logical candidate independently of generation configuration."""
    _require_identifier(question_id, "question_id")
    _require_identifier(generator_id, "generator_id")
    if type(sampling_index) is not int or sampling_index < 0:
        raise ValueError("sampling_index must be a nonnegative integer")
    return sha256_hex(canonical_json_bytes({
        "generator_id": generator_id,
        "original_question_id": question_id,
        "sampling_index": sampling_index,
        "schema": "candidate_id_v1",
    }))


def run_id(stage: str, config: Mapping[str, object]) -> str:
    """Bind a stage to its full canonical configuration."""
    _require_identifier(stage, "stage")
    return sha256_hex(canonical_json_bytes({
        "config": dict(config),
        "schema": "run_id_v1",
        "stage": stage,
    }))


def pair_id(generation_run_id: str, original_question_id: str, response_pos_id: str,
            response_neg_id: str, pair_schema_version: str) -> str:
    """Bind the ordered selected candidates to the generation run and schema."""
    payload = {
        "generation_run_id": generation_run_id,
        "original_question_id": original_question_id,
        "pair_schema_version": pair_schema_version,
        "response_neg_id": response_neg_id,
        "response_pos_id": response_pos_id,
    }
    for field, value in payload.items():
        _require_identifier(value, field)
    return sha256_hex(canonical_json_bytes(payload))
