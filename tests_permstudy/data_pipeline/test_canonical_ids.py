"""Stable byte and identity contracts using only synthetic inputs."""

import pytest

from permstudy.data_pipeline.canonical import (
    canonical_json_bytes, canonical_jsonl_bytes, normalize_text_v1, sha256_hex,
)
from permstudy.data_pipeline.ids import (
    candidate_id, math_question_identity, pair_id, reclor_content_hash, run_id,
)


@pytest.mark.parametrize("raw, normalized", [
    ("  A\r\n  B  ", "A\n  B"),
    ("\tCafe\u0301\rB\r\nC\t", "Café\nB\nC"),
    ("A  \t B\n\nC", "A  \t B\n\nC"),
    (" \r\n\t", ""),
])
def test_normalization_v1_preserves_internal_whitespace(raw, normalized):
    assert normalize_text_v1(raw) == normalized


def test_canonical_json_has_sorted_compact_utf8_bytes_without_newline():
    assert canonical_json_bytes({"z": "é", "a": {"b": 2, "a": 1}}) == (
        b'{"a":{"a":1,"b":2},"z":"\xc3\xa9"}'
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_canonical_json_rejects_nonfinite_numbers(value):
    with pytest.raises(ValueError):
        canonical_json_bytes({"number": value})


def test_canonical_jsonl_sorts_requested_key_with_lf_record_terminators():
    records = [{"id": "b", "rank": 1}, {"rank": 2, "id": "a"}]
    assert canonical_jsonl_bytes(iter(records), "rank") == (
        b'{"id":"b","rank":1}\n{"id":"a","rank":2}\n'
    )
    assert canonical_jsonl_bytes(records, "id") == (
        b'{"id":"a","rank":2}\n{"id":"b","rank":1}\n'
    )


def test_empty_canonical_jsonl_has_no_record_terminator():
    assert canonical_jsonl_bytes([], "id") == b""


def test_sha256_uses_full_lowercase_digest():
    assert sha256_hex(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_reclor_hash_preserves_field_boundaries():
    assert reclor_content_hash("ab", "c", ("d", "e", "f", "g")) != reclor_content_hash(
        "a", "bc", ("d", "e", "f", "g")
    )


def test_reclor_hash_pins_structured_normalized_payload():
    # Independently hashed literal canonical payload with the documented schema.
    assert reclor_content_hash(" ab\r\n", " c ", (" d ", "e", "f", "g")) == (
        "2304a8f7910eabaa5ae060521a88baaef0f899e570155d0464ec37b82557052a"
    )


def test_reclor_hash_preserves_answer_order_and_array_boundaries():
    base = reclor_content_hash("context", "question", ("ab", "c", "d", "e"))
    assert base != reclor_content_hash("context", "question", ("a", "bc", "d", "e"))
    assert base != reclor_content_hash("context", "question", ("c", "ab", "d", "e"))


def test_math_identity_retains_full_normalized_problem_hash():
    assert math_question_identity(" \r\nabc\r\n ") == (
        "math:train:ba7816bf8f01cfea414140de5dae2223b00361a3",
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    )


def test_math_identity_normalizes_unicode_and_newlines():
    assert math_question_identity(" Cafe\u0301\r\nB ") == math_question_identity("Café\nB")
    assert math_question_identity("A  B") != math_question_identity("A B")


@pytest.mark.parametrize("question, generator, index", [("other", "gen", 0), ("q", "other", 0), ("q", "gen", 1)])
def test_candidate_identity_binds_each_logical_coordinate(question, generator, index):
    assert candidate_id(question, generator, index) != candidate_id("q", "gen", 0)


def test_candidate_identity_pins_versioned_payload_hash():
    assert candidate_id("q", "gen", 0) == "d1748f353782747d05f03c188078bd4d28993daa58570b699ba9a19d8b8f7561"


@pytest.mark.parametrize("index", [True, False, -1, 0.5, "0"])
def test_candidate_identity_rejects_invalid_sampling_indices(index):
    with pytest.raises(ValueError):
        candidate_id("q", "gen", index)


@pytest.mark.parametrize("field", [0, 1])
@pytest.mark.parametrize("invalid", ["", " \t", None, 1])
def test_candidate_identity_requires_nonempty_string_identifiers(field, invalid):
    arguments = ["q", "gen", 0]
    arguments[field] = invalid
    with pytest.raises(ValueError):
        candidate_id(*arguments)


def test_generation_config_changes_run_namespace_without_changing_logical_candidate():
    first = (run_id("generation", {"temperature": 0.5}), candidate_id("q", "gen", 0))
    second = (run_id("generation", {"temperature": 0.7}), candidate_id("q", "gen", 0))
    assert first[0] != second[0]
    assert first[1] == second[1]
    assert first != second


def test_run_identity_ignores_mapping_insertion_order_and_binds_stage():
    config = {"seed": 42, "decoding": {"top_p": 0.9, "temperature": 0.5}}
    reordered = {"decoding": {"temperature": 0.5, "top_p": 0.9}, "seed": 42}
    assert run_id("generation", config) == run_id("generation", reordered)
    assert run_id("generation", config) != run_id("verification", config)


def test_run_identity_pins_versioned_payload_hash():
    assert run_id("generation", {"seed": 42}) == "cd1715278eee99f9bbcb67bd25661a2dbd284ac0f8270ae58278de228b645c62"


@pytest.mark.parametrize("invalid", ["", " \t", None, 1])
def test_run_identity_requires_nonempty_string_stage(invalid):
    with pytest.raises(ValueError):
        run_id(invalid, {})


@pytest.mark.parametrize("coordinates", [
    ("run2", "q", "pos", "neg", "pair_v1"),
    ("run1", "other", "pos", "neg", "pair_v1"),
    ("run1", "q", "other", "neg", "pair_v1"),
    ("run1", "q", "pos", "other", "pair_v1"),
    ("run1", "q", "pos", "neg", "pair_v2"),
    ("run1", "q", "neg", "pos", "pair_v1"),
])
def test_pair_identity_binds_run_question_ordered_candidates_and_schema(coordinates):
    assert pair_id(*coordinates) != pair_id("run1", "q", "pos", "neg", "pair_v1")


def test_pair_identity_pins_run_scoped_payload_hash():
    assert pair_id("run1", "q", "pos", "neg", "pair_v1") == "e3b3877f4a3e853136354a9d96fe65f61c14f4db4ee6ad2f9e3237fee1d1defa"


@pytest.mark.parametrize("field", range(5))
@pytest.mark.parametrize("invalid", ["", " \t", None, 1])
def test_pair_identity_requires_nonempty_string_coordinates(field, invalid):
    arguments = ["run1", "q", "pos", "neg", "pair_v1"]
    arguments[field] = invalid
    with pytest.raises(ValueError):
        pair_id(*arguments)
