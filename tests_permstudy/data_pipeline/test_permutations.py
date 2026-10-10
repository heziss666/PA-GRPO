"""Exact AB/BA construction and immutable permutation lineage tests."""

from dataclasses import replace
import json

import pytest

from permstudy.data_pipeline.ids import pair_id
from permstudy.data_pipeline.schema import PairRecord, Source, Split


GENERATION_RUN_ID = "1" * 64
VERIFICATION_RUN_ID = "2" * 64
PAIR_RUN_ID = "3" * 64
QUESTION_HASH = "4" * 64
SPLIT_HASH = "5" * 64
PAIR_MANIFEST_HASH = "6" * 64
TOKENIZER_REVISION = "7" * 40
UPSTREAM_BINDINGS = {"pairs": PAIR_MANIFEST_HASH}


def pair(
    question_id="reclor:train:synthetic_1",
    *,
    positive_candidate_id="positive",
    negative_candidate_id="negative",
):
    identity = pair_id(
        GENERATION_RUN_ID,
        question_id,
        positive_candidate_id,
        negative_candidate_id,
        "pair_v1",
    )
    return PairRecord(
        schema_version="pair_v1",
        pair_run_id=PAIR_RUN_ID,
        pair_id=identity,
        generation_run_id=GENERATION_RUN_ID,
        verification_run_id=VERIFICATION_RUN_ID,
        original_question_id=question_id,
        question_content_hash=QUESTION_HASH,
        source=Source.RECLOR,
        split=Split.TRAIN,
        split_manifest_hash=SPLIT_HASH,
        positive_candidate_id=positive_candidate_id,
        negative_candidate_id=negative_candidate_id,
        positive_response_hash="8" * 64,
        negative_response_hash="9" * 64,
        positive_generator_id="qwen2.5-7b-instruct",
        negative_generator_id="llama-3.1-8b-instruct",
        positive_token_count=11,
        negative_token_count=12,
        tokenizer_repo="Qwen/Qwen2.5-7B-Instruct",
        tokenizer_revision=TOKENIZER_REVISION,
    )


def test_builds_exact_mirrored_ab_ba_records_in_canonical_order():
    from permstudy.data_pipeline.permutations import build_permutations

    source = pair()

    ab, ba = build_permutations(source, upstream_bindings=UPSTREAM_BINDINGS)

    assert (ab.permutation_id, ab.permutation_label, ab.correct_surface) == (0, "AB", "A")
    assert (ab.surface_a_candidate_id, ab.surface_b_candidate_id) == ("positive", "negative")
    assert (ba.permutation_id, ba.permutation_label, ba.correct_surface) == (1, "BA", "B")
    assert (ba.surface_a_candidate_id, ba.surface_b_candidate_id) == ("negative", "positive")
    assert ab.permutation_run_id == ba.permutation_run_id
    assert {
        (record.pair_id, record.generation_run_id, record.original_question_id)
        for record in (ab, ba)
    } == {(source.pair_id, GENERATION_RUN_ID, source.original_question_id)}
    assert {(record.split, record.split_manifest_hash) for record in (ab, ba)} == {
        (Split.TRAIN, SPLIT_HASH)
    }


def test_permutation_stage_role_binds_pair_manifest_hash():
    from permstudy.data_pipeline.ids import run_id
    from permstudy.data_pipeline.permutations import permutation_stage_config

    baseline = permutation_stage_config(UPSTREAM_BINDINGS)
    repeated = permutation_stage_config(dict(reversed(tuple(UPSTREAM_BINDINGS.items()))))
    changed = permutation_stage_config({"pairs": "a" * 64})

    assert baseline == repeated
    assert baseline["upstream_bindings"] == {"pairs": PAIR_MANIFEST_HASH}
    assert run_id("permutations", baseline) == run_id("permutations", repeated)
    assert run_id("permutations", baseline) != run_id("permutations", changed)


def test_build_rejects_pair_identity_that_does_not_match_semantic_candidates():
    from permstudy.data_pipeline.permutations import PermutationIntegrityError, build_permutations

    corrupted = replace(pair(), positive_candidate_id="different_positive")

    with pytest.raises(PermutationIntegrityError, match="pair identity"):
        build_permutations(corrupted, upstream_bindings=UPSTREAM_BINDINGS)


def test_build_rejects_unknown_pair_schema_before_emitting_records():
    from permstudy.data_pipeline.permutations import PermutationIntegrityError, build_permutations

    corrupted = replace(pair(), schema_version="pair_v2")

    with pytest.raises(PermutationIntegrityError, match="pair schema"):
        build_permutations(corrupted, upstream_bindings=UPSTREAM_BINDINGS)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("split", Split.INTERNAL_HOLDOUT, "split lineage"),
        ("split_manifest_hash", "b" * 64, "split lineage"),
        ("generation_run_id", "c" * 64, "generation lineage"),
        ("original_question_id", "reclor:train:other", "question lineage"),
    ],
)
def test_manifest_rejects_cross_lineage_ab_ba_records(field, value, message):
    from permstudy.data_pipeline.permutations import (
        PermutationIntegrityError,
        build_permutations,
        permutation_manifest_bytes,
    )

    ab, ba = build_permutations(pair(), upstream_bindings=UPSTREAM_BINDINGS)
    corrupted_ba = replace(ba, **{field: value})

    with pytest.raises(PermutationIntegrityError, match=message):
        permutation_manifest_bytes((ab, corrupted_ba))


def test_manifest_is_canonical_stable_and_sorted_by_question_then_permutation():
    from permstudy.data_pipeline.permutations import (
        build_permutations,
        permutation_manifest_bytes,
        permutation_manifest_hash,
    )

    first = build_permutations(pair(), upstream_bindings=UPSTREAM_BINDINGS)
    second = build_permutations(
        pair(
            "reclor:train:synthetic_2",
            positive_candidate_id="positive_2",
            negative_candidate_id="negative_2",
        ),
        upstream_bindings=UPSTREAM_BINDINGS,
    )
    scrambled = (second[1], first[1], second[0], first[0])
    canonical = first + second

    rendered = permutation_manifest_bytes(scrambled)
    rows = [json.loads(line) for line in rendered.decode("utf-8").splitlines()]

    assert [(row["original_question_id"], row["permutation_id"]) for row in rows] == [
        ("reclor:train:synthetic_1", 0),
        ("reclor:train:synthetic_1", 1),
        ("reclor:train:synthetic_2", 0),
        ("reclor:train:synthetic_2", 1),
    ]
    assert rendered.endswith(b"\n")
    assert permutation_manifest_hash(scrambled) == permutation_manifest_hash(canonical)
    assert permutation_manifest_hash(scrambled) == "365c08c91468a4ce3fc19fd42759cf9c52db741855c1174bf6905d4eaa34cf95"


def test_manifest_rejects_incomplete_or_duplicate_permutation_pair():
    from permstudy.data_pipeline.permutations import (
        PermutationIntegrityError,
        build_permutations,
        permutation_manifest_bytes,
    )

    ab, ba = build_permutations(pair(), upstream_bindings=UPSTREAM_BINDINGS)

    with pytest.raises(PermutationIntegrityError, match="exactly AB and BA"):
        permutation_manifest_bytes((ab,))
    with pytest.raises(PermutationIntegrityError, match="exactly AB and BA"):
        permutation_manifest_bytes((ab, ba, ba))
