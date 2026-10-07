"""Role-tagged downstream lineage contract; fixtures are synthetic hashes only."""

import importlib

import pytest

from permstudy.data_pipeline.ids import run_id


MATH_HASH = "1" * 64
RECLOR_HASH = "2" * 64
OTHER_HASH = "3" * 64
ROLES = {"math_questions", "reclor_questions"}


def role_bound_stage_config(*args, **kwargs):
    """Import inside the test so the initial RED is a test failure, not collection error."""
    try:
        module = importlib.import_module("permstudy.data_pipeline.lineage")
    except ModuleNotFoundError:
        pytest.fail("permstudy.data_pipeline.lineage is not implemented")
    return module.role_bound_stage_config(*args, **kwargs)


def test_role_bound_config_has_literal_canonical_shape_and_copies_inputs():
    parameters = {"split_seed": 42}
    bindings = {"reclor_questions": RECLOR_HASH, "math_questions": MATH_HASH}

    config = role_bound_stage_config(parameters, bindings, ROLES)

    assert config == {
        "lineage_schema": "role_tagged_upstream_v1",
        "parameters": {"split_seed": 42},
        "upstream_bindings": {
            "math_questions": MATH_HASH,
            "reclor_questions": RECLOR_HASH,
        },
    }
    parameters["split_seed"] = 7
    bindings["math_questions"] = OTHER_HASH
    assert config["parameters"] == {"split_seed": 42}
    assert config["upstream_bindings"]["math_questions"] == MATH_HASH


def test_binding_insertion_order_does_not_change_run_id():
    first = role_bound_stage_config(
        {"split_seed": 42},
        {"math_questions": MATH_HASH, "reclor_questions": RECLOR_HASH},
        ROLES,
    )
    second = role_bound_stage_config(
        {"split_seed": 42},
        {"reclor_questions": RECLOR_HASH, "math_questions": MATH_HASH},
        tuple(reversed(sorted(ROLES))),
    )

    assert first == second
    assert run_id("split", first) == run_id("split", second)
    assert sorted(first["upstream_bindings"].values()) == [MATH_HASH, RECLOR_HASH]


def test_changed_upstream_hash_changes_run_id():
    baseline = role_bound_stage_config(
        {"split_seed": 42},
        {"math_questions": MATH_HASH, "reclor_questions": RECLOR_HASH},
        ROLES,
    )
    changed = role_bound_stage_config(
        {"split_seed": 42},
        {"math_questions": OTHER_HASH, "reclor_questions": RECLOR_HASH},
        ROLES,
    )

    assert run_id("split", baseline) != run_id("split", changed)


def test_swapping_hash_roles_changes_run_id_even_with_same_untyped_hash_list():
    baseline = role_bound_stage_config(
        {"split_seed": 42},
        {"math_questions": MATH_HASH, "reclor_questions": RECLOR_HASH},
        ROLES,
    )
    swapped = role_bound_stage_config(
        {"split_seed": 42},
        {"math_questions": RECLOR_HASH, "reclor_questions": MATH_HASH},
        ROLES,
    )

    assert sorted(baseline["upstream_bindings"].values()) == sorted(swapped["upstream_bindings"].values())
    assert run_id("split", baseline) != run_id("split", swapped)


@pytest.mark.parametrize(
    ("bindings", "allowed_roles"),
    [
        ({"math_questions": MATH_HASH}, ROLES),
        (
            {
                "math_questions": MATH_HASH,
                "reclor_questions": RECLOR_HASH,
                "unexpected": OTHER_HASH,
            },
            ROLES,
        ),
        ({"math_questions": MATH_HASH, "reclor_questions": "A" * 64}, ROLES),
        ({"math_questions": MATH_HASH, "reclor_questions": "f" * 63}, ROLES),
        ({"math_questions": MATH_HASH, "reclor_questions": RECLOR_HASH}, {"math questions", "reclor_questions"}),
    ],
)
def test_role_bound_config_rejects_missing_unknown_or_invalid_bindings(bindings, allowed_roles):
    with pytest.raises(ValueError):
        role_bound_stage_config({"split_seed": 42}, bindings, allowed_roles)


@pytest.mark.parametrize("parameters", [None, [], "split_seed=42"])
def test_role_bound_config_requires_mapping_parameters(parameters):
    with pytest.raises(TypeError):
        role_bound_stage_config(parameters, {"math_questions": MATH_HASH, "reclor_questions": RECLOR_HASH}, ROLES)


def test_existing_task_1_to_6_run_id_golden_is_unchanged():
    assert run_id("generation", {"seed": 42}) == "cd1715278eee99f9bbcb67bd25661a2dbd284ac0f8270ae58278de228b645c62"
