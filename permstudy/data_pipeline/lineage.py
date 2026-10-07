"""Role-tagged immutable-manifest bindings for downstream stage configs."""

from collections.abc import Collection, Mapping
import re


LINEAGE_SCHEMA = "role_tagged_upstream_v1"

_HASH = re.compile(r"[0-9a-f]{64}\Z")
_ROLE = re.compile(r"[a-z][a-z0-9_]*\Z")


def role_bound_stage_config(
    parameters: Mapping[str, object],
    upstream_bindings: Mapping[str, str],
    allowed_roles: Collection[str],
) -> dict[str, object]:
    """Return a canonical config whose upstream hashes retain their stage roles."""
    if not isinstance(parameters, Mapping):
        raise TypeError("parameters must be a mapping")
    if not isinstance(upstream_bindings, Mapping):
        raise TypeError("upstream_bindings must be a mapping")
    if isinstance(allowed_roles, (str, bytes)) or not isinstance(allowed_roles, Collection):
        raise TypeError("allowed_roles must be a collection")

    roles = set(allowed_roles)
    if not roles or any(not isinstance(role, str) or not _ROLE.fullmatch(role) for role in roles):
        raise ValueError("allowed_roles contains an invalid role")
    if set(upstream_bindings) != roles:
        raise ValueError("upstream_bindings must contain exactly the allowed roles")
    if any(not isinstance(value, str) or not _HASH.fullmatch(value) for value in upstream_bindings.values()):
        raise ValueError("upstream manifest hashes must be lowercase SHA256 values")

    return {
        "lineage_schema": LINEAGE_SCHEMA,
        "parameters": dict(parameters),
        "upstream_bindings": {role: upstream_bindings[role] for role in sorted(upstream_bindings)},
    }
