"""Resolved external storage checks and fail-closed public output boundaries."""

from collections.abc import Mapping
from dataclasses import dataclass
import math
import os
from pathlib import Path, PureWindowsPath
import re

from .canonical import sha256_hex


class IsolationError(ValueError):
    """Runtime storage does not satisfy external isolation."""


class SafeLoggingError(ValueError):
    """A payload cannot safely cross the public output boundary."""


@dataclass(frozen=True)
class IsolationReport:
    repo_root_hash: str
    data_root_hash: str
    effective_cache_kinds: tuple[str, ...]


PUBLIC_LOG_KEYS = frozenset({
    "candidate_id", "original_question_id", "generator_id", "shard_id",
    "status", "error_type", "retry_count", "count", "elapsed_seconds",
})
PUBLIC_MANIFEST_KEYS = frozenset({
    "run_id", "source", "source_revision", "split_manifest_hash",
    "question_count", "candidate_count", "pair_count", "generator_model",
    "generator_revision", "status_counts", "aggregate_statistics",
    "artifact_sha256", "real_generation_performed", "phase_status",
})
PUBLIC_EVIDENCE_KEYS = frozenset({
    "schema_version", "git_sha", "platform", "python_version",
    "package_versions", "run_ids", "source_revisions", "source_file_hashes",
    "tree_manifest_hash", "question_counts", "duplicate_counts",
    "stratification_level", "fallback_reasons", "split_manifest_hash",
    "semantic_candidate_set_hash", "artifact_hashes", "completion_counts",
    "verification_status_counts", "question_verification_status_counts",
    "pair_count", "permutation_count", "real_generation_performed", "phase_status",
})
_SENSITIVE_WORDS = frozenset({
    "question", "response", "gold", "prompt", "solution", "context", "answer",
    "questions", "responses", "prompts", "solutions", "contexts", "answers",
    "token", "secret", "password", "path",
})
_SAFE_STRING = re.compile(r"[A-Za-z0-9_.:/+@=,()\-]+\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")


def resolve_path(path: Path) -> Path:
    """Resolve aliases and non-existent descendants before comparisons."""
    return Path(path).resolve(strict=False)


def _normalized(path: Path) -> str:
    return os.path.normcase(str(resolve_path(path)))


def _contains(parent: str, child: str) -> bool:
    try:
        return os.path.commonpath((parent, child)) == parent
    except ValueError:
        # Separate Windows drives have no containment relationship.
        return False


def _effective_caches(env: Mapping[str, str]) -> dict[str, Path]:
    cache_base = Path(env.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    hf_home = Path(env.get("HF_HOME", str(cache_base / "huggingface")))
    hub = Path(env.get("HF_HUB_CACHE", env.get("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))))
    legacy_bert = env.get("PYTORCH_PRETRAINED_BERT_CACHE", str(hub))
    legacy_transformers = env.get("PYTORCH_TRANSFORMERS_CACHE", legacy_bert)
    caches = {
        "HF_HOME": hf_home,
        "HUGGINGFACE_HUB_CACHE": hub,
        "TRANSFORMERS_CACHE": Path(env.get("TRANSFORMERS_CACHE", legacy_transformers)),
        "HF_DATASETS_CACHE": Path(env.get("HF_DATASETS_CACHE", str(hf_home / "datasets"))),
    }
    # Check supplied legacy/new aliases too: another installed version may use them.
    for kind in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "PYTORCH_PRETRAINED_BERT_CACHE", "PYTORCH_TRANSFORMERS_CACHE"):
        if kind in env:
            caches[kind] = Path(env[kind])
    return caches


def validate_external_roots(repo_root: Path, data_root: Path, env: Mapping[str, str]) -> IsolationReport:
    """Validate roots and effective HF caches without exposing runtime paths."""
    repo, data = _normalized(repo_root), _normalized(data_root)
    if _contains(repo, data) or _contains(data, repo):
        raise IsolationError("repository and data roots must not contain one another")
    caches = _effective_caches(env)
    if any(_contains(repo, _normalized(path)) for path in caches.values()):
        raise IsolationError("effective cache must be external to the repository")
    return IsolationReport(
        sha256_hex(repo.encode("utf-8")), sha256_hex(data.encode("utf-8")), tuple(sorted(caches)),
    )


def sanitize_exception(error: BaseException) -> str:
    """Map exception types only; never inspect or format third-party messages."""
    if isinstance(error, TimeoutError):
        return "timeout"
    if isinstance(error, MemoryError):
        return "oom"
    if isinstance(error, ImportError):
        return "dependency_error"
    if isinstance(error, PermissionError):
        return "access_denied"
    # External libraries often expose their own subclasses of these categories.
    names = {cls.__name__ for cls in type(error).__mro__}
    for category, type_names in (
        ("timeout", {"Timeout", "TimeoutException", "ReadTimeout", "ConnectTimeout"}),
        ("oom", {"OutOfMemoryError"}),
        ("access_denied", {"GatedRepoError", "AuthenticationError", "AccessDeniedError"}),
        ("dependency_error", {"DependencyError"}),
    ):
        if names & type_names:
            return category
    return "unexpected_error"


def _safe_string(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(_SAFE_STRING.fullmatch(value))
        and not value.startswith("/")
        and not PureWindowsPath(value).drive
        and "hf_" not in value.lower()
        and not any(part == ".." for part in value.split("/"))
    )


def _number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _count(value: object) -> bool:
    return type(value) is int and value >= 0


def safe_log_fields(fields: Mapping[str, object]) -> dict[str, object]:
    """Reject unknown fields and unsafe values; never silently drop secrets."""
    if not isinstance(fields, Mapping) or not set(fields) <= PUBLIC_LOG_KEYS:
        raise SafeLoggingError("public log contains unsupported fields")
    result = {}
    for key, value in fields.items():
        valid = (_count(value) if key in {"count", "retry_count"}
                 else _number(value) if key == "elapsed_seconds" else _safe_string(value))
        if not valid:
            raise SafeLoggingError("public log contains unsafe values")
        result[key] = value
    return result


def _nested(value: object, kind: str = "scalar") -> object:
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            words = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key) if isinstance(key, str) else ""
            if not _safe_string(key) or set(re.split(r"[^a-z0-9]+", words.lower())) & _SENSITIVE_WORDS:
                raise SafeLoggingError("public manifest contains unsafe nested fields")
            result[key] = _nested(item, kind)
        return result
    if isinstance(value, (list, tuple)):
        return [_nested(item, kind) for item in value]
    if kind == "count":
        valid = _count(value)
    elif kind == "hash":
        valid = isinstance(value, str) and bool(_HASH.fullmatch(value))
    elif kind == "version":
        valid = _safe_string(value)
    else:
        valid = value is None or type(value) is bool or _number(value) or _safe_string(value)
    if not valid:
        raise SafeLoggingError("public manifest contains unsafe nested values")
    return value


def sanitize_public_manifest(payload: Mapping[str, object]) -> dict[str, object]:
    """Copy a single approved schema with recursively validated safe values."""
    if not isinstance(payload, Mapping) or not (
        set(payload) <= PUBLIC_MANIFEST_KEYS or set(payload) <= PUBLIC_EVIDENCE_KEYS
    ):
        raise SafeLoggingError("public manifest contains unsupported fields")
    result = {}
    for key, value in payload.items():
        if key == "real_generation_performed":
            if type(value) is not bool:
                raise SafeLoggingError("public manifest contains unsafe values")
            result[key] = value
            continue
        kind = ("count" if key.endswith(("_count", "_counts")) else
                "hash" if key.endswith(("_hash", "_hashes", "_sha256")) else
                "version" if key.endswith(("_version", "_versions")) else "scalar")
        scalar_fields = {
            "run_id", "source", "source_revision", "generator_model", "generator_revision",
            "phase_status", "git_sha", "platform", "python_version", "schema_version",
        }
        if key in scalar_fields and not _safe_string(value):
            raise SafeLoggingError("public manifest contains unsafe scalar values")
        if key.endswith("_count") and not _count(value):
            raise SafeLoggingError("public manifest contains unsafe count values")
        if key.endswith(("_hash", "_sha256")) and not (isinstance(value, str) and _HASH.fullmatch(value)):
            raise SafeLoggingError("public manifest contains unsafe hash values")
        result[key] = _nested(value, kind)
    return result
