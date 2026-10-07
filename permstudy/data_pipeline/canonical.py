"""Versioned text normalization and deterministic UTF-8 serialization."""

from collections.abc import Iterable, Mapping
import hashlib
import json
import unicodedata


def normalize_text_v1(value: str) -> str:
    """Normalize Unicode and line endings, preserving internal whitespace."""
    return unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n").strip()


def canonical_json_bytes(value: object) -> bytes:
    """Serialize JSON without optional whitespace, escapes, or a final newline."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_jsonl_bytes(records: Iterable[Mapping[str, object]], sort_key: str) -> bytes:
    """Sort records by the requested key and terminate each with one LF."""
    return b"".join(
        canonical_json_bytes(dict(record)) + b"\n"
        for record in sorted(records, key=lambda record: record[sort_key])
    )


def sha256_hex(data: bytes) -> str:
    """Return the full lowercase SHA256 digest of exact bytes."""
    return hashlib.sha256(data).hexdigest()
