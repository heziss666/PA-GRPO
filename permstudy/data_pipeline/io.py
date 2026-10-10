"""Crash-safe manifest, append-only JSONL, recovery, and single-writer shard locks.

Two module-local error types keep this layer's failures unambiguous:

* ``IntegrityError`` reports data that is wrong at rest: a malformed committed
  line, a record whose hash does not match its payload, or an artifact that does
  not match the digest recorded for it. It is deliberately not
  ``isolation.py``'s storage-boundary error.
* ``LockError`` reports that an operation cannot proceed now: an exclusive shard
  lock is held, or a lock file cannot be trusted well enough to recover it.

Every durability boundary in this module is a real ``flush`` followed by
``os.fsync``: records are appended, manifests are swapped in atomically, and the
only in-place mutation is truncating a JSONL back to its last committed line.
Records carry a ``record_hash`` over their canonical payload, and manifest
digests exclude the envelope fields that would otherwise reference the digest
itself.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import socket
import tempfile
from types import MappingProxyType

from psutil import pid_exists

from .canonical import canonical_json_bytes, sha256_hex
from .isolation import resolve_path


class IntegrityError(ValueError):
    """Stored data does not match its recorded digest or canonical shape."""


class LockError(RuntimeError):
    """A shard lock is held, or cannot be trusted well enough to recover it."""


# ``os.open`` on Windows opens in the CRT's text mode unless this flag is set,
# which would rewrite record bytes on the way to disk.
_BINARY = getattr(os, "O_BINARY", 0)
_EXCLUSIVE_CREATE = os.O_CREAT | os.O_EXCL | os.O_WRONLY | _BINARY
# Manifest envelope fields that must never feed the manifest's own digest.
_ENVELOPE_FIELDS = frozenset({"created_at_utc", "output_manifest_hash"})
_LOCK_FIELDS = frozenset({"run_id", "shard_id", "host", "pid", "created_at"})
_LOCK_IDENTITY_FIELDS = frozenset({"run_id", "shard_id"})
_MAX_LOCK_ATTEMPTS = 3


@dataclass(frozen=True)
class JsonlScan:
    """Validated records, the private quarantine path, and the file digest."""

    records: tuple[dict[str, object], ...]
    quarantined_tail_path: Path | None
    file_sha256: str


@dataclass(frozen=True)
class ShardLock:
    """A held shard lock and the exact envelope stored for it."""

    lock_path: Path
    metadata: Mapping[str, object]

    @classmethod
    def acquire(cls, lock_path, metadata, recover_stale=False) -> "ShardLock":
        """Create the lock exclusively, optionally replacing a dead holder's lock.

        ``metadata`` carries only the ``run_id``/``shard_id`` identity. The
        holder's ``host``, ``pid``, and ``created_at`` are stamped here, because
        staleness is only meaningful when the pid is the acquiring process's.
        Recovery needs all three of ``recover_stale=True``, matching run/shard
        identity, and a pid that ``psutil`` reports as dead.

        Before removing a stale lock, recovery re-reads it and refuses to proceed
        unless its bytes are identical to the lock it just validated. That
        narrows the recover-versus-recover race, but it does not close it: the
        mandated ``O_CREAT|O_EXCL`` plus ``psutil`` primitives expose no atomic
        compare-and-delete, so two recoverers can still interleave between this
        re-read and the unlink, and each can end up holding the shard. Stale-lock
        recovery is therefore an explicit single-operator action: two recoverers
        must never run against one shard concurrently.
        """
        identity = _lock_identity(metadata)
        path = Path(lock_path)
        envelope = {**identity, "host": socket.gethostname(), "pid": os.getpid(), "created_at": _utc_now()}
        payload = canonical_json_bytes(envelope)
        for _ in range(_MAX_LOCK_ATTEMPTS):
            try:
                descriptor = os.open(path, _EXCLUSIVE_CREATE, 0o600)
            except FileExistsError as exc:
                if not recover_stale:
                    raise LockError("shard lock is held by another process") from exc
                try:
                    stale = path.read_bytes()
                except FileNotFoundError:
                    # The holder released it between the create and the read.
                    continue
                recorded = _parse_lock(stale)
                if identity["run_id"] != recorded["run_id"] or identity["shard_id"] != recorded["shard_id"]:
                    raise LockError("shard lock belongs to a different run or shard") from exc
                if pid_exists(recorded["pid"]):
                    raise LockError("shard lock is held by a live process") from exc
                # Never remove a lock whose bytes are not the ones just validated:
                # a recoverer that lost the race must fail loudly instead of
                # deleting another process's fresh lock.
                try:
                    current = path.read_bytes()
                except FileNotFoundError as missing:
                    raise LockError("shard lock changed during recovery") from missing
                if current != stale:
                    raise LockError("shard lock changed during recovery")
                _discard(path)
                continue
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
            except BaseException:
                # A half-written lock would have no usable identity and could
                # never be recovered, so drop it before propagating.
                _discard(path)
                raise
            return cls(path, MappingProxyType(envelope))
        raise LockError("shard lock could not be acquired under contention")

    def release(self) -> None:
        """Remove the lock file only while this process still owns it."""
        try:
            recorded = _read_lock(self.lock_path)
        except FileNotFoundError as exc:
            raise LockError("shard lock file is missing") from exc
        if (
            recorded["pid"] != os.getpid()
            or recorded["run_id"] != self.metadata["run_id"]
            or recorded["shard_id"] != self.metadata["shard_id"]
        ):
            raise LockError("shard lock is not owned by this process")
        try:
            self.lock_path.unlink()
        except FileNotFoundError as exc:
            raise LockError("shard lock file is missing") from exc

    def __enter__(self) -> "ShardLock":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Release on exit without ever masking an exception from the body."""
        if exc_type is None:
            self.release()
            return
        try:
            self.release()
        except LockError:
            pass


def write_atomic_manifest(path, payload) -> str:
    """Write the manifest envelope atomically and return its payload digest.

    ``created_at_utc`` and ``output_manifest_hash`` are stored in the envelope
    but excluded from the digest, so the digest never references itself and is
    reproducible for identical content.
    """
    path = Path(path)
    if not isinstance(payload, Mapping):
        raise TypeError("manifest payload must be a mapping")
    envelope = dict(payload)
    created_at = envelope.get("created_at_utc")
    if not isinstance(created_at, str) or not created_at.strip():
        created_at = _utc_now()
    digest = sha256_hex(
        canonical_json_bytes({key: value for key, value in envelope.items() if key not in _ENVELOPE_FIELDS})
    )
    envelope["created_at_utc"] = created_at
    envelope["output_manifest_hash"] = digest
    # Serialize before touching the filesystem so a bad payload changes nothing.
    _replace_atomically(path, canonical_json_bytes(envelope))
    return digest


def append_record(path, payload) -> str:
    """Append one canonical record line durably and return its record hash."""
    path = Path(path)
    if not isinstance(payload, Mapping):
        raise TypeError("record payload must be a mapping")
    # The I/O layer owns record_hash: a caller-supplied value is never trusted.
    record = {key: value for key, value in payload.items() if key != "record_hash"}
    digest = sha256_hex(canonical_json_bytes(record))
    _require_committed_tail(path)
    with path.open("ab") as stream:
        stream.write(canonical_json_bytes({**record, "record_hash": digest}) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    return digest


def scan_jsonl(path, recover_incomplete_tail=False) -> JsonlScan:
    """Validate every record and, on request, quarantine an uncommitted tail.

    Only bytes after the last line feed can be an uncommitted tail. A malformed
    line that is already terminated by a line feed is a committed record and
    stays an integrity failure; it is never repaired automatically.
    """
    path = Path(path)
    data = path.read_bytes()
    complete = data.rfind(b"\n") + 1
    tail = data[complete:]
    records = tuple(
        _parse_record(raw, number)
        for number, raw in enumerate(data[:complete].split(b"\n")[:-1], start=1)
    )
    if not tail:
        return JsonlScan(records, None, sha256_hex(data))
    if not recover_incomplete_tail:
        raise IntegrityError("incomplete final line requires recovery with recover_incomplete_tail=True")
    quarantined = _quarantine_tail(path, tail)
    # Quarantine first, then shrink the source, and only then report success:
    # the repaired original is durable before a caller may append again.
    with path.open("r+b") as stream:
        stream.truncate(complete)
        stream.flush()
        os.fsync(stream.fileno())
    return JsonlScan(records, quarantined, sha256_hex(data[:complete]))


def verify_artifact_ref(data_root, ref) -> None:
    """Fail unless the artifact resolves inside ``data_root`` and matches its digest."""
    if not isinstance(ref.relative_path, str) or not isinstance(ref.sha256, str):
        raise IntegrityError("artifact reference is malformed")
    root = resolve_path(Path(data_root))
    target = resolve_path(root / ref.relative_path)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise IntegrityError(f"artifact path escapes the data root: {ref.relative_path}") from exc
    try:
        actual = sha256_hex(target.read_bytes())
    except (FileNotFoundError, IsADirectoryError) as exc:
        raise IntegrityError("artifact is not a readable file") from exc
    if ref.sha256.lower() != actual:
        raise IntegrityError("artifact sha256 does not match the file on disk")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _discard(path: Path) -> None:
    """Remove a file this process created and no longer wants; keep the first error."""
    try:
        path.unlink()
    except OSError:
        pass


def _replace_atomically(path: Path, data: bytes) -> None:
    """Write a sibling temp file, fsync it, then replace the destination."""
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        _discard(temporary)
        raise


def _require_committed_tail(path: Path) -> None:
    """Refuse to extend a file whose final line was never committed."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb") as stream:
        stream.seek(-1, os.SEEK_END)
        if stream.read(1) == b"\n":
            return
    raise IntegrityError("cannot append after an incomplete final line: explicit recovery is required")


def _quarantine_tail(path: Path, tail: bytes) -> Path:
    """Write the removed bytes to the next free quarantine file beside the source."""
    for index in itertools.count():
        candidate = path.with_name(f"{path.stem}.{index}.quarantine")
        try:
            descriptor = os.open(candidate, _EXCLUSIVE_CREATE, 0o600)
        except FileExistsError:
            continue
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(tail)
            stream.flush()
            os.fsync(stream.fileno())
        return candidate


def _object_pairs(pairs):
    """Reject duplicate keys so a parsed record has exactly one interpretation."""
    record = {}
    for key, value in pairs:
        if key in record:
            raise ValueError("duplicate object key")
        record[key] = value
    return record


def _reject_constant(value):
    raise ValueError(f"non-standard JSON constant {value}")


def _parse_record(raw: bytes, number: int) -> dict[str, object]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise IntegrityError(f"malformed middle line {number}: invalid UTF-8") from exc
    try:
        record = json.loads(text, object_pairs_hook=_object_pairs, parse_constant=_reject_constant)
    except ValueError as exc:
        raise IntegrityError(f"malformed middle line {number}: not valid JSON") from exc
    if not isinstance(record, dict):
        raise IntegrityError(f"malformed middle line {number}: not a JSON object")
    if "record_hash" not in record:
        raise IntegrityError(f"missing record hash at middle line {number}")
    payload = {key: value for key, value in record.items() if key != "record_hash"}
    if record["record_hash"] != sha256_hex(canonical_json_bytes(payload)):
        raise IntegrityError(f"record hash mismatch at middle line {number}")
    return record


def _lock_identity(metadata) -> dict[str, str]:
    if not isinstance(metadata, Mapping) or set(metadata) != _LOCK_IDENTITY_FIELDS:
        raise LockError("lock metadata must contain exactly run_id and shard_id")
    identity = {}
    for name in ("run_id", "shard_id"):
        value = metadata[name]
        if not isinstance(value, str) or not value.strip():
            raise LockError("lock identity values must be non-empty text")
        identity[name] = value
    return identity


def _read_lock(lock_path: Path) -> dict[str, object]:
    """Read a lock file, rejecting anything that does not match the contract."""
    return _parse_lock(lock_path.read_bytes())


def _parse_lock(raw: bytes) -> dict[str, object]:
    """Validate exact lock envelope bytes against the lock contract."""
    try:
        envelope = json.loads(raw.decode("utf-8"), object_pairs_hook=_object_pairs, parse_constant=_reject_constant)
    except (UnicodeDecodeError, ValueError) as exc:
        raise LockError("lock file is not valid JSON") from exc
    if not isinstance(envelope, dict):
        raise LockError("lock file is not a JSON object")
    if _LOCK_FIELDS - set(envelope):
        raise LockError("lock file is missing required identity fields")
    if set(envelope) - _LOCK_FIELDS:
        raise LockError("lock file has unsupported fields")
    for name in ("run_id", "shard_id", "host", "created_at"):
        if not isinstance(envelope[name], str) or not envelope[name].strip():
            raise LockError("lock identity values must be non-empty text")
    if type(envelope["pid"]) is not int or envelope["pid"] <= 0:
        raise LockError("lock pid must be a positive integer")
    return envelope
