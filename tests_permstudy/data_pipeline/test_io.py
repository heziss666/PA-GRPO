"""Crash safety for manifests, append-only JSONL, recovery, and shard locks.

Every case here uses real files, real ``fsync`` calls, and real ``O_EXCL``
creation; nothing about the durability boundary is mocked away.
"""

import importlib
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import socket
import subprocess

import psutil
import pytest

from permstudy.data_pipeline.canonical import canonical_json_bytes, sha256_hex
from permstudy.data_pipeline.schema import ArtifactRef


def load_io():
    """Import inside the call so the initial RED run fails as an import error."""
    return importlib.import_module("permstudy.data_pipeline.io")


def dead_pid():
    """Return a positive PID that this machine reports as unused."""
    candidate = 60000
    while psutil.pid_exists(candidate):
        candidate += 4
        assert candidate < 90000, "no unused PID available for this test"
    return candidate


def stale_lock(tmp_path, omit=(), **overrides):
    """Write a lock file directly, the way a crashed holder would leave it."""
    payload = {
        "run_id": "run-1",
        "shard_id": "shard-0",
        "host": "crashed-host",
        "pid": dead_pid(),
        "created_at": "2026-10-06T00:00:00.000000Z",
    }
    payload.update(overrides)
    for name in omit:
        payload.pop(name)
    path = tmp_path / "shard-0.lock"
    path.write_bytes(canonical_json_bytes(payload))
    return path, payload


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


# --- append-only JSONL records -------------------------------------------------


def test_record_hash_is_stable_and_matches_canonical_payload(tmp_path):
    io = load_io()
    expected = sha256_hex(canonical_json_bytes({"a": "x", "b": [1, 2]}))
    first = io.append_record(tmp_path / "one.jsonl", {"b": [1, 2], "a": "x"})
    second = io.append_record(tmp_path / "two.jsonl", {"a": "x", "b": [1, 2]})
    assert first == second == expected
    line = canonical_json_bytes({"a": "x", "b": [1, 2], "record_hash": expected}) + b"\n"
    assert (tmp_path / "one.jsonl").read_bytes() == line
    assert (tmp_path / "two.jsonl").read_bytes() == line


def test_append_record_owns_the_record_hash_and_round_trips(tmp_path):
    io = load_io()
    source = tmp_path / "source.jsonl"
    digest = io.append_record(source, {"id": "a", "nested": {"k": None}})
    record = io.scan_jsonl(source).records[0]
    assert record == {"id": "a", "nested": {"k": None}, "record_hash": digest}
    target = tmp_path / "target.jsonl"
    assert io.append_record(target, record) == digest
    assert target.read_bytes() == source.read_bytes()
    assert io.append_record(target, {**record, "record_hash": "0" * 64}) == digest
    assert target.read_bytes() == source.read_bytes() * 2


def test_append_record_persists_the_line_before_returning(tmp_path, monkeypatch):
    io = load_io()
    path = tmp_path / "records.jsonl"
    flushed = []
    real_fsync = os.fsync

    def spy(fd):
        flushed.append(path.read_bytes())
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", spy)
    digest = io.append_record(path, {"id": "a"})
    assert flushed == [canonical_json_bytes({"id": "a", "record_hash": digest}) + b"\n"]


def test_append_record_refuses_to_extend_an_incomplete_tail(tmp_path):
    io = load_io()
    path = tmp_path / "records.jsonl"
    io.append_record(path, {"id": "a"})
    with path.open("ab") as stream:
        stream.write(b'{"id":')
    before = path.read_bytes()
    with pytest.raises(io.IntegrityError, match="cannot append"):
        io.append_record(path, {"id": "b"})
    assert path.read_bytes() == before


def test_append_record_rejects_a_non_mapping_payload(tmp_path):
    io = load_io()
    with pytest.raises(TypeError):
        io.append_record(tmp_path / "records.jsonl", [("id", "a")])


# --- scanning and tail recovery ------------------------------------------------


def test_incomplete_tail_is_quarantined_but_middle_corruption_fails(tmp_path):
    io = load_io()

    path = tmp_path / "records.jsonl"
    io.append_record(path, {"id": "a"})
    with path.open("ab") as stream:
        stream.write(b'{"id":')
    scan = io.scan_jsonl(path, recover_incomplete_tail=True)
    assert scan.quarantined_tail_path is not None
    assert scan.quarantined_tail_path.read_bytes() == b'{"id":'
    assert path.read_bytes().endswith(b"\n")
    assert io.scan_jsonl(path).records == scan.records

    path.write_bytes(b'{"id":\n{"id":"b"}\n')
    with pytest.raises(io.IntegrityError, match="middle line"):
        io.scan_jsonl(path, recover_incomplete_tail=True)


def test_scan_rejects_an_incomplete_tail_without_explicit_recovery(tmp_path):
    io = load_io()
    path = tmp_path / "records.jsonl"
    io.append_record(path, {"id": "a"})
    with path.open("ab") as stream:
        stream.write(b'{"id":')
    before = path.read_bytes()
    with pytest.raises(io.IntegrityError, match="recover_incomplete_tail"):
        io.scan_jsonl(path)
    assert path.read_bytes() == before
    assert sorted(tmp_path.glob("*.quarantine")) == []


def test_recovery_quarantines_exact_bytes_and_leaves_valid_jsonl(tmp_path):
    io = load_io()
    path = tmp_path / "shard-0.jsonl"
    io.append_record(path, {"id": "a"})
    io.append_record(path, {"id": "b"})
    tail = b'{"id":"c","record_ha'
    with path.open("ab") as stream:
        stream.write(tail)

    scan = io.scan_jsonl(path, recover_incomplete_tail=True)

    repaired = path.read_bytes()
    assert repaired.endswith(b"\n")
    assert scan.quarantined_tail_path.read_bytes() == tail
    assert scan.quarantined_tail_path.parent == tmp_path
    assert [record["id"] for record in scan.records] == ["a", "b"]
    assert [json.loads(line) for line in repaired.splitlines()] == list(scan.records)
    assert scan.file_sha256 == sha256_hex(repaired)

    # The repaired file is valid JSONL before any resume, and appends continue.
    assert io.scan_jsonl(path).file_sha256 == scan.file_sha256
    ref = ArtifactRef("shard-0.jsonl", scan.file_sha256, len(scan.records))
    assert io.verify_artifact_ref(tmp_path, ref) is None
    assert io.append_record(path, {"id": "c"}) == io.scan_jsonl(path).records[-1]["record_hash"]
    assert [record["id"] for record in io.scan_jsonl(path).records] == ["a", "b", "c"]


def test_repeated_recovery_never_overwrites_earlier_quarantine(tmp_path):
    io = load_io()
    path = tmp_path / "records.jsonl"
    io.append_record(path, {"id": "a"})
    tails = [b'{"id":', b'{"broken":']
    quarantined = []
    for tail in tails:
        with path.open("ab") as stream:
            stream.write(tail)
        quarantined.append(io.scan_jsonl(path, recover_incomplete_tail=True).quarantined_tail_path)
    assert len(set(quarantined)) == 2
    assert [item.name for item in quarantined] == ["records.0.quarantine", "records.1.quarantine"]
    assert [item.read_bytes() for item in quarantined] == tails


def test_quarantine_skips_an_existing_index_without_overwriting_it(tmp_path):
    io = load_io()
    path = tmp_path / "records.jsonl"
    io.append_record(path, {"id": "a"})
    existing = tmp_path / "records.0.quarantine"
    existing.write_bytes(b"earlier evidence")
    with path.open("ab") as stream:
        stream.write(b'{"id":')
    scan = io.scan_jsonl(path, recover_incomplete_tail=True)
    assert scan.quarantined_tail_path.name == "records.1.quarantine"
    assert existing.read_bytes() == b"earlier evidence"


def test_recovery_fsyncs_quarantine_then_truncates_then_fsyncs_the_original(tmp_path, monkeypatch):
    io = load_io()
    path = tmp_path / "records.jsonl"
    io.append_record(path, {"id": "a"})
    tail = b'{"id":'
    complete = path.read_bytes()
    with path.open("ab") as stream:
        stream.write(tail)

    observed = []
    real_fsync = os.fsync

    def spy(fd):
        observed.append((
            [item.read_bytes() for item in sorted(tmp_path.glob("*.quarantine"))],
            path.read_bytes(),
        ))
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", spy)
    io.scan_jsonl(path, recover_incomplete_tail=True)

    assert observed == [
        ([tail], complete + tail),   # quarantine durable before the source is touched
        ([tail], complete),          # source already truncated when it is fsynced
    ]


@pytest.mark.parametrize(
    "middle",
    [b'{"id":', b'{"id":"x"}', b'{"id":"x","record_hash":"' + b"0" * 64 + b'"}',
     b"[1,2,3]", b"\xff\xfe\x00", b"", b'{"id":"x","id":"y"}'],
    ids=["truncated-json", "missing-hash", "bad-hash", "not-an-object", "invalid-utf8",
         "blank-line", "duplicate-key"],
)
def test_malformed_middle_line_is_never_repaired(tmp_path, middle):
    io = load_io()
    path = tmp_path / "records.jsonl"
    first = io.append_record(path, {"id": "a"})
    last = io.append_record(path, {"id": "z"})
    path.write_bytes(
        canonical_json_bytes({"id": "a", "record_hash": first}) + b"\n"
        + middle + b"\n"
        + canonical_json_bytes({"id": "z", "record_hash": last}) + b"\n"
    )
    before = path.read_bytes()
    with pytest.raises(io.IntegrityError, match="middle line"):
        io.scan_jsonl(path, recover_incomplete_tail=True)
    assert path.read_bytes() == before
    assert sorted(tmp_path.glob("*.quarantine")) == []


def test_malformed_final_complete_line_is_not_treated_as_a_tail(tmp_path):
    io = load_io()
    path = tmp_path / "records.jsonl"
    io.append_record(path, {"id": "a"})
    path.write_bytes(path.read_bytes() + b'{"id":\n')
    before = path.read_bytes()
    with pytest.raises(io.IntegrityError, match="middle line"):
        io.scan_jsonl(path, recover_incomplete_tail=True)
    assert path.read_bytes() == before
    assert sorted(tmp_path.glob("*.quarantine")) == []


def test_scan_rejects_a_tampered_record_hash(tmp_path):
    io = load_io()
    path = tmp_path / "records.jsonl"
    io.append_record(path, {"id": "a"})
    path.write_bytes(canonical_json_bytes({"id": "a", "record_hash": "0" * 64}) + b"\n")
    with pytest.raises(io.IntegrityError, match="record hash mismatch"):
        io.scan_jsonl(path)


def test_scan_of_an_empty_file_reports_no_records(tmp_path):
    io = load_io()
    path = tmp_path / "records.jsonl"
    path.write_bytes(b"")
    scan = io.scan_jsonl(path)
    assert scan.records == ()
    assert scan.quarantined_tail_path is None
    assert scan.file_sha256 == sha256_hex(b"")


# --- immutable manifests -------------------------------------------------------


def test_manifest_fsyncs_the_temp_file_before_an_atomic_replace(tmp_path, monkeypatch):
    io = load_io()
    path = tmp_path / "run" / "manifest.json"
    path.parent.mkdir()
    events = []
    real_fsync, real_replace = os.fsync, os.replace

    def spy_fsync(fd):
        events.append("fsync")
        return real_fsync(fd)

    def spy_replace(source, destination):
        temporary = Path(source)
        assert temporary.parent == path.parent          # same filesystem: replace is atomic
        assert temporary != path
        assert not path.exists()                        # nothing is exposed before the swap
        events.append(("replace", temporary.read_bytes()))
        return real_replace(source, destination)

    monkeypatch.setattr(os, "fsync", spy_fsync)
    monkeypatch.setattr(os, "replace", spy_replace)
    io.write_atomic_manifest(path, {"stage": "split", "counts": {"a": 1}})

    assert [event if isinstance(event, str) else event[0] for event in events] == ["fsync", "replace"]
    assert events[1][1] == path.read_bytes()
    assert [item.name for item in path.parent.iterdir()] == ["manifest.json"]


def test_manifest_digest_excludes_the_envelope_and_is_reproducible(tmp_path):
    io = load_io()
    base = {
        "schema_version": "run_manifest_v1",
        "stage": "split",
        "run_id": "r1",
        "config_hash": "a" * 64,
        "upstream_manifest_hashes": [],
        "artifacts": [],
        "counts": {"questions": 3},
    }
    expected = sha256_hex(canonical_json_bytes(base))
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    assert io.write_atomic_manifest(
        first, {**base, "created_at_utc": "2026-10-06T00:00:00.000000Z", "output_manifest_hash": "b" * 64}
    ) == expected
    assert io.write_atomic_manifest(
        second, {**base, "created_at_utc": "2027-01-01T12:00:00.000000Z"}
    ) == expected

    envelope = read_json(first)
    assert envelope == {
        **base,
        "created_at_utc": "2026-10-06T00:00:00.000000Z",
        "output_manifest_hash": expected,
    }
    assert first.read_bytes() == canonical_json_bytes(envelope)
    assert read_json(second)["created_at_utc"] == "2027-01-01T12:00:00.000000Z"

    # Re-writing identical content reproduces identical bytes.
    assert io.write_atomic_manifest(first, envelope) == expected
    assert read_json(first) == envelope
    # Every field outside the envelope participates in the digest.
    assert io.write_atomic_manifest(
        tmp_path / "third.json", {**base, "counts": {"questions": 4}}
    ) != expected


def test_manifest_stamps_a_utc_created_at_when_the_caller_omits_one(tmp_path):
    io = load_io()
    path = tmp_path / "manifest.json"
    payload = {"stage": "split", "counts": {}}
    digest = io.write_atomic_manifest(path, payload)
    envelope = read_json(path)
    assert set(envelope) == {"stage", "counts", "created_at_utc", "output_manifest_hash"}
    created = datetime.fromisoformat(envelope["created_at_utc"].replace("Z", "+00:00"))
    assert created.tzinfo is not None
    assert created.utcoffset() == timedelta(0)
    assert envelope["output_manifest_hash"] == digest == sha256_hex(canonical_json_bytes(payload))


def test_manifest_serialization_failure_leaves_the_destination_untouched(tmp_path):
    io = load_io()
    path = tmp_path / "manifest.json"
    io.write_atomic_manifest(path, {"stage": "split"})
    before = path.read_bytes()
    with pytest.raises(TypeError):
        io.write_atomic_manifest(path, {"stage": object()})
    assert path.read_bytes() == before
    assert [item.name for item in tmp_path.iterdir()] == ["manifest.json"]


def test_manifest_rejects_a_non_mapping_payload(tmp_path):
    io = load_io()
    with pytest.raises(TypeError):
        io.write_atomic_manifest(tmp_path / "manifest.json", ["stage"])


# --- artifact verification -----------------------------------------------------


def test_verify_artifact_ref_accepts_a_relative_path(tmp_path):
    io = load_io()
    data_root = tmp_path / "data"
    artifact = data_root / "stage" / "records.jsonl"
    artifact.parent.mkdir(parents=True)
    io.append_record(artifact, {"id": "a"})
    ref = ArtifactRef("stage/records.jsonl", io.scan_jsonl(artifact).file_sha256, 1)
    assert io.verify_artifact_ref(data_root, ref) is None


def test_verify_artifact_ref_rejects_a_digest_mismatch(tmp_path):
    io = load_io()
    data_root = tmp_path / "data"
    data_root.mkdir()
    io.append_record(data_root / "records.jsonl", {"id": "a"})
    with pytest.raises(io.IntegrityError, match="sha256"):
        io.verify_artifact_ref(data_root, ArtifactRef("records.jsonl", "0" * 64, 1))


def test_verify_artifact_ref_rejects_a_missing_artifact(tmp_path):
    io = load_io()
    data_root = tmp_path / "data"
    data_root.mkdir()
    with pytest.raises(io.IntegrityError, match="not a readable file"):
        io.verify_artifact_ref(data_root, ArtifactRef("absent.jsonl", "a" * 64, 0))


def test_verify_artifact_ref_rejects_a_path_that_escapes_the_data_root(tmp_path):
    io = load_io()
    data_root = tmp_path / "data"
    data_root.mkdir()
    outside = tmp_path / "outside.jsonl"
    digest = io.append_record(outside, {"id": "leak"})
    # The digest matches the file exactly; only containment can reject this ref.
    with pytest.raises(io.IntegrityError, match="escapes the data root"):
        io.verify_artifact_ref(data_root, ArtifactRef("../outside.jsonl", digest, 1))


def test_verify_artifact_ref_rejects_a_malformed_reference(tmp_path):
    io = load_io()
    data_root = tmp_path / "data"
    data_root.mkdir()
    with pytest.raises(io.IntegrityError, match="malformed"):
        io.verify_artifact_ref(data_root, ArtifactRef("records.jsonl", None, 0))


@pytest.mark.skipif(os.name != "nt", reason="Windows junction behavior")
def test_verify_artifact_ref_rejects_a_junction_escape(tmp_path):
    io = load_io()
    data_root = tmp_path / "data"
    data_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    digest = io.append_record(outside / "records.jsonl", {"id": "leak"})
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(data_root / "link"), str(outside)], capture_output=True
    )
    if created.returncode:
        pytest.skip("OS does not permit junctions")
    with pytest.raises(io.IntegrityError, match="escapes the data root"):
        io.verify_artifact_ref(data_root, ArtifactRef("link/records.jsonl", digest, 1))


# --- single-writer shard locks -------------------------------------------------


def test_lock_stores_exactly_the_contract_fields(tmp_path):
    io = load_io()
    lock_path = tmp_path / "shard-0.lock"
    lock = io.ShardLock.acquire(lock_path, {"run_id": "run-1", "shard_id": "shard-0"})

    envelope = read_json(lock_path)
    assert set(envelope) == {"run_id", "shard_id", "host", "pid", "created_at"}
    assert envelope["run_id"] == "run-1"
    assert envelope["shard_id"] == "shard-0"
    assert envelope["host"] == socket.gethostname()
    assert envelope["pid"] == os.getpid()
    created = datetime.fromisoformat(envelope["created_at"].replace("Z", "+00:00"))
    assert created.utcoffset() == timedelta(0)
    assert lock_path.read_bytes() == canonical_json_bytes(envelope)
    assert b"\r" not in lock_path.read_bytes()
    assert dict(lock.metadata) == envelope
    assert lock.lock_path == lock_path

    lock.release()
    assert not lock_path.exists()


def test_lock_collision_without_recovery_leaves_the_holder_untouched(tmp_path):
    io = load_io()
    lock_path = tmp_path / "shard-0.lock"
    io.ShardLock.acquire(lock_path, {"run_id": "run-1", "shard_id": "shard-0"})
    before = lock_path.read_bytes()
    with pytest.raises(io.LockError, match="held"):
        io.ShardLock.acquire(lock_path, {"run_id": "run-1", "shard_id": "shard-0"})
    assert lock_path.read_bytes() == before


@pytest.mark.parametrize(
    "metadata",
    [{}, {"run_id": "run-1"}, {"shard_id": "shard-0"}, {"run_id": "run-1", "shard_id": ""},
     {"run_id": "run-1", "shard_id": "shard-0", "pid": 1234},
     {"run_id": "run-1", "shard_id": "shard-0", "extra": 1}],
    ids=["empty", "run-only", "shard-only", "blank", "pid-override", "extra-key"],
)
def test_lock_metadata_must_be_exactly_the_identity(tmp_path, metadata):
    io = load_io()
    lock_path = tmp_path / "shard-0.lock"
    with pytest.raises(io.LockError):
        io.ShardLock.acquire(lock_path, metadata)
    assert not lock_path.exists()


def test_lock_is_not_deleted_while_its_process_is_alive(tmp_path):
    io = load_io()
    lock_path = tmp_path / "shard-0.lock"
    io.ShardLock.acquire(lock_path, {"run_id": "run-1", "shard_id": "shard-0"})
    before = lock_path.read_bytes()
    with pytest.raises(io.LockError, match="live"):
        io.ShardLock.acquire(lock_path, {"run_id": "run-1", "shard_id": "shard-0"}, recover_stale=True)
    assert lock_path.read_bytes() == before


def test_stale_lock_is_never_deleted_without_explicit_recovery(tmp_path):
    io = load_io()
    lock_path, _ = stale_lock(tmp_path)
    before = lock_path.read_bytes()
    with pytest.raises(io.LockError, match="held"):
        io.ShardLock.acquire(lock_path, {"run_id": "run-1", "shard_id": "shard-0"})
    assert lock_path.read_bytes() == before


@pytest.mark.parametrize(
    "overrides",
    [{"run_id": "run-2"}, {"shard_id": "shard-9"}],
    ids=["other-run", "other-shard"],
)
def test_lock_recovery_rejects_a_different_run_or_shard(tmp_path, overrides):
    io = load_io()
    lock_path, _ = stale_lock(tmp_path, **overrides)
    before = lock_path.read_bytes()
    with pytest.raises(io.LockError, match="different run or shard"):
        io.ShardLock.acquire(lock_path, {"run_id": "run-1", "shard_id": "shard-0"}, recover_stale=True)
    assert lock_path.read_bytes() == before


@pytest.mark.parametrize("missing", ["run_id", "shard_id", "host", "pid", "created_at"])
def test_lock_recovery_rejects_incomplete_identity(tmp_path, missing):
    io = load_io()
    lock_path, _ = stale_lock(tmp_path, omit=(missing,))
    before = lock_path.read_bytes()
    with pytest.raises(io.LockError, match="missing required identity fields"):
        io.ShardLock.acquire(lock_path, {"run_id": "run-1", "shard_id": "shard-0"}, recover_stale=True)
    assert lock_path.read_bytes() == before


@pytest.mark.parametrize(
    "payload",
    [b'{"run_id": "run-1"', b'["run_id"]', b"",
     canonical_json_bytes({"run_id": "run-1", "shard_id": "shard-0", "host": "h",
                           "pid": "1234", "created_at": "2026-10-06T00:00:00.000000Z"})],
    ids=["malformed-json", "not-an-object", "empty", "text-pid"],
)
def test_lock_recovery_rejects_an_unusable_lock_file(tmp_path, payload):
    io = load_io()
    lock_path = tmp_path / "shard-0.lock"
    lock_path.write_bytes(payload)
    with pytest.raises(io.LockError):
        io.ShardLock.acquire(lock_path, {"run_id": "run-1", "shard_id": "shard-0"}, recover_stale=True)
    assert lock_path.read_bytes() == payload


def test_lock_recovery_flag_is_also_accepted_positionally(tmp_path):
    io = load_io()
    lock_path, _ = stale_lock(tmp_path)
    lock = io.ShardLock.acquire(lock_path, {"run_id": "run-1", "shard_id": "shard-0"}, True)
    assert read_json(lock_path)["pid"] == os.getpid()
    lock.release()
    assert not lock_path.exists()


def test_lock_recovery_replaces_a_dead_holders_lock(tmp_path):
    io = load_io()
    lock_path, payload = stale_lock(tmp_path)
    assert not psutil.pid_exists(payload["pid"])
    lock = io.ShardLock.acquire(lock_path, {"run_id": "run-1", "shard_id": "shard-0"}, recover_stale=True)
    envelope = read_json(lock_path)
    assert envelope["pid"] == os.getpid()
    assert envelope["run_id"] == "run-1"
    assert envelope["shard_id"] == "shard-0"
    lock.release()
    assert not lock_path.exists()


def test_lock_recovery_consults_psutil_with_the_recorded_pid(tmp_path, monkeypatch):
    io = load_io()
    lock_path, payload = stale_lock(tmp_path)
    observed = []
    monkeypatch.setattr(io, "pid_exists", lambda pid: observed.append(pid) or False)
    lock = io.ShardLock.acquire(lock_path, {"run_id": "run-1", "shard_id": "shard-0"}, recover_stale=True)
    assert observed == [payload["pid"]]
    lock.release()


def test_lock_recovery_refuses_a_pid_that_psutil_reports_alive(tmp_path, monkeypatch):
    io = load_io()
    lock_path, _ = stale_lock(tmp_path)
    before = lock_path.read_bytes()
    monkeypatch.setattr(io, "pid_exists", lambda pid: True)
    with pytest.raises(io.LockError, match="live"):
        io.ShardLock.acquire(lock_path, {"run_id": "run-1", "shard_id": "shard-0"}, recover_stale=True)
    assert lock_path.read_bytes() == before


def test_failed_lock_write_does_not_leave_a_poison_lock(tmp_path, monkeypatch):
    io = load_io()
    lock_path = tmp_path / "shard-0.lock"
    real_fsync = os.fsync

    def failing_fsync(fd):
        raise OSError("injected")

    monkeypatch.setattr(os, "fsync", failing_fsync)
    with pytest.raises(OSError):
        io.ShardLock.acquire(lock_path, {"run_id": "run-1", "shard_id": "shard-0"})
    monkeypatch.setattr(os, "fsync", real_fsync)
    assert not lock_path.exists()
    io.ShardLock.acquire(lock_path, {"run_id": "run-1", "shard_id": "shard-0"}).release()
    assert not lock_path.exists()


def test_lock_release_refuses_a_lock_owned_by_another_process(tmp_path):
    io = load_io()
    lock_path = tmp_path / "shard-0.lock"
    lock = io.ShardLock.acquire(lock_path, {"run_id": "run-1", "shard_id": "shard-0"})
    lock_path.write_bytes(canonical_json_bytes({**read_json(lock_path), "pid": dead_pid()}))
    before = lock_path.read_bytes()
    with pytest.raises(io.LockError, match="owned"):
        lock.release()
    assert lock_path.read_bytes() == before


def test_lock_release_reports_a_missing_lock_file(tmp_path):
    io = load_io()
    lock_path = tmp_path / "shard-0.lock"
    lock = io.ShardLock.acquire(lock_path, {"run_id": "run-1", "shard_id": "shard-0"})
    lock.release()
    with pytest.raises(io.LockError, match="missing"):
        lock.release()


def test_lock_context_manager_releases_the_lock(tmp_path):
    io = load_io()
    lock_path = tmp_path / "shard-0.lock"
    with io.ShardLock.acquire(lock_path, {"run_id": "run-1", "shard_id": "shard-0"}) as lock:
        assert lock_path.exists()
        assert lock.lock_path == lock_path
    assert not lock_path.exists()


def test_lock_context_manager_never_masks_a_body_failure(tmp_path):
    io = load_io()
    lock_path = tmp_path / "shard-0.lock"
    with pytest.raises(RuntimeError, match="boom"):
        with io.ShardLock.acquire(lock_path, {"run_id": "run-1", "shard_id": "shard-0"}):
            lock_path.unlink()
            raise RuntimeError("boom")
    assert not lock_path.exists()


def test_error_taxonomy_stays_module_local():
    io = load_io()
    isolation = importlib.import_module("permstudy.data_pipeline.isolation")
    assert issubclass(io.IntegrityError, Exception)
    assert not issubclass(io.IntegrityError, isolation.IsolationError)
    assert not issubclass(io.LockError, io.IntegrityError)
