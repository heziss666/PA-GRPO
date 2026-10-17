"""Synthetic data only: explicit warning exception, real Parquet round-trip."""

from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest

from permstudy.data_pipeline import trainer_export
from scripts_permstudy.data import _common as io
from tests_permstudy.data_pipeline.test_trainer_export import _records


def test_missing_approval_and_changed_gate_fail_before_reading_dataset(tmp_path):
    from scripts_permstudy.phase2 import export_smoke_only as export

    with pytest.raises(io.CLIContractError):
        export.run(SimpleNamespace(accept_math_warn_smoke_only=False), tmp_path)
    assert not list(tmp_path.iterdir())
    report = {
        "run_id": export.CPU_RUN,
        "status": "PASS_WITH_WARNINGS",
        "input_hashes": {"pair_view": "a" * 64},
        "GPU_authorized": False,
        "training_authorized": False,
        "parquet_exported": False,
    }
    export.validate_gate(report, {"pair_view": "a" * 64})
    for field, value in (("status", "PASS"), ("GPU_authorized", True)):
        bad = deepcopy(report)
        bad[field] = value
        with pytest.raises(io.IntegrityError):
            export.validate_gate(bad, {"pair_view": "a" * 64})
    with pytest.raises(io.IntegrityError):
        export.validate_gate(report, {"pair_view": "b" * 64})


def test_smoke_parquet_roundtrip_freezes_views_and_rejects_tampering(tmp_path):
    from scripts_permstudy.phase2.export_smoke_only import export_checked_rows

    records = _records()
    rows = trainer_export.build_trainer_rows(*records)
    folder = tmp_path / "SMOKE_ONLY_NOT_FOR_FORMAL_EXPERIMENT"
    views = {"pairs.jsonl": b"synthetic-pair-view\n"}
    result = export_checked_rows(tmp_path, folder, rows, views, {"synthetic": True})
    assert result["row_count"] == 4
    assert result["pair_count"] == 2
    assert result["roundtrip_exact"] is True
    assert result["usage"] == ["SMOKE_ONLY", "NOT_FOR_FORMAL_EXPERIMENT"]
    assert result["GPU_authorized"] is False
    parquet = folder / "SMOKE_ONLY_NOT_FOR_FORMAL_EXPERIMENT.parquet"
    assert len(pd.read_parquet(parquet)) == 4
    assert (folder / "pairs.jsonl").read_bytes() == b"synthetic-pair-view\n"
    before = parquet.read_bytes()
    assert (
        export_checked_rows(tmp_path, folder, rows, views, {"synthetic": True})
        == result
    )
    assert parquet.read_bytes() == before
    # Existing published files must never be silently repaired or overwritten.
    with parquet.open("ab") as stream:
        stream.write(b"corruption")
    corrupted = parquet.read_bytes()
    with pytest.raises(io.IntegrityError):
        export_checked_rows(tmp_path, folder, rows, views, {"synthetic": True})
    assert parquet.read_bytes() == corrupted


def test_roundtrip_content_change_never_publishes_release(tmp_path, monkeypatch):
    from scripts_permstudy.phase2.export_smoke_only import export_checked_rows

    rows = trainer_export.build_trainer_rows(*_records())
    changed = deepcopy(rows)
    changed[0]["prompt"][1]["content"] += " changed content"
    real_read = pd.read_parquet

    def changed_read(path):
        frame = real_read(path)
        frame.at[0, "prompt"] = changed[0]["prompt"]
        return frame

    monkeypatch.setattr(pd, "read_parquet", changed_read)
    folder = tmp_path / "SMOKE_ONLY_NOT_FOR_FORMAL_EXPERIMENT"
    with pytest.raises(io.IntegrityError):
        export_checked_rows(tmp_path, folder, rows, {}, {"synthetic": True})
    assert not (folder / "manifest.json").exists()
