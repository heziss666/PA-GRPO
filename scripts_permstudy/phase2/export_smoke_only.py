"""One explicitly approved WARN exception; never changes the CPU Gate policy."""

from collections import Counter
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from permstudy.data_pipeline import trainer_export
from permstudy.data_pipeline.canonical import canonical_json_bytes, sha256_hex
from permstudy.data_pipeline.isolation import sanitize_public_manifest
from scripts_permstudy.data import _common as io
from scripts_permstudy.phase2 import smoke_release_gate as gate

BASELINE = "9ef2f9415990fb8e2ab442d714e59e50db9f343b"
CPU_RUN = "7d4c4f9812f89a26133677baf84a33a988df99f23b3c3b7d86da815fd78e3421"
GATE_REPORT_SHA256 = "2a632507f0a820bbe2780368a350c693a1e5a999273704f44f48cb930850ac72"
USAGE = ["SMOKE_ONLY", "NOT_FOR_FORMAL_EXPERIMENT"]
AUTHORIZATION = {
    "baseline": BASELINE,
    "cpu_gate_run_id": CPU_RUN,
    "cpu_gate_report_sha256": GATE_REPORT_SHA256,
    "user_approval": "explicit_stage_A_confirmation",
    "scope": "one_time_math_WARN_exception_freeze_and_export_only",
    "usage": USAGE,
    "GPU_authorized": False,
    "training_authorized": False,
    "formal_experiment_authorized": False,
    "historical_formal_gates_unchanged": True,
}


def validate_gate(report, hashes):
    if (
        report.get("run_id") != CPU_RUN
        or report.get("status") != "PASS_WITH_WARNINGS"
        or report.get("input_hashes") != hashes
        or any(
            report.get(k) is not False
            for k in ("GPU_authorized", "training_authorized", "parquet_exported")
        )
    ):
        raise io.IntegrityError()


def _native(value):
    if isinstance(value, (np.ndarray, list)):
        return [_native(v) for v in value]
    if isinstance(value, dict):
        return {k: _native(v) for k, v in value.items()}
    return value


def export_checked_rows(root, folder, rows, views, provenance):
    """Reuse the trainer schema/writer; publish a receipt only after exact readback."""
    folder = io.rooted(root, folder.relative_to(root).as_posix())
    trainer_export.validate_trainer_rows(rows)
    row_hash = sha256_hex(canonical_json_bytes(rows))
    receipt_path = folder / "manifest.json"
    previous = io.read_json(receipt_path) if receipt_path.exists() else None
    if previous is not None:
        if (
            previous["provenance"] != provenance
            or previous["trainer_rows_sha256"] != row_hash
        ):
            raise io.IntegrityError()
        for name, expected in previous["artifact_hashes"].items():
            target = io.rooted(root, (folder / name).relative_to(root).as_posix())
            if sha256_hex(target.read_bytes()) != expected:
                raise io.IntegrityError()
    hashes = {}
    for name, data in views.items():
        ref = io.persist_bytes(
            root, (folder / name).relative_to(root).as_posix(), data, 1
        )
        hashes[name] = ref.sha256
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "SMOKE_ONLY_NOT_FOR_FORMAL_EXPERIMENT.parquet"
    if not target.exists():
        trainer_export._write_parquet_atomically(target, rows)
    restored = _native(pd.read_parquet(target).to_dict("records"))
    trainer_export.validate_trainer_rows(restored)
    if restored != rows:
        raise io.IntegrityError()
    hashes[target.name] = sha256_hex(target.read_bytes())
    receipt = {
        "schema_version": "smoke_only_export_receipt_v1",
        "usage": USAGE,
        "GPU_authorized": False,
        "training_authorized": False,
        "formal_experiment_authorized": False,
        "historical_formal_gates_unchanged": True,
        "provenance": provenance,
        "trainer_rows_sha256": row_hash,
        "row_count": len(rows),
        "pair_count": len({r["extra_info"]["pair_id"] for r in rows}),
        "roundtrip_exact": True,
        "artifact_hashes": hashes,
    }
    io.persist_bytes(
        root,
        receipt_path.relative_to(root).as_posix(),
        canonical_json_bytes(receipt),
        1,
    )
    return receipt


def run(args, root):
    if not args.accept_math_warn_smoke_only:
        raise io.CLIContractError()
    gate_path = io.rooted(root, f"phase2/smoke_release/{CPU_RUN}/report-private.json")
    if sha256_hex(gate_path.read_bytes()) != GATE_REPORT_SHA256:
        raise io.IntegrityError()
    report = io.read_json(gate_path)
    _, rows, hashes = gate.load_checked_dataset(root)
    validate_gate(report, hashes)
    if len(rows) != 760 or Counter(r["extra_info"]["source"] for r in rows) != {
        "math": 292,
        "reclor": 468,
    }:
        raise io.IntegrityError()
    views = {}
    for name, expected in (
        ("pairs", "pair_view"),
        ("permutations", "permutation_view"),
    ):
        data = io.rooted(
            root, f"{gate.PREFIX}/combined-provisional-{name}.jsonl"
        ).read_bytes()
        if sha256_hex(data) != hashes[expected]:
            raise io.IntegrityError()
        views[f"frozen-{name}.jsonl"] = data
    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=io.REPO, text=True
    ).strip()
    provenance = {
        "authorization": AUTHORIZATION,
        "gate_status": report["status"],
        "source_gate_status": {
            s: r["format_only"]["status"] for s, r in report["source_results"].items()
        },
        "input_hashes": hashes,
        "git_sha": git_sha,
        "script_sha256": sha256_hex(Path(__file__).read_bytes()),
        "prompt_template_hash": trainer_export.PROMPT_TEMPLATE_HASH,
        "row_schema": trainer_export.ROW_SCHEMA_VERSION,
    }
    identity = sha256_hex(canonical_json_bytes(provenance))
    folder = io.rooted(root, f"phase2/smoke_only_exports/{identity}")
    receipt = export_checked_rows(root, folder, rows, views, provenance)
    public = sanitize_public_manifest(
        {
            "schema_version": "smoke_only_stage_A_v1",
            "git_sha": git_sha,
            "run_ids": {"cpu_gate": CPU_RUN, "smoke_only_export": identity},
            "artifact_hashes": {
                "cpu_gate_report": GATE_REPORT_SHA256,
                "frozen_pairs": receipt["artifact_hashes"]["frozen-pairs.jsonl"],
                "frozen_permutations": receipt["artifact_hashes"][
                    "frozen-permutations.jsonl"
                ],
                "parquet": receipt["artifact_hashes"][
                    "SMOKE_ONLY_NOT_FOR_FORMAL_EXPERIMENT.parquet"
                ],
                "trainer_rows": receipt["trainer_rows_sha256"],
                "export_receipt": sha256_hex((folder / "manifest.json").read_bytes()),
            },
            "question_counts": {"math": 146, "reclor": 234},
            "pair_count": 380,
            "permutation_count": 760,
            "completion_counts": {
                "roundtrip_rows": 760,
                "parquet_exported": 1,
                "unsure_restored": 0,
                "GPU_runs": 0,
                "training_steps": 0,
            },
            "phase_status": "SMOKE_ONLY_NOT_FOR_FORMAL_EXPERIMENT_MATH_WARN_ACCEPTED",
            "real_generation_performed": False,
        }
    )
    io.persist_bytes(
        root,
        (folder / "summary-public.json").relative_to(root).as_posix(),
        canonical_json_bytes(public),
        1,
    )
    return {"run_id": identity, "count": 760}


def build_parser():
    parser = io.parser("export_smoke_only")
    parser.add_argument("--accept-math-warn-smoke-only", action="store_true")
    return parser


def main(argv=None):
    return io.execute(build_parser, run, argv)


if __name__ == "__main__":
    sys.exit(main())
