"""Validated command boundary for audit_generator_distribution."""

from collections.abc import Sequence
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts_permstudy.data import _common as io
from scripts_permstudy.data.verify_candidates import load_verification
from permstudy.data_pipeline import audit
from permstudy.data_pipeline.ids import run_id
from permstudy.data_pipeline.schema import AuditDecision, AuditSelectionRecord


def load_decisions(root, reference):
    scan = io.scan_jsonl(io.rooted(root, reference))
    return [
        io.checked(
            AuditDecision.from_dict,
            {k: v for k, v in row.items() if k != "record_hash"},
            integrity=True,
        )
        for row in scan.records
    ]


def rebuild_selection(root, verification, seed):
    generation, _, _, gold, verified = load_verification(root, verification)
    selection = io.checked(
        audit.build_audit_selection,
        verified,
        gold,
        generation["run_id"],
        audit_seed=seed,
        verification_manifest_hash=verification["output_manifest_hash"],
        integrity=True,
    )
    snapshot_hash = audit.verification_snapshot_hash(verified, gold)
    config = io.checked(
        audit.audit_stage_config,
        generation["run_id"],
        verification["run_id"],
        snapshot_hash,
        audit_seed=seed,
        verification_manifest_hash=verification["output_manifest_hash"],
    )
    identity = run_id("audit", config)
    if any(item.audit_run_id != identity for item in selection):
        raise io.IntegrityError()
    return config, selection


def load_audit(root, manifest):
    verification = io.upstream(root, manifest, "verification")
    config, selection = rebuild_selection(
        root, verification, manifest["typed_config"]["parameters"]["audit_seed"]
    )
    if (
        config != manifest["typed_config"]
        or io.records(root, manifest, "selection.jsonl", AuditSelectionRecord)
        != selection
    ):
        raise io.IntegrityError()
    if io.select_refs(manifest, "summary.json"):
        decisions = io.records(root, manifest, "decisions.jsonl", AuditDecision)
        summary = audit.validate_audit_decisions(selection, decisions)
        raw = io.read_json(
            io.rooted(root, io.select_refs(manifest, "summary.json")[0].relative_path)
        )
        if raw != summary_payload(summary):
            raise io.IntegrityError()


def summary_payload(summary):
    return {
        "required_count": summary.required_count,
        "completed_count": summary.completed_count,
        "verdict_counts": dict(summary.verdict_counts),
        "confirmed_disagreements": summary.confirmed_disagreements,
        "systematic_issue": summary.systematic_issue,
    }


def build_parser() -> io.argparse.ArgumentParser:
    parser = io.parser("audit_generator_distribution")
    parser.add_argument("--verification-manifest", required=True)
    parser.add_argument("--decisions")
    parser.add_argument("--audit-seed", type=int, default=42)
    return parser


def dispatch(args, root):
    if args.audit_seed < 0:
        raise io.CLIContractError()
    verification = io.load_manifest(root, args.verification_manifest, "verification")
    config, selection = rebuild_selection(root, verification, args.audit_seed)
    identity = run_id("audit", config)
    prefix = f"audit/{identity}"
    # Validate the complete decision set against a fresh Task 14 reconstruction
    # before writing any summary or claiming completion.
    decisions = load_decisions(root, args.decisions) if args.decisions else None
    summary = (
        audit.validate_audit_decisions(selection, decisions)
        if decisions is not None
        else None
    )
    refs = [io.persist_records(root, f"{prefix}/selection.jsonl", selection)]
    counts = {"selected": len(selection)}
    filename = "manifest.json"
    if summary is not None:
        refs.append(io.persist_records(root, f"{prefix}/decisions.jsonl", decisions))
        refs.append(
            io.persist_bytes(
                root,
                f"{prefix}/summary.json",
                io.canonical_json_bytes(summary_payload(summary)),
                1,
            )
        )
        counts["completed"] = summary.completed_count
        filename = "completed.json"
    return io.output(
        io.persist_manifest(
            root,
            "audit",
            config,
            refs,
            counts,
            {"verification": args.verification_manifest},
            filename=filename,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    return io.execute(build_parser, dispatch, argv)


if __name__ == "__main__":
    raise SystemExit(main())
