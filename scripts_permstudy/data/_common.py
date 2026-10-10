"""Shared CLI I/O boundaries; stage algorithms live in permstudy.data_pipeline."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import json
import os
from pathlib import Path, PureWindowsPath
import re
import tempfile
import traceback

from permstudy.data_pipeline.audit import AuditIntegrityError
from permstudy.data_pipeline.canonical import canonical_json_bytes, sha256_hex
from permstudy.data_pipeline.generation import (
    GenerationConfigurationError,
    RunMismatchError,
)
from permstudy.data_pipeline.ids import run_id
from permstudy.data_pipeline.io import (
    IntegrityError,
    LockError,
    append_record,
    scan_jsonl,
    verify_artifact_ref,
    write_atomic_manifest,
)
from permstudy.data_pipeline.isolation import (
    IsolationError,
    SafeLoggingError,
    validate_external_roots,
)
from permstudy.data_pipeline.pairs import PairIntegrityError
from permstudy.data_pipeline.permutations import PermutationIntegrityError
from permstudy.data_pipeline.schema import ArtifactRef, RunManifest
from permstudy.data_pipeline.sources.math import MathContractError
from permstudy.data_pipeline.sources.reclor import ReclorContractError
from permstudy.data_pipeline.trainer_export import TrainerExportIntegrityError
from permstudy.data_pipeline.verification import (
    DependencyContractError,
    MathVerificationError,
    ReClorVerificationError,
)

REPO = Path(__file__).resolve().parents[2]
HASH = re.compile(r"[0-9a-f]{64}\Z")
ROLES = {
    "split": {"math_questions", "reclor_questions"},
    "generation": {"split"},
    "verification": {"generation"},
    "pairs": {"generation", "verification"},
    "permutations": {"pairs"},
    "audit": {"verification"},
    "trainer_export": {"split", "generation", "pairs", "permutations"},
}


class CLIContractError(ValueError):
    """Invalid CLI configuration, without user-controlled public text."""


class PhaseBoundaryError(CLIContractError):
    """The requested operation belongs to a later task or phase."""


class Parser(argparse.ArgumentParser):
    def error(self, message):
        # argparse's default includes arbitrary argument values and absolute paths.
        self.exit(2, "contract_error\n")


def parser(name, *, common=True):
    result = Parser(prog=f"{name}.py")
    if common:
        add_common(result)
    return result


def add_common(result):
    result.add_argument("--data-root", default=os.environ.get("PAGRPO_DATA_ROOT"))
    result.add_argument(
        "--private-log", help="Opt-in private traceback file relative to data root"
    )


def positive_int(value):
    try:
        number = int(value)
        if number <= 0:
            raise ValueError
        return number
    except ValueError as exc:
        raise argparse.ArgumentTypeError("positive integer required") from exc


def rooted(root, value, *, relative=True):
    if not isinstance(value, (str, Path)) or not str(value):
        raise CLIContractError()
    raw = str(value)
    supplied = Path(raw)
    absolute = supplied.is_absolute() or bool(PureWindowsPath(raw).drive)
    if relative and (absolute or "\\" in raw):
        raise CLIContractError()
    if ".." in Path(raw).parts:
        raise CLIContractError()
    target = (supplied if absolute else root / supplied).resolve()
    if target == root or (relative and not target.is_relative_to(root)):
        raise CLIContractError()
    return target


def checked(call, *args, integrity=False, **kwargs):
    """Translate package schema/config ValueErrors at a known input boundary."""
    try:
        return call(*args, **kwargs)
    except (ValueError, TypeError, KeyError) as exc:
        raise (IntegrityError if integrity else CLIContractError)() from exc


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def read_json(path):
    def reject(value):
        raise ValueError

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique,
            parse_constant=reject,
        )
    except (ValueError, UnicodeError) as exc:
        raise IntegrityError() from exc


def artifact(root, path, count=1):
    return ArtifactRef(
        path.relative_to(root).as_posix(), sha256_hex(path.read_bytes()), count
    )


def load_manifest(root, reference, stage=None, *, ancestry=()):
    path = rooted(root, reference)
    if path in ancestry or len(ancestry) > 12:
        raise IntegrityError()
    payload = read_json(path)
    core = checked(
        lambda: RunManifest.from_dict(
            {key: payload[key] for key in RunManifest.__annotations__}
        ),
        integrity=True,
    )
    if stage is not None and core.stage != stage:
        raise IntegrityError()
    digest = sha256_hex(
        canonical_json_bytes(
            {
                key: value
                for key, value in payload.items()
                if key not in {"created_at_utc", "output_manifest_hash"}
            }
        )
    )
    if digest != core.output_manifest_hash:
        raise IntegrityError()
    for ref in core.artifacts:
        rooted(root, ref.relative_path)
        verify_artifact_ref(root, ref)
    if len({ref.relative_path for ref in core.artifacts}) != len(core.artifacts):
        raise IntegrityError()
    if core.stage == "sources":
        if payload.get("source") not in {"math", "reclor"}:
            raise IntegrityError()
        return payload
    expected_roles = ROLES.get(core.stage)
    bindings = payload.get("upstream_bindings")
    upstreams = payload.get("upstream_manifests")
    if (
        expected_roles is None
        or not isinstance(bindings, dict)
        or set(bindings) != expected_roles
    ):
        raise IntegrityError()
    if not isinstance(upstreams, dict) or set(upstreams) != expected_roles:
        raise IntegrityError()
    if tuple(sorted(bindings.values())) != core.upstream_manifest_hashes:
        raise IntegrityError()
    for role, raw in upstreams.items():
        ref = checked(ArtifactRef.from_dict, raw, integrity=True)
        verify_artifact_ref(root, ref)
        expected_stage = "sources" if role.endswith("_questions") else role
        upstream = load_manifest(
            root, ref.relative_path, expected_stage, ancestry=(*ancestry, path)
        )
        if upstream["output_manifest_hash"] != bindings[role]:
            raise IntegrityError()
        if role.endswith("_questions") and upstream["source"] != role.removesuffix(
            "_questions"
        ):
            raise IntegrityError()
    config = payload.get("typed_config")
    if not isinstance(config, dict) or config.get("upstream_bindings") != bindings:
        raise IntegrityError()
    if (
        sha256_hex(canonical_json_bytes(config)) != core.config_hash
        or run_id(core.stage, config) != core.run_id
    ):
        raise IntegrityError()
    return payload


def upstream(root, manifest, role):
    ref = checked(
        ArtifactRef.from_dict, manifest["upstream_manifests"][role], integrity=True
    )
    return load_manifest(
        root, ref.relative_path, "sources" if role.endswith("_questions") else role
    )


def records(root, manifest, basename, record_type, *, required=True):
    refs = [
        ArtifactRef.from_dict(raw)
        for raw in manifest["artifacts"]
        if Path(raw["relative_path"]).name == basename
    ]
    if required and not refs:
        raise IntegrityError()
    result = []
    for ref in refs:
        verify_artifact_ref(root, ref)
        scan = scan_jsonl(rooted(root, ref.relative_path))
        if len(scan.records) != ref.record_count:
            raise IntegrityError()
        result.extend(
            checked(
                record_type.from_dict,
                {k: v for k, v in row.items() if k != "record_hash"},
                integrity=True,
            )
            for row in scan.records
        )
    return result


def select_refs(manifest, *basenames):
    return [
        ArtifactRef.from_dict(ref)
        for ref in manifest["artifacts"]
        if Path(ref["relative_path"]).name in basenames
    ]


def persist_bytes(root, relative, data, count):
    path = rooted(root, relative)
    if path.exists():
        if path.read_bytes() != data:
            raise IntegrityError()
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    return artifact(root, path, count)


def persist_records(root, relative, rows):
    rows = [row.to_dict() if hasattr(row, "to_dict") else row for row in rows]
    path = rooted(root, relative)
    if path.exists():
        scan = scan_jsonl(path)
        if [
            {k: v for k, v in row.items() if k != "record_hash"} for row in scan.records
        ] != rows:
            raise IntegrityError()
        return artifact(root, path, len(rows))
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary)
    try:
        for row in rows:
            append_record(temporary, row)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return artifact(root, path, len(rows))


def persist_manifest(
    root,
    stage,
    config,
    refs,
    counts,
    upstream_paths,
    *,
    extra=None,
    filename="manifest.json",
):
    identity = run_id(stage, config)
    payload = dict(
        schema_version="run_manifest_v1",
        stage=stage,
        run_id=identity,
        config_hash=sha256_hex(canonical_json_bytes(config)),
        typed_config=config,
        upstream_bindings=config["upstream_bindings"],
        upstream_manifest_hashes=sorted(config["upstream_bindings"].values()),
        artifacts=[ref.to_dict() for ref in refs],
        counts=counts,
        upstream_manifests={
            role: artifact(root, rooted(root, ref)).to_dict()
            for role, ref in upstream_paths.items()
        },
    )
    payload.update(extra or {})
    path = rooted(root, f"{stage}/{identity}/{filename}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = read_json(path)
        if {
            k: v
            for k, v in old.items()
            if k not in {"created_at_utc", "output_manifest_hash"}
        } != payload:
            raise IntegrityError()
        return old
    write_atomic_manifest(path, payload)
    return read_json(path)


def output(manifest):
    return {"run_id": manifest["run_id"], "count": sum(manifest["counts"].values())}


def execute(build_parser, dispatch, argv=None):
    root = None
    log = None
    try:
        try:
            args = build_parser().parse_args(argv)
        except SystemExit as exc:
            return int(exc.code)
        if not args.data_root:
            raise CLIContractError()
        root = Path(args.data_root).resolve()
        validate_external_roots(REPO, root, os.environ)
        if args.private_log:
            log = rooted(root, args.private_log)
        # Dependency progress and warnings can contain payloads, paths, or credentials.
        with (
            open(os.devnull, "w", encoding="utf-8") as sink,
            redirect_stdout(sink),
            redirect_stderr(sink),
        ):
            result = dispatch(args, root)
        public = {"status": "ok"}
        if result:
            if "run_id" in result:
                if not HASH.fullmatch(result["run_id"]):
                    raise SafeLoggingError()
                public["run_id"] = result["run_id"]
            if "count" in result:
                if type(result["count"]) is not int or result["count"] < 0:
                    raise SafeLoggingError()
                public["count"] = result["count"]
        print(json.dumps(public, sort_keys=True))
        return 0
    except Exception as error:
        if isinstance(error, PhaseBoundaryError):
            code, label = 2, "PhaseBoundaryError"
        elif isinstance(
            error,
            (
                CLIContractError,
                IsolationError,
                SafeLoggingError,
                ReclorContractError,
                MathContractError,
                GenerationConfigurationError,
            ),
        ):
            code, label = 2, "contract_error"
        elif isinstance(
            error,
            (
                IntegrityError,
                RunMismatchError,
                PairIntegrityError,
                PermutationIntegrityError,
                TrainerExportIntegrityError,
                AuditIntegrityError,
                MathVerificationError,
                ReClorVerificationError,
            ),
        ):
            code, label = 3, "integrity_error"
        elif isinstance(
            error, (DependencyContractError, LockError, OSError, ImportError)
        ):
            code, label = 4, "dependency_access_error"
        else:
            code, label = 4, "unexpected_error"
        cause = error.__cause__
        while cause is not None:
            if isinstance(cause, (OSError, ImportError, DependencyContractError)):
                code, label = 4, "dependency_access_error"
                break
            cause = cause.__cause__
        if log is not None:
            try:
                log = rooted(root, log.relative_to(root).as_posix())
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open("a", encoding="utf-8") as stream:
                    traceback.print_exc(file=stream)
            except OSError:
                pass
        print(json.dumps({"error_type": label, "status": "failed"}, sort_keys=True))
        return code
