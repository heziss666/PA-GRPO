"""CPU-only smoke release v1. No historical verifier/gate changes or GPU calls."""

from collections import Counter
from importlib.metadata import version
import math
import platform
import re
import subprocess
import sys
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from permstudy.data_pipeline.canonical import canonical_json_bytes, sha256_hex
from permstudy.data_pipeline import trainer_export
from permstudy.data_pipeline.isolation import sanitize_public_manifest
from permstudy.data_pipeline.schema import (
    PairRecord,
    PermutationRecord,
    VerificationStatus,
)
from scripts_permstudy.data import _common as io
from scripts_permstudy.phase2 import run_real_smoke as real

PREFIX = "phase2/expansion_500_20261017"
CONFIG = {
    "schema_version": "smoke_release_conditions_v1",
    "seed": 42,
    "folds": 5,
    "bootstrap_repetitions": 2000,
    "bootstrap_unit": "original_question_id",
    "bootstrap_kind": "fixed_oof_percentile_95",
    "warning_balanced_accuracy": 0.65,
    "fail_balanced_accuracy": 0.75,
    "threshold_scope": "project_preregistered_engineering_not_universal_statistical",
    "classifier": "LogisticRegression_C1_liblinear_max_iter_1000",
    "text_diagnostic": "tfidf_word_1_2_min_df2_max_features5000_sublinear_tf",
    "policy": "only_PASS_can_freeze_export_no_GPU_authorization",
}


def structure_features(text, token_count):
    """Numeric structure only: no lexical tokens, numeric values or answer letters."""
    if not isinstance(text, str) or type(token_count) is not int or token_count < 0:
        raise ValueError("invalid structural input")
    lines = text.splitlines()
    features = {
        "token_length": token_count,
        "character_length": len(text),
        "line_count": len(lines),
        "blank_lines": sum(not s.strip() for s in lines),
        "paragraph_count": len(re.findall(r"\S(?:.*?)(?=\n\s*\n|\Z)", text, re.S)),
        "list_items": sum(
            bool(re.match(r"\s*(?:[-*+]\s|\d+[.)]\s)", s)) for s in lines
        ),
        "heading_lines": sum(bool(re.match(r"\s*#{1,6}\s", s)) for s in lines),
        "code_fences": text.count("```"),
        "latex_backslashes": text.count("\\"),
        "latex_dollars": text.count("$"),
        "whitespace": sum(c.isspace() for c in text),
    }
    for name, symbol in (
        ("period", "."),
        ("comma", ","),
        ("colon", ":"),
        ("semicolon", ";"),
        ("exclamation", "!"),
        ("question_mark", "?"),
        ("paren_open", "("),
        ("paren_close", ")"),
        ("square_open", "["),
        ("square_close", "]"),
        ("brace_open", "{"),
        ("brace_close", "}"),
    ):
        features[name] = text.count(symbol)
    return features


def style_status(score):
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("invalid score")
    return (
        "FAIL" if score >= 0.75 else "PASS_WITH_WARNINGS" if score >= 0.65 else "PASS"
    )


def release_status(integrity_ok, format_scores):
    statuses = [style_status(s) for s in format_scores]
    if not statuses or not integrity_ok or "FAIL" in statuses:
        return "FAIL"
    return "PASS_WITH_WARNINGS" if "PASS_WITH_WARNINGS" in statuses else "PASS"


def grouped_folds(groups, labels):
    groups, labels = np.asarray(groups), np.asarray(labels)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(cv.split(np.zeros(len(labels)), labels, groups))
    if any(set(groups[a]) & set(groups[b]) for a, b in folds):
        raise ValueError("question crossed folds")
    return folds


def question_bootstrap(groups, labels, predictions, repetitions=2000):
    """Resample complete questions and both labels, never AB/BA rows."""
    groups, labels, predictions = map(np.asarray, (groups, labels, predictions))
    units = []
    for group in sorted(set(groups)):
        ix = np.flatnonzero(groups == group)
        if len(ix) != 2 or sorted(labels[ix].tolist()) != [0, 1]:
            raise ValueError("bootstrap needs one chosen/rejected per question")
        units.append(float(np.mean(labels[ix] == predictions[ix])))
    if not units or repetitions < 1:
        raise ValueError("empty bootstrap")
    # Every question has one record per class, so mean within-question accuracy
    # equals Balanced Accuracy for each full-question bootstrap resample.
    rng = np.random.default_rng(42)
    samples = np.asarray(units)[
        rng.integers(0, len(units), (repetitions, len(units)))
    ].mean(axis=1)
    return np.quantile(samples, [0.025, 0.975]).tolist()


def diagnose_source(rows, bootstrap_repetitions=2000):
    rows = sorted(rows, key=lambda r: (r["original_question_id"], r["label"]))
    keys = [(r["original_question_id"], r["label"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate question/label")
    groups = np.array([r["original_question_id"] for r in rows])
    y = np.array([r["label"] for r in rows])
    # Enforce paired grain before fitting anything.
    question_bootstrap(groups, y, y, repetitions=1)
    folds = grouped_folds(groups, y)
    feature_names = sorted(structure_features("", 0))
    x_format = np.array(
        [
            [structure_features(r["text"], r["token_count"])[f] for f in feature_names]
            for r in rows
        ],
        dtype=float,
    )
    inputs = {
        "generator_only": np.array([[r["generator"]] for r in rows]),
        "format_only": x_format,
        "full_text_diagnostic_only": np.array([r["text"] for r in rows]),
    }
    output = [
        {"original_question_id": r["original_question_id"], "label": r["label"]}
        for r in rows
    ]
    report = {
        "question_count": len(set(groups)),
        "record_count": len(rows),
        "fold_question_counts": [len(set(groups[t])) for _, t in folds],
        "format_feature_names": feature_names,
    }
    for kind, x in inputs.items():
        predictions = np.full(len(rows), -1)
        coefficients = []
        for fold, (train, test) in enumerate(folds):
            encoder = (
                OneHotEncoder(handle_unknown="ignore")
                if kind == "generator_only"
                else StandardScaler()
                if kind == "format_only"
                else TfidfVectorizer(
                    ngram_range=(1, 2), min_df=2, max_features=5000, sublinear_tf=True
                )
            )
            model = make_pipeline(
                encoder,
                LogisticRegression(
                    C=1.0, solver="liblinear", max_iter=1000, random_state=42
                ),
            )
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(x[train], y[train])
            pred = model.predict(x[test])
            predictions[test] = pred
            if kind == "format_only":
                coefficients.append(model[-1].coef_[0])
            for index, value in zip(test, pred):
                output[index]["fold"] = fold
                output[index][kind] = int(value)
        if (predictions < 0).any():
            raise ValueError("missing oof predictions")
        report[kind] = {
            "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
            "question_bootstrap_ci95": question_bootstrap(
                groups, y, predictions, bootstrap_repetitions
            ),
        }
        if kind == "format_only":
            weights = np.mean(coefficients, axis=0)
            report[kind]["mean_standardized_coefficients"] = dict(
                zip(feature_names, map(float, weights))
            )
            report[kind]["status"] = style_status(report[kind]["balanced_accuracy"])
    return report, output


def _read_records(path, record_type):
    return [
        record_type.from_dict({k: v for k, v in r.items() if k != "record_hash"})
        for r in io.scan_jsonl(path).records
    ]


def load_checked_dataset(root):
    """Join the approved view to immutable artifacts and real screening records."""
    base = root / PREFIX
    receipt = io.read_json(base / "quality-screen-receipt.json")
    pairs_file, perms_file = (
        base / "combined-provisional-pairs.jsonl",
        base / "combined-provisional-permutations.jsonl",
    )
    hashes = {
        "selection_receipt": sha256_hex(
            (base / "quality-screen-receipt.json").read_bytes()
        )
    }
    for name, file, expected in (
        ("pair_view", pairs_file, receipt["combined_pair_view_sha256"]),
        ("permutation_view", perms_file, receipt["combined_permutation_view_sha256"]),
    ):
        if sha256_hex(file.read_bytes()) != expected:
            raise io.IntegrityError()
        hashes[name] = expected
    pairs, perms = (
        _read_records(pairs_file, PairRecord),
        _read_records(perms_file, PermutationRecord),
    )
    if (
        len(pairs) != 380
        or len({p.original_question_id for p in pairs}) != 380
        or len(perms) != 760
    ):
        raise io.IntegrityError()
    if Counter(p.source.value for p in pairs) != {"math": 146, "reclor": 234}:
        raise io.IntegrityError()
    all_questions, all_candidates, approved_ids, rows = {}, {}, set(), []
    for gen, ver in sorted(
        {(p.generation_run_id, p.verification_run_id) for p in pairs}
    ):
        selected = [
            p
            for p in pairs
            if p.generation_run_id == gen and p.verification_run_id == ver
        ]
        directory = root / f"phase2/{gen}/{ver}"
        new = (directory / "quality-screen-final.json").exists()
        approval_file = directory / (
            "quality-screen-final.json" if new else "selected-positive-ai-screen.json"
        )
        approval = io.read_json(approval_file)
        if (approval["generation_run_id"], approval["verification_run_id"]) != (
            gen,
            ver,
        ):
            raise io.IntegrityError()
        if new:
            allowed = {
                n["pair_id"]
                for n in approval["notes"]
                if n["outcome"] == "RETAIN_PROVISIONAL"
                and n["positive"]["verdict"] == "PASS"
                and n["negative"]["verdict"] == "INCORRECT_CONFIRMED"
            }
        else:
            negative_file = directory / "retained-negative-label-check.json"
            negative = io.read_json(negative_file)
            hashes[f"negative_check_{ver}"] = sha256_hex(negative_file.read_bytes())
            allowed = {
                n["pair_id"]
                for n in approval["notes"]
                if n["pair_ai_verdict"] == "PASS"
            }
            if allowed != {n["pair_id"] for n in negative["notes"]} or (
                negative["generation_run_id"],
                negative["verification_run_id"],
            ) != (gen, ver):
                raise io.IntegrityError()
        if {p.pair_id for p in selected} != allowed:
            raise io.IntegrityError()
        approved_ids.update(allowed)
        hashes[f"approval_{ver}"] = sha256_hex(approval_file.read_bytes())
        verification = io.load_manifest(
            root, f"verification/{ver}/manifest.json", "verification"
        )
        generation, split, plan, questions, candidates, gold, verified = (
            real._load_real_verification(root, verification, real.load_phase2_split)
        )
        for manifest in (verification, generation, split):
            hashes[manifest["run_id"]] = manifest["output_manifest_hash"]
        vmap = {(v.generation_run_id, v.candidate_id): v for v in verified}
        cmap = {c.plan.key: c for c in candidates}
        for pair in selected:
            if pair.split.value != "train":
                raise io.IntegrityError()
            for label, cid, expected in (
                (1, pair.positive_candidate_id, VerificationStatus.CORRECT),
                (0, pair.negative_candidate_id, VerificationStatus.INCORRECT),
            ):
                c, v = cmap[(gen, cid)], vmap[(gen, cid)]
                if (
                    c.finish_reason == "length"
                    or v.verification_status is not expected
                    or c.plan.split.value != "train"
                ):
                    raise io.IntegrityError()
                rows.append(
                    {
                        "original_question_id": pair.original_question_id,
                        "label": label,
                        "source": pair.source.value,
                        "generator": c.plan.generator_id,
                        "text": c.response,
                        "token_count": pair.positive_token_count
                        if label
                        else pair.negative_token_count,
                    }
                )
        for pair_run in {p.pair_run_id for p in selected}:
            m = io.load_manifest(root, f"pairs/{pair_run}/manifest.json", "pairs")
            hashes[pair_run] = m["output_manifest_hash"]
            official = {
                p.pair_id: p for p in io.records(root, m, "pairs.jsonl", PairRecord)
            }
            if any(
                official.get(p.pair_id) != p
                for p in selected
                if p.pair_run_id == pair_run
            ):
                raise io.IntegrityError()
        all_questions.update({q.original_question_id: q for q in questions})
        all_candidates.update(cmap)
    if approved_ids != {p.pair_id for p in pairs}:
        raise io.IntegrityError()
    for perm_run in {p.permutation_run_id for p in perms}:
        m = io.load_manifest(
            root, f"permutations/{perm_run}/manifest.json", "permutations"
        )
        hashes[perm_run] = m["output_manifest_hash"]
        official = {
            (p.pair_id, p.permutation_id): p
            for p in io.records(root, m, "permutations.jsonl", PermutationRecord)
        }
        if any(
            official.get((p.pair_id, p.permutation_id)) != p
            for p in perms
            if p.permutation_run_id == perm_run
        ):
            raise io.IntegrityError()
    trainer_rows = trainer_export.build_trainer_rows(
        list(all_questions.values()), list(all_candidates.values()), pairs, perms
    )
    if len(trainer_rows) != 760:
        raise io.IntegrityError()
    return rows, trainer_rows, hashes


def run(args, root):
    package_versions = {
        name: version(name)
        for name in ("scikit-learn", "numpy", "scipy", "joblib", "threadpoolctl")
    }
    rows, trainer_rows, hashes = load_checked_dataset(root)
    reports, predictions = {}, {}
    for source in ("math", "reclor"):
        reports[source], predictions[source] = diagnose_source(
            [r for r in rows if r["source"] == source]
        )
    status = release_status(
        True, [r["format_only"]["balanced_accuracy"] for r in reports.values()]
    )
    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=io.REPO, text=True
    ).strip()
    code_hash = sha256_hex(__import__("pathlib").Path(__file__).read_bytes())
    identity = sha256_hex(
        canonical_json_bytes(
            {
                "config": CONFIG,
                "inputs": hashes,
                "code_hash": code_hash,
                "package_versions": package_versions,
            }
        )
    )
    prefix = f"phase2/smoke_release/{identity}"
    diagnostic = {
        "config": CONFIG,
        "run_id": identity,
        "git_sha": git_sha,
        "script_sha256": code_hash,
        "input_hashes": hashes,
        "status": status,
        "source_results": reports,
        "integrity": {
            "pairs": 380,
            "abba_rows": 760,
            "excluded_unsure_restored": 0,
            "duplicate_question_ids": 0,
            "holdout_overlap": 0,
            "known_rejected_included": 0,
        },
        "historical_formal_gates_unchanged": True,
        "formal_audit_completed": False,
        "GPU_authorized": False,
        "training_authorized": False,
        "parquet_exported": False,
    }
    io.persist_bytes(
        root, f"{prefix}/oof-private.json", canonical_json_bytes(predictions), 1
    )
    if status == "PASS":
        # A smoke-only selected-data receipt, not a rewrite of the old RunManifests.
        # Reuse the existing renderer, schema validator and atomic Parquet writer.
        import pandas as pd

        folder = io.rooted(root, prefix)
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / "smoke-train.parquet"
        trainer_export._write_parquet_atomically(target, trainer_rows)

        def native(value):
            if isinstance(value, np.ndarray):
                return [native(v) for v in value]
            if isinstance(value, dict):
                return {k: native(v) for k, v in value.items()}
            if isinstance(value, list):
                return [native(v) for v in value]
            return value

        restored = native(pd.read_parquet(target).to_dict("records"))
        trainer_export.validate_trainer_rows(restored)
        if restored != trainer_rows:
            raise io.IntegrityError()
        hashes["smoke_parquet"] = sha256_hex(target.read_bytes())
        diagnostic["parquet_exported"] = True
    io.persist_bytes(
        root, f"{prefix}/report-private.json", canonical_json_bytes(diagnostic), 1
    )
    # Floats/feature names remain in private report and reviewed Markdown, never
    # stuffed into integer-only PUBLIC_EVIDENCE_KEYS completion_counts.

    public = sanitize_public_manifest(
        {
            "schema_version": CONFIG["schema_version"],
            "git_sha": git_sha,
            "python_version": platform.python_version(),
            "package_versions": package_versions,
            "run_ids": {"smoke_release": identity},
            "artifact_hashes": hashes,
            "question_counts": {"math": 146, "reclor": 234},
            "pair_count": 380,
            "permutation_count": 760,
            "completion_counts": {
                "new_retained": 275,
                "new_quarantined": 101,
                "new_unsure": 17,
                "parquet_exported": int(diagnostic["parquet_exported"]),
                "GPU_runs": 0,
                "training_steps": 0,
            },
            "phase_status": f"CPU_SMOKE_RELEASE_{status}",
            "real_generation_performed": False,
        }
    )
    io.persist_bytes(
        root, f"{prefix}/summary-public.json", canonical_json_bytes(public), 1
    )
    return {"run_id": identity, "count": 380}


def main(argv=None):
    return io.execute(lambda: io.parser("smoke_release_gate"), run, argv)


if __name__ == "__main__":
    sys.exit(main())
