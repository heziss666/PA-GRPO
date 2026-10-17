# Stage A: explicitly approved Smoke-only data exception

## Authorization and unchanged boundaries

The user explicitly confirmed execution of Stage A after reviewing the
"Smoke-only data release, short training validation and Pair-v2" plan.
Baseline: `9ef2f9415990fb8e2ab442d714e59e50db9f343b`.

This approval accepts the MATH format warning **once, only to freeze and export
the existing 380 screened pairs for a future training plumbing Smoke**. It does
not authorize GPU use, training, Pair-v2, formal four-method experiments, or a
main merge. It does not certify the provisional AI screening as a completed
formal human Audit.

The historical CPU result remains `CPU_SMOKE_RELEASE_PASS_WITH_WARNINGS`:
MATH format-only BA 71.92% (WARN), ReClor 53.85% (PASS). The 65% / 75%
engineering thresholds, CPU Gate implementation, old verification labels,
parsers and historical formal FAIL records are not modified.

## Export method

`scripts_permstudy/phase2/export_smoke_only.py` is a separate, explicitly opted-in
entry point. It does not add a bypass flag to the CPU Gate. It pins the approved
private Gate report by SHA256, rebuilds the checked dataset using the existing
immutable artifact/screening loader, and requires exact equality of all upstream
hash bindings with the approved Gate inputs.

The selection is 146 MATH + 234 ReClor pairs: 380 unique Train questions and
760 AB/BA rows. All 17 UNSURE pairs and previously quarantined pairs remain
excluded. Existing Pair/Candidate identities, response hashes, generation and
verification provenance, split lineage and AB/BA mappings are retained. No
cross-run candidate recombination or new response selection is performed.

The existing trainer row builder, schema validator and atomic Parquet writer
are reused. Frozen pair/permutation views are exact byte copies. The new export
is stored only below the external private data root, in its own namespace.
Both the Parquet filename and adjacent export receipt mark it `SMOKE_ONLY`
and `NOT_FOR_FORMAL_EXPERIMENT`; row schema and prompt contents are unchanged.
The adjacent `manifest.json` is a bounded export receipt, not a replacement for
the upstream pipeline RunManifests or formal Audit/Gates.

Publication of that receipt requires exact Parquet readback equality, including
every prompt, answer surface, nested identity and provenance field. Existing
published files are hash-checked and never silently overwritten or repaired.

Command (the data-root environment variable must identify the external private
root; no machine-specific path is committed):

```bash
python -m scripts_permstudy.phase2.export_smoke_only \
  --accept-math-warn-smoke-only \
  --private-log private_logs/smoke-only-stage-A.private.log
```

Stage A ends after CPU readback verification. Stage B GPU execution and training,
Stage C Pair-v2, and Stage D formal experiments remain on HOLD pending separate
explicit authorization.

## Focused checks

The new synthetic-only tests first failed because the export entry point did
not exist. After implementation, all 3 export tests and 14 existing CPU Gate
tests passed (17 total). They exercise the real Parquet writer/readback,
explicit opt-in, changed Gate/input rejection, idempotent reuse without
rewriting, corrupted-file rejection and changed-content rejection before
receipt publication. Ruff and format checks passed. A separate read-only
review found no important correctness, privacy or scope issues and independently
ran the 3 export tests successfully. These are local results, not GitHub CI.
The final run including the existing production-preparation tests passed
27 tests. The Phase 1 full test suite was not repeated.

Execution code: `711ca485b4dba00bd3e0d0acd7acb65914a3453f`.
Parquet dependencies: pandas `2.3.3`, PyArrow `22.0.0`.

## CPU acceptance result

Stage A completed. Export identity:
`ad8355ce93da7a1050ac95d628f3b905b188761673f2238a56197b72d31031e1`.

| Check | Result |
|---|---|
| Frozen pairs | 380: 146 MATH + 234 ReClor |
| Exact Parquet round-trip | 760/760 rows identical |
| Source rows | 292 MATH + 468 ReClor |
| Unique original Train question IDs | 380 |
| Duplicate questions / Holdout overlap | 0 / 0 |
| UNSURE restored | 0 |
| Parquet size | 592,824 bytes |
| GPU runs / training steps | 0 / 0 |
| CPU Gate status | PASS_WITH_WARNINGS, unchanged |
| Formal Audit/Gates release | Not granted |

Parquet SHA256:
`15b3b0944fed63d8269109c05df5298e768bde9a87455b8e0e5c60e919ce06fb`.

Full Trainer-row semantic SHA256 (canonical nested records):
`11bfdb30ee6ff459a3c16b006098f4b153b2c7776e5fbe6ebd43dd498798e4f8`.

Frozen Pair JSONL SHA256:
`db0f294450144c9553a848f3ce1d436551824b0ee18ee97c387adec7bbf2988b`.

Frozen permutation JSONL SHA256:
`5505e619af505e7f8581ae839f8494d2f5cc69ccf3a3aad07c3a8f4acba74c20`.

A separate CPU readback revalidated all row identities, per-pair permutations,
source counts and the canonical semantic hash. The original Gate report, OOF
predictions, combined pair/permutation views and screening receipt were checked
again against their pre-execution hashes: all 5 were unchanged. The original
Gate script, Phase 1 package/CLI paths and public Gate summary have no diff
relative to the approved baseline.

The allowlisted public receipt is
`artifacts/data_pipeline/phase2/smoke_only_stage_A/summary.json`. It matches the
private export's public summary exactly as parsed JSON. The full Parquet,
frozen canonical records and detailed export receipt remain outside Git.

**STOP: Stage A only.** This dataset is now frozen for a possible plumbing-only
Smoke, not released for formal research training. No Stage B/C/D execution is
included or automatically authorized.
