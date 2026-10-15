# First 200-per-source batch: CPU verification and positive AI screen

Analysis baseline: `14aa2d51b1f7b15ae2bc1b0860115e8c38d34eb6`.
This is the local CPU follow-up to the preceding generation receipt, not a new
generation run. The user confirmed that the server was shut down. No SSH/GPU
session, model generation, parser change, gate change or training was started
in this step. Provider billing was not independently inspected.

## Original automatic results

| Source | Planned Train questions | Saved candidates | Verified candidates | Correct | Incorrect | Invalid | Ambiguous | Automatic pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MATH | 200 | 800 | 792 | 439 | 130 | 218 | 5 | 57 |
| ReClor | 200 | 800 | 800 | 561 | 233 | 6 | 0 | 95 |
| Total | 400 | 1,600 | 1,592 | 1,000 | 363 | 224 | 5 | 152 |

These are the original verifier labels, **not post-screen corrected labels**.
Candidate-level `error` is zero. Two MATH questions have
`gold_verification_error` under the unchanged parser; all eight of their
candidates are excluded from verification and pair construction. In the JSON,
`reference_failed` is the sanitized count alias for that question-level status.

Automatic pair yield uses all 200 planned questions per source:
MATH 57/200 = 28.5%; ReClor 95/200 = 47.5%. The resulting canonical AB/BA sets
contain 114 and 190 records respectively. This does not mean that 200 pairs per
source have been produced. All these questions came from the frozen Train split.

The main missing side is an incorrect candidate: 111 MATH and 87 ReClor
questions lack one; 30 MATH and 18 ReClor questions lack a correct candidate.
The two remaining MATH questions are the reference failures noted above.
MATH Llama invalids are 154/400 successful generated candidates (38.5%);
ReClor Llama invalids are 6/400 (1.5%). Those source-specific denominators are
generated candidates, not only binary labels. Raw truncated finishes remain
invalid and excluded. No threshold was weakened to improve these figures.

## Selected-positive AI screening, not human release

All 152 selected positive responses were read against their problems and
official references. For the initial deterministic ten-pair sample per source,
both responses were read. The subsequent screen covered selected positives
only; it did not fully review the other 132 selected negative responses.

| Source | Automatic pairs | AI-provisionally retained | AI-blocked pairs | Unsure / held |
| --- | ---: | ---: | ---: | ---: |
| MATH | 57 | 36 | 19 | 2 |
| ReClor | 95 | 69 | 19 | 7 |
| Total | 152 | 105 | 38 | 9 |

The 38 blocked pairs comprise 37 positive reasoning flags and one additional
MATH question quarantined for an AI-confirmed false `incorrect` label. The
37 positive flags concern explicit false arithmetic, contradictory conclusions,
reversed conditions, misread options or other substantive reasoning errors,
despite a correct final number/letter. Minor imprecision or incomplete proof
was noted but not automatically treated as a confirmed defect. Nine ambiguous
reasoning/input-completeness cases remain held, not mislabeled as confirmed
failures. These are AI assessments, not an independent human adjudication.

The false-incorrect case arises when a candidate gives the defined matrix's
name rather than its explicit entries. Exact symbolic matrix multiplication
independently verifies the candidate's equivalence. The generic scalar-symbol
comparison does not use the definition supplied in the question. The entire
question and its automatic pair are quarantined; original labels, raw text
and manifests remain unchanged. No general alias parser was added.

Private notes bind each reviewed pair to its generation/verification run,
candidate identities, original pair ID and source artifact hashes. Separate
provisional PairRecord views retain 36 MATH and 69 ReClor pairs. They are
filtered views of existing records, **not new formal runs or trainer releases**.
This screen does not claim a measured population error rate or guarantee that
all remaining reasoning or negative labels are correct.

## Integrity, audit boundary and backup

The existing pinned Qwen2.5-7B tokenizer was actually loaded offline for the
existing minimum-token-gap pair selector. Formal generation, verification,
pair, permutation and audit-selection manifests were reloaded with artifact
integrity and role-bound lineage checks. Original manifest bytes and the raw
archive hash remain unchanged; 49 historical shard hashes were also rechecked.
Neither verifier labels nor canonical pair/permutation files were overwritten.

The formal audit selections contain 245 MATH and 26 ReClor items. No formal
decisions or human signoffs have been supplied. AI screen notes were **not**
converted into fabricated human `AGREE` decisions or an arbitrary AuditSummary.
Consequently no Functional/Statistical Gate evaluation, finalize, trainer export,
or training was run for this batch. Known label risk and incomplete formal
review preclude any training-release or Gate PASS claim. Previous smoke FAIL
records remain intact.

The complete CPU records, audit selections, pair/permutation records and actual
AI annotations were backed up locally outside Git; every archived member was
compared byte-for-byte with its source. Backup SHA256:

`5ea0f2eb1b3c0cc6e58490a0802d962b2acb01ff87c0584afe49cba5840ffdfa`

The raw generation backup is separate and unchanged. All real problems,
references, responses, detailed reasons, candidate identities and private
paths remain outside Git. The adjacent JSON uses the existing public evidence
allowlist. This commit is evidence only: no pipeline implementation or new
framework is introduced, and no new full Phase 1 test run is claimed.

## Current boundary

Local verification, automatic pairing and all-selected-positive AI screening
are complete. Formal audit, full pair-quality release, gates and training
remain incomplete/HOLD. Main merge remains HOLD. Further sampling, parser
changes or GPU use were not initiated. The next decision should address these
specific quality holds and the remaining reliable-pair gap, rather than treating
either the 152 automatic pairs or 105 provisional pairs as training-approved.
