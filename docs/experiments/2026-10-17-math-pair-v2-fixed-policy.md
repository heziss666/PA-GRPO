# MATH Pair-v2: fixed CPU intervention, before real diagnostics

User authorization: rematch the existing 146 retained MATH Train questions
only. Pair-v1, 234 ReClor pairs, original candidates/verification labels,
isolation decisions and the frozen 380-pair Smoke export remain immutable.
No GPU, candidate generation, formal training or new Gate framework.

## Fixed distance

For each nonnegative integer feature, use exact rational
`abs(a-b)/(1+a+b)`, bounded in [0,1). Group distances are the unweighted mean
of their features. No dataset-fitted scale, numeric answer values, lexical
tokens, embeddings or semantic similarity are used.

- Length: token count and character count.
- LaTeX: backslash, dollar, opening brace and closing brace counts.
- Layout: line, blank line, paragraph, list item, heading, code fence and
  whitespace counts.
- Punctuation: period, comma, colon, semicolon, exclamation, question mark,
  opening/closing parentheses and opening/closing square brackets.
- Generator: 0 when equal, 1 otherwise.

`D = (length + LaTeX + layout + punctuation + generator) / 5`.
Text normalization and pinned Qwen2.5 tokenizer follow the existing contract.
Eligibility/dedup reuse the frozen package logic, including minimum composite
representative identity and exclusion of length/invalid/error candidates.

## Fixed selection and review

Within the same question, generation run and verification run, enumerate
eligible correct/incorrect representatives. Sort by exact D, original pair
first on ties, token gap, positive ID, negative ID. Propose at most one new
pair per question. Review both proposed responses against the original
problem and official solution. Accept only positive PASS plus negative
INCORRECT_CONFIRMED; any defect or uncertainty retains the original pair.
Do not seek another alternative after a failed proposal in this intervention.
No original question is dropped; new review holds apply to proposals only.
Reviews are AI-assisted/provisional, not independent human certification.
If a new explicit defect contradicts an original candidate's approval, stop
before diagnosis for adjudication rather than falling back to that known-bad
candidate. Original files still remain immutable.

The policy is a selection version, not a change to the PairRecord schema.
Original generation/verification lineage is inherited, semantic pair IDs
are recomputed for changed candidate identities, and a new pair-run identity
binds the fixed policy, upstream hashes and quality decisions. AB/BA reuse
the existing construction and validation. No new Trainer Parquet is written.

## One diagnostic and interpretation

After reviews, run the existing source diagnostic once: same 146 question
groups, five fixed grouped folds, training-fold-only preprocessing and
2,000 whole-question bootstrap resamples. Read the stored Pair-v1 results;
do not refit them. A persistent start marker blocks a second real fit, even
after an error. Report all three classifier scores/intervals, replacement
and fallback counts, retention and generator composition. The established
65%/75% format thresholds remain unchanged.
Diagnostic and descriptive response text is raw model output, exactly as in
the baseline; normalization is used for matching/hash only, not a hidden
format-cleaning intervention.

This uses an already-inspected cohort and is an internal intervention
diagnostic, not untouched out-of-sample confirmation. Do not retune weights
after viewing it. Stop for review after reporting; formal Audit/Gates and
formal experiment authorization are unaffected.
