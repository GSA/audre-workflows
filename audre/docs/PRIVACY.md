# Privacy-preserving reporting rules

Privacy in AUDRE is a pipeline stage, not a manual review step. The reason is
practical: a review step is applied once, while a pipeline stage is applied every
time the analysis is re-run, including the run that produces the final figures.

## 1. What never enters the analysis

Conversation content is consumed by `preprocess.build_documents`, passed to the
labeling stage, and goes no further. The analysis spine returned by
`preprocess.build_conversations` contains no text at all, and `schema.validate`
**fails** if a content or identifier column appears in any input table.

`use_case_title` is the one content-bearing label kind. It is useful for
qualitative review of what a cluster or transition contains, and it should not be
published or aggregated.

## 2. Pseudonymization

```python
from audre import privacy

salt = privacy.new_salt()          # generate once per analysis
conversations = privacy.deidentify(
    conversations,
    salt=salt,
    unit_map={"Real Estate Office": "Unit B"},   # optional relabelling
)
```

`pseudonymize` uses HMAC-SHA256 rather than a bare hash. This matters: with a
bare hash, anyone holding a list of candidate identifiers — an employee
directory, for instance — can hash each one and confirm membership. Keying the
digest with a secret salt defeats that.

Keep the salt out of version control. Discard it once re-linking is no longer
needed; after that the pseudonyms are non-reversible even by the analyst.

Pseudonyms are stable *within* a salt, so longitudinal joins work, and differ
*across* salts, so two releases cannot be linked to each other.

## 3. Minimum-cell rules

Every aggregate must clear two thresholds:

| Rule | Default | Why |
| --- | --- | --- |
| `n_conversations >= 5` | `privacy.MIN_CONVERSATIONS` | A cell backed by one or two conversations is unstable and close to individual. |
| `n_users >= 3` | `privacy.MIN_USERS` | **The load-bearing rule.** A conversation threshold alone is satisfied by a single prolific user, which makes the "aggregate" effectively personal. |

All analysis functions apply these by default (`suppress=True`). Pass
`suppress=False` only for internal diagnostics, never for output.

```python
# drop failing rows
out = privacy.suppress_small_cells(frame)

# or keep them, blanking counts and relabelling the category
out = privacy.suppress_small_cells(frame, mode="mask", label_columns=("label_value",))
```

Masking preserves the fact that a category existed without revealing its size,
which is preferable when the *absence* of a row would itself be informative.

Immediately before writing a figure or table:

```python
privacy.assert_reportable(frame)   # raises if any row is below threshold
```

This is a deliberate hard gate. It fails loudly so that a forgotten suppression
cannot reach a publication.

## 4. Stratum eligibility

Distinct from cell suppression. `stratify` refuses to analyze a stratum at all
below `MIN_STRATUM_CONVERSATIONS` (50) and `MIN_STRATUM_USERS` (10), because a
network built from a handful of users is both statistically unstable and highly
re-identifying. Excluded strata are reported in
`result.attrs["strata_skipped_too_small"]` — including strata the caller never
named — so the exclusion is visible rather than silent.

## 5. Missingness is reported, not hidden

```python
privacy.missingness_report(frame)
```

Suppression and missingness interact. A column that is 60% missing can make a
surviving cell unrepresentative even when it clears the count thresholds, so
report coverage alongside results. `resources.cost_coverage` does the same for
observed-versus-estimated cost.

## 6. What AUDRE does not provide

- **Formal differential privacy.** The thresholds here are k-anonymity-style
  minimum-cell rules. They are not a DP mechanism and carry no privacy budget or
  formal guarantee. Do not describe them as differentially private.
- **Protection against a determined insider** who holds both the salt and the
  underlying export.
- **Protection against inference from many overlapping releases.** Repeatedly
  publishing overlapping aggregates of the same population can erode
  suppression even when each individual release satisfies the rules.
- **Legal or policy sufficiency.** Thresholds are defaults, not compliance.
  Confirm the required values with your privacy office.

## Checklist before publishing

- [ ] Salt generated per analysis, excluded from version control
- [ ] `unit_map` complete if organizational names must be neutral
- [ ] No content column in any input table (`audre validate` passes)
- [ ] `use_case_title` excluded from every reported table
- [ ] `privacy.assert_reportable` called on every figure input
- [ ] Suppression rule and thresholds stated in the methods text
- [ ] Cost provenance and missingness reported alongside cost figures
- [ ] Transition window stated as an analytical choice, with a sensitivity sweep
