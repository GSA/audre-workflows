# AUDRE

**A**nalyzing **U**sage, **D**epth, **R**esource intensity, and **E**ngagement.

A reproducible, privacy-aware pipeline for turning enterprise GenAI chat
telemetry into context-specific usage representations and candidate workflow
hypotheses.

Counts of users and conversations establish that a platform is being used. They
do not say what work is being supported, whether use is isolated or recurring,
how activities relate over time, or which uses should receive limited evaluation
resources. AUDRE combines conversation-level labels, interaction depth,
occupational activity mapping, resource intensity, organizational context, and
directed transitions between temporally adjacent conversations into views that
address those questions.

## What AUDRE does not claim

The framework separates what is **recorded** from what is **assigned**,
**derived**, and **inferred**:

| | |
| --- | --- |
| **Recorded** | Users, conversations, messages, timestamps, models |
| **Assigned** | Task labels and taxonomy categories |
| **Derived** | Transitions, co-occurrence, network metrics, token and cost estimates |
| **Inferred** | Candidate workflows |
| **Independent** | Outcomes, effectiveness, ROI — *not produced by this package* |

Operational logs cannot establish whether AI improved time, quality, accuracy, or
any downstream outcome. Those questions need independently measured outcomes and
a baseline or counterfactual. Nothing here estimates them. Recurring transitions
are hypotheses for validation by interview, process mapping, or targeted
measurement — not confirmed business processes.

## Install

Requires Python 3.10+.

```bash
cd audre
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # add ",llm" for the LLM labeler
```

## Quick start

```bash
python examples/run_pipeline.py
```

This runs every stage against the committed test fixture and writes tidy CSVs to
`examples/output/tables/`. The fixture exercises pipeline wiring only — it is far
too small and regular to support any finding.

Or via the CLI:

```bash
audre schema
audre validate --conversations conversations.csv --labels labels.csv
audre label    --documents documents.csv --output labels.csv --labeler rules
audre analyze  --conversations conversations.csv --labels labels.csv \
               --resources resources.csv --output-dir output/tables
```

Then render figures:

```bash
AUDRE_TABLE_DIR=output/tables AUDRE_PLOT_DIR=output/plots \
    Rscript R/01_adoption_engagement.R
```

## Pipeline

```
preprocess  ->  privacy  ->  labeling  ->  analysis  ->  figures (R)
```

The ordering is deliberate. Privacy preparation happens before analysis, so no
intermediate artifact holds a direct identifier, and every analysis stage emits
aggregates that have already passed a minimum-cell rule.

| Stage | Module | Purpose |
| --- | --- | --- |
| Preprocess | `preprocess` | Raw export (nested JSON or message-level) into the schema. Content is isolated into a separate documents frame. |
| Privacy | `privacy` | Salted pseudonymization, content removal, minimum-cell suppression, reporting gates. |
| Labeling | `labeling` | `LLMLabeler` (closed-set taxonomy + open topic tags + titles) or `RuleLabeler` (deterministic, offline). |
| **A**doption / **E**ngagement | `adoption` | Volume, depth, temporal trend, model share. |
| **R**esource intensity | `resources` | Token and cost measures with observed-versus-estimated provenance. |
| Transitions | `transitions` | Directed relationships between temporally adjacent conversations. |
| Co-occurrence | `cooccurrence` | Undirected same-conversation relationships. |
| Networks | `networks` | Graph construction and node metrics for either edge shape. |
| Context | `stratify` | Any of the above within organizational or model strata. |
| Figures | `R/` | ggplot2 / ggraph. Draws only; performs no aggregation. |

Volume and depth are always reported as a pair, because they rank activities
differently: volume surfaces broadly represented uses, depth surfaces less
frequent but more interaction-intensive ones. The same holds for volume and
resource intensity — the most common activity is not necessarily the most
expensive.

## Privacy defaults

Every aggregate must clear both thresholds:

- `n_conversations >= 5`
- `n_users >= 3` — the load-bearing rule. A conversation threshold alone is
  satisfied by a single prolific user, which makes the "aggregate" personal.

Strata below 50 conversations or 10 users are not analyzed at all. These are
k-anonymity-style minimum-cell rules, **not** differential privacy: there is no
privacy budget and no formal guarantee. See [`docs/PRIVACY.md`](docs/PRIVACY.md).

## Labeling

The prompts used for the original analysis are included, generalized so the
closed-set value lists render from a swappable taxonomy file rather than being
pasted into the prompt:

- `prompts/taxonomy_classification.jinja` — closed-set work activity, request
  topic, occupation group
- `prompts/topic_tags.jinja` — open-vocabulary topic tags
- `prompts/use_case_title.jinja` — short descriptive title

```bash
export AUDRE_LLM_BASE_URL=https://your-endpoint/v1
export AUDRE_LLM_API_KEY=...
```

```python
from audre.labeling import LLMLabeler, RuleLabeler, Taxonomy

labels = LLMLabeler(model="llama_4_maverick").label(documents)
labels = RuleLabeler().label(documents)                    # offline, deterministic

Taxonomy.load().coverage(labels)   # which categories were used, and which never were
```

Two taxonomies ship, and any YAML of the same shape works. Swapping requires no
prompt editing.

| File | When to use |
| --- | --- |
| `taxonomies/default.yaml` | **Recommended.** Maps to published occupational classifications (O\*NET work activities, SOC major groups), which is what makes the results comparable with external descriptions of work. |
| `taxonomies/neutral.yaml` | Coarser and jurisdiction-neutral. For settings the published classifications describe poorly, or where a national taxonomy is out of place. |

Taxonomy assignments describe work activities represented in a conversation.
They do **not** identify a user's occupation, grade, or responsibilities.

## Documentation

- [`docs/SCHEMA.md`](docs/SCHEMA.md) — the three input tables and label kinds
- [`docs/PRIVACY.md`](docs/PRIVACY.md) — reporting rules and pre-publication checklist
- [`docs/PROVENANCE.md`](docs/PROVENANCE.md) — what this generalizes, and what was
  deliberately left out

## Limitations

- Observational and descriptive. Captures activity only within the observed
  platform: not work done in other systems, not time spent reviewing output, not
  whether output was used.
- The transition window is an analytical definition, not evidence of process
  dependence. Use `transitions.window_sweep` to show how conclusions move as it
  varies.
- Conversation-level multi-label tagging obscures within-conversation ordering
  and generates multiple eligible transitions per conversation pair.
- Token and cost figures are estimates that depend on transcript availability,
  tokenization assumptions, model metadata, and price inputs. They approximate
  transcript-level inference cost only and exclude all other program expense.
- Assigned labels require validation. `Taxonomy.coverage` surfaces off-taxonomy
  values; human review of a sample is still necessary.
- The R figure layer has not been executed in the environment where it was
  written. Run it once before relying on it.
- Statistical validation (synthetic-pattern recovery, sensitivity analysis) is
  not implemented here.

## Testing

```bash
python -m pytest                         # 77 tests
python tests/fixtures/make_fixtures.py   # regenerate fixtures
```

## License

[CC0 1.0 Universal](../LICENSE) — public domain dedication. Use, modify, and
redistribute freely, including commercially, without permission or attribution.
No trademark or patent rights are waived; provided as-is without warranty.

Attribution is not required but is appreciated for academic reuse. This package
accompanies a methods contribution submitted to a NeurIPS workshop; citation
details will be added once the submission status is resolved.

Note that the bundled `taxonomies/default.yaml` reproduces category names from
published occupational classifications (O\*NET work activities and SOC major
groups), which are U.S. Government works in the public domain. CC0 applies to
this package's own code and data files.
