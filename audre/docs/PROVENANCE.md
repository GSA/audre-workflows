# Where this code came from

`audre/` is a generalization of internal case-study analysis code. This file
records what was carried over and what changed, so that the open-source package
can be reviewed on its own terms and the internal scripts do not need to be
published to make sense of it.

The source folders are private and git-ignored.

## Mapping

| Open-source module | Generalized from | What changed |
| --- | --- | --- |
| `preprocess.py` | `preprocess_for_topic_modeling.ipynb`, `create_conversation_level_data.py`, `process_new_prod_usecases*.ipynb` | Roster/employment filtering removed entirely (organization-specific and identity-bearing). Timestamp parsing and message flattening kept and hardened. Content is now isolated in a separate documents frame instead of flowing alongside the analysis table. |
| `preprocess.parse_model_family` | `parse_model` in the notebooks | Was a hardcoded chain of vendor names and version strings, so every new model release silently fell into `"Other"`. Now splits on the leading token so unseen identifiers group correctly. |
| `labeling/llm.py` | `classify_chat_csv_taxonomy_to_jsonl.py`, `categorize_chat_csv_to_jsonl.py` | Vendored settings module and hardcoded local file paths replaced with arguments and environment variables. Retry with backoff added. The "preserve the raw response on a parse failure" behaviour was kept — it is the right call and avoids re-spending tokens. |
| `prompts/taxonomy_classification.jinja` | `classication_category_taxonomy.jinja` | Allowed-value lists are now rendered from a swappable taxonomy file rather than pasted into the prompt, so changing the scheme no longer means editing the prompt. Added an explicit instruction not to infer the user's job. |
| `prompts/topic_tags.jinja` | `tags_generation.jinja` | Added an instruction to exclude organization, team, person, and product names from tags. |
| `prompts/use_case_title.jinja` | `use_case_categorization.jinja` | Organization-specific examples replaced with neutral ones; added an instruction to exclude identifying detail. |
| `taxonomies/default.yaml` | value lists inside the taxonomy prompt | Extracted to data. Swappable, and documented as swappable for settings where the bundled classifications do not fit. |
| `adoption.py` | `R/01_model_preference.R`, `R/02_tags_workflow.R` | Aggregation moved from R to Python so the R layer only draws. Distinct-user counts added throughout (they did not exist and are the main re-identification defence). Depth now reports median and interquartile range, not just a mean. |
| `cooccurrence.py` | `tags_to_undirected_cooccur.py` **and its 9 near-duplicate variants** | The ten scripts differed only by a hardcoded office or model filter. Collapsed into one function plus `stratify.by_stratum`. Added Jaccard normalization, since raw co-occurrence counts are dominated by high-prevalence labels. |
| `networks.py` | `R/03_tag_network.R`, `R/03a`, `R/03b`, `R/05_tag_network_cost.r`, `R/05a` | Graph metrics moved to Python and deduplicated. Directed graphs now supported (the originals were undirected only) with in/out degree and strength. |
| `resources.py` | `R/04_basic_costs.r`, cost cells in `process_new_prod_usecases0714.ipynb` | Median imputation is now **opt-in** rather than the default, and always carries a provenance flag. Prices are supplied by the caller instead of hardcoded, because a stale price table silently corrupts every downstream cost figure. |
| `stratify.py` | the per-office and per-model script copies | New. Adds a stratum-eligibility gate and reports skipped strata. |
| `transitions.py` | — | **New.** The directed 24-hour transition analysis was described in the case study but was not present in the shared code, so it was written from the method description. Adds a configurable window, an `adjacent_only` option, asymmetry diagnostics, and a window sweep. |
| `privacy.py` | — | **New.** The source scripts had no suppression, no pseudonymization, and no minimum-cell rules. |
| `schema.py` | `user_prod_data_dictionary.csv` | The data dictionary described columns informally; this is an enforced schema with validation and referential-integrity checks. |
| `R/*.R` | `workflow_analysis_playaround/R/*` | Hardcoded dated filenames, organization names, and platform names removed; paths are environment-configurable. All aggregation removed. Interpretation limits moved into figure captions so they travel with the image. |

## Deliberately not carried over

- **Employment-roster joins.** Filtering conversations to current employees
  required organizational HR extracts. It is organization-specific and identity-
  bearing, and no analysis layer needs it.
- **Ridge-regression cost imputation.** The notebook fitted a Ridge model to
  predict conversation cost and, in the same notebook, measured it as barely
  distinguishable from a median baseline. Shipping it would imply more precision
  than it earns. `resolve_cost` offers median imputation, opt-in and flagged.
- **Embedding-based topic clustering** (`urania_v2_embeddings.py`). A separate
  method with a separate purpose, and it depends on a locally cached transformer
  checkpoint. Out of scope here.
- **Hardcoded model price tables.** Published prices change; a stale bundled
  table would silently produce wrong cost figures. Callers supply prices.

## Known gaps

- The R figure layer has not been executed in this environment (R is not
  installed here). It is a direct adaptation of scripts that ran against the same
  table shapes, but it should be run once before being relied upon.
- Statistical validation — synthetic-pattern recovery, window sensitivity,
  missingness sensitivity — is not implemented. `transitions.window_sweep`
  produces the input for a window sensitivity analysis but does not summarize it.
- The `RuleLabeler` keyword vocabulary is a reproducibility fixture. It matches
  surface keywords only and will miss paraphrase and implication, so it should
  not be used to produce substantive findings on real data.
