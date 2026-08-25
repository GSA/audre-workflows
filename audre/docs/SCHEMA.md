# Minimal input schema

AUDRE requires three narrow tables. It does **not** require prompts, responses,
conversation titles, or any user attribute beyond an opaque identifier. That is
the point: an organization can run the labeling stage inside its own boundary and
then share, archive, or publish an analysis built only from these tables.

Print the schema at any time:

```bash
audre schema
```

## `conversations` — the analysis spine

One row per chat thread.

| Column | Requirement | Notes |
| --- | --- | --- |
| `conversation_id` | required | Unique. One distinct chat thread. |
| `user_id` | required | Opaque and pseudonymous. Needed because transitions are computed *within* a user and because minimum-user thresholds are the main re-identification defence. |
| `created_at` | required | Timestamp, coerced to UTC. Drives the transition and temporal layers. |
| `unit` | optional | Organizational context: team, office, business line. AUDRE never interprets it, so pseudonymous unit names can be substituted upstream. |
| `n_messages` | optional | Total messages, all roles. The default depth measure. |
| `n_user_messages` | optional | User-authored messages only. |
| `model` | optional | Model identifier. |
| `model_family` | optional | Grouped model identifier. `preprocess.parse_model_family` derives this if absent. |

## `labels` — long format

One row per `(conversation_id, label_kind, label_value)`. Long format is
required because a conversation may carry several unordered topic tags while
taxonomy assignments are single-valued.

| Column | Requirement | Notes |
| --- | --- | --- |
| `conversation_id` | required | Foreign key to `conversations`. |
| `label_kind` | required | See below. |
| `label_value` | required | The assigned value. |
| `label_source` | optional | What assigned it, e.g. `llm:<model>` or `rules:keyword-v1`. Strongly recommended: labels are assigned, not recorded, and figures should be able to say by what. |
| `label_confidence` | optional | Numeric, if the labeler provides one. |

### Label kinds

| Kind | Cardinality | Meaning |
| --- | --- | --- |
| `topic_tag` | **multi-valued** | Open-vocabulary topic. Counts are *not* mutually exclusive and sum to more than the number of conversations. |
| `work_activity` | single-valued | Closed-set work activity. Describes the activity in the conversation, **not** the user's occupation. |
| `request_topic` | single-valued | Closed-set domain of the request. |
| `occupation_group` | single-valued | Closed-set occupational group associated with the *subject matter*. |
| `use_case_title` | single-valued | Free-text title. Content-bearing: use for qualitative review only, never publish or aggregate. |

Non-standard kinds pass through with a warning but have no built-in semantics.

## `resources` — optional

One row per conversation.

| Column | Requirement | Notes |
| --- | --- | --- |
| `conversation_id` | required | Foreign key. |
| `prompt_tokens`, `completion_tokens`, `total_tokens` | optional | Observed or estimated. |
| `cost_usd` | optional | Observed cost from platform records. |
| `cost_source` | optional | Set by `resources.resolve_cost` if absent. |

`resources.resolve_cost` produces `analysis_cost_usd` plus a `cost_source`
provenance flag (`observed`, `estimated`, `median_imputed`). Estimated and
observed values are never silently merged, and `resources.cost_coverage` reports
the split so a reader can see what a cost total is actually built from.

## Rejected columns

Validation **fails** if any of these appear, because they carry conversation
content or direct identifiers: `content`, `message_text`, `prompt`, `response`,
`chat`, `chat_parsed`, `email`, `employee_email`, `full_name`, `title`,
`conversation_title`.

Content lives only in the documents frame produced by
`preprocess.build_documents`, which feeds the labeling stage and nothing else.

## Getting from a raw export to this schema

`audre.preprocess` handles both common export shapes:

```python
from audre import preprocess, schema

# Nested export: one row per conversation with a JSON message history
messages = preprocess.messages_from_nested(raw, chat_column="chat")

# Long export: one row per message already — just parse the timestamps
messages["created_at"] = preprocess.parse_timestamps(messages["created_at_raw"])

conversations = preprocess.build_conversations(messages, unit_map=unit_lookup)
documents = preprocess.build_documents(messages)   # content; labeling only
```

Convert an existing wide label table with `preprocess.labels_from_frame`, which
explodes list-valued and comma-separated cells:

```python
labels = preprocess.labels_from_frame(
    existing,
    mapping={"tags": "topic_tag", "onet": "work_activity", "request": "request_topic"},
    label_source="llm:llama_4_maverick",
)
```
