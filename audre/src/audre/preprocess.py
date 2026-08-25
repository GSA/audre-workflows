"""Turn a raw chat export into AUDRE's schema.

Chat platforms export conversations in one of two shapes:

* **long/message-level** — one row per message, with ``role`` and ``content``
  (typical of a SQL or warehouse query);
* **nested** — one row per conversation with a JSON blob holding the message
  history (typical of a document-store dump).

Both are handled. Everything content-bearing is consumed here and does not
propagate: :func:`build_documents` yields text for the labeling step only, and
:func:`build_conversations` returns the analysis spine with no text at all. That
split is what allows an organization to run labeling internally and then share
or archive only the non-content tables.

File attachments and base64 image payloads are dropped rather than
stringified, since they inflate character counts and can embed content that no
downstream stage is designed to protect.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

import pandas as pd

from . import schema

USER_ROLES = ("user", "human")
CONTENT_KEYS = ("text", "content", "value", "message")
DROP_KEYS = frozenset({"files", "attachments", "images", "url", "data", "image_url"})

MIN_DOCUMENT_WORDS = 10


def parse_timestamps(values: pd.Series) -> pd.Series:
    """Parse mixed Unix seconds, Unix milliseconds, and date strings to UTC.

    Real exports mix all three, often within one column. Values are routed by
    magnitude rather than assumed, because misreading milliseconds as seconds
    shifts timestamps by decades and silently destroys the transition layer.
    """

    result = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns, UTC]")
    numeric = pd.to_numeric(values, errors="coerce")

    is_numeric = numeric.notna()
    if is_numeric.any():
        seconds = numeric.copy()
        # Anything at or above 1e11 seconds would be year 5138; treat as ms.
        seconds.loc[seconds >= 1e11] = seconds.loc[seconds >= 1e11] / 1000
        result.loc[is_numeric] = pd.to_datetime(
            seconds.loc[is_numeric], unit="s", errors="coerce", utc=True
        )

    is_string = ~is_numeric & values.notna()
    if is_string.any():
        result.loc[is_string] = pd.to_datetime(
            values.loc[is_string], errors="coerce", utc=True
        )

    return result


def _first_available(mapping: Any, keys: Iterable[str], default: Any = None) -> Any:
    if not isinstance(mapping, dict):
        return default
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return default


def extract_text(content: Any) -> str | None:
    """Flatten a message ``content`` field to plain text, dropping binary parts."""

    if content is None:
        return None

    if isinstance(content, str):
        return content.strip() or None

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
            elif isinstance(item, dict):
                candidate = _first_available(item, CONTENT_KEYS)
                if isinstance(candidate, str) and candidate.strip():
                    parts.append(candidate.strip())
        return "\n".join(parts).strip() or None

    if isinstance(content, dict):
        candidate = _first_available(content, CONTENT_KEYS)
        if isinstance(candidate, str):
            return candidate.strip() or None
        remainder = {k: v for k, v in content.items() if k not in DROP_KEYS}
        return json.dumps(remainder, ensure_ascii=False) if remainder else None

    return str(content).strip() or None


def _messages_of(chat: Any) -> list[tuple[str, dict]]:
    """Return ``(message_id, message)`` pairs from a nested chat object."""

    if not isinstance(chat, dict):
        return []

    messages = None
    history = chat.get("history")
    if isinstance(history, dict) and isinstance(history.get("messages"), (dict, list)):
        messages = history["messages"]
    elif isinstance(chat.get("messages"), (dict, list)):
        messages = chat["messages"]

    if isinstance(messages, dict):
        return [(str(key), value) for key, value in messages.items() if isinstance(value, dict)]
    if isinstance(messages, list):
        return [
            (str(message.get("id", index)), message)
            for index, message in enumerate(messages)
            if isinstance(message, dict)
        ]
    return []


def messages_from_nested(
    frame: pd.DataFrame,
    *,
    chat_column: str = "chat",
    conversation_id_column: str = "conversation_id",
    user_only: bool = False,
) -> pd.DataFrame:
    """Explode a nested chat export into a message-level table."""

    records = []
    for row in frame.to_dict("records"):
        chat = row.get(chat_column)
        if isinstance(chat, str):
            try:
                chat = json.loads(chat)
            except json.JSONDecodeError:
                continue

        for index, (message_id, message) in enumerate(_messages_of(chat)):
            role = _first_available(message, ("role", "sender", "author"))
            if role is None:
                continue
            role = str(role).strip().lower()
            if user_only and role not in USER_ROLES:
                continue

            text = extract_text(_first_available(message, CONTENT_KEYS))
            if not text:
                continue

            records.append(
                {
                    "conversation_id": row.get(conversation_id_column),
                    "message_id": message_id,
                    "message_index": index,
                    "role": role,
                    "content": text,
                    "created_at_raw": _first_available(
                        message, ("timestamp", "created_at", "createdAt", "date")
                    ),
                    "model": _first_available(message, ("model", "model_name")),
                }
            )

    out = pd.DataFrame(records)
    if not out.empty:
        out["created_at"] = parse_timestamps(out["created_at_raw"])
        out = out.sort_values(
            ["conversation_id", "created_at", "message_index"],
            na_position="last",
            ignore_index=True,
        )
    return out


def build_conversations(
    messages: pd.DataFrame,
    *,
    user_map: pd.DataFrame | None = None,
    unit_map: pd.DataFrame | None = None,
    model_column: str = "model",
) -> pd.DataFrame:
    """Aggregate a message-level table into the AUDRE ``conversations`` spine.

    ``created_at`` is the earliest message timestamp, ``n_messages`` counts all
    roles, and ``n_user_messages`` counts only user-authored messages. Both
    depth measures are kept because they answer different questions: total
    messages reflects the size of the exchange, user messages reflect how much
    the person actually put in.

    The primary model is the one carrying the most messages in the conversation,
    which matches how the case-study data attributed multi-model threads.
    """

    required = {"conversation_id", "role", "content"}
    missing = required - set(messages.columns)
    if missing:
        raise KeyError(f"messages is missing {sorted(missing)}")

    working = messages.copy()
    if "created_at" not in working.columns:
        raise KeyError("messages must contain 'created_at'; call parse_timestamps first")

    working["is_user"] = working["role"].str.lower().isin(USER_ROLES)
    working["character_count"] = working["content"].fillna("").str.len()

    aggregations = {
        "created_at": ("created_at", "min"),
        "last_message_at": ("created_at", "max"),
        "n_messages": ("content", "size"),
        "n_user_messages": ("is_user", "sum"),
        "character_count": ("character_count", "sum"),
    }
    if "user_id" in working.columns:
        aggregations["user_id"] = ("user_id", "first")

    out = working.groupby("conversation_id", as_index=False).agg(**aggregations)

    if model_column in working.columns:
        primary = (
            working.dropna(subset=[model_column])
            .groupby(["conversation_id", model_column])
            .size()
            .rename("n")
            .reset_index()
            .sort_values(["conversation_id", "n"], ascending=[True, False])
            .groupby("conversation_id", as_index=False)
            .first()[["conversation_id", model_column]]
            .rename(columns={model_column: "model"})
        )
        n_models = (
            working.dropna(subset=[model_column])
            .groupby("conversation_id")[model_column]
            .nunique()
            .rename("n_models_used")
            .reset_index()
        )
        out = out.merge(primary, on="conversation_id", how="left").merge(
            n_models, on="conversation_id", how="left"
        )

    if user_map is not None:
        out = out.merge(user_map, on="conversation_id", how="left")
    if unit_map is not None:
        join_key = "user_id" if "user_id" in unit_map.columns else "conversation_id"
        out = out.merge(unit_map, on=join_key, how="left")

    if "model" in out.columns:
        out["model_family"] = parse_model_family(out["model"])

    return schema.coerce(out, schema.CONVERSATIONS)


def parse_model_family(models: pd.Series) -> pd.Series:
    """Group model identifiers into families by leading token.

    Deliberately generic. The case-study version hardcoded a handful of vendor
    names and version strings, which meant every new model release silently
    landed in "Other". Splitting on the first delimiter generalizes to unseen
    identifiers; supply an explicit mapping instead when families do not follow
    that convention.
    """

    normalized = models.astype("string").str.strip().str.lower()
    family = normalized.str.split(r"[-_./:\s]", n=1, regex=True).str[0]
    return family.str.replace(r"\d+$", "", regex=True).str.strip().replace("", pd.NA)


def build_documents(
    messages: pd.DataFrame,
    *,
    user_only: bool = True,
    min_words: int = MIN_DOCUMENT_WORDS,
) -> pd.DataFrame:
    """Build one text document per conversation, for the labeling step only.

    Parameters
    ----------
    user_only:
        Keep only user-authored messages. On by default: including model output
        makes documents reflect what the system generated as much as what the
        person asked for, which biases topic labels toward model style.
    min_words:
        Drop documents shorter than this. Very short documents produce unstable
        labels, so they are excluded from labeling rather than labeled badly.

    Returns
    -------
    DataFrame
        ``conversation_id``, ``document``, ``n_messages``, ``word_count``. This
        frame carries conversation content and must not be published or joined
        into a reported table.
    """

    working = messages.copy()
    if user_only:
        working = working[working["role"].str.lower().isin(USER_ROLES)]

    out = (
        working.dropna(subset=["content"])
        .sort_values(["conversation_id", "created_at", "message_index"], na_position="last")
        .groupby("conversation_id", as_index=False)
        .agg(
            document=("content", lambda values: "\n\n".join(v.strip() for v in values if v.strip())),
            n_messages=("content", "size"),
        )
    )

    out["word_count"] = out["document"].str.split().str.len()
    n_before = len(out)
    out = out[out["word_count"] >= min_words].reset_index(drop=True)

    out.attrs["n_dropped_short"] = n_before - len(out)
    out.attrs["min_words"] = min_words
    out.attrs["contains_content"] = True
    return out


def labels_from_frame(
    frame: pd.DataFrame,
    *,
    mapping: dict[str, str],
    label_source: str = "external",
) -> pd.DataFrame:
    """Convert a wide label table into AUDRE's long ``labels`` format.

    Parameters
    ----------
    mapping:
        ``{source_column: label_kind}``. List-valued and comma-separated cells
        are exploded, so an existing multi-tag column converts without
        pre-processing.
    """

    frames = []
    for column, label_kind in mapping.items():
        if column not in frame.columns:
            continue
        subset = frame[["conversation_id", column]].rename(columns={column: "label_value"})
        subset["label_value"] = subset["label_value"].map(_as_list)
        subset = subset.explode("label_value")
        subset["label_kind"] = label_kind
        frames.append(subset)

    if not frames:
        return pd.DataFrame(
            columns=["conversation_id", "label_kind", "label_value", "label_source"]
        )

    out = pd.concat(frames, ignore_index=True)
    out["label_source"] = label_source
    out = out.dropna(subset=["label_value"])
    out["label_value"] = out["label_value"].astype("string").str.strip()
    out = out[out["label_value"] != ""]
    return out.drop_duplicates(ignore_index=True)[
        ["conversation_id", "label_kind", "label_value", "label_source"]
    ]


def _as_list(value: Any) -> list[Any]:
    """Normalize a cell to a list, tolerating stringified lists from CSV round-trips."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text.replace("'", '"'))
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
        if "," in text:
            return [part.strip() for part in text.split(",") if part.strip()]
        return [text]
    return [value]
