"""Undirected label co-occurrence within conversations.

Complements :mod:`audre.transitions`. Co-occurrence describes labels applied to
the *same* conversation; transitions describe labels applied to *consecutive*
conversations. Reporting both makes the distinction explicit, since an
undirected network cannot distinguish "these topics appear together" from
"this topic leads to that one".
"""

from __future__ import annotations

from itertools import combinations

import pandas as pd

from . import privacy


def build_cooccurrence(
    conversations: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    label_kind: str = "topic_tag",
    suppress: bool = True,
    min_users: int = privacy.MIN_USERS,
) -> pd.DataFrame:
    """Build an undirected label co-occurrence edge table.

    Pairs are emitted in a canonical (sorted) order so that ``(a, b)`` and
    ``(b, a)`` aggregate into a single edge. ``jaccard`` is included because raw
    counts are dominated by high-prevalence labels: two common tags co-occur
    often simply by being common, and the normalized measure separates that from
    a genuine association.
    """

    subset = labels.loc[
        labels["label_kind"] == label_kind, ["conversation_id", "label_value"]
    ].drop_duplicates()
    if subset.empty:
        raise ValueError(f"No labels of kind {label_kind!r}")

    spine_columns = ["conversation_id", "user_id"] + (
        ["unit"] if "unit" in conversations.columns else []
    )
    merged = subset.merge(conversations[spine_columns], on="conversation_id", how="inner")

    label_totals = (
        merged.groupby("label_value")["conversation_id"].nunique().rename("n_label")
    )

    rows = []
    for (conversation_id, user_id), group in merged.groupby(
        ["conversation_id", "user_id"], sort=False
    ):
        label_values = sorted(group["label_value"].dropna().unique())
        if len(label_values) < 2:
            continue
        for label_a, label_b in combinations(label_values, 2):
            rows.append(
                {
                    "label_a": label_a,
                    "label_b": label_b,
                    "conversation_id": conversation_id,
                    "user_id": user_id,
                }
            )

    if not rows:
        return _empty_cooccurrence()

    instances = pd.DataFrame(rows)
    edges = (
        instances.groupby(["label_a", "label_b"], dropna=False)
        .agg(
            n_conversations=("conversation_id", "nunique"),
            n_users=("user_id", "nunique"),
        )
        .reset_index()
    )

    edges = edges.merge(
        label_totals.rename("n_label_a"), left_on="label_a", right_index=True, how="left"
    ).merge(
        label_totals.rename("n_label_b"), left_on="label_b", right_index=True, how="left"
    )
    union = edges["n_label_a"] + edges["n_label_b"] - edges["n_conversations"]
    edges["jaccard"] = edges["n_conversations"] / union.where(union > 0)

    edges = edges.sort_values(
        ["n_conversations", "n_users"], ascending=False, ignore_index=True
    )
    edges.attrs["label_kind"] = label_kind

    if suppress:
        edges = privacy.suppress_small_cells(edges, min_users=min_users)
    return edges


def _empty_cooccurrence() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "label_a",
            "label_b",
            "n_conversations",
            "n_users",
            "n_label_a",
            "n_label_b",
            "jaccard",
        ]
    )
