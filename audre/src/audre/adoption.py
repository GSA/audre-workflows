"""Adoption and engagement views: how much, by whom, and how deeply.

These are the "A" (usage) and "E" (engagement) layers of AUDRE. They are kept
separate because they answer different questions and, in practice, rank
activities differently: volume identifies broadly represented uses, whereas
interaction depth identifies less common but more interaction-intensive ones.
Reporting only one of the two systematically hides a class of activity.
"""

from __future__ import annotations

import pandas as pd

from . import privacy


def _labels_of_kind(labels: pd.DataFrame, label_kind: str) -> pd.DataFrame:
    subset = labels.loc[labels["label_kind"] == label_kind]
    if subset.empty:
        raise ValueError(
            f"No labels of kind {label_kind!r}. Available kinds: "
            f"{sorted(labels['label_kind'].dropna().unique())}"
        )
    return subset.drop_duplicates(subset=["conversation_id", "label_value"])


def _within_group_share(
    frame: pd.DataFrame, by: str | None, count_column: str = "n_conversations"
) -> pd.Series:
    """Share of ``count_column`` within each ``by`` group, or overall if None."""

    if by is None:
        total = frame[count_column].sum()
        return frame[count_column] / total if total else frame[count_column] * 0.0
    totals = frame.groupby(by, dropna=False)[count_column].transform("sum")
    return frame[count_column] / totals.replace(0, pd.NA)


def label_frequency(
    conversations: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    label_kind: str = "topic_tag",
    by: str | None = None,
    suppress: bool = True,
) -> pd.DataFrame:
    """Count conversations and distinct users per label.

    Multi-valued kinds such as ``topic_tag`` are not mutually exclusive, so the
    returned counts sum to more than the number of conversations. ``share`` is
    therefore the share *within* the grouping level and should be read as
    "proportion of label assignments", not "proportion of conversations".
    """

    subset = _labels_of_kind(labels, label_kind)
    group_columns = ["label_value"] + ([by] if by else [])

    merged = subset.merge(
        conversations[["conversation_id", "user_id"] + ([by] if by and by != "user_id" else [])],
        on="conversation_id",
        how="inner",
    )

    out = (
        merged.groupby(group_columns, dropna=False)
        .agg(
            n_conversations=("conversation_id", "nunique"),
            n_users=("user_id", "nunique"),
        )
        .reset_index()
    )

    out["share"] = _within_group_share(out, by)
    out = out.sort_values("n_conversations", ascending=False, ignore_index=True)
    return privacy.suppress_small_cells(out) if suppress else out


def interaction_depth(
    conversations: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    label_kind: str = "topic_tag",
    depth_column: str = "n_messages",
    suppress: bool = True,
) -> pd.DataFrame:
    """Summarize conversation depth per label.

    Depth is reported as mean, median and interquartile range together. A mean
    alone is misleading here because message counts are strongly right-skewed,
    and a single long conversation can dominate a small label.

    Longer conversations may reflect task complexity, iteration, or repeated
    clarification. Depth is not a measure of difficulty, quality, or success.
    """

    if depth_column not in conversations.columns:
        raise KeyError(f"conversations has no depth column {depth_column!r}")

    subset = _labels_of_kind(labels, label_kind)
    merged = subset.merge(
        conversations[["conversation_id", "user_id", depth_column]],
        on="conversation_id",
        how="inner",
    )

    out = (
        merged.groupby("label_value", dropna=False)
        .agg(
            n_conversations=("conversation_id", "nunique"),
            n_users=("user_id", "nunique"),
            mean_depth=(depth_column, "mean"),
            median_depth=(depth_column, "median"),
            p25_depth=(depth_column, lambda s: s.quantile(0.25)),
            p75_depth=(depth_column, lambda s: s.quantile(0.75)),
            max_depth=(depth_column, "max"),
        )
        .reset_index()
        .sort_values("mean_depth", ascending=False, ignore_index=True)
    )

    return privacy.suppress_small_cells(out) if suppress else out


def adoption_over_time(
    conversations: pd.DataFrame,
    *,
    freq: str = "W",
    by: str | None = None,
    suppress: bool = True,
) -> pd.DataFrame:
    """Conversation and active-user counts per period.

    ``n_users`` is the number of users active within each period, so the values
    are not additive across periods: summing them would double-count a user who
    is active in more than one period.
    """

    if "created_at" not in conversations.columns:
        raise KeyError("conversations must contain 'created_at'")

    working = conversations.dropna(subset=["created_at"]).copy()
    working["period"] = working["created_at"].dt.to_period(freq).dt.start_time

    group_columns = ["period"] + ([by] if by else [])
    out = (
        working.groupby(group_columns, dropna=False)
        .agg(
            n_conversations=("conversation_id", "nunique"),
            n_users=("user_id", "nunique"),
        )
        .reset_index()
        .sort_values(group_columns, ignore_index=True)
    )

    return privacy.suppress_small_cells(out) if suppress else out


def multi_model_adoption(
    conversations: pd.DataFrame,
    *,
    model_column: str = "model_family",
    suppress: bool = True,
) -> pd.DataFrame:
    """Distribution of how many distinct models each user engaged with.

    Whether users concentrate on one model or spread across several is a
    different signal from raw model share, and speaks to whether model choice
    is deliberate or incidental.
    """

    if model_column not in conversations.columns:
        raise KeyError(f"conversations has no model column {model_column!r}")

    per_user = (
        conversations.dropna(subset=[model_column])
        .groupby("user_id")[model_column]
        .nunique()
        .rename("n_models")
        .reset_index()
    )

    out = (
        per_user.groupby("n_models")
        .agg(n_users=("user_id", "nunique"))
        .reset_index()
        .sort_values("n_models", ignore_index=True)
    )
    out["share_of_users"] = out["n_users"] / out["n_users"].sum()

    if suppress:
        return privacy.suppress_small_cells(
            out, count_column="n_users", user_column=None, min_conversations=privacy.MIN_USERS
        )
    return out


def model_share(
    conversations: pd.DataFrame,
    *,
    model_column: str = "model_family",
    by: str | None = None,
    suppress: bool = True,
) -> pd.DataFrame:
    """Share of conversations by model, optionally within a context stratum."""

    if model_column not in conversations.columns:
        raise KeyError(f"conversations has no model column {model_column!r}")

    group_columns = [model_column] + ([by] if by else [])
    out = (
        conversations.dropna(subset=[model_column])
        .groupby(group_columns, dropna=False)
        .agg(
            n_conversations=("conversation_id", "nunique"),
            n_users=("user_id", "nunique"),
        )
        .reset_index()
    )

    out["share"] = _within_group_share(out, by)
    out = out.sort_values("n_conversations", ascending=False, ignore_index=True)
    return privacy.suppress_small_cells(out) if suppress else out


def leading_label_by_unit(
    conversations: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    label_kind: str = "work_activity",
    unit_column: str = "unit",
    suppress: bool = True,
) -> pd.DataFrame:
    """Most frequent label within each organizational unit, with its share.

    Intended for single-valued taxonomy kinds, where ``share`` is interpretable
    as a proportion of the unit's mapped conversations.
    """

    counts = label_frequency(
        conversations, labels, label_kind=label_kind, by=unit_column, suppress=suppress
    )
    if counts.empty:
        return counts

    leading = (
        counts.sort_values(["n_conversations", "label_value"], ascending=[False, True])
        .groupby(unit_column, dropna=False, as_index=False)
        .first()
    )
    return leading.sort_values("n_conversations", ascending=False, ignore_index=True)
