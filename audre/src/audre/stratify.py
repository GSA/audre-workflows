"""Run any AUDRE analysis within organizational or model context strata.

The case-study code this package generalizes contained ten near-identical
scripts that differed only by a hardcoded office or model filter. That pattern
guarantees drift: a fix applied to one copy silently misses the other nine.

:func:`by_stratum` replaces all of them. Any function whose first argument is a
conversations frame can be applied per stratum, and small strata are skipped
rather than published, because a network built from a handful of users is both
unstable and re-identifying.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import pandas as pd

from . import privacy

#: Minimum users required before a stratum is analyzed at all. Distinct from
#: cell-level suppression: this gate is about whether a stratum can support a
#: stable estimate, not whether one aggregate row is safe to show.
MIN_STRATUM_USERS = 10
MIN_STRATUM_CONVERSATIONS = 50


def stratum_sizes(conversations: pd.DataFrame, by: str) -> pd.DataFrame:
    """Report size and eligibility of each stratum before any analysis runs."""

    if by not in conversations.columns:
        raise KeyError(f"conversations has no column {by!r}")

    out = (
        conversations.groupby(by, dropna=False)
        .agg(
            n_conversations=("conversation_id", "nunique"),
            n_users=("user_id", "nunique"),
        )
        .reset_index()
        .sort_values("n_conversations", ascending=False, ignore_index=True)
    )
    out["eligible"] = (out["n_conversations"] >= MIN_STRATUM_CONVERSATIONS) & (
        out["n_users"] >= MIN_STRATUM_USERS
    )
    return out


def eligible_strata(
    conversations: pd.DataFrame,
    by: str,
    *,
    min_users: int = MIN_STRATUM_USERS,
    min_conversations: int = MIN_STRATUM_CONVERSATIONS,
) -> list[str]:
    """List strata large enough to analyze."""

    sizes = stratum_sizes(conversations, by)
    keep = (sizes["n_conversations"] >= min_conversations) & (
        sizes["n_users"] >= min_users
    )
    return sizes.loc[keep, by].dropna().tolist()


def by_stratum(
    analysis: Callable[..., pd.DataFrame],
    conversations: pd.DataFrame,
    *args,
    by: str,
    strata: Iterable[str] | None = None,
    min_users: int = MIN_STRATUM_USERS,
    min_conversations: int = MIN_STRATUM_CONVERSATIONS,
    **kwargs,
) -> pd.DataFrame:
    """Apply ``analysis`` within each eligible stratum and concatenate results.

    Parameters
    ----------
    analysis:
        Any AUDRE analysis whose first positional argument is a conversations
        frame, for example :func:`audre.transitions.build_transitions` or
        :func:`audre.adoption.interaction_depth`.
    by:
        Stratifying column on ``conversations``, typically ``"unit"`` or
        ``"model_family"``.
    strata:
        Explicit stratum list. Defaults to every eligible stratum. Values passed
        here are still size-checked, so naming a small stratum will not force it
        into the output.

    Returns
    -------
    DataFrame
        The concatenated results with a ``stratum`` and ``stratum_variable``
        column added. Results are stratum-specific and must not be compared as
        if computed on a common scale; each stratum has its own denominators.
    """

    requested = list(strata) if strata is not None else None
    allowed = eligible_strata(
        conversations, by, min_users=min_users, min_conversations=min_conversations
    )
    targets = [value for value in (requested or allowed) if value in allowed]

    skipped = sorted(set(requested or allowed) - set(targets))
    frames = []

    for value in targets:
        subset = conversations[conversations[by] == value]
        result = analysis(subset, *args, **kwargs)
        if result is None or result.empty:
            continue
        frames.append(result.assign(stratum=value, stratum_variable=by))

    if not frames:
        out = pd.DataFrame()
    else:
        out = pd.concat(frames, ignore_index=True)

    out.attrs["stratum_variable"] = by
    out.attrs["strata_analyzed"] = targets
    out.attrs["strata_skipped_too_small"] = skipped
    out.attrs["stratum_gate"] = (
        f"n_conversations >= {min_conversations} and n_users >= {min_users}"
    )
    return out


def label_share_matrix(
    conversations: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    by: str = "unit",
    label_kind: str = "topic_tag",
    top_n_labels: int = 15,
    suppress: bool = True,
) -> pd.DataFrame:
    """Long-format label share per stratum, for heatmap figures.

    Shares are computed *within* each stratum, so a small unit is not visually
    swamped by a large one. The trade-off is that column values are not
    comparable as absolute volumes; pair the heatmap with stratum sizes.
    """

    from .adoption import label_frequency

    counts = label_frequency(
        conversations, labels, label_kind=label_kind, by=by, suppress=suppress
    )
    if counts.empty:
        return counts

    top_labels = (
        counts.groupby("label_value")["n_conversations"]
        .sum()
        .nlargest(top_n_labels)
        .index
    )
    out = counts[counts["label_value"].isin(top_labels)].copy()
    return out.sort_values([by, "share"], ascending=[True, False], ignore_index=True)
