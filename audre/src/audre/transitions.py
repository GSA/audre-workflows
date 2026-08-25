"""Directed task transitions between temporally adjacent conversations.

The "T" layer of AUDRE, and the only layer that connects otherwise isolated
interactions. A directed transition from label ``A`` to label ``B`` is counted
when a conversation carrying ``B`` follows a conversation carrying ``A`` for the
*same user* within a bounded time window.

Interpretation limits, which the API is designed to keep visible:

* Temporal adjacency is not process dependence. Two conversations occurring in
  sequence need not belong to the same business process, and neither need have
  informed the other. Recurring transitions are *candidate workflows* —
  hypotheses for validation by interview, process mapping, or targeted outcome
  measurement — not confirmed workflows.
* The window length is an analytical choice, not a property of the data. Use
  :func:`window_sweep` to show how conclusions move as it varies; a transition
  that appears only at one window length is an artifact of that choice.
* Multi-label conversations generate every eligible label pair, so counts are
  label-level relationships rather than one transition per conversation pair.
  ``n_users`` is the honest support measure: a high count from two users is not
  an organizational pattern.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from . import privacy

DEFAULT_WINDOW_HOURS = 24.0


def _prepare(
    conversations: pd.DataFrame, labels: pd.DataFrame, label_kind: str
) -> pd.DataFrame:
    subset = labels.loc[labels["label_kind"] == label_kind, ["conversation_id", "label_value"]]
    if subset.empty:
        raise ValueError(
            f"No labels of kind {label_kind!r}. Available kinds: "
            f"{sorted(labels['label_kind'].dropna().unique())}"
        )

    required = {"conversation_id", "user_id", "created_at"}
    missing = required - set(conversations.columns)
    if missing:
        raise KeyError(f"conversations is missing {sorted(missing)}")

    spine = conversations.dropna(subset=["user_id", "created_at"])[
        ["conversation_id", "user_id", "created_at"]
        + (["unit"] if "unit" in conversations.columns else [])
    ]

    merged = subset.drop_duplicates().merge(spine, on="conversation_id", how="inner")
    return merged.sort_values(["user_id", "created_at", "conversation_id"], ignore_index=True)


def conversation_pairs(
    conversations: pd.DataFrame,
    *,
    window_hours: float = DEFAULT_WINDOW_HOURS,
    adjacent_only: bool = True,
) -> pd.DataFrame:
    """Return eligible ordered conversation pairs per user.

    Parameters
    ----------
    adjacent_only:
        When true (default), pair each conversation only with the one that
        immediately follows it for that user. When false, pair it with every
        later conversation inside the window.

        The default is the conservative choice. Allowing all pairs within a
        window makes counts grow roughly quadratically with a user's activity
        in a busy day, so the most active users come to dominate the network for
        reasons that have nothing to do with workflow structure.
    """

    if window_hours <= 0:
        raise ValueError("window_hours must be positive")

    working = conversations.dropna(subset=["user_id", "created_at"])[
        ["conversation_id", "user_id", "created_at"]
    ].sort_values(["user_id", "created_at", "conversation_id"], ignore_index=True)

    window = pd.Timedelta(hours=window_hours)
    rows = []

    for _, group in working.groupby("user_id", sort=False):
        records = group.to_dict("records")
        for index, first in enumerate(records):
            for second in records[index + 1 :]:
                gap = second["created_at"] - first["created_at"]
                if gap > window:
                    break
                rows.append(
                    {
                        "user_id": first["user_id"],
                        "from_conversation_id": first["conversation_id"],
                        "to_conversation_id": second["conversation_id"],
                        "gap_minutes": gap.total_seconds() / 60.0,
                    }
                )
                if adjacent_only:
                    break

    return pd.DataFrame(
        rows,
        columns=[
            "user_id",
            "from_conversation_id",
            "to_conversation_id",
            "gap_minutes",
        ],
    )


def build_transitions(
    conversations: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    label_kind: str = "topic_tag",
    window_hours: float = DEFAULT_WINDOW_HOURS,
    adjacent_only: bool = True,
    exclude_self: bool = True,
    suppress: bool = True,
    min_users: int = privacy.MIN_USERS,
) -> pd.DataFrame:
    """Build a directed label-transition edge table.

    Returns one row per ordered ``(from_label, to_label)`` pair with:

    ``n_transitions``
        Number of eligible label-pair instances.
    ``n_users``
        Distinct users contributing at least one instance. This is the support
        measure to filter and report on.
    ``conditional_probability``
        ``n_transitions`` divided by all outgoing transitions from
        ``from_label``. Read as "given that a transition left A, how often did
        it arrive at B" — not as a probability that A causes B.
    ``median_gap_minutes``
        Median elapsed time between the paired conversations, which helps
        distinguish tight task sequences from same-day coincidence.
    """

    prepared = _prepare(conversations, labels, label_kind)
    pairs = conversation_pairs(
        conversations, window_hours=window_hours, adjacent_only=adjacent_only
    )

    if pairs.empty:
        return _empty_transitions()

    label_lookup = (
        prepared.groupby("conversation_id")["label_value"].apply(list).to_dict()
    )

    rows = []
    for pair in pairs.itertuples(index=False):
        from_labels = label_lookup.get(pair.from_conversation_id)
        to_labels = label_lookup.get(pair.to_conversation_id)
        if not from_labels or not to_labels:
            continue
        for from_label, to_label in product(from_labels, to_labels):
            if exclude_self and from_label == to_label:
                continue
            rows.append(
                {
                    "from_label": from_label,
                    "to_label": to_label,
                    "user_id": pair.user_id,
                    "gap_minutes": pair.gap_minutes,
                }
            )

    if not rows:
        return _empty_transitions()

    instances = pd.DataFrame(rows)
    edges = (
        instances.groupby(["from_label", "to_label"], dropna=False)
        .agg(
            n_transitions=("user_id", "size"),
            n_users=("user_id", "nunique"),
            median_gap_minutes=("gap_minutes", "median"),
            mean_gap_minutes=("gap_minutes", "mean"),
        )
        .reset_index()
    )

    outgoing = edges.groupby("from_label")["n_transitions"].transform("sum")
    edges["conditional_probability"] = edges["n_transitions"] / outgoing

    edges = edges.sort_values(
        ["n_transitions", "n_users"], ascending=False, ignore_index=True
    )
    edges.attrs.update(
        {
            "label_kind": label_kind,
            "window_hours": window_hours,
            "adjacent_only": adjacent_only,
            "exclude_self": exclude_self,
        }
    )

    if suppress:
        edges = privacy.suppress_small_cells(
            edges,
            count_column="n_transitions",
            user_column="n_users",
            min_conversations=privacy.MIN_CONVERSATIONS,
            min_users=min_users,
        )
    return edges


def _empty_transitions() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "from_label",
            "to_label",
            "n_transitions",
            "n_users",
            "median_gap_minutes",
            "mean_gap_minutes",
            "conditional_probability",
        ]
    )


def asymmetry(edges: pd.DataFrame) -> pd.DataFrame:
    """Compare each edge with its reverse.

    Direction is the whole reason to build a directed network. If ``A -> B`` and
    ``B -> A`` occur about equally often, the pair reflects co-occurrence rather
    than sequence, and should not be described as a workflow direction.
    ``asymmetry_ratio`` near 0 means balanced; near 1 means strongly one-way.
    """

    reverse = edges[["from_label", "to_label", "n_transitions"]].rename(
        columns={
            "from_label": "to_label",
            "to_label": "from_label",
            "n_transitions": "n_reverse_transitions",
        }
    )
    out = edges.merge(reverse, on=["from_label", "to_label"], how="left")
    out["n_reverse_transitions"] = out["n_reverse_transitions"].fillna(0)

    total = out["n_transitions"] + out["n_reverse_transitions"]
    out["asymmetry_ratio"] = np.where(
        total > 0,
        (out["n_transitions"] - out["n_reverse_transitions"]).abs() / total,
        np.nan,
    )
    return out.sort_values("n_transitions", ascending=False, ignore_index=True)


def window_sweep(
    conversations: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    windows_hours: tuple[float, ...] = (6.0, 12.0, 24.0, 48.0),
    label_kind: str = "topic_tag",
    **kwargs,
) -> pd.DataFrame:
    """Rebuild the transition table at several window lengths.

    The window is an analyst-chosen parameter, so its influence must be
    reported rather than assumed away. An edge that appears at only one window
    length is an artifact of that choice, not a finding. The returned frame
    carries a ``window_hours`` column so rankings can be compared across
    windows.
    """

    frames = []
    for window in windows_hours:
        edges = build_transitions(
            conversations, labels, label_kind=label_kind, window_hours=window, **kwargs
        )
        edges = edges.assign(window_hours=window)
        frames.append(edges)

    if not frames:
        return _empty_transitions().assign(window_hours=pd.Series(dtype=float))
    return pd.concat(frames, ignore_index=True)
