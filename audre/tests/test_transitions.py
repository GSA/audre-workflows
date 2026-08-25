from __future__ import annotations

import pandas as pd
import pytest

from audre import transitions


def _make(rows: list[tuple[str, str, str, list[str]]]):
    """Build (conversations, labels) from (conversation_id, user, iso_time, tags)."""

    conversations = pd.DataFrame(
        [
            {"conversation_id": cid, "user_id": user, "created_at": pd.Timestamp(ts, tz="UTC")}
            for cid, user, ts, _ in rows
        ]
    )
    labels = pd.DataFrame(
        [
            {"conversation_id": cid, "label_kind": "topic_tag", "label_value": tag}
            for cid, _, _, tags in rows
            for tag in tags
        ]
    )
    return conversations, labels


def test_direction_is_respected():
    conversations, labels = _make(
        [
            ("c1", "u1", "2026-01-01 09:00", ["a"]),
            ("c2", "u1", "2026-01-01 10:00", ["b"]),
        ]
    )
    edges = transitions.build_transitions(conversations, labels, suppress=False)

    assert edges[["from_label", "to_label"]].values.tolist() == [["a", "b"]]


def test_transitions_never_cross_users():
    conversations, labels = _make(
        [
            ("c1", "u1", "2026-01-01 09:00", ["a"]),
            ("c2", "u2", "2026-01-01 10:00", ["b"]),
        ]
    )
    edges = transitions.build_transitions(conversations, labels, suppress=False)

    assert edges.empty, "a transition requires the same user"


def test_window_boundary_excludes_later_conversations():
    conversations, labels = _make(
        [
            ("c1", "u1", "2026-01-01 09:00", ["a"]),
            ("c2", "u1", "2026-01-03 09:00", ["b"]),
        ]
    )
    inside = transitions.build_transitions(
        conversations, labels, window_hours=72, suppress=False
    )
    outside = transitions.build_transitions(
        conversations, labels, window_hours=24, suppress=False
    )

    assert len(inside) == 1
    assert outside.empty


def test_adjacent_only_limits_pairing_to_the_next_conversation():
    conversations, labels = _make(
        [
            ("c1", "u1", "2026-01-01 09:00", ["a"]),
            ("c2", "u1", "2026-01-01 10:00", ["b"]),
            ("c3", "u1", "2026-01-01 11:00", ["c"]),
        ]
    )
    adjacent = transitions.build_transitions(conversations, labels, suppress=False)
    all_pairs = transitions.build_transitions(
        conversations, labels, adjacent_only=False, suppress=False
    )

    assert set(map(tuple, adjacent[["from_label", "to_label"]].values)) == {
        ("a", "b"),
        ("b", "c"),
    }
    # a->c only appears once non-adjacent pairing is enabled
    assert ("a", "c") in set(map(tuple, all_pairs[["from_label", "to_label"]].values))


def test_multi_label_conversations_produce_every_eligible_pair():
    conversations, labels = _make(
        [
            ("c1", "u1", "2026-01-01 09:00", ["a", "b"]),
            ("c2", "u1", "2026-01-01 10:00", ["c"]),
        ]
    )
    edges = transitions.build_transitions(conversations, labels, suppress=False)

    assert set(map(tuple, edges[["from_label", "to_label"]].values)) == {
        ("a", "c"),
        ("b", "c"),
    }


def test_self_transitions_are_excluded_by_default():
    conversations, labels = _make(
        [
            ("c1", "u1", "2026-01-01 09:00", ["a"]),
            ("c2", "u1", "2026-01-01 10:00", ["a"]),
        ]
    )
    excluded = transitions.build_transitions(conversations, labels, suppress=False)
    included = transitions.build_transitions(
        conversations, labels, exclude_self=False, suppress=False
    )

    assert excluded.empty
    assert included[["from_label", "to_label"]].values.tolist() == [["a", "a"]]


def test_conditional_probabilities_sum_to_one_per_source(conversations, labels):
    edges = transitions.build_transitions(conversations, labels, suppress=False)
    totals = edges.groupby("from_label")["conditional_probability"].sum()

    assert ((totals - 1.0).abs() < 1e-9).all()


def test_suppression_removes_low_support_edges(conversations, labels):
    unsuppressed = transitions.build_transitions(conversations, labels, suppress=False)
    suppressed = transitions.build_transitions(conversations, labels, suppress=True)

    assert len(suppressed) <= len(unsuppressed)
    assert (suppressed["n_users"] >= 3).all()


def test_asymmetry_is_one_for_a_purely_one_way_edge():
    conversations, labels = _make(
        [
            ("c1", "u1", "2026-01-01 09:00", ["a"]),
            ("c2", "u1", "2026-01-01 10:00", ["b"]),
        ]
    )
    edges = transitions.build_transitions(conversations, labels, suppress=False)
    out = transitions.asymmetry(edges)

    assert out.loc[0, "asymmetry_ratio"] == pytest.approx(1.0)


def test_asymmetry_is_zero_for_a_balanced_pair():
    conversations, labels = _make(
        [
            ("c1", "u1", "2026-01-01 09:00", ["a"]),
            ("c2", "u1", "2026-01-01 10:00", ["b"]),
            ("c3", "u2", "2026-01-01 09:00", ["b"]),
            ("c4", "u2", "2026-01-01 10:00", ["a"]),
        ]
    )
    edges = transitions.build_transitions(conversations, labels, suppress=False)
    out = transitions.asymmetry(edges)

    assert out["asymmetry_ratio"].tolist() == pytest.approx([0.0, 0.0])


def test_window_sweep_labels_each_window(conversations, labels):
    swept = transitions.window_sweep(
        conversations, labels, windows_hours=(6.0, 24.0), suppress=False
    )
    assert set(swept["window_hours"].unique()) == {6.0, 24.0}


def test_zero_window_is_rejected(conversations):
    with pytest.raises(ValueError, match="positive"):
        transitions.conversation_pairs(conversations, window_hours=0)
