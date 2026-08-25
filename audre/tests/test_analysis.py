from __future__ import annotations

import pandas as pd
import pytest

from audre import adoption, cooccurrence, networks, resources, stratify


# -- adoption / engagement -------------------------------------------------


def test_label_frequency_counts_conversations_and_users(conversations, labels):
    out = adoption.label_frequency(conversations, labels, suppress=False)

    assert {"label_value", "n_conversations", "n_users", "share"} <= set(out.columns)
    assert (out["n_users"] <= out["n_conversations"]).all()
    assert out["share"].sum() == pytest.approx(1.0)


def test_label_frequency_shares_sum_to_one_within_each_stratum(conversations, labels):
    out = adoption.label_frequency(conversations, labels, by="unit", suppress=False)
    totals = out.groupby("unit")["share"].sum()

    assert ((totals - 1.0).abs() < 1e-9).all()


def test_volume_and_depth_rank_labels_differently(conversations, labels):
    """The central claim of the adoption/engagement pair.

    The fixture deliberately makes some labels frequent-and-shallow and others
    rare-and-deep. If these two rankings ever coincide, one of the two views is
    redundant and the pairing is not worth reporting.
    """

    frequency = adoption.label_frequency(conversations, labels, suppress=False)
    depth = adoption.interaction_depth(conversations, labels, suppress=False)

    assert frequency["label_value"].tolist() != depth["label_value"].tolist()


def test_interaction_depth_reports_a_spread_not_just_a_mean(conversations, labels):
    out = adoption.interaction_depth(conversations, labels, suppress=False)

    assert {"median_depth", "p25_depth", "p75_depth"} <= set(out.columns)
    assert (out["p25_depth"] <= out["median_depth"]).all()
    assert (out["median_depth"] <= out["p75_depth"]).all()


def test_missing_depth_column_raises(conversations, labels):
    with pytest.raises(KeyError):
        adoption.interaction_depth(
            conversations.drop(columns=["n_messages"]), labels, suppress=False
        )


def test_unknown_label_kind_lists_available_kinds(conversations, labels):
    with pytest.raises(ValueError, match="topic_tag"):
        adoption.label_frequency(conversations, labels, label_kind="nope")


def test_adoption_over_time_is_ordered(conversations):
    out = adoption.adoption_over_time(conversations, suppress=False)
    assert out["period"].is_monotonic_increasing


# -- co-occurrence ---------------------------------------------------------


def test_cooccurrence_pairs_are_canonically_ordered(conversations, labels):
    out = cooccurrence.build_cooccurrence(conversations, labels, suppress=False)
    assert (out["label_a"] < out["label_b"]).all(), "a<b prevents duplicate edges"


def test_cooccurrence_jaccard_is_a_proportion(conversations, labels):
    out = cooccurrence.build_cooccurrence(conversations, labels, suppress=False)
    assert out["jaccard"].between(0, 1).all()


def test_single_label_conversations_produce_no_cooccurrence_edges():
    conversations = pd.DataFrame(
        {
            "conversation_id": ["c1", "c2"],
            "user_id": ["u1", "u2"],
            "created_at": pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True),
        }
    )
    labels = pd.DataFrame(
        {
            "conversation_id": ["c1", "c2"],
            "label_kind": ["topic_tag"] * 2,
            "label_value": ["a", "b"],
        }
    )
    out = cooccurrence.build_cooccurrence(conversations, labels, suppress=False)
    assert out.empty


# -- networks --------------------------------------------------------------


def test_graph_respects_min_weight(conversations, labels):
    edges = cooccurrence.build_cooccurrence(conversations, labels, suppress=False)
    permissive = networks.build_graph(edges, min_weight=1)
    strict = networks.build_graph(edges, min_weight=edges["n_conversations"].max())

    assert strict.number_of_edges() <= permissive.number_of_edges()


def test_directed_edges_yield_a_directed_graph_with_in_out_metrics(
    conversations, labels
):
    from audre import transitions

    edges = transitions.build_transitions(conversations, labels, suppress=False)
    graph = networks.build_graph(edges, min_weight=1)
    metrics = networks.node_metrics(graph)

    assert graph.is_directed()
    assert {"in_degree", "out_degree", "in_strength", "out_strength"} <= set(
        metrics.columns
    )


def test_node_metrics_are_sorted_by_strength(conversations, labels):
    edges = cooccurrence.build_cooccurrence(conversations, labels, suppress=False)
    metrics = networks.node_metrics(networks.build_graph(edges, min_weight=1))

    assert metrics["strength"].is_monotonic_decreasing


def test_top_n_subgraph_keeps_only_internal_edges(conversations, labels):
    edges = cooccurrence.build_cooccurrence(conversations, labels, suppress=False)
    reduced = networks.top_n_subgraph(edges, top_n=3, min_weight=1)

    nodes = set(reduced["label_a"]) | set(reduced["label_b"])
    assert len(nodes) <= 3


def test_unrecognized_edge_columns_raise():
    with pytest.raises(KeyError, match="from_label"):
        networks.build_graph(pd.DataFrame({"x": ["a"], "y": ["b"], "weight": [1]}))


# -- resources -------------------------------------------------------------


def test_resolve_cost_marks_provenance(resource_records):
    resolved = resources.resolve_cost(resource_records)

    assert set(resolved["cost_source"].dropna()) <= {
        resources.OBSERVED,
        resources.ESTIMATED,
        resources.IMPUTED,
    }
    assert resolved.loc[resolved["has_observed_cost"], "cost_source"].eq(
        resources.OBSERVED
    ).all()


def test_median_imputation_is_off_by_default(resource_records):
    default = resources.resolve_cost(resource_records)
    opted_in = resources.resolve_cost(resource_records, impute_median=True)

    assert default["analysis_cost_usd"].isna().any()
    assert not opted_in["analysis_cost_usd"].isna().any()
    assert (opted_in["cost_source"] == resources.IMPUTED).any()


def test_cost_coverage_shares_sum_to_one(resource_records):
    coverage = resources.cost_coverage(
        resources.resolve_cost(resource_records, impute_median=True)
    )
    assert coverage["share_of_conversations"].sum() == pytest.approx(1.0)


def test_cost_coverage_requires_resolve_first(resource_records):
    with pytest.raises(KeyError, match="resolve_cost"):
        resources.cost_coverage(resource_records)


def test_estimate_cost_leaves_unpriced_models_null(conversations, resource_records):
    prices = resources.price_table({"model-alpha-1": (0.5, 2.0)})
    out = resources.estimate_cost(resource_records, conversations, prices)

    unpriced = out[~out["priced_model"]]
    assert unpriced["estimated_cost_usd"].isna().all(), "unpriced must be null, not zero"


def test_estimate_tokens_rejects_nonpositive_divisor():
    with pytest.raises(ValueError, match="positive"):
        resources.estimate_tokens(pd.Series([100]), chars_per_token=0)


def test_resource_intensity_by_label_reports_coverage(
    conversations, labels, resource_records
):
    resolved = resources.resolve_cost(resource_records)
    out = resources.resource_intensity_by_label(
        resolved, labels, conversations=conversations, label_kind="topic_tag", suppress=False
    )

    assert "observed_cost_coverage" in out.columns
    assert out["observed_cost_coverage"].between(0, 1).all()


# -- stratification --------------------------------------------------------


def test_small_strata_are_excluded_from_analysis(conversations, labels):
    from audre import transitions

    tiny = conversations.copy()
    tiny.loc[tiny.index[:3], "unit"] = "Unit Z"

    out = stratify.by_stratum(
        transitions.build_transitions,
        tiny,
        labels,
        by="unit",
        suppress=False,
    )

    assert "Unit Z" not in set(out.get("stratum", []))
    assert "Unit Z" in out.attrs["strata_skipped_too_small"]


def test_by_stratum_tags_each_result_with_its_stratum(conversations, labels):
    out = stratify.by_stratum(
        adoption.label_frequency, conversations, labels, by="unit", suppress=False
    )

    assert set(out["stratum"]) == set(out.attrs["strata_analyzed"])
    assert (out["stratum_variable"] == "unit").all()


def test_stratum_sizes_flags_eligibility(conversations):
    sizes = stratify.stratum_sizes(conversations, "unit")
    assert "eligible" in sizes.columns
    assert sizes["n_conversations"].is_monotonic_decreasing


def test_units_differ_in_their_label_mix(conversations, labels):
    """Context stratification must actually separate units.

    The fixture gives each unit a distinct topic profile. If the leading label
    were identical everywhere, the stratified view would add nothing.
    """

    leading = adoption.leading_label_by_unit(
        conversations, labels, label_kind="topic_tag", suppress=False
    )
    assert leading["label_value"].nunique() > 1
