"""Graph construction and node metrics for label networks.

Works on either edge table shape:

* directed transitions from :func:`audre.transitions.build_transitions`
  (``from_label``/``to_label``)
* undirected co-occurrence from :func:`audre.cooccurrence.build_cooccurrence`
  (``label_a``/``label_b``)

Layout is intentionally not computed here. Force-directed positions are
stochastic and stratum-specific, so coordinates from two different strata are
not comparable and should never be read as spatial similarity. Figures are
produced by the R layer, which sets its own seed and states this in the caption.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd

DIRECTED_COLUMNS = ("from_label", "to_label")
UNDIRECTED_COLUMNS = ("label_a", "label_b")


def _edge_columns(edges: pd.DataFrame) -> tuple[str, str, bool]:
    if set(DIRECTED_COLUMNS) <= set(edges.columns):
        return (*DIRECTED_COLUMNS, True)
    if set(UNDIRECTED_COLUMNS) <= set(edges.columns):
        return (*UNDIRECTED_COLUMNS, False)
    raise KeyError(
        "edges must contain either ('from_label', 'to_label') or ('label_a', 'label_b')"
    )


def _default_weight_column(edges: pd.DataFrame) -> str:
    for candidate in ("n_transitions", "n_conversations", "weight"):
        if candidate in edges.columns:
            return candidate
    raise KeyError("no weight column found; pass weight_column explicitly")


def build_graph(
    edges: pd.DataFrame,
    *,
    weight_column: str | None = None,
    min_weight: int = 2,
    node_attributes: pd.DataFrame | None = None,
) -> nx.Graph | nx.DiGraph:
    """Build a weighted graph from an edge table.

    ``min_weight`` defaults to 2 so that single-instance edges, which are both
    the least reliable and the most re-identifying, are excluded by default.
    """

    source_column, target_column, directed = _edge_columns(edges)
    weight_column = weight_column or _default_weight_column(edges)

    working = edges.dropna(subset=[source_column, target_column, weight_column])
    working = working[working[source_column] != working[target_column]]
    working = working[working[weight_column] >= min_weight]

    graph = nx.from_pandas_edgelist(
        working,
        source=source_column,
        target=target_column,
        edge_attr=[
            column
            for column in working.columns
            if column not in {source_column, target_column}
        ],
        create_using=nx.DiGraph if directed else nx.Graph,
    )
    nx.set_edge_attributes(
        graph,
        {
            (row[source_column], row[target_column]): float(row[weight_column])
            for _, row in working.iterrows()
        },
        name="weight",
    )

    if node_attributes is not None and not node_attributes.empty:
        key = node_attributes.columns[0]
        for _, row in node_attributes.iterrows():
            node = row[key]
            if node in graph:
                graph.nodes[node].update(row.drop(labels=[key]).to_dict())

    graph.graph["min_weight"] = min_weight
    graph.graph["weight_column"] = weight_column
    return graph


def node_metrics(graph: nx.Graph | nx.DiGraph) -> pd.DataFrame:
    """Compute per-label network metrics.

    Betweenness and closeness use ``1 / weight`` as edge distance, since a
    heavier edge means a *closer* relationship. Using the raw weight as a
    distance would invert the meaning of both metrics.
    """

    if graph.number_of_nodes() == 0:
        return pd.DataFrame(
            columns=["label", "degree", "strength", "betweenness", "closeness"]
        )

    distance = {
        (u, v): 1.0 / weight if weight else float("inf")
        for u, v, weight in graph.edges(data="weight", default=1.0)
    }
    nx.set_edge_attributes(graph, distance, name="distance")

    betweenness = nx.betweenness_centrality(graph, weight="distance")
    closeness = nx.closeness_centrality(graph, distance="distance")

    rows = []
    for node in graph.nodes:
        record: dict[str, object] = {"label": node}
        if graph.is_directed():
            record["in_degree"] = graph.in_degree(node)
            record["out_degree"] = graph.out_degree(node)
            record["in_strength"] = graph.in_degree(node, weight="weight")
            record["out_strength"] = graph.out_degree(node, weight="weight")
            record["degree"] = record["in_degree"] + record["out_degree"]
            record["strength"] = record["in_strength"] + record["out_strength"]
        else:
            record["degree"] = graph.degree(node)
            record["strength"] = graph.degree(node, weight="weight")
        record["betweenness"] = betweenness.get(node)
        record["closeness"] = closeness.get(node)
        record.update(
            {
                key: value
                for key, value in graph.nodes[node].items()
                if key not in record
            }
        )
        rows.append(record)

    return pd.DataFrame(rows).sort_values("strength", ascending=False, ignore_index=True)


def top_n_subgraph(
    edges: pd.DataFrame,
    *,
    top_n: int = 20,
    weight_column: str | None = None,
    min_weight: int = 2,
) -> pd.DataFrame:
    """Restrict an edge table to the ``top_n`` labels by network strength.

    Selecting nodes by strength and then keeping only the edges *between* them
    keeps published figures legible. It also drops peripheral structure, so a
    top-N figure should not be described as the network.
    """

    source_column, target_column, _ = _edge_columns(edges)
    graph = build_graph(edges, weight_column=weight_column, min_weight=min_weight)
    metrics = node_metrics(graph)
    if metrics.empty:
        return edges.iloc[0:0]

    keep = set(metrics.nlargest(top_n, "strength")["label"])
    return edges[
        edges[source_column].isin(keep) & edges[target_column].isin(keep)
    ].reset_index(drop=True)


def to_edge_csv(edges: pd.DataFrame, path: str) -> None:
    """Write an edge table for the R figure layer."""

    edges.to_csv(path, index=False)
