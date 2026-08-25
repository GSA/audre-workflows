# ==========================================================
# 03_networks.R
#
# Label network figures. Handles both edge-table shapes:
#   directed transitions  (from_label, to_label, n_transitions)
#   undirected co-occurrence (label_a, label_b, n_conversations)
#
# Interpretation limits carried in every caption:
#   * Force-directed layouts are stochastic and stratum-specific.
#     Node positions are NOT comparable across panels, and
#     proximity does not mean similarity.
#   * Temporal adjacency is not process dependence. Directed
#     edges are candidate workflows, not confirmed processes.
#
# Inputs:
#   transition_edges.csv          (optional)
#   transition_node_metrics.csv   (optional)
#   cooccurrence_edges.csv        (optional)
#   cooccurrence_node_metrics.csv (optional)
#   resource_intensity.csv        (optional; scales node size)
# ==========================================================

source("R/00_setup.R")

TOP_N_NODES <- as.integer(Sys.getenv("AUDRE_TOP_N_NODES", "20"))
TOP_N_EDGES <- as.integer(Sys.getenv("AUDRE_TOP_N_EDGES", "40"))

# ----------------------------------------------------------
# Shared plotting helper
# ----------------------------------------------------------

plot_network <- function(edges,
                         node_metrics,
                         source_column,
                         target_column,
                         weight_column,
                         directed,
                         title,
                         subtitle,
                         caption,
                         node_size_column = NULL,
                         node_size_label = "Network strength") {

    # Node selection by strength, then keep only edges between the
    # selected nodes. This keeps a published figure legible, but it
    # drops peripheral structure, so the result is a view of the
    # network rather than the network.
    keep_nodes <- node_metrics %>%
        slice_max(strength, n = TOP_N_NODES, with_ties = FALSE) %>%
        pull(label)

    edges_top <- edges %>%
        filter(
            .data[[source_column]] %in% keep_nodes,
            .data[[target_column]] %in% keep_nodes
        ) %>%
        slice_max(.data[[weight_column]], n = TOP_N_EDGES, with_ties = FALSE)

    if (nrow(edges_top) == 0) {
        message("No edges survive the node/edge selection; skipping ", title)
        return(invisible(NULL))
    }

    size_column <- if (!is.null(node_size_column) &&
                       node_size_column %in% names(node_metrics)) {
        node_size_column
    } else {
        "strength"
    }

    vertices <- node_metrics %>%
        filter(label %in% union(edges_top[[source_column]], edges_top[[target_column]])) %>%
        mutate(node_size = .data[[size_column]]) %>%
        relocate(label)

    # igraph requires the first two edge columns to be the endpoints.
    edges_for_graph <- edges_top %>%
        rename(from = !!rlang::sym(source_column), to = !!rlang::sym(target_column)) %>%
        relocate(from, to)

    graph <- graph_from_data_frame(
        d = edges_for_graph,
        vertices = vertices,
        directed = directed
    )

    set.seed(NETWORK_SEED)

    plot <- ggraph(graph, layout = "fr") +
        {
            if (directed) {
                geom_edge_fan(
                    aes(
                        width = .data[[weight_column]],
                        alpha = conditional_probability
                    ),
                    arrow = arrow(length = unit(3, "mm"), type = "closed"),
                    end_cap = circle(4, "mm"),
                    colour = "grey35"
                )
            } else {
                geom_edge_link(
                    aes(width = .data[[weight_column]]),
                    alpha = 0.4,
                    colour = "grey35"
                )
            }
        } +
        geom_node_point(aes(size = node_size), colour = PRIMARY_FILL, alpha = 0.9) +
        geom_node_text(aes(label = name), repel = TRUE, size = 3.4) +
        scale_edge_width(range = c(0.3, 2.6)) +
        labs(
            title = title,
            subtitle = subtitle,
            caption = caption,
            edge_width = weight_column,
            size = node_size_label
        ) +
        theme_void() +
        theme(plot.background = element_rect(fill = "white", colour = NA))

    if (directed) {
        plot <- plot + scale_edge_alpha(range = c(0.25, 0.85), name = "P(to | from)")
    }

    plot
}

# ----------------------------------------------------------
# Optional node sizing by resource intensity
# ----------------------------------------------------------

resource_intensity <- read_table("resource_intensity.csv", required = FALSE)

attach_resources <- function(node_metrics) {
    if (is.null(resource_intensity) ||
        !"mean_cost_usd" %in% names(resource_intensity)) {
        return(node_metrics)
    }
    node_metrics %>%
        left_join(
            resource_intensity %>% select(label = label_value, mean_cost_usd),
            by = "label"
        )
}

# ----------------------------------------------------------
# Directed transition network
# ----------------------------------------------------------

transition_edges <- read_table("transition_edges.csv", required = FALSE)
transition_nodes <- read_table("transition_node_metrics.csv", required = FALSE)

if (!is.null(transition_edges) && !is.null(transition_nodes) && nrow(transition_edges) > 0) {
    transition_nodes <- attach_resources(transition_nodes)

    p_transitions <- plot_network(
        edges = transition_edges,
        node_metrics = transition_nodes,
        source_column = "from_label",
        target_column = "to_label",
        weight_column = "n_transitions",
        directed = TRUE,
        title = "Directed task-transition network",
        subtitle = paste0(
            "Top ", TOP_N_NODES, " labels by directed strength. An edge counts a ",
            "conversation carrying the target label following one carrying the ",
            "source label, for the same user, within the analysis window."
        ),
        caption = paste(
            "Temporal adjacency is not process dependence: edges are candidate",
            "workflows requiring validation, not confirmed business processes.",
            "Layout is stochastic and stratum-specific; positions are not",
            "comparable across panels."
        ),
        node_size_column = "mean_cost_usd",
        node_size_label = if ("mean_cost_usd" %in% names(transition_nodes)) {
            "Mean estimated cost (USD)"
        } else {
            "Network strength"
        }
    )

    if (!is.null(p_transitions)) {
        save_plot(p_transitions, "transition_network.png", width = 11, height = 9)
    }
}

# ----------------------------------------------------------
# Undirected co-occurrence network
# ----------------------------------------------------------

cooccurrence_edges <- read_table("cooccurrence_edges.csv", required = FALSE)
cooccurrence_nodes <- read_table("cooccurrence_node_metrics.csv", required = FALSE)

if (!is.null(cooccurrence_edges) && !is.null(cooccurrence_nodes) && nrow(cooccurrence_edges) > 0) {
    cooccurrence_nodes <- attach_resources(cooccurrence_nodes)

    p_cooccurrence <- plot_network(
        edges = cooccurrence_edges,
        node_metrics = cooccurrence_nodes,
        source_column = "label_a",
        target_column = "label_b",
        weight_column = "n_conversations",
        directed = FALSE,
        title = "Label co-occurrence network",
        subtitle = paste0(
            "Top ", TOP_N_NODES, " labels by weighted degree. An edge counts ",
            "labels applied to the same conversation."
        ),
        caption = paste(
            "Co-occurrence is undirected and cannot distinguish 'these topics",
            "appear together' from 'this topic leads to that one'. Layout is",
            "stochastic; positions are not comparable across panels."
        ),
        node_size_column = "mean_cost_usd",
        node_size_label = if ("mean_cost_usd" %in% names(cooccurrence_nodes)) {
            "Mean estimated cost (USD)"
        } else {
            "Network strength"
        }
    )

    if (!is.null(p_cooccurrence)) {
        save_plot(p_cooccurrence, "cooccurrence_network.png", width = 11, height = 9)
    }
}

message("03_networks.R complete.")
