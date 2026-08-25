# ==========================================================
# 04_resources.R
#
# Resource-intensity figures.
#
# The point of this view is that frequently observed activities
# are not uniformly the most resource-intensive. A volume figure
# and a cost figure answer different prioritization questions,
# and the frequency/intensity plane makes the difference legible.
#
# Cost provenance is plotted first, because a total built from
# 20% observed records supports very different claims from one
# built from 95%.
#
# Inputs:
#   cost_coverage.csv
#   resource_intensity.csv
# ==========================================================

source("R/00_setup.R")

TOP_N <- as.integer(Sys.getenv("AUDRE_TOP_N", "20"))

# ----------------------------------------------------------
# Cost provenance
# ----------------------------------------------------------

cost_coverage <- read_table("cost_coverage.csv")

p_coverage <- cost_coverage %>%
    ggplot(aes(
        x = reorder(cost_source, n_conversations),
        y = n_conversations,
        fill = cost_source
    )) +
    geom_col(colour = "grey20") +
    coord_flip() +
    scale_y_continuous(labels = comma) +
    labs(
        title = "Cost records by provenance",
        subtitle = paste(
            "Observed values come from platform records; estimated values are",
            "derived from token counts and published prices."
        ),
        caption = paste(
            "Estimated values approximate transcript-level inference cost only.",
            "They are not billing records and exclude all other program expense."
        ),
        x = NULL,
        y = "Conversations"
    ) +
    guides(fill = "none")

save_plot(p_coverage, "cost_coverage.png", width = 8, height = 4.5)

# ----------------------------------------------------------
# Frequency against resource intensity
#
# The key figure. Both axes are logarithmic because both
# distributions are strongly right-skewed; bubble area carries
# volume so the reader is not asked to infer it from position.
# ----------------------------------------------------------

resource_intensity <- read_table("resource_intensity.csv")

has_tokens <- "mean_tokens" %in% names(resource_intensity)
has_cost <- "mean_cost_usd" %in% names(resource_intensity)

if (has_tokens && has_cost) {
    p_plane <- resource_intensity %>%
        filter(mean_tokens > 0, mean_cost_usd > 0) %>%
        ggplot(aes(x = mean_tokens, y = mean_cost_usd)) +
        geom_point(aes(size = n_conversations), colour = PRIMARY_FILL, alpha = 0.7) +
        geom_text_repel(aes(label = label_value), size = 3, max.overlaps = 16) +
        scale_x_log10(labels = comma) +
        scale_y_log10(labels = dollar_format(accuracy = 0.0001)) +
        scale_size_continuous(range = c(2, 12), labels = comma) +
        labs(
            title = "Estimated resource intensity by label",
            subtitle = "Both axes on a log scale; bubble area is conversation volume.",
            caption = paste(
                "The most frequently observed activity is not necessarily the",
                "most resource-intensive.", LABEL_SOURCE_NOTE
            ),
            x = "Mean estimated tokens per conversation (log scale)",
            y = "Mean estimated cost per conversation (log scale)",
            size = "Conversations"
        )

    save_plot(p_plane, "resource_intensity_plane.png", width = 10, height = 7.5)
}

# ----------------------------------------------------------
# Highest per-conversation intensity
#
# Ranked by per-conversation cost rather than total, so that a
# high-volume/low-intensity label does not crowd out a genuinely
# expensive one.
# ----------------------------------------------------------

if (has_cost) {
    p_top_cost <- resource_intensity %>%
        slice_max(mean_cost_usd, n = TOP_N) %>%
        ggplot(aes(x = reorder(label_value, mean_cost_usd), y = mean_cost_usd)) +
        geom_col(fill = PRIMARY_FILL) +
        coord_flip() +
        scale_y_continuous(labels = dollar_format(accuracy = 0.0001)) +
        labs(
            title = "Highest mean estimated cost per conversation",
            subtitle = paste0("Top ", TOP_N, " labels by per-conversation cost, not total spend."),
            caption = "Estimated values are not billing records.",
            x = NULL,
            y = "Mean estimated cost per conversation (USD)"
        )

    save_plot(p_top_cost, "top_cost_per_conversation.png", width = 9, height = 7)
}

# ----------------------------------------------------------
# Observed-cost coverage by label
#
# Plotted next to the cost figures because a label whose cost is
# mostly estimated should carry less weight in prioritization.
# ----------------------------------------------------------

if ("observed_cost_coverage" %in% names(resource_intensity)) {
    p_label_coverage <- resource_intensity %>%
        slice_max(n_conversations, n = TOP_N) %>%
        ggplot(aes(
            x = reorder(label_value, observed_cost_coverage),
            y = observed_cost_coverage
        )) +
        geom_col(fill = "grey55") +
        coord_flip() +
        scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1)) +
        labs(
            title = "Share of cost that is observed rather than estimated",
            subtitle = paste0("Top ", TOP_N, " labels by conversation volume."),
            x = NULL,
            y = "Share of conversations with an observed cost"
        )

    save_plot(p_label_coverage, "observed_cost_coverage_by_label.png", width = 9, height = 7)
}

message("04_resources.R complete.")
