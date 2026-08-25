# ==========================================================
# 00_setup.R
# Shared setup for the AUDRE figure layer.
#
# The R scripts consume the tidy CSVs written by the Python
# pipeline. They perform no filtering, joining, or aggregation
# of their own: every number in a figure is one that already
# passed the privacy layer's minimum-cell rules. Keeping
# aggregation out of the figure layer is what makes the
# figures auditable against the tables.
# ==========================================================

suppressPackageStartupMessages({
    library(tidyverse)
    library(rlang)
    library(scales)
    library(igraph)
    library(ggraph)
    library(ggrepel)
})

# All paths are configurable so the scripts can run against
# any output directory. Nothing is hardcoded to one dataset.
TABLE_DIR <- Sys.getenv("AUDRE_TABLE_DIR", "output/tables")
PLOT_DIR <- Sys.getenv("AUDRE_PLOT_DIR", "output/plots")

# Shown in figure captions so a reader can tell which labeling
# system produced the categories being plotted.
LABEL_SOURCE_NOTE <- Sys.getenv(
    "AUDRE_LABEL_SOURCE_NOTE",
    "Labels are analytically assigned, not recorded."
)

dir.create(PLOT_DIR, recursive = TRUE, showWarnings = FALSE)

theme_set(theme_minimal(base_size = 13))

PRIMARY_FILL <- "#3b5fa8"
SEQUENTIAL_LOW <- "white"
SEQUENTIAL_HIGH <- "#3b5fa8"

# Layouts are stochastic, so a seed is fixed for every network
# figure. Positions remain stratum-specific and must not be
# compared across panels.
NETWORK_SEED <- 123

read_table <- function(filename, required = TRUE) {
    path <- file.path(TABLE_DIR, filename)

    if (!file.exists(path)) {
        if (required) {
            stop(
                "Missing input table: ", path, "\n",
                "Run the Python pipeline first (see audre/examples/run_pipeline.py).",
                call. = FALSE
            )
        }
        message("Skipping optional table: ", path)
        return(NULL)
    }

    readr::read_csv(path, show_col_types = FALSE)
}

save_plot <- function(plot, filename, width = 10, height = 6) {
    ggsave(
        filename = file.path(PLOT_DIR, filename),
        plot = plot,
        width = width,
        height = height,
        dpi = 300,
        bg = "white"
    )
    message("Wrote ", file.path(PLOT_DIR, filename))
}

# Fails loudly rather than silently plotting a suppressed row as
# zero, which would misrepresent a hidden cell as an empty one.
assert_no_suppressed <- function(data, columns) {
    for (column in intersect(columns, names(data))) {
        if (any(is.na(data[[column]]))) {
            stop(
                "Column '", column, "' contains NA, which indicates masked ",
                "small cells. Filter or re-export before plotting.",
                call. = FALSE
            )
        }
    }
    invisible(data)
}

message("AUDRE figure layer ready. Tables: ", TABLE_DIR, " | Plots: ", PLOT_DIR)
