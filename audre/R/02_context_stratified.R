# ==========================================================
# 02_context_stratified.R
#
# Context-stratified views: how usage differs across
# organizational units or model families.
#
# Shares are computed within each stratum by the Python layer,
# so a small unit is not visually swamped by a large one. The
# trade-off is that cells are comparable as proportions but not
# as volumes, which is why stratum sizes are plotted alongside.
#
# Inputs:
#   stratum_sizes.csv
#   label_share_by_stratum.csv
#   leading_label_by_unit.csv     (optional)
# ==========================================================

source("R/00_setup.R")

STRATUM_LABEL <- Sys.getenv("AUDRE_STRATUM_LABEL", "Organizational unit")

# ----------------------------------------------------------
# Stratum sizes
#
# Published first, and deliberately. Every stratified figure
# that follows is conditional on these denominators, and strata
# too small to support a stable estimate are excluded upstream.
# ----------------------------------------------------------

stratum_sizes <- read_table("stratum_sizes.csv")

p_sizes <- stratum_sizes %>%
    filter(eligible) %>%
    ggplot(aes(x = reorder(stratum, n_conversations), y = n_conversations)) +
    geom_col(fill = PRIMARY_FILL) +
    coord_flip() +
    scale_y_continuous(labels = comma) +
    labs(
        title = paste0("Conversation volume by ", tolower(STRATUM_LABEL)),
        subtitle = "Only strata large enough to support a stable estimate are shown.",
        x = NULL,
        y = "Conversations"
    )

save_plot(p_sizes, "stratum_sizes.png", width = 9, height = 6)

# ----------------------------------------------------------
# Label share heatmap
# ----------------------------------------------------------

label_share <- read_table("label_share_by_stratum.csv")
stratum_column <- setdiff(names(label_share), c("label_value", "n_conversations", "n_users", "share"))[1]

p_heatmap <- label_share %>%
    ggplot(aes(
        x = .data[[stratum_column]],
        y = reorder(label_value, share),
        fill = share
    )) +
    geom_tile(colour = "white") +
    geom_text(aes(label = percent(share, accuracy = 0.1)), size = 2.8) +
    scale_fill_gradient(
        low = SEQUENTIAL_LOW,
        high = SEQUENTIAL_HIGH,
        labels = percent_format()
    ) +
    labs(
        title = paste0("Label share by ", tolower(STRATUM_LABEL)),
        subtitle = "Share is computed within each stratum, so columns sum to 1 but are not comparable as volumes.",
        caption = LABEL_SOURCE_NOTE,
        x = STRATUM_LABEL,
        y = NULL,
        fill = "Share"
    ) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

save_plot(p_heatmap, "label_share_by_stratum.png", width = 12, height = 8)

# ----------------------------------------------------------
# Leading label per unit (optional)
#
# Meaningful only for single-valued taxonomy kinds, where share
# is a proportion of the unit's mapped conversations.
# ----------------------------------------------------------

leading <- read_table("leading_label_by_unit.csv", required = FALSE)

if (!is.null(leading)) {
    unit_column <- setdiff(names(leading), c("label_value", "n_conversations", "n_users", "share"))[1]

    p_leading <- leading %>%
        ggplot(aes(
            x = reorder(.data[[unit_column]], share),
            y = share,
            fill = label_value
        )) +
        geom_col() +
        coord_flip() +
        scale_y_continuous(labels = percent_format(accuracy = 1)) +
        labs(
            title = paste0("Leading work activity within each ", tolower(STRATUM_LABEL)),
            subtitle = "Bar length is the activity's share of the stratum's mapped conversations.",
            caption = paste(
                LABEL_SOURCE_NOTE,
                "Activities describe conversation content, not users' occupations."
            ),
            x = NULL,
            y = "Share of mapped conversations",
            fill = "Work activity"
        )

    save_plot(p_leading, "leading_label_by_unit.png", width = 11, height = 7)
}

message("02_context_stratified.R complete.")
