# ==========================================================
# 01_adoption_engagement.R
#
# Adoption (volume) and engagement (depth) figures.
#
# Volume and depth are plotted as a matched pair on purpose.
# They rank activities differently: volume surfaces broadly
# represented uses, depth surfaces less frequent but more
# interaction-intensive ones. Publishing only one of the two
# systematically hides a class of activity.
#
# Inputs (from the Python pipeline):
#   label_frequency.csv
#   interaction_depth.csv
#   adoption_over_time.csv        (optional)
#   model_share.csv               (optional)
#   multi_model_adoption.csv      (optional)
# ==========================================================

source("R/00_setup.R")

TOP_N <- as.integer(Sys.getenv("AUDRE_TOP_N", "20"))

# ----------------------------------------------------------
# Label frequency
# ----------------------------------------------------------

label_frequency <- read_table("label_frequency.csv")

p_frequency <- label_frequency %>%
    slice_max(n_conversations, n = TOP_N) %>%
    ggplot(aes(x = reorder(label_value, n_conversations), y = n_conversations)) +
    geom_col(fill = PRIMARY_FILL) +
    coord_flip() +
    scale_y_continuous(labels = comma) +
    labs(
        title = "Most frequently assigned conversation labels",
        subtitle = paste0(
            "Top ", TOP_N, " labels. Multi-valued labels are not mutually ",
            "exclusive, so counts sum to more than the number of conversations."
        ),
        caption = LABEL_SOURCE_NOTE,
        x = NULL,
        y = "Conversations"
    )

save_plot(p_frequency, "label_frequency.png", width = 9, height = 7)

# ----------------------------------------------------------
# Interaction depth
#
# Median with interquartile range rather than a bare mean:
# message counts are strongly right-skewed, so a mean alone
# lets one long conversation define a small label.
# ----------------------------------------------------------

interaction_depth <- read_table("interaction_depth.csv")

p_depth <- interaction_depth %>%
    slice_max(mean_depth, n = TOP_N) %>%
    ggplot(aes(x = reorder(label_value, mean_depth), y = median_depth)) +
    geom_linerange(aes(ymin = p25_depth, ymax = p75_depth), colour = "grey65", linewidth = 1) +
    geom_point(aes(size = n_conversations), colour = PRIMARY_FILL) +
    coord_flip() +
    scale_size_continuous(labels = comma) +
    labs(
        title = "Interaction depth by conversation label",
        subtitle = paste0(
            "Point = median messages per conversation; line = interquartile ",
            "range. Depth reflects iteration, not difficulty or success."
        ),
        caption = LABEL_SOURCE_NOTE,
        x = NULL,
        y = "Messages per conversation",
        size = "Conversations"
    )

save_plot(p_depth, "interaction_depth.png", width = 9, height = 7)

# ----------------------------------------------------------
# Volume against depth
#
# The single most useful view of the pair: labels off the
# diagonal are the ones a volume-only report would misrank.
# ----------------------------------------------------------

volume_depth <- label_frequency %>%
    select(label_value, n_conversations) %>%
    inner_join(
        interaction_depth %>% select(label_value, median_depth, mean_depth),
        by = "label_value"
    )

p_volume_depth <- volume_depth %>%
    ggplot(aes(x = n_conversations, y = median_depth)) +
    geom_point(colour = PRIMARY_FILL, alpha = 0.75, size = 2.5) +
    geom_text_repel(aes(label = label_value), size = 3, max.overlaps = 18) +
    scale_x_log10(labels = comma) +
    labs(
        title = "Conversation volume against interaction depth",
        subtitle = "Volume on a log scale. Labels far from the trend rank differently on the two measures.",
        caption = LABEL_SOURCE_NOTE,
        x = "Conversations (log scale)",
        y = "Median messages per conversation"
    )

save_plot(p_volume_depth, "volume_vs_depth.png", width = 10, height = 7)

# ----------------------------------------------------------
# Adoption over time (optional)
#
# Active users are per-period and therefore not additive:
# summing them across periods would double-count returning users.
# ----------------------------------------------------------

adoption_over_time <- read_table("adoption_over_time.csv", required = FALSE)

if (!is.null(adoption_over_time)) {
    stratified <- "stratum" %in% names(adoption_over_time)

    p_time <- adoption_over_time %>%
        ggplot(aes(x = as.Date(period), y = n_conversations)) +
        {
            if (stratified) {
                geom_line(aes(colour = stratum), linewidth = 0.9)
            } else {
                geom_line(colour = PRIMARY_FILL, linewidth = 0.9)
            }
        } +
        scale_y_continuous(labels = comma) +
        labs(
            title = "Conversation volume over time",
            subtitle = "Counts per period. Active-user counts are not additive across periods.",
            x = NULL,
            y = "Conversations",
            colour = "Context"
        )

    save_plot(p_time, "adoption_over_time.png", width = 11, height = 5)
}

# ----------------------------------------------------------
# Model share (optional)
# ----------------------------------------------------------

model_share <- read_table("model_share.csv", required = FALSE)

if (!is.null(model_share)) {
    p_model <- model_share %>%
        ggplot(aes(x = reorder(model_family, share), y = share)) +
        geom_col(fill = PRIMARY_FILL) +
        coord_flip() +
        scale_y_continuous(labels = percent_format(accuracy = 0.1)) +
        labs(
            title = "Share of conversations by model family",
            x = NULL,
            y = "Share of conversations"
        )

    save_plot(p_model, "model_share.png", width = 8, height = 5)
}

# ----------------------------------------------------------
# Multi-model adoption (optional)
#
# Distinct from model share: whether users concentrate on one
# model or spread across several speaks to whether model choice
# is deliberate.
# ----------------------------------------------------------

multi_model <- read_table("multi_model_adoption.csv", required = FALSE)

if (!is.null(multi_model)) {
    p_multi <- multi_model %>%
        ggplot(aes(x = factor(n_models), y = share_of_users)) +
        geom_col(fill = PRIMARY_FILL) +
        scale_y_continuous(labels = percent_format(accuracy = 1)) +
        labs(
            title = "Number of distinct model families used per user",
            x = "Distinct model families",
            y = "Share of users"
        )

    save_plot(p_multi, "multi_model_adoption.png", width = 8, height = 5)
}

message("01_adoption_engagement.R complete.")
