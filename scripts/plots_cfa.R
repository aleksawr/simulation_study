## ===========================================================
## CHECK & VISUALIZE RESULTS — CFA version (categorical x)
## Expects perf_by_dataset.csv produced by 03_validate_cfa.R
## Columns: dataset_id, model, score_type, RMSE, R2, (optional) N, R2_target
## ===========================================================

out_dir <- "data/out_cfa"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages(require(ggplot2))

## -----------------------------------------------------------
## 0) Data
## -----------------------------------------------------------
perf_path <- file.path(out_dir, "perf_by_dataset.csv")
if (!file.exists(perf_path)) {
  stop("Missing file: ", perf_path, "\nRun 03_validate_cfa.R before plotting.")
}
perf <- utils::read.csv(perf_path, stringsAsFactors = FALSE)

## Sanity check required columns
need <- c("model","score_type","R2","RMSE")
if (!all(need %in% names(perf))) {
  stop("perf_by_dataset.csv missing required columns: ", paste(setdiff(need, names(perf)), collapse=", "))
}

## Order score types for nicer plotting
score_levels <- c("items","sum","factor")
perf$score_type <- factor(perf$score_type, levels = intersect(score_levels, unique(perf$score_type)))

## Device: prefer ragg if available (robust UTF-8)
dev_fun <- if (requireNamespace("ragg", quietly = TRUE)) ragg::agg_png else grDevices::png

## -----------------------------------------------------------
## 1) Aggregated results with mean +/- SE (CATEGORICAL AXIS, NO LINES)
## -----------------------------------------------------------
agg_mean <- aggregate(cbind(R2, RMSE) ~ score_type + model, data = perf, FUN = mean)
agg_se   <- aggregate(cbind(R2, RMSE) ~ score_type + model, data = perf,
                      FUN = function(x) sd(x)/sqrt(length(x)))
names(agg_mean)[3:4] <- paste0(names(agg_mean)[3:4], "_mean")
names(agg_se)[3:4]   <- paste0(names(agg_se)[3:4],   "_se")

agg <- merge(agg_mean, agg_se, by = c("score_type","model"), sort = TRUE)

R2_long <- data.frame(
  score_type = agg$score_type,
  model      = agg$model,
  metric     = "R2",
  value      = agg$R2_mean,
  se         = agg$R2_se
)
RMSE_long <- data.frame(
  score_type = agg$score_type,
  model      = agg$model,
  metric     = "RMSE",
  value      = agg$RMSE_mean,
  se         = agg$RMSE_se
)
agg_err <- rbind(R2_long, RMSE_long)

pd <- position_dodge(width = 0.5)

p_err <- ggplot(agg_err, aes(x = score_type, y = value, color = model)) +
  geom_pointrange(
    aes(ymin = value - se, ymax = value + se),
    position = pd, size = 0.4
  ) +
  facet_wrap(~ metric, scales = "free_y", ncol = 2) +
  labs(
    title = "Predictive performance by score type (mean +/- SE across datasets)",
    x = "Score type",
    y = NULL,
    color = "Model"
  ) +
  theme_minimal(base_size = 14) +
  theme(panel.grid.minor = element_blank())

print(p_err)
ggsave(file.path(out_dir, "perf_grid_with_se_cfa.png"), p_err,
       width = 10, height = 5.5, dpi = 300, device = dev_fun)

## -----------------------------------------------------------
## 2) Replicate-level distributions (categorical x)
## -----------------------------------------------------------
# (a) R2
p_reps_r2 <- ggplot(perf, aes(x = score_type, y = R2, color = model)) +
  geom_boxplot(
    aes(group = interaction(model, score_type)),
    position = position_dodge(width = 0.55),
    width = 0.38, alpha = 0.25, outlier.shape = NA
  ) +
  geom_point(
    position = position_jitterdodge(jitter.width = 0.08, dodge.width = 0.55),
    alpha = 0.55, size = 1.6
  ) +
  labs(
    title = "Distribution of predictive R^2 across datasets (categorical score type)",
    x = "Score type", y = expression(R^2), color = "Model"
  ) +
  theme_minimal(base_size = 14) +
  theme(panel.grid.minor = element_blank())

print(p_reps_r2)
ggsave(file.path(out_dir, "perf_replicates_R2_cfa.png"),
       p_reps_r2, width = 9, height = 6, dpi = 300, device = dev_fun)

# (b) RMSE
p_reps_rmse <- ggplot(perf, aes(x = score_type, y = RMSE, color = model)) +
  geom_boxplot(
    aes(group = interaction(model, score_type)),
    position = position_dodge(width = 0.55),
    width = 0.38, alpha = 0.25, outlier.shape = NA
  ) +
  geom_point(
    position = position_jitterdodge(jitter.width = 0.08, dodge.width = 0.55),
    alpha = 0.55, size = 1.6
  ) +
  labs(
    title = "Distribution of predictive RMSE across datasets (categorical score type)",
    x = "Score type", y = "RMSE", color = "Model"
  ) +
  theme_minimal(base_size = 14) +
  theme(panel.grid.minor = element_blank())

print(p_reps_rmse)
ggsave(file.path(out_dir, "perf_replicates_RMSE_cfa.png"),
       p_reps_rmse, width = 9, height = 6, dpi = 300, device = dev_fun)

## -----------------------------------------------------------
## 3) Rainclouds (optional, if ggdist installed)
## -----------------------------------------------------------
if (requireNamespace("ggdist", quietly = TRUE)) {
  library(ggdist)
  # R2 rainclouds
  p_r2_rain <- ggplot(perf, aes(x = score_type, y = R2)) +
    stat_halfeye(adjust = 0.7, width = 0.6, justification = -0.2,
                 .width = 0, point_colour = NA) +
    geom_boxplot(width = 0.15, outlier.shape = NA) +
    geom_point(position = position_jitter(width = 0.06, height = 0),
               alpha = 0.4, size = 1.3) +
    facet_wrap(~ model, ncol = 2) +
    labs(title = "Predictive R^2 (rainclouds; score types)",
         x = "Score type", y = expression(R^2)) +
    theme_minimal(base_size = 14) +
    theme(panel.grid.minor = element_blank())
  ggsave(file.path(out_dir, "perf_replicates_R2_raincloud_cfa.png"), p_r2_rain,
         width = 10, height = 6, dpi = 300, device = dev_fun)
  
  # RMSE rainclouds
  p_rmse_rain <- ggplot(perf, aes(x = score_type, y = RMSE)) +
    stat_halfeye(adjust = 0.7, width = 0.6, justification = -0.2,
                 .width = 0, point_colour = NA) +
    geom_boxplot(width = 0.15, outlier.shape = NA) +
    geom_point(position = position_jitter(width = 0.06, height = 0),
               alpha = 0.4, size = 1.3) +
    facet_wrap(~ model, ncol = 2) +
    labs(title = "Predictive RMSE (rainclouds; score types)",
         x = "Score type", y = "RMSE") +
    theme_minimal(base_size = 14) +
    theme(panel.grid.minor = element_blank())
  ggsave(file.path(out_dir, "perf_replicates_RMSE_raincloud_cfa.png"), p_rmse_rain,
         width = 10, height = 6, dpi = 300, device = dev_fun)
}

cat("\nSaved CFA plots to:", normalizePath(out_dir), "\n")
