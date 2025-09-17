## ===========================================================
## CHECK & VISUALIZE RESULTS 
## ===========================================================

out_dir <- "data/out"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages(require(ggplot2))

## -----------------------------------------------------------
## 1) Aggregated results with mean ± SE
## -----------------------------------------------------------
perf <- utils::read.csv(file.path(out_dir, "perf_by_dataset.csv"),
                        stringsAsFactors = FALSE)

## --- R2 stats ---
R2_mean <- aggregate(R2 ~ rho_X + model, data = perf, FUN = mean)
R2_se   <- aggregate(R2 ~ rho_X + model, data = perf, FUN = function(x) sd(x)/sqrt(length(x)))
names(R2_mean)[3] <- "R2_mean"; names(R2_se)[3] <- "R2_se"
R2_stats <- merge(R2_mean, R2_se, by = c("rho_X","model"), sort = TRUE)

## --- RMSE stats ---
RMSE_mean <- aggregate(RMSE ~ rho_X + model, data = perf, FUN = mean)
RMSE_se   <- aggregate(RMSE ~ rho_X + model, data = perf, FUN = function(x) sd(x)/sqrt(length(x)))
names(RMSE_mean)[3] <- "RMSE_mean"; names(RMSE_se)[3] <- "RMSE_se"
RMSE_stats <- merge(RMSE_mean, RMSE_se, by = c("rho_X","model"), sort = TRUE)

## --- Long format for plotting ---
R2_long <- data.frame(
  rho_X  = R2_stats$rho_X,
  model  = R2_stats$model,
  metric = "R2",
  value  = R2_stats$R2_mean,
  se     = R2_stats$R2_se
)
RMSE_long <- data.frame(
  rho_X  = RMSE_stats$rho_X,
  model  = RMSE_stats$model,
  metric = "RMSE",
  value  = RMSE_stats$RMSE_mean,
  se     = RMSE_stats$RMSE_se
)
agg_err <- rbind(R2_long, RMSE_long)

## --- Plot mean ± SE ---
p_err <- ggplot(agg_err, aes(x = rho_X, y = value, color = model, group = model)) +
  geom_point(size = 3) +
  geom_line(linewidth = 1) +
  geom_errorbar(aes(ymin = value - se, ymax = value + se), width = 0.02) +
  scale_x_continuous(breaks = sort(unique(agg_err$rho_X))) +
  facet_wrap(~ metric, scales = "free_y", ncol = 2) +
  labs(
    title = "Predictive performance by reliability (mean ± SE across replicates)",
    x = "Predictor reliability (rho_X)",
    y = NULL,
    color = "Model"
  ) +
  theme_minimal(base_size = 14)

print(p_err)
ggsave(file.path(out_dir, "perf_grid_with_se.png"), p_err,
       width = 10, height = 5.5, dpi = 300)


## -----------------------------------------------------------
## 2) Replicate-level distributions (dots + boxplots)
## -----------------------------------------------------------

## --- R² ---
p_reps_r2_jitter <- ggplot(perf, aes(x = factor(rho_X), y = R2, color = model)) +
  geom_jitter(width = 0.1, alpha = 0.5, size = 2) +
  geom_boxplot(alpha = 0.2, outlier.shape = NA) +
  labs(
    title = "Distribution of predictive R2 across replicates",
    x = "Predictor reliability (rho_X)",
    y = expression(R^2),
    color = "Model"
  ) +
  theme_minimal(base_size = 14)

print(p_reps_r2_jitter)
ggsave(file.path(out_dir, "perf_replicates_R2_jitter.png"),
       p_reps_r2_jitter, width = 9, height = 6, dpi = 300)

## --- RMSE ---
p_reps_rmse_jitter <- ggplot(perf, aes(x = factor(rho_X), y = RMSE, color = model)) +
  geom_jitter(width = 0.1, alpha = 0.5, size = 2) +
  geom_boxplot(alpha = 0.2, outlier.shape = NA) +
  labs(
    title = "Distribution of predictive RMSE across replicates",
    x = "Predictor reliability (rho_X)",
    y = "RMSE",
    color = "Model"
  ) +
  theme_minimal(base_size = 14)

print(p_reps_rmse_jitter)
ggsave(file.path(out_dir, "perf_replicates_RMSE_jitter.png"),
       p_reps_rmse_jitter, width = 9, height = 6, dpi = 300)


cat("\nSaved final plots to:", normalizePath(out_dir), "\n")
