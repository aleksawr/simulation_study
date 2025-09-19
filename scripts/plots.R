## ===========================================================
## CHECK & VISUALIZE RESULTS (rewritten: categorical x, no lines)
## ===========================================================

out_dir <- "data/out"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages(require(ggplot2))

## -----------------------------------------------------------
## 0) Data
## -----------------------------------------------------------
perf <- utils::read.csv(file.path(out_dir, "perf_by_dataset.csv"),
                        stringsAsFactors = FALSE)

## -----------------------------------------------------------
## 1) Aggregated results with mean +/- SE (CATEGORICAL AXIS, NO LINES)
## -----------------------------------------------------------
R2_mean <- aggregate(R2 ~ rho_X + model, data = perf, FUN = mean)
R2_se   <- aggregate(R2 ~ rho_X + model, data = perf, FUN = function(x) sd(x)/sqrt(length(x)))
names(R2_mean)[3] <- "R2_mean"; names(R2_se)[3] <- "R2_se"
R2_stats <- merge(R2_mean, R2_se, by = c("rho_X","model"), sort = TRUE)

RMSE_mean <- aggregate(RMSE ~ rho_X + model, data = perf, FUN = mean)
RMSE_se   <- aggregate(RMSE ~ rho_X + model, data = perf, FUN = function(x) sd(x)/sqrt(length(x)))
names(RMSE_mean)[3] <- "RMSE_mean"; names(RMSE_se)[3] <- "RMSE_se"
RMSE_stats <- merge(RMSE_mean, RMSE_se, by = c("rho_X","model"), sort = TRUE)

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

# Behandle rho_X som KATEGORI
rho_levels <- as.character(sort(unique(agg_err$rho_X)))
agg_err$rho_f <- factor(as.character(agg_err$rho_X), levels = rho_levels)

# Punkt + errorbar per kategori og modell (ingen linje mellom kategoriene)
pd <- position_dodge(width = 0.5)

p_err <- ggplot(agg_err, aes(x = rho_f, y = value, color = model)) +
  geom_pointrange(
    aes(ymin = value - se, ymax = value + se),
    position = pd, size = 0.4
  ) +
  facet_wrap(~ metric, scales = "free_y", ncol = 2) +
  labs(
    title = "Predictive performance by reliability (mean +/- SE across replicates)",
    x = "Predictor reliability (rho_X)",
    y = NULL,
    color = "Model"
  ) +
  theme_minimal(base_size = 14) +
  theme(panel.grid.minor = element_blank())

print(p_err)

# Device: bruk ragg hvis tilgjengelig (for robust UTF-8), ellers standard
dev_fun <- if (requireNamespace("ragg", quietly = TRUE)) ragg::agg_png else grDevices::png
ggsave(file.path(out_dir, "perf_grid_with_se.png"), p_err,
       width = 10, height = 5.5, dpi = 300, device = dev_fun)

## -----------------------------------------------------------
## 2) Replicate-level distributions (centered; categorical x)
## -----------------------------------------------------------
# Bruk kategorisk x for å unngå illusjon av linearitet
perf$rho_f <- factor(as.character(perf$rho_X), levels = rho_levels)

# (a) R2
p_reps_r2_centered <- ggplot(
  perf, aes(x = rho_f, y = R2, color = model)
) +
  geom_boxplot(
    aes(group = interaction(model, rho_f)),
    position = position_dodge(width = 0.55),
    width = 0.38,
    alpha = 0.25,
    outlier.shape = NA
  ) +
  geom_point(
    position = position_jitterdodge(jitter.width = 0.08, dodge.width = 0.55),
    alpha = 0.55, size = 1.6
  ) +
  labs(
    title = "Distribution of predictive R^2 across replicates (categorical reliability)",
    x = "Predictor reliability (rho_X)",
    y = expression(R^2),
    color = "Model"
  ) +
  theme_minimal(base_size = 14) +
  theme(panel.grid.minor = element_blank())

print(p_reps_r2_centered)
ggsave(file.path(out_dir, "perf_replicates_R2_centered.png"),
       p_reps_r2_centered, width = 9, height = 6, dpi = 300, device = dev_fun)

# (b) RMSE
p_reps_rmse_centered <- ggplot(
  perf, aes(x = rho_f, y = RMSE, color = model)
) +
  geom_boxplot(
    aes(group = interaction(model, rho_f)),
    position = position_dodge(width = 0.55),
    width = 0.38,
    alpha = 0.25,
    outlier.shape = NA
  ) +
  geom_point(
    position = position_jitterdodge(jitter.width = 0.08, dodge.width = 0.55),
    alpha = 0.55, size = 1.6
  ) +
  labs(
    title = "Distribution of predictive RMSE across replicates (categorical reliability)",
    x = "Predictor reliability (rho_X)",
    y = "RMSE",
    color = "Model"
  ) +
  theme_minimal(base_size = 14) +
  theme(panel.grid.minor = element_blank())

print(p_reps_rmse_centered)
ggsave(file.path(out_dir, "perf_replicates_RMSE_centered.png"),
       p_reps_rmse_centered, width = 9, height = 6, dpi = 300, device = dev_fun)

cat("\nSaved final plots to:", normalizePath(out_dir), "\n")




# install.packages("ggdist") # run once if needed
library(ggdist)

# R2 rainclouds
p_r2_rain <- ggplot(perf, aes(x = rho_f, y = R2)) +
  stat_halfeye(adjust = 0.7, width = 0.6, justification = -0.2,
               .width = 0, point_colour = NA) +
  geom_boxplot(width = 0.15, outlier.shape = NA) +
  geom_point(position = position_jitter(width = 0.06, height = 0), alpha = 0.4, size = 1.3) +
  facet_wrap(~ model, ncol = 2) +
  labs(title = "Predictive R^2 (rainclouds; categorical reliability)",
       x = "Predictor reliability (rho_X)", y = expression(R^2)) +
  theme_minimal(base_size = 14) +
  theme(panel.grid.minor = element_blank())
ggsave(file.path(out_dir, "perf_replicates_R2_raincloud.png"), p_r2_rain,
       width = 10, height = 6, dpi = 300)

# RMSE rainclouds
p_rmse_rain <- ggplot(perf, aes(x = rho_f, y = RMSE)) +
  stat_halfeye(adjust = 0.7, width = 0.6, justification = -0.2,
               .width = 0, point_colour = NA) +
  geom_boxplot(width = 0.15, outlier.shape = NA) +
  geom_point(position = position_jitter(width = 0.06, height = 0), alpha = 0.4, size = 1.3) +
  facet_wrap(~ model, ncol = 2) +
  labs(title = "Predictive RMSE (rainclouds; categorical reliability)",
       x = "Predictor reliability (rho_X)", y = "RMSE") +
  theme_minimal(base_size = 14) +
  theme(panel.grid.minor = element_blank())
ggsave(file.path(out_dir, "perf_replicates_RMSE_raincloud.png"), p_rmse_rain,
       width = 10, height = 6, dpi = 300)

