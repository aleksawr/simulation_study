# ===== 03_validate.R =====
rm(list = ls(all.names = TRUE))                                  ## clear workspace

pred_dir  <- "data/pred"                                         ## directory where prediction files were saved
index_fn  <- file.path(pred_dir, "pred_index.csv")               ## index listing per-dataset prediction file paths
manifest  <- utils::read.csv("data/sim/manifest.csv", stringsAsFactors = FALSE)  ## simulation manifest (metadata)
pred_idx  <- utils::read.csv(index_fn, stringsAsFactors = FALSE) ## prediction index created by 02_train.R

## Compute evaluation metrics from true vs predicted
metricify <- function(y, yhat) {
  mse <- mean((y - yhat)^2)                                      ## mean squared error
  r2  <- 1 - mse / var(y)                                        ## out-of-sample R^2 (1 - MSE/Var(Y_test))
  c(RMSE = sqrt(mse), R2 = r2)                                   ## return RMSE and R^2
}

rows <- list(); k <- 1L                                          ## container for per-dataset performance rows
for (i in seq_len(nrow(pred_idx))) {                             ## iterate over all prediction entries
  row <- pred_idx[i, ]                                           ## one row from prediction index
  man <- manifest[manifest$dataset_id == row$dataset_id, , drop = FALSE]  ## match to manifest (for R2_true, etc.)
  # OLS
  plm <- utils::read.csv(row$pred_lm, stringsAsFactors = FALSE)  ## read OLS predictions (y, yhat) for this dataset
  m_lm <- metricify(plm$y, plm$yhat)                             ## compute RMSE and R^2 for OLS
  rows[[k]] <- data.frame(                                       ## store metrics for OLS
    model      = "lm",                                           ## model identifier
    dataset_id = row$dataset_id,                                 ## dataset ID
    rho_X      = row$rho_X,                                      ## reliability used in this dataset
    replicate  = row$replicate,                                  ## replicate number
    RMSE       = m_lm["RMSE"],                                   ## RMSE (test)
    R2         = m_lm["R2"],                                     ## R^2 (test)
    R2_true    = man$R2_target,                                  ## latent target R^2 (from simulation settings)
    stringsAsFactors = FALSE
  ); k <- k + 1L
  
  # XGB (optional)
  if (!is.null(row$pred_xgb) &&                                  ## only if xgboost predictions exist
      !is.na(row$pred_xgb) &&
      nzchar(row$pred_xgb) &&
      file.exists(row$pred_xgb)) {
    
    pxg <- utils::read.csv(row$pred_xgb, stringsAsFactors = FALSE) ## read xgboost predictions (y, yhat)
    m_xg <- metricify(pxg$y, pxg$yhat)                              ## compute RMSE and R^2 for xgboost
    rows[[k]] <- data.frame(                                        ## store metrics for xgboost
      model      = "xgboost",
      dataset_id = row$dataset_id,
      rho_X      = row$rho_X,
      replicate  = row$replicate,
      RMSE       = m_xg["RMSE"],
      R2         = m_xg["R2"],
      R2_true    = man$R2_target,
      stringsAsFactors = FALSE
    ); k <- k + 1L
  }
}

perf <- do.call(rbind, rows)                                       ## bind all per-dataset metric rows
out_dir <- "data/out"                                              ## output directory for validation results
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)        ## ensure output directory exists
utils::write.csv(perf, file.path(out_dir, "perf_by_dataset.csv"),  ## save detailed per-dataset metrics
                 row.names = FALSE, fileEncoding = "UTF-8")

# Aggregate by rho_X & model
agg <- aggregate(perf[, c("RMSE","R2")],                           ## average metrics within (rho_X, model)
                 by = list(rho_X = perf$rho_X, model = perf$model),
                 FUN = mean, na.rm = TRUE)
utils::write.csv(agg, file.path(out_dir, "perf_agg.csv"),          ## save aggregated metrics
                 row.names = FALSE, fileEncoding = "UTF-8")

# Quick base-R plot (OLS vs reliability)
png(file.path(out_dir, "R2_vs_rhoX.png"), width = 900, height = 600)  ## open PNG device for plot
op <- par(no.readonly = TRUE); on.exit(par(op), add = TRUE)            ## save par settings and restore on exit
with(subset(agg, model == "lm"), {                                     ## use aggregated OLS rows only
  plot(rho_X, R2, type = "b",                                          ## line+points: R^2 vs reliability
       xlab = "Predictor reliability (rho_X)",
       ylab = "Out-of-sample R-squared",
       main = "Effect of predictor reliability on test R^2 (OLS)")
})
dev.off()                                                              ## write PNG and close device

cat("Wrote detailed and aggregated metrics to", out_dir, "\n")         ## final message
