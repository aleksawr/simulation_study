# ===== 03_validate_cfa.R (robust) =====
rm(list = ls(all.names = TRUE))

pred_dir <- "data/pred_cfa"
index_fn <- file.path(pred_dir, "pred_index.csv")
manifest <- utils::read.csv("data/sim_cfa/manifest.csv", stringsAsFactors = FALSE)
if (!file.exists(index_fn)) stop("Missing prediction index: ", index_fn)
pred_idx <- utils::read.csv(index_fn, stringsAsFactors = FALSE)

metricify <- function(y, yhat) {
  mse <- mean((y - yhat)^2)
  r2  <- 1 - mse / var(y)
  c(RMSE = sqrt(mse), R2 = r2)
}
read_pred_if_exists <- function(path) {
  if (is.null(path) || is.na(path) || !nzchar(path) || !file.exists(path)) return(NULL)
  utils::read.csv(path, stringsAsFactors = FALSE)
}

rows <- list(); k <- 1L
for (i in seq_len(nrow(pred_idx))) {
  row <- pred_idx[i, ]
  p <- read_pred_if_exists(row$pred_file); if (is.null(p)) next
  man <- manifest[manifest$dataset_id == row$dataset_id, , drop = FALSE]
  
  m <- metricify(p$y, p$yhat)
  rows[[k]] <- data.frame(
    dataset_id = row$dataset_id,
    model      = row$model,
    score_type = row$score_type,
    N          = if ("N" %in% names(row)) row$N else if ("N" %in% names(man)) man$N else NA,
    n_train    = if ("n_train" %in% names(row)) row$n_train else NA,
    n_test     = if ("n_test" %in% names(row)) row$n_test else NA,
    seed_split = if ("seed_split" %in% names(row)) row$seed_split else NA,
    R2_target  = if ("R2_target_Y" %in% names(man)) man$R2_target_Y else NA,
    RMSE       = unname(m["RMSE"]),
    R2         = unname(m["R2"]),
    stringsAsFactors = FALSE
  ); k <- k + 1L
}

perf <- if (length(rows)) do.call(rbind, rows) else
  data.frame(dataset_id=character(), model=character(), score_type=character(),
             N=integer(), n_train=integer(), n_test=integer(),
             seed_split=integer(), R2_target=numeric(), RMSE=numeric(), R2=numeric(),
             stringsAsFactors = FALSE)

out_dir <- "data/out_cfa"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

utils::write.csv(perf, file.path(out_dir, "perf_by_dataset.csv"),
                 row.names = FALSE, fileEncoding = "UTF-8")

agg_model <- aggregate(perf[, c("RMSE","R2")],
                       by = list(model = perf$model),
                       FUN = mean, na.rm = TRUE)
utils::write.csv(agg_model, file.path(out_dir, "perf_agg_by_model.csv"),
                 row.names = FALSE, fileEncoding = "UTF-8")

if ("score_type" %in% names(perf)) {
  agg_model_score <- aggregate(perf[, c("RMSE","R2")],
                               by = list(model = perf$model,
                                         score_type = perf$score_type),
                               FUN = mean, na.rm = TRUE)
  utils::write.csv(agg_model_score, file.path(out_dir, "perf_agg_by_model_score.csv"),
                   row.names = FALSE, fileEncoding = "UTF-8")
}

cat("Wrote detailed and aggregated metrics to", out_dir, "\n")








# ===== 02_train_cfa.R =====
# Train OLS and optionally XGBoost on:
#   (a) raw item scores, (b) sum scores, (c) CFA factor scores.
# Uses the CFA-style manifest (file_items, file_latent) from data/sim_cfa/.
# Produces predictions + an index with score_type = "items" | "sum" | "factor".

rm(list = ls(all.names = TRUE))
source("scripts/utils_pilot.R")   ## needs: train_test_idx()

## ------------------ config ------------------
manifest <- utils::read.csv("data/sim_cfa/manifest.csv", stringsAsFactors = FALSE)
pred_dir <- "data/pred_cfa"
dir.create(pred_dir, recursive = TRUE, showWarnings = FALSE)

use_xgboost      <- TRUE    ## toggle ML
use_sum_scores   <- TRUE    ## toggle sum/mean scores
use_factor_scores<- TRUE    ## toggle CFA factor scores (requires lavaan)
test_ratio       <- 0.30

## ------------------ helpers ------------------
## OLS
fit_lm_predict <- function(Xtr, ytr, Xte, yte) {
  # Build a data.frame for lm
  tr <- data.frame(Y = ytr, Xtr); te <- data.frame(Y = yte, Xte)
  fml <- as.formula(paste("Y ~", paste(names(Xtr), collapse = " + ")))
  fit <- lm(fml, data = tr)
  yhat <- predict(fit, newdata = te)
  list(y = yte, yhat = yhat)
}

## XGB
fit_xgb_predict <- function(Xtr, ytr, Xte, yte) {
  if (!requireNamespace("xgboost", quietly = TRUE))
    stop("Install 'xgboost' or set use_xgboost <- FALSE")
  dtr <- xgboost::xgb.DMatrix(as.matrix(Xtr), label = ytr)
  dte <- xgboost::xgb.DMatrix(as.matrix(Xte), label = yte)
  bst <- xgboost::xgb.train(
    data = dtr, nrounds = 300,
    params = list(booster="gbtree", objective="reg:squarederror",
                  eta=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8),
    verbose = 0
  )
  yhat <- predict(bst, dte)
  list(y = yte, yhat = yhat)
}

## Sum/mean scores for a 2x3 simple-structure design:
##  - F1_sum = mean(X1:X3); F2_sum = mean(X4:X6)
## If you want sums, replace rowMeans with rowSums and rename.
make_sum_scores_2x3 <- function(df_items) {
  stopifnot(all(paste0("X", 1:6) %in% names(df_items)))
  data.frame(
    F1_sum = rowMeans(df_items[, c("X1","X2","X3")]),
    F2_sum = rowMeans(df_items[, c("X4","X5","X6")])
  )
}

## Factor scores via lavaan (fit on TRAIN ONLY; score train + test)
## Model fixed to known 2x3 simple structure:
lavaan_factor_scores_2x3 <- function(tr_items, te_items) {
  if (!requireNamespace("lavaan", quietly = TRUE))
    stop("Install 'lavaan' or set use_factor_scores <- FALSE")
  model_txt <- '
    F1 =~ X1 + X2 + X3
    F2 =~ X4 + X5 + X6
  '
  fit <- lavaan::cfa(model_txt, data = tr_items, std.lv = TRUE)
  # Scores for train and test under the same fitted model (no leakage from test)
  F_tr <- as.data.frame(lavaan::lavPredict(fit, newdata = tr_items))
  F_te <- as.data.frame(lavaan::lavPredict(fit, newdata = te_items))
  # Ensure consistent names
  names(F_tr) <- sub("\\.fs", "", names(F_tr)); names(F_te) <- names(F_tr)
  list(Ftr = F_tr, Fte = F_te)
}

## Extract item matrix and Y from a df with Y + X1:Xp
get_XY_from_items <- function(df) {
  Xcols <- grep("^X\\d+$", names(df), value = TRUE)
  list(X = df[, Xcols, drop = FALSE], y = df$Y)
}

## Write predictions to CSV and return file path
write_pred <- function(y, yhat, out_path) {
  utils::write.csv(data.frame(y = y, yhat = yhat),
                   out_path, row.names = FALSE, fileEncoding = "UTF-8")
  out_path
}

## Append one row to index (data.frame)
mk_index_row <- function(dataset_id, model, score_type, file_path, extra = list()) {
  as.data.frame(c(list(
    dataset_id = dataset_id,
    model      = model,
    score_type = score_type,
    pred_file  = file_path
  ), extra), stringsAsFactors = FALSE)
}

## ------------------ main loop ------------------
rows <- list(); k <- 1L

for (i in seq_len(nrow(manifest))) {
  ds <- manifest[i, ]
  df_items <- utils::read.csv(ds$file_items, stringsAsFactors = FALSE)
  
  ## Split once per dataset (same split for all branches)
  idx <- train_test_idx(nrow(df_items), test_ratio, seed = ds$seed + 100)
  tr_df <- df_items[idx$train, , drop = FALSE]
  te_df <- df_items[idx$test,  , drop = FALSE]
  
  ## ========== (A) ITEMS BRANCH ==========
  xy_tr <- get_XY_from_items(tr_df)
  xy_te <- get_XY_from_items(te_df)
  
  # OLS (items)
  lm_items <- fit_lm_predict(xy_tr$X, xy_tr$y, xy_te$X, xy_te$y)
  fn_lm_items <- file.path(pred_dir, sprintf("%s_items_lm.csv", ds$dataset_id))
  write_pred(lm_items$y, lm_items$yhat, fn_lm_items)
  rows[[k]] <- mk_index_row(ds$dataset_id, "lm", "items", fn_lm_items,
                            list(N = ds$N)); k <- k + 1L
  
  # XGB (items)
  if (use_xgboost) {
    xgb_items <- fit_xgb_predict(xy_tr$X, xy_tr$y, xy_te$X, xy_te$y)
    fn_xgb_items <- file.path(pred_dir, sprintf("%s_items_xgb.csv", ds$dataset_id))
    write_pred(xgb_items$y, xgb_items$yhat, fn_xgb_items)
    rows[[k]] <- mk_index_row(ds$dataset_id, "xgboost", "items", fn_xgb_items,
                              list(N = ds$N)); k <- k + 1L
  }
  
  ## ========== (B) SUM-SCORE BRANCH ==========
  if (use_sum_scores) {
    S_tr <- make_sum_scores_2x3(tr_df)
    S_te <- make_sum_scores_2x3(te_df)
    
    # OLS (sum)
    lm_sum <- fit_lm_predict(S_tr, xy_tr$y, S_te, xy_te$y)
    fn_lm_sum <- file.path(pred_dir, sprintf("%s_sum_lm.csv", ds$dataset_id))
    write_pred(lm_sum$y, lm_sum$yhat, fn_lm_sum)
    rows[[k]] <- mk_index_row(ds$dataset_id, "lm", "sum", fn_lm_sum,
                              list(N = ds$N)); k <- k + 1L
    
    # XGB (sum)
    if (use_xgboost) {
      xgb_sum <- fit_xgb_predict(S_tr, xy_tr$y, S_te, xy_te$y)
      fn_xgb_sum <- file.path(pred_dir, sprintf("%s_sum_xgb.csv", ds$dataset_id))
      write_pred(xgb_sum$y, xgb_sum$yhat, fn_xgb_sum)
      rows[[k]] <- mk_index_row(ds$dataset_id, "xgboost", "sum", fn_xgb_sum,
                                list(N = ds$N)); k <- k + 1L
    }
  }
  
  ## ========== (C) FACTOR-SCORE BRANCH ==========
  if (use_factor_scores) {
    if (!requireNamespace("lavaan", quietly = TRUE)) {
      warning("lavaan not installed; skipping factor-score branch for dataset ", ds$dataset_id)
    } else {
      Fsc <- lavaan_factor_scores_2x3(tr_df[, grep("^X\\d+$", names(tr_df)), drop = FALSE],
                                      te_df[, grep("^X\\d+$", names(te_df)), drop = FALSE])
      # OLS (factor)
      lm_fac <- fit_lm_predict(Fsc$Ftr, xy_tr$y, Fsc$Fte, xy_te$y)
      fn_lm_fac <- file.path(pred_dir, sprintf("%s_factor_lm.csv", ds$dataset_id))
      write_pred(lm_fac$y, lm_fac$yhat, fn_lm_fac)
      rows[[k]] <- mk_index_row(ds$dataset_id, "lm", "factor", fn_lm_fac,
                                list(N = ds$N)); k <- k + 1L
      
      # XGB (factor)
      if (use_xgboost) {
        xgb_fac <- fit_xgb_predict(Fsc$Ftr, xy_tr$y, Fsc$Fte, xy_te$y)
        fn_xgb_fac <- file.path(pred_dir, sprintf("%s_factor_xgb.csv", ds$dataset_id))
        write_pred(xgb_fac$y, xgb_fac$yhat, fn_xgb_fac)
        rows[[k]] <- mk_index_row(ds$dataset_id, "xgboost", "factor", fn_xgb_fac,
                                  list(N = ds$N)); k <- k + 1L
      }
    }
  }
}

## ------------------ write index ------------------
pred_index <- do.call(rbind, rows)
utils::write.csv(pred_index, file.path(pred_dir, "pred_index.csv"),
                 row.names = FALSE, fileEncoding = "UTF-8")
cat("Predictions saved to", pred_dir, "\n")
