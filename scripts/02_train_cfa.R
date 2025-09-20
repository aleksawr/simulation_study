# ===== 02_train_cfa.R =====
# Train OLS and optionally XGBoost on:
#   (a) items, (b) sum scores, (c) CFA factor scores, (d) combined (items+sum+factor)
# Uses manifest in data/sim_cfa/. Writes predictions + pred_index.csv in data/pred_cfa/.

rm(list = ls(all.names = TRUE))
source("scripts/utils_pilot.R")   ## must provide: train_test_idx()

## ------------------ config ------------------
manifest_path <- "data/sim_cfa/manifest.csv"
pred_dir      <- "data/pred_cfa"
dir.create(pred_dir, recursive = TRUE, showWarnings = FALSE)

if (!file.exists(manifest_path)) stop("Missing manifest: ", manifest_path)
manifest <- utils::read.csv(manifest_path, stringsAsFactors = FALSE)


use_xgboost        <- TRUE    ## toggle ML
use_sum_scores     <- TRUE    ## toggle sum/mean scores
use_factor_scores  <- TRUE    ## toggle CFA factor scores (requires lavaan)
use_combined       <- TRUE    ## toggle combined branch (items + sum + factor)
test_ratio         <- 0.30

## Optional: train-only standardization for each branch
scale_items  <- FALSE
scale_sum    <- FALSE
scale_factor <- FALSE
scale_all    <- FALSE   ## rarely needed for trees; kept for completeness

## XGBoost params (kept simple/constant for fairness)
xgb_nrounds <- 300
xgb_params  <- list(
  booster="gbtree", objective="reg:squarederror",
  eta=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8
)

## ------------------ helpers ------------------
fit_lm_predict <- function(Xtr, ytr, Xte, yte) {
  tr <- data.frame(Y = ytr, Xtr); te <- data.frame(Y = yte, Xte)
  fml <- as.formula(paste("Y ~", paste(names(Xtr), collapse = " + ")))
  fit <- lm(fml, data = tr)
  yhat <- predict(fit, newdata = te)
  list(y = yte, yhat = yhat)
}

fit_xgb_predict <- function(Xtr, ytr, Xte, yte) {
  if (!requireNamespace("xgboost", quietly = TRUE))
    stop("Install 'xgboost' or set use_xgboost <- FALSE")
  dtr <- xgboost::xgb.DMatrix(as.matrix(Xtr), label = ytr)
  dte <- xgboost::xgb.DMatrix(as.matrix(Xte), label = yte)
  bst <- xgboost::xgb.train(data = dtr, nrounds = xgb_nrounds,
                            params = xgb_params, verbose = 0)
  yhat <- predict(bst, dte)
  list(y = yte, yhat = yhat)
}

## Sum/mean scores for a 2x3 simple-structure design
make_sum_scores_2x3 <- function(df_items) {
  stopifnot(all(paste0("X", 1:6) %in% names(df_items)))
  data.frame(
    F1_sum = rowMeans(df_items[, c("X1","X2","X3")]),
    F2_sum = rowMeans(df_items[, c("X4","X5","X6")])
  )
}

## Factor scores via lavaan (fit on TRAIN ONLY; score train + test)
lavaan_factor_scores_2x3 <- function(tr_items, te_items) {
  if (!requireNamespace("lavaan", quietly = TRUE))
    stop("Install 'lavaan' or set use_factor_scores <- FALSE")
  model_txt <- '
    F1 =~ X1 + X2 + X3
    F2 =~ X4 + X5 + X6
  '
  fit <- tryCatch(
    lavaan::cfa(model_txt, data = tr_items, std.lv = TRUE),
    error = function(e) e
  )
  if (inherits(fit, "error")) {
    warning("lavaan CFA failed: ", conditionMessage(fit))
    return(NULL)
  }
  F_tr <- as.data.frame(lavaan::lavPredict(fit, newdata = tr_items))
  F_te <- as.data.frame(lavaan::lavPredict(fit, newdata = te_items))
  names(F_tr) <- sub("\\.fs$", "", names(F_tr))
  names(F_te) <- names(F_tr)
  list(Ftr = F_tr, Fte = F_te)
}

## Extract item matrix and Y from a df with Y + X1:Xp
get_XY_from_items <- function(df) {
  Xcols <- grep("^X\\d+$", names(df), value = TRUE)
  list(X = df[, Xcols, drop = FALSE], y = df$Y)
}

## Train-only standardization (optional)
scale_train_test <- function(Xtr, Xte) {
  mu  <- sapply(Xtr, mean);  sdv <- sapply(Xtr, sd)
  sdv[sdv == 0] <- 1
  Xtr_s <- as.data.frame(scale(Xtr, center = mu, scale = sdv))
  Xte_s <- as.data.frame(scale(Xte, center = mu, scale = sdv))
  list(Xtr = Xtr_s, Xte = Xte_s)
}

write_pred <- function(y, yhat, out_path) {
  utils::write.csv(data.frame(y = y, yhat = yhat),
                   out_path, row.names = FALSE, fileEncoding = "UTF-8")
  out_path
}

mk_index_row <- function(dataset_id, model, score_type, file_path, extra = list()) {
  as.data.frame(c(list(
    dataset_id = dataset_id,
    model      = model,         # "lm" or "xgboost"
    score_type = score_type,    # "items" | "sum" | "factor" | "all"
    pred_file  = file_path
  ), extra), stringsAsFactors = FALSE)
}

## ------------------ main loop ------------------
rows <- list(); k <- 1L

for (i in seq_len(nrow(manifest))) {
  ds <- manifest[i, ]
  
  if (!file.exists(ds$file_items)) {
    warning("Missing items file for dataset_id=", ds$dataset_id, ": ", ds$file_items)
    next
  }
  df_items <- utils::read.csv(ds$file_items, stringsAsFactors = FALSE)
  
  ## Split once per dataset (same split for all branches)
  split_seed <- ds$seed + 100
  idx   <- train_test_idx(nrow(df_items), test_ratio, seed = split_seed)
  tr_df <- df_items[idx$train, , drop = FALSE]
  te_df <- df_items[idx$test,  , drop = FALSE]
  
  n_train <- nrow(tr_df); n_test <- nrow(te_df)
  
  ## ========== (A) ITEMS ==========
  xy_tr <- get_XY_from_items(tr_df)
  xy_te <- get_XY_from_items(te_df)
  if (scale_items) {
    sts <- scale_train_test(xy_tr$X, xy_te$X)
    xy_tr$X <- sts$Xtr; xy_te$X <- sts$Xte
  }
  # OLS
  lm_items <- fit_lm_predict(xy_tr$X, xy_tr$y, xy_te$X, xy_te$y)
  fn_lm_items <- file.path(pred_dir, sprintf("%s_items_lm.csv", ds$dataset_id))
  write_pred(lm_items$y, lm_items$yhat, fn_lm_items)
  rows[[k]] <- mk_index_row(ds$dataset_id, "lm", "items", fn_lm_items,
                            list(N = ds$N, n_train = n_train, n_test = n_test, seed_split = split_seed)); k <- k + 1L
  # XGB
  if (use_xgboost) {
    xgb_items <- fit_xgb_predict(xy_tr$X, xy_tr$y, xy_te$X, xy_te$y)
    fn_xgb_items <- file.path(pred_dir, sprintf("%s_items_xgb.csv", ds$dataset_id))
    write_pred(xgb_items$y, xgb_items$yhat, fn_xgb_items)
    rows[[k]] <- mk_index_row(ds$dataset_id, "xgboost", "items", fn_xgb_items,
                              list(N = ds$N, n_train = n_train, n_test = n_test, seed_split = split_seed)); k <- k + 1L
  }
  
  ## ========== (B) SUM ==========
  if (use_sum_scores) {
    S_tr <- make_sum_scores_2x3(tr_df)
    S_te <- make_sum_scores_2x3(te_df)
    if (scale_sum) {
      sts <- scale_train_test(S_tr, S_te)
      S_tr <- sts$Xtr; S_te <- sts$Xte
    }
    # OLS
    lm_sum <- fit_lm_predict(S_tr, xy_tr$y, S_te, xy_te$y)
    fn_lm_sum <- file.path(pred_dir, sprintf("%s_sum_lm.csv", ds$dataset_id))
    write_pred(lm_sum$y, lm_sum$yhat, fn_lm_sum)
    rows[[k]] <- mk_index_row(ds$dataset_id, "lm", "sum", fn_lm_sum,
                              list(N = ds$N, n_train = n_train, n_test = n_test, seed_split = split_seed)); k <- k + 1L
    # XGB
    if (use_xgboost) {
      xgb_sum <- fit_xgb_predict(S_tr, xy_tr$y, S_te, xy_te$y)
      fn_xgb_sum <- file.path(pred_dir, sprintf("%s_sum_xgb.csv", ds$dataset_id))
      write_pred(xgb_sum$y, xgb_sum$yhat, fn_xgb_sum)
      rows[[k]] <- mk_index_row(ds$dataset_id, "xgboost", "sum", fn_xgb_sum,
                                list(N = ds$N, n_train = n_train, n_test = n_test, seed_split = split_seed)); k <- k + 1L
    }
  }
  
  ## ========== (C) FACTOR ==========
  if (use_factor_scores) {
    if (!requireNamespace("lavaan", quietly = TRUE)) {
      warning("lavaan not installed; skipping factor-score branch for dataset ", ds$dataset_id)
    } else {
      X_tr_items <- tr_df[, grep("^X\\d+$", names(tr_df)), drop = FALSE]
      X_te_items <- te_df[, grep("^X\\d+$", names(te_df)), drop = FALSE]
      Fsc <- lavaan_factor_scores_2x3(X_tr_items, X_te_items)
      if (!is.null(Fsc)) {
        F_tr <- Fsc$Ftr; F_te <- Fsc$Fte
        if (scale_factor) {
          sts <- scale_train_test(F_tr, F_te)
          F_tr <- sts$Xtr; F_te <- sts$Xte
        }
        # OLS
        lm_fac <- fit_lm_predict(F_tr, xy_tr$y, F_te, xy_te$y)
        fn_lm_fac <- file.path(pred_dir, sprintf("%s_factor_lm.csv", ds$dataset_id))
        write_pred(lm_fac$y, lm_fac$yhat, fn_lm_fac)
        rows[[k]] <- mk_index_row(ds$dataset_id, "lm", "factor", fn_lm_fac,
                                  list(N = ds$N, n_train = n_train, n_test = n_test, seed_split = split_seed)); k <- k + 1L
        # XGB
        if (use_xgboost) {
          xgb_fac <- fit_xgb_predict(F_tr, xy_tr$y, F_te, xy_te$y)
          fn_xgb_fac <- file.path(pred_dir, sprintf("%s_factor_xgb.csv", ds$dataset_id))
          write_pred(xgb_fac$y, xgb_fac$yhat, fn_xgb_fac)
          rows[[k]] <- mk_index_row(ds$dataset_id, "xgboost", "factor", fn_xgb_fac,
                                    list(N = ds$N, n_train = n_train, n_test = n_test, seed_split = split_seed)); k <- k + 1L
        }
      }
    }
  }
  
  ## ========== (D) COMBINED: items + sum + factor ==========
  if (use_combined) {
    # Start from items (always present)
    X_all_tr <- xy_tr$X
    X_all_te <- xy_te$X
    # Add sum scores if they were computed in this iteration
    if (exists("S_tr") && exists("S_te")) {
      X_all_tr <- cbind(X_all_tr, S_tr)
      X_all_te <- cbind(X_all_te, S_te)
    }
    # Add factor scores if they were computed in this iteration
    if (exists("F_tr") && exists("F_te")) {
      X_all_tr <- cbind(X_all_tr, F_tr)
      X_all_te <- cbind(X_all_te, F_te)
    }
    # Ensure unique, aligned names
    names(X_all_tr) <- make.unique(names(X_all_tr))
    names(X_all_te) <- names(X_all_tr)
    
    if (scale_all) {
      sts <- scale_train_test(X_all_tr, X_all_te)
      X_all_tr <- sts$Xtr; X_all_te <- sts$Xte
    }
    
    # OLS baseline on combined set (optional but informative)
    lm_all <- fit_lm_predict(X_all_tr, xy_tr$y, X_all_te, xy_te$y)
    fn_lm_all <- file.path(pred_dir, sprintf("%s_all_lm.csv", ds$dataset_id))
    write_pred(lm_all$y, lm_all$yhat, fn_lm_all)
    rows[[k]] <- mk_index_row(ds$dataset_id, "lm", "all", fn_lm_all,
                              list(N = ds$N, n_train = n_train, n_test = n_test, seed_split = split_seed)); k <- k + 1L
    
    # XGBoost on combined features
    if (use_xgboost) {
      xgb_all <- fit_xgb_predict(X_all_tr, xy_tr$y, X_all_te, xy_te$y)
      fn_xgb_all <- file.path(pred_dir, sprintf("%s_all_xgb.csv", ds$dataset_id))
      write_pred(xgb_all$y, xgb_all$yhat, fn_xgb_all)
      rows[[k]] <- mk_index_row(ds$dataset_id, "xgboost", "all", fn_xgb_all,
                                list(N = ds$N, n_train = n_train, n_test = n_test, seed_split = split_seed)); k <- k + 1L
    }
  }
  
  ## Clean up branch-local objects so the next dataset iteration starts fresh
  if (exists("S_tr")) rm(S_tr, S_te)
  if (exists("F_tr")) rm(F_tr, F_te)
  if (exists("Fsc"))  rm(Fsc)
}

## ------------------ write index ------------------
pred_index <- if (length(rows)) do.call(rbind, rows) else
  data.frame(dataset_id=character(), model=character(), score_type=character(),
             pred_file=character(), N=integer(), n_train=integer(), n_test=integer(),
             seed_split=integer(), stringsAsFactors = FALSE)

utils::write.csv(pred_index, file.path(pred_dir, "pred_index.csv"),
                 row.names = FALSE, fileEncoding = "UTF-8")
cat("Predictions saved to", pred_dir, "\n")

