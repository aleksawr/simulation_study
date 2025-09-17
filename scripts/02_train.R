# ===== 02_train.R =====
rm(list = ls(all.names = TRUE))                      ## clear the R workspace (remove all objects)
source("scripts/utils_pilot.R")                      ## load helper functions (e.g., train/test split)

manifest <- utils::read.csv("data/sim/manifest.csv", stringsAsFactors = FALSE)  ## read simulation manifest (metadata + file paths)
pred_dir <- "data/pred"                              ## directory to store prediction files
dir.create(pred_dir, recursive = TRUE, showWarnings = FALSE)                     ## ensure prediction directory exists

use_xgboost <- FALSE  # set TRUE if you have xgboost installed               ## toggle: train optional xgboost model

## -------- Linear model: fit on train, predict on test --------
fit_lm_predict <- function(df, test_ratio, seed) {
  Xcols <- grep("^X\\d+$", names(df), value = TRUE)  ## select observed predictors X1..Xp
  idx   <- train_test_idx(nrow(df), test_ratio, seed = seed + 123)             ## reproducible split (seed offset for model)
  tr    <- df[idx$train, , drop = FALSE]                                       ## training data
  te    <- df[idx$test,  , drop = FALSE]                                       ## test data
  fml <- as.formula(paste("Y ~", paste(Xcols, collapse = " + ")))              ## formula: Y ~ X1 + X2 + ... + Xp
  fit <- lm(fml, data = tr)                                                    ## fit OLS on training set
  yhat <- predict(fit, newdata = te)                                           ## predict on test set
  list(y = te$Y, yhat = yhat)                                                  ## return test targets and predictions
}

## -------- XGBoost: fit on train, predict on test (optional) --------
fit_xgb_predict <- function(df, test_ratio, seed) {
  if (!requireNamespace("xgboost", quietly = TRUE))                            ## check if xgboost is available
    stop("Install 'xgboost' or set use_xgboost <- FALSE")
  set.seed(seed + 456)                                                         ## seed offset for xgb branch
  Xcols <- grep("^X\\d+$", names(df), value = TRUE)                            ## predictor columns
  idx   <- train_test_idx(nrow(df), test_ratio, seed = seed + 456)             ## reproducible split for xgb
  Xtr <- as.matrix(df[idx$train, Xcols, drop = FALSE]); ytr <- df$Y[idx$train] ## training matrices/vectors
  Xte <- as.matrix(df[idx$test,  Xcols, drop = FALSE]); yte <- df$Y[idx$test]  ## test matrices/vectors
  dtr <- xgboost::xgb.DMatrix(Xtr, label = ytr)                                ## xgboost DMatrix for train
  dte <- xgboost::xgb.DMatrix(Xte, label = yte)                                ## xgboost DMatrix for test
  bst <- xgboost::xgb.train(                                                   ## train gradient-boosted trees (regression)
    data = dtr, nrounds = 300,                                                ## 300 boosting rounds
    params = list(booster="gbtree", objective="reg:squarederror",             ## squared error loss for regression
                  eta=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8),## learning rate, depth, sampling
    verbose = 0
  )
  yhat <- predict(bst, dte)                                                    ## predict on test DMatrix
  list(y = yte, yhat = yhat)                                                   ## return test targets and predictions
}

rows <- list(); k <- 1L                                                        ## container for index rows; row counter
for (i in seq_len(nrow(manifest))) {                                           ## iterate over all simulated datasets
  ds <- manifest[i, ]                                                          ## metadata for the i-th dataset
  df <- utils::read.csv(ds$file, stringsAsFactors = FALSE)                     ## load the simulated dataset (Y + X1..Xp)
  # Re-use dataset seed so the split is reproducible per dataset
  test_ratio <- 0.30                                                           ## hold out 30% for testing
  
  # OLS
  lm_out <- fit_lm_predict(df, test_ratio, seed = ds$seed)                     ## train/predict with OLS
  pred_lm_fn <- file.path(pred_dir, sprintf("%s_lm.csv", ds$dataset_id))      ## file path to save OLS predictions
  utils::write.csv(data.frame(y = lm_out$y, yhat = lm_out$yhat),               ## save columns: true y, predicted yhat
                   pred_lm_fn, row.names = FALSE, fileEncoding = "UTF-8")
  
  # Optional ML
  pred_ml_fn <- NA_character_                                                  ## default: no ML file
  if (use_xgboost) {                                                           ## optionally run xgboost
    ml_out <- fit_xgb_predict(df, test_ratio, seed = ds$seed)                  ## train/predict with xgboost
    pred_ml_fn <- file.path(pred_dir, sprintf("%s_xgb.csv", ds$dataset_id))    ## file path to save xgb predictions
    utils::write.csv(data.frame(y = ml_out$y, yhat = ml_out$yhat),             ## save columns: true y, predicted yhat
                     pred_ml_fn, row.names = FALSE, fileEncoding = "UTF-8")
  }
  
  rows[[k]] <- data.frame(                                                     ## add an index row describing outputs
    dataset_id = ds$dataset_id,                                                ## which dataset the predictions correspond to
    rho_X      = ds$rho_X,                                                     ## reliability level used in that dataset
    replicate  = ds$replicate,                                                 ## replicate ID
    pred_lm    = pred_lm_fn,                                                   ## path to OLS predictions file
    pred_xgb   = pred_ml_fn,                                                   ## path to xgb predictions file (or NA)
    stringsAsFactors = FALSE
  ); k <- k + 1L
}

index <- do.call(rbind, rows)                                                  ## combine all rows to create the prediction index
utils::write.csv(index, file.path(pred_dir, "pred_index.csv"),                 ## save index so downstream scripts can locate preds
                 row.names = FALSE, fileEncoding = "UTF-8")
cat("Predictions saved to", pred_dir, "\n")                                    ## final message
