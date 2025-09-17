# =========================================================
# Simple simulation: vary ONLY predictor reliability (rho_X)
# - Latent predictors X* (p variables), here independent (r_betweenX = 0)
# - Outcome: Y = beta' * X* + eps, with a chosen target R^2
# - Observed predictors: X = sqrt(rho)*X* + sqrt(1 - rho)*E
# - For each reliability level, fit lm(Y ~ X) and summarize
# =========================================================

rm(list = ls(all = TRUE))
set.seed(12345)

# ----------------------- User settings -----------------------
N            <- 1000                 # sample size
p            <- 4                    # number of relevant predictors
r_betweenX   <- 0.00                 # correlation among latent predictors (0 for easier start?)
beta         <- c(0.6, 0.5, 0.4, 0.3)# true effects for X* -> Y (length must be p)
R2_target    <- 0.50                 # true R^2 in the latent model (Y on X*)
rho_grid     <- c(0.6, 0.8, 1.0)     # ONLY factor we vary (predictor reliability)
replicates   <- 1                    # set >1 to average results across runs
save_data    <- FALSE                # TRUE to save per-condition CSVs
out_dir      <- "data_simple"        # folder for saving if save_data = TRUE
make_plot    <- TRUE                 # quick plot of R^2 vs reliability at the end

# ------------------ Helper functions (base R) -----------------
# Build a p x p correlation matrix: 1 on diagonal, r elsewhere
make_corr <- function(p, r) {
  if (p == 1) return(matrix(1, 1, 1))
  M <- matrix(r, nrow = p, ncol = p)
  diag(M) <- 1
  M
}

# Draw N x p multivariate normal with correlation matrix R (unit variances)
rmvnorm_simple <- function(N, R) {
  # R should be positive-definite; chol gives upper triangular L such that R = t(L) %*% L
  L <- chol(R)
  Z <- matrix(rnorm(N * ncol(R)), nrow = N, ncol = ncol(R))
  Z %*% L
}

# Choose sigma_epsilon so that the latent model achieves the target R^2
compute_sigma_for_R2 <- function(X_star, beta, R2) {
  linpred <- as.vector(X_star %*% beta)
  var_lp  <- var(linpred)
  # R2 = var(lp) / (var(lp) + sigma^2)  =>  sigma^2 = var(lp)*(1 - R2)/R2
  sigma2 <- var_lp * (1 - R2) / R2
  sqrt(sigma2)
}

# Simulate one dataset for a given reliability rho_X (applied to ALL predictors)
simulate_once <- function(N, p, r_betweenX, beta, R2_target, rho_X) {
  # 1) Latent predictors X* ~ MVN(0, R)
  R <- make_corr(p, r_betweenX)
  X_star <- rmvnorm_simple(N, R)  # N x p, Var ~ 1, Cor ~ R
  
  # 2) Choose epsilon SD to hit the target latent R^2
  sigma_eps <- compute_sigma_for_R2(X_star, beta, R2_target)
  
  # 3) Generate Y from the latent model (no measurement error in Y)
  eps <- rnorm(N, 0, sigma_eps)
  Y   <- as.vector(X_star %*% beta) + eps
  
  # 4) Add measurement error to predictors to achieve reliability rho_X
  #    X_obs = sqrt(rho) * X_star + sqrt(1 - rho) * E, with E ~ N(0, 1)
  E <- matrix(rnorm(N * p), nrow = N, ncol = p)
  X <- sqrt(rho_X) * X_star + sqrt(1 - rho_X) * E
  
  # Name columns and return both observed and latent (for optional checks)
  colnames(X)      <- paste0("X", 1:p)
  colnames(X_star) <- paste0("X", 1:p, "_latent")
  data.frame(Y = Y, X, X_star)
}

# Fit OLS with observed X and return a tidy one-row summary
fit_and_summarize <- function(dat) {
  Xcols <- grep("^X\\d+$", colnames(dat), value = TRUE)
  fml   <- as.formula(paste0("Y ~ ", paste0(Xcols, collapse = " + ")))
  fit   <- lm(fml, data = dat)
  summ  <- summary(fit)
  
  # Metrics
  metrics <- data.frame(
    R2_est = unname(summ$r.squared),
    R2_adj = unname(summ$adj.r.squared),
    RMSE   = sqrt(mean(summ$residuals^2))
  )
  
  # Coefficients -> 1-row data.frame
  coefs <- coef(fit)                 # named vector: c("(Intercept)"=..., "X1"=..., ...)
  coefs_df <- as.data.frame(t(coefs))
  # Rename "(Intercept)" -> "Intercept" for consistency
  names(coefs_df) <- sub("^\\(Intercept\\)$", "Intercept", names(coefs_df))
  
  # Combine and return (columns will be: R2_est, R2_adj, RMSE, Intercept, X1..Xp)
  cbind(metrics, coefs_df)
}


# --------------------------- Run loop -------------------------
if (save_data) dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

summaries <- list()
idx <- 1

for (rho in rho_grid) {
  for (rep in seq_len(replicates)) {
    dat <- simulate_once(
      N = N, p = p, r_betweenX = r_betweenX,
      beta = beta, R2_target = R2_target, rho_X = rho
    )
    if (save_data) {
      fn <- file.path(out_dir, sprintf("sim_rho%s_rep%s_N%s.csv",
                                       gsub("\\.", "p", sprintf("%.2f", rho)), rep, N))
      utils::write.csv(dat[, c("Y", paste0("X", 1:p))], fn, row.names = FALSE, fileEncoding = "UTF-8")
    }
    s <- fit_and_summarize(dat)
    s$rho_X      <- rho
    s$replicate  <- rep
    s$N          <- N
    s$R2_true    <- R2_target
    s$p          <- p
    s$r_betweenX <- r_betweenX
    s$note       <- "Only predictor reliability varies; latent model linear; Y without measurement error"
    summaries[[idx]] <- s
    idx <- idx + 1
  }
}

summary_tab <- do.call(rbind, summaries)

# --------------------------- Output ---------------------------
cat("\n=== Summary by reliability (rho_X) ===\n")
print(summary_tab[, c("rho_X","replicate","N","R2_true","R2_est","R2_adj","RMSE","Intercept", paste0("X",1:p))],
      row.names = FALSE)

# Optional quick plot: estimated R^2 vs reliability
if (make_plot) {
  # Aggregate over replicates if replicates > 1
  if (replicates > 1) {
    agg <- aggregate(R2_est ~ rho_X, data = summary_tab, FUN = mean)
    plot(agg$rho_X, agg$R2_est, type = "b",
         xlab = "Predictor reliability (rho_X)",
         ylab = "Estimated R-squared (Y ~ observed X)",
         main = "Effect of reliability on observed model fit")
  } else {
    plot(summary_tab$rho_X, summary_tab$R2_est, type = "b",
         xlab = "Predictor reliability (rho_X)",
         ylab = "Estimated R-squared (Y ~ observed X)",
         main = "Effect of reliability on observed model fit")
  }
}

# ---------------------- Sanity check (optional) ----------------------
# Empirical reliability proxy for X1 (Var(X1*) ~ 1): cor(X1, X1_latent)^2
dat_check <- simulate_once(N, p, r_betweenX, beta, R2_target, rho_X = rho_grid[1])
cor(dat_check$X1, dat_check$X1_latent)^2

