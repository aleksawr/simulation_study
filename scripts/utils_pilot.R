# ===== utils_pilot.R =====
set.seed(12345)

# Constant-correlation matrix (validate)
make_corr <- function(p, r) {
  if (p == 1) return(matrix(1, 1, 1))
  stopifnot(r > -1/(p - 1))
  M <- matrix(r, p, p); diag(M) <- 1; M
}

# MVN via chol, unit variances
rmvnorm_simple <- function(N, R) {
  L <- chol(R)
  Z <- matrix(rnorm(N * ncol(R)), nrow = N, ncol = ncol(R))
  Z %*% L
}

# Choose eps SD to hit target latent R^2 given realized X*
compute_sigma_for_R2 <- function(X_star, beta, R2) {
  linpred <- as.vector(X_star %*% beta)
  var_lp  <- var(linpred)
  sigma2  <- var_lp * (1 - R2) / R2
  sqrt(sigma2)
}

# Deterministic split per dataset_id for reproducibility
train_test_idx <- function(n, test_ratio = 0.30, seed) {
  set.seed(seed)
  ntest <- floor(test_ratio * n)
  test  <- sample.int(n, ntest)
  list(test = test, train = setdiff(seq_len(n), test))
}
