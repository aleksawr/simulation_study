# ===== 01_simulate_cfa.R =====
# Minimal simulation of 2 latent factors -> 6 items (3 per factor),
# with easy loadings (.80, .70, .60). Creates an outcome Y from the true factors
# with a target R^2, and writes CSVs + a manifest.
#
# NO PACKAGES REQUIRED (we implement a tiny mvn via Cholesky).

rm(list = ls(all.names = TRUE))

# --- settings ---
N             <- 1000                         ## sample size
r_F12         <- 0.40                         ## correlation between factors F1 and F2
load_F1       <- c(0.80, 0.70, 0.60)          ## loadings for items X1..X3 on F1
load_F2       <- c(0.80, 0.70, 0.60)          ## loadings for items X4..X6 on F2
R2_target_Y   <- 0.32                         ## desired true R^2 for Y from (F1, F2)
beta_Y        <- c(0.6, 0.2)                  ## coefficients for Y = 0.6*F1 + 0.2*F2 + error
seed_base     <- 1234                         ## base seed for reproducibility
replicates    <- 10                           ## how many datasets to simulate
out_dir       <- "data/sim_cfa"               ## output folder
manifest_fn   <- file.path(out_dir, "manifest.csv")

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# --- tiny helpers (no packages) ---

## 1) draw from MVN(0, Sigma) using Cholesky
rmvnorm_chol <- function(n, Sigma, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  L <- chol(Sigma)                        # upper-triangular (R uses upper by default)
  Z <- matrix(rnorm(n * ncol(Sigma)), nrow = n)  # iid N(0,1)
  Z %*% L                                 # N x p matrix with covariance Sigma
}

## 2) compute residual SD to hit target R^2 in a linear model Y = eta + eps
##    target R^2 = var(eta) / var(Y)  -> var(Y) = var(eta) / R2_target
##    so var(eps) = var(Y) - var(eta) = var(eta)*(1/R2_target - 1)
sigma_for_R2 <- function(eta, R2_target) {
  v_eta <- var(as.numeric(eta))
  v_eps <- v_eta * (1 / R2_target - 1)
  sqrt(max(v_eps, 0))                     # guard against tiny negatives due to rounding
}

## 3) build factor covariance (2x2) from correlation
Sigma_F <- matrix(c(1, r_F12,
                    r_F12, 1), nrow = 2, byrow = TRUE)

## 4) build loading matrix (6x2), simple structure, no cross-loadings
Lambda <- rbind(
  c(load_F1[1], 0),
  c(load_F1[2], 0),
  c(load_F1[3], 0),
  c(0, load_F2[1]),
  c(0, load_F2[2]),
  c(0, load_F2[3])
)

## 5) residual variances so that each item has variance ~ 1: theta = 1 - lambda^2
Theta <- diag(1 - rowSums(Lambda^2))

# --- simulate one dataset ---
simulate_once <- function(N, Sigma_F, Lambda, Theta, beta_Y, R2_target_Y, seed) {
  set.seed(seed)
  # latent factors F ~ MVN(0, Sigma_F)
  F <- rmvnorm_chol(N, Sigma_F)                 # N x 2: columns are F1, F2
  colnames(F) <- c("F1_true", "F2_true")
  
  # items: X = Lambda * F + eps, with eps ~ MVN(0, Theta) and Theta diagonal
  # Since Theta is diagonal, we can add independent errors per item:
  E <- matrix(rnorm(N * nrow(Lambda)), nrow = N) %*% diag(sqrt(diag(Theta)))
  X <- F %*% t(Lambda) + E                      # N x 6
  colnames(X) <- paste0("X", 1:6)
  
  # outcome Y from true factors
  eta <- as.numeric(F %*% beta_Y)               # linear predictor
  sigma_eps <- sigma_for_R2(eta, R2_target_Y)   # choose noise for target R^2
  Y <- eta + rnorm(N, 0, sigma_eps)
  
  # diagnostics
  R2_emp <- var(eta) / var(Y)                   # empirical R^2 achieved
  list(
    data = data.frame(Y = Y, X, F),
    diag = c(R2_emp = R2_emp, sd_eps = sigma_eps)
  )
}

# --- generate replicates + write CSVs + manifest ---
manifest_rows <- list(); k <- 1L

for (rep in seq_len(replicates)) {
  dataset_id <- sprintf("cfa_2x3_rep%03d", rep)
  seed <- seed_base + rep
  
  sim <- simulate_once(
    N = N, Sigma_F = Sigma_F, Lambda = Lambda, Theta = Theta,
    beta_Y = beta_Y, R2_target_Y = R2_target_Y, seed = seed
  )
  
  # Write two files: (1) items+Y, (2) true factors (optional, handy for evaluation)
  fn_items <- file.path(out_dir, sprintf("%s_items.csv", dataset_id))
  fn_latent <- file.path(out_dir, sprintf("%s_latent.csv", dataset_id))
  
  utils::write.csv(sim$data[, c("Y", paste0("X", 1:6))],
                   fn_items, row.names = FALSE, fileEncoding = "UTF-8")
  utils::write.csv(sim$data[, c("F1_true", "F2_true")],
                   fn_latent, row.names = FALSE, fileEncoding = "UTF-8")
  
  manifest_rows[[k]] <- data.frame(
    dataset_id   = dataset_id,
    file_items   = fn_items,
    file_latent  = fn_latent,
    N            = N,
    r_F12        = r_F12,
    load_F1_1    = Lambda[1,1], load_F1_2 = Lambda[2,1], load_F1_3 = Lambda[3,1],
    load_F2_1    = Lambda[4,2], load_F2_2 = Lambda[5,2], load_F2_3 = Lambda[6,2],
    R2_target_Y  = R2_target_Y,
    R2_emp_Y     = sim$diag["R2_emp"],
    sd_eps_Y     = sim$diag["sd_eps"],
    seed         = seed,
    stringsAsFactors = FALSE
  ); k <- k + 1L
}

manifest <- do.call(rbind, manifest_rows)
utils::write.csv(manifest, manifest_fn, row.names = FALSE, fileEncoding = "UTF-8")
cat("Wrote", nrow(manifest), "datasets to", out_dir, "\n")
