# ===== 01_simulate.R =====

rm(list = ls(all.names = TRUE))            ## clear the R workspace (remove all objects)

# --- settings ---
N            <- 1000                       ## sample size
p            <- 4                          ## number of true predictors
r_betweenX   <- 0.00                       ## latent correlation among predictors (0 = independent)
beta         <- c(0.6, 0.5, 0.4, 0.3)      ## true regression coefficients for latent predictors
R2_target    <- 0.50                       ## desired true R^2 in the latent model
rho_grid     <- c(0.60, 0.80, 1.00)        ## predictor reliability values to vary
replicates   <- 50                         ## number of repetitions per reliability condition
out_dir      <- "data/sim"                 ## directory to save simulated datasets
manifest_fn  <- file.path(out_dir, "manifest.csv")  ## file to save metadata (manifest) of all runs

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)  ## create output folder if not exists
source("scripts/utils_pilot.R")            ## load utility functions (correlation, variance handling, etc.)

## Function to simulate one dataset given N, p, reliability, etc.
simulate_once <- function(N, p, r_betweenX, beta, R2_target, rho_X, seed) {
  set.seed(seed)                           ## fix random seed for reproducibility
  R <- make_corr(p, r_betweenX)            ## build latent correlation matrix
  X_star <- rmvnorm_simple(N, R)           ## generate latent predictors X* with correlation R
  sigma_eps <- compute_sigma_for_R2(X_star, beta, R2_target)  ## compute residual SD to achieve target R^2
  eta <- as.vector(X_star %*% beta)        ## latent linear predictor (true regression without noise)
  Y   <- eta + rnorm(N, 0, sigma_eps)      ## observed outcome with noise
  E <- matrix(rnorm(N * p), N, p)          ## measurement error for predictors
  X <- sqrt(rho_X) * X_star + sqrt(1 - rho_X) * E  ## observed predictors with given reliability
  colnames(X)      <- paste0("X", 1:p)     ## name observed predictors as X1..Xp
  colnames(X_star) <- paste0("X", 1:p, "_latent")  ## name latent predictors as X1..Xp_latent
  R2_latent_emp <- var(eta) / var(Y)       ## empirical latent R^2 (check actual vs target)
  rho_hat <- sapply(1:p, function(j) cor(X[, j], X_star[, j])^2)  ## empirical reliability per predictor
  list(data = data.frame(Y = Y, X, X_star), ## return simulated data
       diag = c(R2_latent_emp = R2_latent_emp,  ## diagnostics: empirical R^2
                rho_hat_mean = mean(rho_hat)))  ## average empirical reliability across predictors
}

man_rows <- list(); k <- 1L                 ## container for manifest rows and counter
for (i in seq_along(rho_grid)) {            ## loop over reliability levels
  rho <- rho_grid[i]
  for (rep in seq_len(replicates)) {        ## loop over replicates within each reliability level
    dataset_id <- sprintf("rho%02d_rep%03d", round(100*rho), rep)  ## unique dataset ID
    seed <- 1000 + i*100 + rep              ## deterministic seed based on rho and replicate index
    sim <- simulate_once(N, p, r_betweenX, beta, R2_target, rho, seed)  ## simulate one dataset
    fn  <- file.path(out_dir, sprintf("%s.csv", dataset_id))  ## filename for this dataset
    utils::write.csv(sim$data[, c("Y", paste0("X",1:p))],   ## save dataset (only Y and observed X)
                     fn, row.names = FALSE, fileEncoding = "UTF-8")
    man_rows[[k]] <- data.frame(            ## add row to manifest with metadata
      dataset_id   = dataset_id,            ## unique ID for dataset
      file         = fn,                    ## filename where dataset is stored
      rho_X        = rho,                   ## reliability level
      replicate    = rep,                   ## replicate index
      N            = N,                     ## sample size
      p            = p,                     ## number of predictors
      r_betweenX   = r_betweenX,            ## latent correlation among predictors
      R2_target    = R2_target,             ## target latent R^2
      R2_latent_emp= sim$diag["R2_latent_emp"], ## empirical latent R^2
      seed         = seed,                  ## random seed used
      stringsAsFactors = FALSE
    ); k <- k + 1L
  }
}

manifest <- do.call(rbind, man_rows)       ## combine all manifest rows into one data frame
utils::write.csv(manifest, manifest_fn, row.names = FALSE, fileEncoding = "UTF-8")  ## save manifest
cat("Wrote", nrow(manifest), "datasets to", out_dir, "\n")  ## print summary message

