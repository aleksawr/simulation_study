# ===== 00_run_all_cfa.R =====
rm(list = ls(all.names = TRUE))

source("scripts/01_simulate_cfa.R")
source("scripts/02_train_cfa.R")
source("scripts/03_validate_cfa.R")

source("scripts/plots.R")
source("scripts/cleanup.R")
