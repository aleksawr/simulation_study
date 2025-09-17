# ===== 00_run_all.R =====
rm(list = ls(all.names = TRUE))

source("scripts/01_simulate.R")
source("scripts/02_train.R")
source("scripts/03_validate.R")

