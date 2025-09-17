## ===========================================================
## cleanup.R  — housekeeping script for simulation study
## ===========================================================

# Root directories
sim_dir   <- "data/sim"
pred_dir  <- "data/pred"
out_dir   <- "data/out"

# 1) Keep manifest files, remove individual replicate CSVs
if (dir.exists(sim_dir)) {
  all_files <- list.files(sim_dir, full.names = TRUE)
  keep      <- grep("manifest.csv$", all_files, value = TRUE)
  to_remove <- setdiff(all_files, keep)
  if (length(to_remove)) {
    file.remove(to_remove)
    cat("Removed", length(to_remove), "simulation datasets.\n")
  }
}

# 2) Keep pred_index.csv, remove individual prediction files
if (dir.exists(pred_dir)) {
  all_files <- list.files(pred_dir, full.names = TRUE)
  keep      <- grep("pred_index.csv$", all_files, value = TRUE)
  to_remove <- setdiff(all_files, keep)
  if (length(to_remove)) {
    file.remove(to_remove)
    cat("Removed", length(to_remove), "prediction files.\n")
  }
}

# 3) Keep only summary outputs in data/out
if (dir.exists(out_dir)) {
  all_files <- list.files(out_dir, full.names = TRUE)
  keep      <- grep("^(perf_|.*\\.png$)", basename(all_files), value = TRUE) # keep perf_*.csv and plots
  to_remove <- setdiff(all_files, file.path(out_dir, keep))
  if (length(to_remove)) {
    file.remove(to_remove)
    cat("Removed", length(to_remove), "other output files.\n")
  }
}

cat("Cleanup complete.\n")
