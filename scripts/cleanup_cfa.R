## ===========================================================
## cleanup_cfa.R — housekeeping for CFA pipeline
## ===========================================================

sim_dir <- "data/sim_cfa"
pred_dir <- "data/pred_cfa"
out_dir <- "data/out_cfa"

## 1) Keep manifest, remove per-replicate CSVs in sim_cfa
if (dir.exists(sim_dir)) {
  all_files <- list.files(sim_dir, full.names = TRUE)
  keep      <- grep("manifest\\.csv$", all_files, value = TRUE)
  to_remove <- setdiff(all_files, keep)
  if (length(to_remove)) {
    removed <- file.remove(to_remove)
    cat("Removed", sum(removed), "simulation files (kept manifest).\n")
  }
}

## 2) Keep pred_index.csv, remove per-prediction files in pred_cfa
if (dir.exists(pred_dir)) {
  all_files <- list.files(pred_dir, full.names = TRUE)
  keep      <- grep("pred_index\\.csv$", all_files, value = TRUE)
  to_remove <- setdiff(all_files, keep)
  if (length(to_remove)) {
    removed <- file.remove(to_remove)
    cat("Removed", sum(removed), "prediction files (kept pred_index.csv).\n")
  }
}

## 3) Keep only summary outputs (perf_*.csv and *.png) in out_cfa
if (dir.exists(out_dir)) {
  all_files <- list.files(out_dir, full.names = TRUE)
  keep_mask <- grepl("^perf_.*\\.csv$|\\.png$", basename(all_files))
  to_remove <- all_files[!keep_mask]
  if (length(to_remove)) {
    removed <- file.remove(to_remove)
    cat("Removed", sum(removed), "other output files (kept perf_*.csv and plots).\n")
  }
}

cat("CFA cleanup complete.\n")
