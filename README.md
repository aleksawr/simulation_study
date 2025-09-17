# Simulation Study: Reliability of Predictors

This repository contains a simple simulation study illustrating how **predictor reliability** affects estimated regression results.

---

## Contents

- **first_simulation.R**  
  Main R script for the simulation.  
  - Simulates latent predictors (X*)  
  - Adds measurement error to create observed predictors (X) with reliability levels 0.6, 0.8, and 1.0  
  - Generates an outcome (Y) with a target latent R² = 0.50  
  - Fits linear models (Y ~ X) and summarizes results  
  - Optionally produces a plot of R² versus predictor reliability

- **data_simple/** *(optional)*  
  Folder for saved datasets if `save_data = TRUE` in the script.

---

## How to Run

1. Open `first_simulation.R` in RStudio.  
2. Adjust the settings at the top of the script if needed:  
   - `N`: sample size  
   - `p`: number of predictors  
   - `r_betweenX`: latent predictor correlation (0 for independent, 0.40 for moderate correlation)  
   - `rho_grid`: reliability values to simulate  
3. Run the script.  
4. Check the console output for a summary table, and the plot window for the reliability vs R² graph.

---

## Notation

In the code there is `rho_X` to denote **predictor reliability**.  
The reliability is written as ρ (the Greek letter *rho*).

Reliability can be defined as:

\[
\rho_{XX'} = \frac{\text{Var(True score)}}{\text{Var(Observed score)}}
\]

- If ρ = 1.0 → perfect measurement (no error).  
- If ρ = 0.8 → 80% of the variance in X is true signal, 20% is measurement error.  
- If ρ = 0.6 → 60% true signal, 40% error.

---

## Results

- Lower reliability in predictors leads to:  
  - Smaller estimated regression coefficients (attenuation).  
  - Lower estimated R² values.  
- With perfect reliability (ρ = 1.0), estimates match the true latent model.

---
