# Simulation Study: Reliability of Predictors in Regression, SEM, and ML

This repository contains a simulation study designed to illustrate, in the context of **regression** and **structural equation modelling (SEM)**, how **predictor reliability** (i.e., the degree to which observed predictors reflect their latent counterparts without measurement error) affects model estimation and predictive performance.

The study reproduces, in simplified form, central ideas discussed in *Beyond the Hype: A Simulation Study Evaluating the Predictive Performance of Machine Learning Models in Psychology* (Jankowsky et al., 2024). It places these ideas in the tradition of **classical regression and SEM**, where measurement error has been studied extensively.

------------------------------------------------------------------------

## Contents

-   **01_simulate.R**\
    Generates synthetic datasets for analysis. Specifically:

    -   Constructs latent predictors (X\*) according to user-defined correlation structure.
    -   Produces observed predictors (X) by adding measurement error such that their reliability equals pre-specified values (`rho_X`).
    -   Creates outcome variables (Y) as linear combinations of the latent predictors with noise adjusted to achieve a target latent R².
    -   Saves all datasets and writes a **manifest file** (`manifest.csv`) describing dataset ID, reliability, replicate number, seed, and empirical checks.


-   **02_train.R**\
    Trains predictive models on the simulated datasets. Specifically:

    -   Reads datasets from the manifest.
    -   Splits each dataset into training and test subsets (default 70/30 split) using reproducible seeds.
    -   Fits **ordinary least squares (OLS) regression** (`Y ~ X1 + ... + Xp`).
    -   Optionally, fits a **gradient boosted tree model (XGBoost)** if the package is installed and enabled.
    -   Saves predictions (true Y and predicted Ŷ for the test set) to CSV files.
    -   Produces an **index file** (`pred_index.csv`) linking each dataset to its prediction outputs.


-   **03_validate.R**\
    Evaluates predictive performance and aggregates results. Specifically:

    -   Loads prediction files and compares predicted values to true outcomes.
    -   Computes two performance metrics: RMSE and R².
    -   Indicates the proportion of variance in Y explained by predictions.
    -   Saves detailed results per dataset (`perf_by_dataset.csv`).
    -   Aggregates results across replicates by reliability and model (`perf_agg.csv`).
    -   Creates a simple plot (`R2_vs_rhoX.png`) showing how predictive R² depends on predictor reliability.


-   **scripts/utils_pilot.R**\
    Contains helper functions used across scripts:

    -   `make_corr()` – builds correlation matrices for latent predictors.
    -   `rmvnorm_simple()` – generates multivariate normal data.
    -   `compute_sigma_for_R2()` – determines error variance to achieve the target latent R².
    -   `train_test_idx()` – creates reproducible train/test splits.


-   **data/**\
    Project output directory, with subfolders:

    -   `data/sim/` – simulated datasets + manifest.
    -   `data/pred/` – prediction outputs + index.
    -   `data/out/` – validation results and plots.

------------------------------------------------------------------------

## How to Run

1.  Run **`01_simulate.R`** to generate synthetic datasets.
    -   Main settings:
        -   `N`: number of observations (sample size).
        -   `p`: number of predictors.
        -   `r_betweenX`: correlation among latent predictors (0 = independent; 0.40 = moderately correlated).
        -   `beta`: true regression coefficients for latent predictors.
        -   `R2_target`: desired proportion of variance explained in the latent outcome model.
        -   `rho_grid`: reliability levels to simulate (e.g., 0.60, 0.80, 1.00).
        -   `replicates`: number of datasets to generate per reliability level.
    -   Outputs: datasets in `data/sim/` and `manifest.csv`.
2.  Run **`02_train.R`** to train models and save predictions.
    -   Models: OLS always, XGBoost optional.
    -   Outputs: per-dataset prediction files in `data/pred/` and `pred_index.csv`.
3.  Run **`03_validate.R`** to compute metrics and visualize results.
    -   Outputs: per-dataset performance (`perf_by_dataset.csv`), aggregated results (`perf_agg.csv`), and R² vs reliability plot (`R2_vs_rhoX.png`).

------------------------------------------------------------------------

## Notation and Theoretical Background

In the code, **`rho_X`** denotes predictor reliability.  
This follows psychometric convention where reliability is expressed as ρ (Greek *rho*).

- **Reliability** is formally defined as:  

  $$\rho = \frac{\text{Var(True score)}}{\text{Var(Observed score)}}$$  

  - ρ = 1.0 → predictors are measured without error (perfect reliability).  
  - ρ = 0.8 → 80% of observed variance reflects the true latent predictor, 20% is error variance.  
  - ρ = 0.6 → 60% is true signal, 40% is error.  

---

### What is OLS and why is it relevant here?

- **Ordinary Least Squares (OLS)** regression is the most common method of linear regression.  
- The model assumes a linear relationship:  

  $$Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + \dots + \beta_pX_p + \varepsilon$$  

- Coefficients are estimated by minimizing the sum of squared residuals:  

  $$\text{minimize } \sum (Y - \hat{Y})^2$$  

- **In regression theory:** measurement error in predictors attenuates regression coefficients and lowers $R^2$.  
- **In SEM:** predictors are modeled as latent variables with explicit reliabilities; attenuation is represented in the measurement model.  
- **In this project:** OLS provides the baseline model because its behavior under measurement error is well understood. It anchors the simulations in the regression/SEM tradition and sets expectations against which ML models can be compared.  

---

### Evaluation Metrics

Predictive performance is evaluated using two standard metrics: **Root Mean Squared Error (RMSE)** and the **Coefficient of Determination (R²)**.  

- **Root Mean Squared Error (RMSE):**  

  $$RMSE = \sqrt{\tfrac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$  

  - Represents the average prediction error, expressed in the same units as the outcome variable.  
  - Larger errors are penalized more strongly due to squaring.  
  - Lower RMSE indicates better predictive accuracy.  

- **Coefficient of Determination (R²):**  

  $$R^2 = 1 - \tfrac{MSE}{Var(Y)}$$  

  - Represents the proportion of variance in the outcome explained by predictions.  
  - Values closer to 1 indicate stronger predictive performance.  
  - Under measurement error, R² decreases systematically, regardless of sample size or model complexity.  

Together, RMSE and R² capture complementary aspects of model performance:  
- RMSE quantifies **absolute prediction error**.  
- R² quantifies **relative explanatory power**.  

Both metrics highlight the consequences of measurement error: even flexible machine learning models cannot achieve high predictive performance when predictor reliability is low.


------------------------------------------------------------------------

## Summary

This project illustrates a central psychometric principle:
**The maximum predictive performance is bounded by the reliability of the predictors.**

-   **In regression:** this is seen as *attenuation bias* - coefficients shrink and predictive power decreases when predictors are noisy.
-   **In SEM:** this is modelled explicitly - observed indicators have reliabilities \< 1, and latent factors capture the true scores.
-   **In ML:** the same limits apply - flexible models cannot exceed the information content of the observed data.

Thus, this project connects **classical measurement theory (regression, SEM)** with **modern predictive modelling (ML)**, showing their common constraint: **measurement error sets the ceiling**.

------------------------------------------------------------------------
