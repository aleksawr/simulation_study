# Simulation Study: Reliability of Predictors in Regression, SEM, and ML

This repository contains a simulation study designed to illustrate, in the context of **regression** and **structural equation modeling (SEM)**, how **predictor reliability** (i.e., the degree to which observed predictors reflect their latent counterparts without measurement error) affects model estimation and predictive performance.

The study reproduces, in simplified form, central ideas discussed in *Beyond the Hype: A Simulation Study Evaluating the Predictive Performance of Machine Learning Models in Psychology* (Jankowsky et al., 2024). It places these ideas in the tradition of **classical regression and SEM**, where measurement error has been studied extensively.

------------------------------------------------------------------------

## Contents

-   **01_simulate.R**\
    Generates synthetic datasets for analysis. Specifically:

    -   Constructs latent predictors (X\*) according to user-defined correlation structure.\
    -   Produces observed predictors (X) by adding measurement error such that their reliability equals pre-specified values (`rho_X`).\
    -   Creates outcome variables (Y) as linear combinations of the latent predictors with noise adjusted to achieve a target latent R².\
    -   Saves all datasets and writes a **manifest file** (`manifest.csv`) describing dataset ID, reliability, replicate number, seed, and empirical checks.

-   **02_train.R**\
    Trains predictive models on the simulated datasets. Specifically:

    -   Reads datasets from the manifest.\
    -   Splits each dataset into training and test subsets (default 70/30 split) using reproducible seeds.\
    -   Fits **ordinary least squares (OLS) regression** (`Y ~ X1 + ... + Xp`).\
    -   Optionally, fits a **gradient boosted tree model (XGBoost)** if the package is installed and enabled.\
    -   Saves predictions (true Y and predicted Ŷ for the test set) to CSV files.\
    -   Produces an **index file** (`pred_index.csv`) linking each dataset to its prediction outputs.

    **What is OLS and why is it relevant here?**

    -   OLS stands for *Ordinary Least Squares regression*.\

    -   It is the most common method of linear regression, where coefficients are chosen to minimize the sum of squared residuals:

$$Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + \dots + \beta_pX_p + \varepsilon$$  

$$\text{minimize } \sum (Y - \hat{Y})^2$$


    -   **In regression theory:** measurement error in predictors attenuates regression coefficients and lowers $R^2$.\

    -   **In SEM:** predictors are modeled as latent variables with explicit reliabilities; attenuation is represented in the measurement model.\

    -   **In this project:** OLS provides the baseline model because its behavior under measurement error is well understood. It anchors the simulations in the regression/SEM tradition and sets expectations against which ML models can be compared.

-   **03_validate.R**\
    Evaluates predictive performance and aggregates results. Specifically:

    -   Loads prediction files and compares predicted values to true outcomes.\
    -   Computes two performance metrics:
        -   **Root Mean Squared Error (RMSE):**

            $$
            RMSE = \sqrt{\frac{1}{n} \sum (y - \hat{y})^2}
            $$

            Measures average prediction error in outcome units.

        -   **Coefficient of Determination (R²):**

            $$
            R^2 = 1 - \frac{MSE}{Var(Y)}
            $$

            Indicates the proportion of variance in Y explained by predictions.
    -   Saves detailed results per dataset (`perf_by_dataset.csv`).\
    -   Aggregates results across replicates by reliability and model (`perf_agg.csv`).\
    -   Creates a simple plot (`R2_vs_rhoX.png`) showing how predictive R² depends on predictor reliability.

-   **scripts/utils_pilot.R**\
    Contains helper functions used across scripts:

    -   `make_corr()` – builds correlation matrices for latent predictors.\
    -   `rmvnorm_simple()` – generates multivariate normal data.\
    -   `compute_sigma_for_R2()` – determines error variance to achieve the target latent R².\
    -   `train_test_idx()` – creates reproducible train/test splits.

-   **data/**\
    Project output directory, with subfolders:

    -   `data/sim/` – simulated datasets + manifest.\
    -   `data/pred/` – prediction outputs + index.\
    -   `data/out/` – validation results and plots.

------------------------------------------------------------------------

## How to Run

1.  Run **`01_simulate.R`** to generate synthetic datasets.
    -   Main settings:
        -   `N`: number of observations (sample size).\
        -   `p`: number of predictors.\
        -   `r_betweenX`: correlation among latent predictors (0 = independent; 0.40 = moderately correlated).\
        -   `beta`: true regression coefficients for latent predictors.\
        -   `R2_target`: desired proportion of variance explained in the latent outcome model.\
        -   `rho_grid`: reliability levels to simulate (e.g., 0.60, 0.80, 1.00).\
        -   `replicates`: number of datasets to generate per reliability level.\
    -   Outputs: datasets in `data/sim/` and `manifest.csv`.
2.  Run **`02_train.R`** to train models and save predictions.
    -   Models: OLS always, XGBoost optional.\
    -   Outputs: per-dataset prediction files in `data/pred/` and `pred_index.csv`.
3.  Run **`03_validate.R`** to compute metrics and visualize results.
    -   Outputs: per-dataset performance (`perf_by_dataset.csv`), aggregated results (`perf_agg.csv`), and R² vs reliability plot (`R2_vs_rhoX.png`).

------------------------------------------------------------------------

## Notation

In the code, **`rho_X`** denotes predictor reliability.\
This follows psychometric convention where reliability is expressed as ρ (Greek *rho*).

Reliability is formally defined as:

$$
\rho = \frac{\text{Var(True score)}}{\text{Var(Observed score)}}
$$

-   ρ = 1.0 → predictors are measured without error (perfect reliability).\
-   ρ = 0.8 → 80% of observed variance reflects the true latent predictor, 20% is error variance.\
-   ρ = 0.6 → 60% is true signal, 40% is error.

------------------------------------------------------------------------

## Results

The simulations demonstrate the following:

-   **Effect of measurement error**\
    Lower reliability reduces the strength of observed predictor–outcome associations.
    -   Regression coefficients are attenuated toward zero.\
    -   Out-of-sample R² declines systematically as reliability decreases.
-   **Comparison of models**\
    Both OLS and XGBoost are constrained by predictor reliability.
    -   Machine learning cannot recover information lost due to measurement error.\
    -   Gains from complex models are limited when predictors themselves are noisy.
-   **Perfect reliability benchmark**\
    At ρ = 1.0, observed estimates align closely with the latent data-generating model, providing a baseline for comparison.

------------------------------------------------------------------------

## Conceptual Summary

This project illustrates a central psychometric principle:\
**the maximum predictive performance is bounded by the reliability of the predictors.**

-   **In regression:** this is seen as *attenuation bias*—coefficients shrink and predictive power decreases when predictors are noisy.\
-   **In SEM:** this is modeled explicitly—observed indicators have reliabilities \< 1, and latent factors capture the true scores.\
-   **In ML:** the same limits apply—flexible models cannot exceed the information content of the observed data.

Thus, this project connects **classical measurement theory (regression, SEM)** with **modern predictive modeling (ML)**, showing their common constraint: **measurement error sets the ceiling**.

------------------------------------------------------------------------
