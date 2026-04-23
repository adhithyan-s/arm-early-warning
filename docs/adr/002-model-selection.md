# ADR 002: Model Selection

## Status
Accepted

## Context
We need a forecasting model for AMR resistance rates 2 years ahead.
The feature matrix has ~6000 rows, 12 features, and significant missing values in lag and slope features (~15-20% NaN).

## Decision
Use LightGBM as the primary model with Ridge regression as baseline.

## Reasoning
- LightGBM handles missing values natively — no imputation needed
- Gradient boosting outperforms random forests on tabular data with non-linear patterns (confirmed by EDA: regime change in
  K. pneumoniae carbapenems post-2018)
- Dataset size (~6000 rows) is too small for neural networks
- Ridge baseline (R²=0.904) vs LightGBM (R²=0.921) confirms non-linear signal exists but linear features dominate

## Consequences
- LightGBM requires libomp on macOS (documented in README setup)
- Model must be retrained if new ECDC data is added annually
- Forecast uncertainty is not natively quantified — future work could add conformal prediction intervals