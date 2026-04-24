import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def _compute_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (Country, Pathogen, Antibiotic) group, compute lagged values of PctResistant at 1, 2, and 3 years back.

    Why: Last year's resistance is the single strongest predictor of this year's resistance. 
    Lags also let the model detect momentum — is resistance accelerating or holding steady?
    """
    group_cols = ["CountryCode", "Pathogen", "Antibiotic"]

    for lag in [1, 2, 3]:
        df[f"lag_{lag}yr"] = (
            df.groupby(group_cols)["PctResistant"]
            .shift(lag)
        )

    return df


def _compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling mean and std over 3 and 5 year windows per group.

    Why: Smooths the year-to-year noise visible in individual country series. 
    Rolling std captures volatility — a high std signals an unstable series where a spike (like K. pneumoniae carbapenems
    post-2018) is more likely.
    """
    group_cols = ["CountryCode", "Pathogen", "Antibiotic"]

    for window in [3, 5]:
        df[f"rolling_mean_{window}yr"] = (
            df.groupby(group_cols)["PctResistant"]
            .transform(
                lambda x: x.shift(1)
                .rolling(window, min_periods=2)
                .mean()
            )
        )
        df[f"rolling_std_{window}yr"] = (
            df.groupby(group_cols)["PctResistant"]
            .transform(
                lambda x: x.shift(1)
                .rolling(window, min_periods=2)
                .std()
            )
        )

    return df


def _compute_trend_slope(series: pd.Series) -> float:
    """
    Fit a simple linear slope to a series of resistance values.
    Returns the slope (percentage points per year).
    Returns NaN if fewer than 3 non-null values.
    """
    valid = series.dropna()
    if len(valid) < 3:
        return np.nan
    x = np.arange(len(valid))
    slope, _ = np.polyfit(x, valid.values, 1)
    return slope


def _compute_slope_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Linear slope of PctResistant over the past 3 and 5 years per group.

    Why: Slope directly captures trend direction. A rising slope on K. pneumoniae carbapenems is an early warning signal. 
    A falling slope on MRSA reflects successful intervention.

    Important: we compute slope on data UP TO but not including the current year (shift(1) before rolling) to prevent data leakage.
    """
    group_cols = ["CountryCode", "Pathogen", "Antibiotic"]

    for window in [3, 5]:
        df[f"slope_{window}yr"] = (
            df.groupby(group_cols)["PctResistant"]
            .transform(
                lambda x: x.shift(1)
                .rolling(window, min_periods=3)
                .apply(_compute_trend_slope, raw=False)
            )
        )

    return df


def _compute_regional_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (Year, Pathogen, Antibiotic), compute the mean resistance across ALL countries — this becomes a cross-country context feature.

    Why: A country with 5% resistance looks very different if the EU average is 3% (high) vs 40% (low). 
    Regional context is a strong signal the model needs.

    We exclude the country itself from its own regional average to prevent leakage.
    """
    group_cols = ["Year", "Pathogen", "Antibiotic"]

    # sum and count accross all countries for each year+combo
    df["_regional_sum"] = df.groupby(group_cols)["PctResistant"].transform("sum")
    df["_regional_count"] = df.groupby(group_cols)["PctResistant"].transform("count")

    # leave-one-out mean: subtract this country's value from the total
    df["regional_avg_excl_self"] = (
        (df["_regional_sum"] - df["PctResistant"].fillna(0)) / (df["_regional_count"] - 1).clip(lower=1)
    )

    df = df.drop(columns=["_regional_sum", "_regional_count"])
    return df


def _compute_data_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features that capture data reliability rather than resistance signal.
    """
    group_cols = ["CountryCode", "Pathogen", "Antibiotic"]

    # Log-scaled isolate count — handle NaN and zeros safely
    df["log_num_isolates"] = np.log1p(
        pd.to_numeric(df["NumIsolates"], errors="coerce").fillna(0)
    )

    # Cumulative count of non-null observations up to (not including) current year
    def cumulative_obs(x):
        shifted = x.shift(1)
        return shifted.notna().cumsum().astype(float)

    df["years_of_data"] = (
        df.groupby(group_cols)["PctResistant"]
        .transform(cumulative_obs)
    )

    return df


def _create_target(df: pd.DataFrame, horizon: int = 2) -> pd.DataFrame:
    """
    Create the forecast target: PctResistant at year T + horizon.

    horizon=2 means we're predicting 2 years ahead — given data up to 2020, predict 2022's resistance rate.

    Why 2 years: gives enough lead time to be actionable for public health planning, while being close enough to be forecastable.
    """
    group_cols = ["CountryCode", "Pathogen", "Antibiotic"]

    df[f"target_pct_resistant_{horizon}yr"] = (
        df.groupby(group_cols)["PctResistant"]
        .shift(-horizon)
    )

    return df


def build_features(df: pd.DataFrame, forecast_horizon: int = 2) -> pd.DataFrame:
    """
    Master function: takes the validated ECDC DataFrame and returns an ML-ready feature matrix.

    Each row represents a (Country, Year, Pathogen, Antibiotic) observation with all engineered features and a forecast target.

    Rows where the target is NaN (the last `horizon` years of each series) 
    are kept in the output — they're used for inference, not training.
    """
    logger.info(f"Building features for {len(df)} rows | horizon={forecast_horizon}yr")

    # sort first - all group operations depend on correct time ordering
    df = df.sort_values(
        ["CountryCode", "Pathogen", "Antibiotic", "Year"]
    ).reset_index(drop=True)

    df = _compute_lag_features(df)
    df = _compute_rolling_features(df)
    df = _compute_slope_features(df)
    df = _compute_regional_features(df)
    df = _compute_data_quality_features(df)
    df = _create_target(df, horizon=forecast_horizon)

    feature_cols = [
        "lag_1yr", "lag_2yr", "lag_3yr",
        "rolling_mean_3yr", "rolling_mean_5yr",
        "rolling_std_3yr", "rolling_std_5yr",
        "slope_3yr", "slope_5yr",
        "regional_avg_excl_self",
        "log_num_isolates",
        "years_of_data",
    ]

    missing_features = df[feature_cols].isnull().sum()
    logger.info(
        f"Feature matrix built: {len(df)} rows | "
        f"{len(feature_cols)} features\n"
        f"Missing values per feature:\n{missing_features.to_string()}"
    )

    return df
