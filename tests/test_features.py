import pytest
import pandas as pd
import numpy as np

from src.features.engineer import (
    build_features,
    _compute_lag_features,
    _compute_trend_slope,
    _compute_regional_features,
    _compute_data_quality_features,
    _create_target,
)


# ---------------------------------------------------------------------------
# Helper — minimal valid DataFrame for feature engineering tests
# ---------------------------------------------------------------------------

def _make_group_df(
        pct_values: list,
        country: str = "Germany",
        country_code: str = "DE",
        pathogen: str = "Klebsiella pneumoniae",
        antibiotic: str = "Carbapenems",
        start_year: int = 2010,
) -> pd.DataFrame:
    """
    Build a single-group time series DataFrame with known values.
    Makes it easy to test exact expected outputs.
    """
    n = len(pct_values)
    return pd.DataFrame({
        "Year":         list(range(start_year, start_year + n)),
        "CountryCode":  [country_code] * n,
        "Country":      [country] * n,
        "Pathogen":     [pathogen] * n,
        "Antibiotic":   [antibiotic] * n,
        "PctResistant": pct_values,
        "NumResistant": [10.0] * n,
        "NumIsolates":  [200.0] * n,
    })


def _make_multi_country_df() -> pd.DataFrame:
    """Two countries, same pathogen+antibiotic, 6 years each."""
    de = _make_group_df(
        [5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        country="Germany",
        country_code="DE"
    )

    fr = _make_group_df(
        [20.0, 22.0, 24.0, 26.0, 28.0, 30.0],
        country="France",
        country_code="FR"
    )

    return pd.concat([de, fr], ignore_index=True)


# ---------------------------------------------------------------------------
# Tests for _compute_lag_features
# ---------------------------------------------------------------------------

class TestLagFeatures:

    def test_lag_columns_created(self):
        df = _make_group_df([5.0, 10.0, 15.0, 20.0, 25.0])
        result = _compute_lag_features(df.copy())
        assert "lag_1yr" in result.columns
        assert "lag_2yr" in result.columns
        assert "lag_3yr" in result.columns

    def test_lag_1yr_correct_values(self):
        df = _make_group_df([5.0, 10.0, 15.0, 20.0, 25.0])
        result = _compute_lag_features(df.copy())
        # first row has no lag - should be NaN
        assert pd.isna(result.iloc[0]["lag_1yr"])
        # second row lag_1 should be first row's value
        assert result.iloc[1]["lag_1yr"] == 5.0
        assert result.iloc[2]["lag_1yr"] == 10.0

    def test_lag_does_not_bleed_across_groups(self):
        # two countries - lag should not carry over from country A to country B
        df = _make_multi_country_df()
        df = df.sort_values(["CountryCode", "Pathogen", "Antibiotic", "Year"])
        result = _compute_lag_features(df.copy())

        # first row of France group should have NaN lag, not Germany's last value
        fr_first = result[(result["CountryCode"] == "FR")].sort_values("Year").iloc[0]
        assert pd.isna(fr_first["lag_1yr"])

    def test_lag_3yr_needs_3_rows(self):
        df = _make_group_df([5.0, 10.0, 15.0, 20.0, 25.0])
        result = _compute_lag_features(df.copy())
        # first 3 rows should have NaN for lag_3yr
        assert pd.isna(result.iloc[0]["lag_3yr"])
        assert pd.isna(result.iloc[1]["lag_3yr"])
        assert pd.isna(result.iloc[2]["lag_3yr"])
        # fourth row should have first row's value
        assert result.iloc[3]["lag_3yr"] == 5.0


# ---------------------------------------------------------------------------
# Tests for _compute_trend_slope
# ---------------------------------------------------------------------------

class TestTrendSlope:

    def test_positive_slope_detected(self):
        # perfectly linear increasing series
        series = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        slope = _compute_trend_slope(series)
        assert slope > 0

    def test_negative_slope_detected(self):
        series = pd.Series([50.0, 40.0, 30.0, 20.0, 10.0])
        slope = _compute_trend_slope(series)
        assert slope < 0

    def test_flat_series_slope_near_zero(self):
        series = pd.Series([10.0, 10.0, 10.0, 10.0, 10.0])
        slope = _compute_trend_slope(series)
        assert abs(slope) < 0.01

    def test_returns_nan_for_too_few_values(self):
        series = pd.Series([10.0, 20.0])  # only 2 values, need min 3
        slope = _compute_trend_slope(series)
        assert np.isnan(slope)

    def test_ignores_nan_values(self):
        series = pd.Series([10.0, np.nan, 30.0, 40.0, 50.0])
        slope = _compute_trend_slope(series)
        # should still compute without crashing
        assert not np.isnan(slope)


# ---------------------------------------------------------------------------
# Tests for _compute_regional_features
# ---------------------------------------------------------------------------

class TestRegionalFeatures:

    def test_regional_avg_column_created(self):
        df = _make_multi_country_df()
        result = _compute_regional_features(df.copy())
        assert "regional_avg_excl_self" in result.columns

    def test_regional_avg_excludes_self(self):
        # Germany=10%, France=30% in year 2010
        # Germany's regional_avg should be 30% (only France)
        # France's regional_avg should be 10% (only Germany)
        de = _make_group_df([10.0], country="Germany", country_code="DE", start_year=2010)
        fr = _make_group_df([30.0], country="France", country_code="FR", start_year=2010)
        df = pd.concat([de, fr], ignore_index=True)
        result = _compute_regional_features(df.copy())

        de_row = result[result["CountryCode"] == "DE"].iloc[0]
        fr_row = result[result["CountryCode"] == "FR"].iloc[0]

        assert de_row["regional_avg_excl_self"] == pytest.approx(30.0)
        assert fr_row["regional_avg_excl_self"] == pytest.approx(10.0)

    def test_no_missing_in_regional_avg(self):
        df = _make_multi_country_df()
        result = _compute_regional_features(df.copy())
        assert result["regional_avg_excl_self"].isna().sum() == 0


# ---------------------------------------------------------------------------
# Tests for _compute_data_quality_features
# ---------------------------------------------------------------------------

class TestDataQualityFeatures:

    def test_log_num_isolates_created(self):
        df = _make_group_df([5.0, 10.0, 15.0])
        result = _compute_data_quality_features(df.copy())
        assert "log_num_isolates" in result.columns

    def test_log_num_isolates_no_missing(self):
        df = _make_group_df([5.0, 10.0, 15.0])
        result = _compute_data_quality_features(df.copy())
        assert result["log_num_isolates"].isna().sum() == 0

    def test_log_num_isolates_handles_nan_isolates(self):
        df = _make_group_df([5.0, 10.0, 15.0])
        df["NumIsolates"] = None  # all missing
        result = _compute_data_quality_features(df.copy())
        # should not crash — NaN treated as 0, log1p(0) = 0
        assert (result["log_num_isolates"] == 0.0).all()

    def test_years_of_data_increases_over_time(self):
        df = _make_group_df([5.0, 10.0, 15.0, 20.0, 25.0])
        result = _compute_data_quality_features(df.copy())
        years_data = result["years_of_data"].tolist()
        # should be non-decreasing
        assert all(
            years_data[i] <= years_data[i + 1]
            for i in range(len(years_data) - 1)
        )

    def test_years_of_data_first_row_is_zero(self):
        df = _make_group_df([5.0, 10.0, 15.0])
        result = _compute_data_quality_features(df.copy())
        assert result.iloc[0]["years_of_data"] == 0.0


# ---------------------------------------------------------------------------
# Tests for _create_target
# ---------------------------------------------------------------------------

class TestCreateTarget:

    def test_target_column_created(self):
        df = _make_group_df([5.0, 10.0, 15.0, 20.0, 25.0])
        result = _create_target(df.copy(), horizon=2)
        assert "target_pct_resistant_2yr" in result.columns

    def test_target_is_future_value(self):
        df = _make_group_df([5.0, 10.0, 15.0, 20.0, 25.0])
        result = _create_target(df.copy(), horizon=2)
        # row 0 target should be row 2's PctResistant = 15.0
        assert result.iloc[0]["target_pct_resistant_2yr"] == 15.0
        assert result.iloc[1]["target_pct_resistant_2yr"] == 20.0

    def test_last_rows_target_is_nan(self):
        df = _make_group_df([5.0, 10.0, 15.0, 20.0, 25.0])
        result = _create_target(df.copy(), horizon=2)
        # last 2 rows have no future data
        assert pd.isna(result.iloc[-1]["target_pct_resistant_2yr"])
        assert pd.isna(result.iloc[-2]["target_pct_resistant_2yr"])

    def test_target_does_not_bleed_across_groups(self):
        df = _make_multi_country_df()
        df = df.sort_values(["CountryCode", "Pathogen", "Antibiotic", "Year"])
        result = _create_target(df.copy(), horizon=2)
        # last row of Germany should have NaN — not France's future values
        de_last = result[result["CountryCode"] == "DE"].sort_values("Year").iloc[-1]
        assert pd.isna(de_last["target_pct_resistant_2yr"])


# ---------------------------------------------------------------------------
# Tests for build_features (integration)
# ---------------------------------------------------------------------------

class TestBuildFeatures:

    def test_returns_dataframe(self):
        df = _make_multi_country_df()
        result = build_features(df.copy())
        assert isinstance(result, pd.DataFrame)

    def test_all_feature_columns_present(self):
        df = _make_multi_country_df()
        result = build_features(df.copy())
        expected = [
            "lag_1yr", "lag_2yr", "lag_3yr",
            "rolling_mean_3yr", "rolling_mean_5yr",
            "rolling_std_3yr", "rolling_std_5yr",
            "slope_3yr", "slope_5yr",
            "regional_avg_excl_self",
            "log_num_isolates", "years_of_data",
            "target_pct_resistant_2yr",
        ]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_row_count_preserved(self):
        df = _make_multi_country_df()
        result = build_features(df.copy())
        assert len(result) == len(df)

    def test_sort_order_is_correct(self):
        # output should be sorted by country+pathogen+antibiotic+year
        df = _make_multi_country_df()
        result = build_features(df.copy())
        for (code, path, anti), group in result.groupby(["CountryCode", "Pathogen", "Antibiotic"]):
            years = group["Year"].tolist()
            assert years == sorted(years), f"Years not sorted for {code}/{path}/{anti}"
