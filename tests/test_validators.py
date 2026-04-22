import pytest
import pandas as pd
import pandera.pandas as pa

from src.data.validators import validate_ecdc_data


# ---------------------------------------------------------------------------
# Helper — builds a minimal valid DataFrame to test against
# ---------------------------------------------------------------------------

def _make_valid_df(n_rows: int = None) -> pd.DataFrame:
    """
    Returns a DataFrame that passes all validation checks.
    Generates enough rows (500+) by default to satisfy the minimum row count.
    Pass n_rows explicitly only when testing the minimum row count check itself.
    """
    if n_rows is not None:
        # Used specifically for testing the too-few-rows case
        return pd.DataFrame({
            "Year":         [2020] * n_rows,
            "CountryCode":  ["DE"] * n_rows,
            "Country":      ["Germany"] * n_rows,
            "Pathogen":     ["Klebsiella pneumoniae"] * n_rows,
            "Antibiotic":   ["Carbapenems"] * n_rows,
            "PctResistant": [5.0] * n_rows,
            "NumResistant": [10.0] * n_rows,
            "NumIsolates":  [200.0] * n_rows,
        })

    # Build a realistic dataset covering all 10 combinations, 25 countries, and 3 years
    import itertools
    combos = [
        ("Klebsiella pneumoniae",  "Carbapenems"),
        ("Klebsiella pneumoniae",  "Fluoroquinolones"),
        ("Klebsiella pneumoniae",  "3rd gen cephalosporins"),
        ("Escherichia coli",       "Carbapenems"),
        ("Escherichia coli",       "Fluoroquinolones"),
        ("Escherichia coli",       "3rd gen cephalosporins"),
        ("Staphylococcus aureus",  "Meticillin"),
        ("Enterococcus faecium",   "Vancomycin"),
        ("Pseudomonas aeruginosa", "Carbapenems"),
        ("Acinetobacter spp.",     "Carbapenems"),
    ]
    countries = [
        ("DE", "Germany"), ("FR", "France"), ("IT", "Italy"),
        ("ES", "Spain"),   ("PL", "Poland"), ("GR", "Greece"),
        ("NL", "Netherlands"), ("BE", "Belgium"), ("SE", "Sweden"),
        ("AT", "Austria"),
    ]
    years = [2018, 2019, 2020, 2021, 2022, 2023]

    rows = []
    for (pathogen, antibiotic), (code, country), year in itertools.product(
        combos, countries, years
    ):
        rows.append({
            "Year":         year,
            "CountryCode":  code,
            "Country":      country,
            "Pathogen":     pathogen,
            "Antibiotic":   antibiotic,
            "PctResistant": 5.0,
            "NumResistant": 10.0,
            "NumIsolates":  200.0,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests — valid data passes
# ---------------------------------------------------------------------------

class TestValidDataPasses:

    def test_valid_dataframe_passes(self):
        df = _make_valid_df()
        result = validate_ecdc_data(df)
        assert result is not None
        assert len(result) == 600

    def test_missing_pct_resistant_is_allowed(self):
        # PctResistant is nullable — missing values are expected
        df = _make_valid_df()
        df.loc[0, "PctResistant"] = None
        result = validate_ecdc_data(df)
        assert pd.isna(result.iloc[0]["PctResistant"])

    def test_missing_num_isolates_is_allowed(self):
        # NumIsolates is nullable — missing when PctResistant is 0
        df = _make_valid_df()
        df.loc[0, "NumIsolates"] = None
        result = validate_ecdc_data(df)
        assert pd.isna(result.iloc[0]["NumIsolates"])


# ---------------------------------------------------------------------------
# Tests — invalid data fails
# ---------------------------------------------------------------------------

class TestInvalidDataFails:

    def test_pct_resistant_above_100_fails(self):
        df = _make_valid_df()
        df.loc[0, "PctResistant"] = 150.0  # impossible percentage
        with pytest.raises(Exception):
            validate_ecdc_data(df)

    def test_negative_pct_resistant_fails(self):
        df = _make_valid_df()
        df.loc[0, "PctResistant"] = -5.0
        with pytest.raises(Exception):
            validate_ecdc_data(df)

    def test_negative_num_resistant_fails(self):
        df = _make_valid_df()
        df.loc[0, "NumResistant"] = -1.0
        with pytest.raises(Exception):
            validate_ecdc_data(df)

    def test_unknown_pathogen_fails(self):
        df = _make_valid_df()
        df.loc[0, "Pathogen"] = "Unknown bacteria"
        with pytest.raises(Exception):
            validate_ecdc_data(df)

    def test_year_out_of_range_fails(self):
        df = _make_valid_df()
        df.loc[0, "Year"] = 1850  # clearly wrong
        with pytest.raises(Exception):
            validate_ecdc_data(df)

    def test_num_resistant_exceeds_num_isolates_fails(self):
        df = _make_valid_df()
        # 500 resistant out of only 200 total — physically impossible
        df.loc[0, "NumResistant"] = 500.0
        df.loc[0, "NumIsolates"]  = 200.0
        with pytest.raises(Exception):
            validate_ecdc_data(df)

    def test_too_few_rows_fails(self):
        # Only 3 rows — below our minimum threshold of 500
        df = _make_valid_df(n_rows=3)
        with pytest.raises(ValueError, match="expected at least 500"):
            validate_ecdc_data(df)