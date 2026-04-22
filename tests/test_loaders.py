import pytest
import pandas as pd
from pathlib import Path
import tempfile
import os

from src.data.loaders import _parse_ecdc_file, _load_combination, load_ecdc_data


# ---------------------------------------------------------------------------
# Helpers — build minimal fake ECDC CSVs in a temp directory
# ---------------------------------------------------------------------------


def _write_fake_csv(directory: Path, filename: str, rows: list[dict]):
    """Write a minimal ECDC-format CSV to a temp directory."""
    df = pd.DataFrame(rows)
    df.to_csv(directory / filename, index=False)


def _make_fake_ecdc_row(
        region_code: str,
        region_name: str,
        year: int,
        num_value: float,
        indicator: str = "R - resistant isolates",
        unit: str = "N",
) -> dict:
    return {
        "HealthTopic": "Antimicrobial resistance",
        "Population": "Klebsiella pneumoniae|Carbapenems",
        "Indicator": indicator,
        "Unit": unit,
        "Time": year,
        "RegionCode": region_code,
        "RegionName": region_name,
        "NumValue": num_value,
        "TxtValue": "",
    }


# ---------------------------------------------------------------------------
# Tests for _parse_ecdc_file
# ---------------------------------------------------------------------------

class TestParseEcdcFile:

    def test_returns_expected_columns(self, tmp_path):
        rows = [_make_fake_ecdc_row("DE", "Germany", 2020, 5.0)]
        _write_fake_csv(tmp_path, "test.csv", rows)

        result = _parse_ecdc_file(tmp_path / "test.csv")

        assert set(result.columns) == {"Year", "CountryCode", "Country", "Value"}

    def test_year_is_integer(self, tmp_path):
        rows = [_make_fake_ecdc_row("DE", "Germany", 2020, 5.0)]
        _write_fake_csv(tmp_path, "test.csv", rows)

        result = _parse_ecdc_file(tmp_path / "test.csv")

        assert result["Year"].dtype == int

    def test_value_is_float(self, tmp_path):
        # value comes in a string in some CSVs - should be cast to float
        rows = [_make_fake_ecdc_row("DE", "Germany", 2020, "12.500000")]
        _write_fake_csv(tmp_path, "test.csv", rows)

        result = _parse_ecdc_file(tmp_path / "test.csv")

        assert result["Value"].dtype == float

    def test_drops_row_with_missing_num_value(self, tmp_path):
        rows = [
            _make_fake_ecdc_row("DE", "Germany", 2020, 5.0),
            _make_fake_ecdc_row("FR", "France",  2020, None),
        ]
        _write_fake_csv(tmp_path, "test.csv", rows)

        result = _parse_ecdc_file(tmp_path / "test.csv")

        assert len(result) == 1
        assert result.iloc[0]["CountryCode"] == "DE"

    def test_multiple_countries_and_years(self, tmp_path):
        rows = [
            _make_fake_ecdc_row("DE", "Germany", 2018, 2.0),
            _make_fake_ecdc_row("DE", "Germany", 2019, 3.0),
            _make_fake_ecdc_row("FR", "France",  2018, 5.0),
        ]
        _write_fake_csv(tmp_path, "test.csv", rows)

        result = _parse_ecdc_file(tmp_path / "test.csv")

        assert len(result) == 3

    
# ---------------------------------------------------------------------------
# Tests for _load_combination
# ---------------------------------------------------------------------------

class TestLoadCombination:

    def _write_combination(self, tmp_path, key, n_rows, pct_rows):
        _write_fake_csv(tmp_path, f"{key}_N.csv", n_rows)
        _write_fake_csv(tmp_path, f"{key}_pct.csv", pct_rows)

    def test_returns_expected_columns(self, tmp_path):
        n_rows = [_make_fake_ecdc_row("DE", "Germany", 2020, 10.0, unit="N")]
        p_rows = [_make_fake_ecdc_row("DE", "Germany", 2020, 25.0, unit="%")]
        self._write_combination(tmp_path, "TEST_Combo", n_rows, p_rows)

        result = _load_combination(
            tmp_path, "TEST_Combo", "Klebsiella pneumoniae", "Carbapenems"
        )

        expected_cols = {
            "Year", "CountryCode", "Country",
            "Pathogen", "Antibiotic",
            "PctResistant", "NumResistant", "NumIsolates",
        }
        assert set(result.columns) == expected_cols

    def test_pathogen_and_antibiotic_attached(self, tmp_path):
        n_rows = [_make_fake_ecdc_row("DE", "Germany", 2020, 10.0)]
        p_rows = [_make_fake_ecdc_row("DE", "Germany", 2020, 25.0)]
        self._write_combination(tmp_path, "TEST_Combo", n_rows, p_rows)

        result = _load_combination(
            tmp_path, "TEST_Combo", "Klebsiella pneumoniae", "Carbapenems"
        )

        assert result.iloc[0]["Pathogen"]   == "Klebsiella pneumoniae"
        assert result.iloc[0]["Antibiotic"] == "Carbapenems"
    
    def test_num_isolates_derived_correctly(self, tmp_path):
        # 10 resistant out of 25% → total should be 40
        n_rows = [_make_fake_ecdc_row("DE", "Germany", 2020, 10.0)]
        p_rows = [_make_fake_ecdc_row("DE", "Germany", 2020, 25.0)]
        self._write_combination(tmp_path, "TEST_Combo", n_rows, p_rows)

        result = _load_combination(
            tmp_path, "TEST_Combo", "Klebsiella pneumoniae", "Carbapenems"
        )

        assert result.iloc[0]["NumIsolates"] == 40.0

    def test_num_isolates_is_none_when_pct_is_zero(self, tmp_path):
        # 0 resistant, 0% → can't derive total, should be None
        n_rows = [_make_fake_ecdc_row("DE", "Germany", 2020, 0.0)]
        p_rows = [_make_fake_ecdc_row("DE", "Germany", 2020, 0.0)]
        self._write_combination(tmp_path, "TEST_Combo", n_rows, p_rows)

        result = _load_combination(
            tmp_path, "TEST_Combo", "Klebsiella pneumoniae", "Carbapenems"
        )

        assert pd.isna(result.iloc[0]["NumIsolates"])

    def test_returns_none_when_n_file_missing(self, tmp_path):
        # Only write the pct file — N file is missing
        p_rows = [_make_fake_ecdc_row("DE", "Germany", 2020, 25.0)]
        _write_fake_csv(tmp_path, "TEST_Combo_pct.csv", p_rows)

        result = _load_combination(
            tmp_path, "TEST_Combo", "Klebsiella pneumoniae", "Carbapenems"
        )

        assert result is None

    def test_returns_none_when_pct_file_missing(self, tmp_path):
        # Only write the N file — pct file is missing
        n_rows = [_make_fake_ecdc_row("DE", "Germany", 2020, 10.0)]
        _write_fake_csv(tmp_path, "TEST_Combo_N.csv", n_rows)

        result = _load_combination(
            tmp_path, "TEST_Combo", "Klebsiella pneumoniae", "Carbapenems"
        )

        assert result is None

    
# ---------------------------------------------------------------------------
# Tests for load_ecdc_data
# ---------------------------------------------------------------------------

class TestLoadEcdcData:

    def _write_all_combinations(self, tmp_path):
        """Write minimal fake CSVs for all 10 combinations."""
        from src.data.loaders import DATASET_MAP
        for key in DATASET_MAP:
            n_rows = [_make_fake_ecdc_row("DE", "Germany", 2020, 5.0)]
            p_rows = [_make_fake_ecdc_row("DE", "Germany", 2020, 10.0)]
            _write_fake_csv(tmp_path, f"{key}_N.csv",   n_rows)
            _write_fake_csv(tmp_path, f"{key}_pct.csv", p_rows)

    def test_returns_dataframe(self, tmp_path):
        self._write_all_combinations(tmp_path)
        result = load_ecdc_data(str(tmp_path))
        assert isinstance(result, pd.DataFrame)

    def test_has_correct_columns(self, tmp_path):
        self._write_all_combinations(tmp_path)
        result = load_ecdc_data(str(tmp_path))
        expected = {
            "Year", "CountryCode", "Country",
            "Pathogen", "Antibiotic",
            "PctResistant", "NumResistant", "NumIsolates",
        }
        assert set(result.columns) == expected

    def test_all_10_combinations_present(self, tmp_path):
        self._write_all_combinations(tmp_path)
        result = load_ecdc_data(str(tmp_path))
        assert result[["Pathogen", "Antibiotic"]].drop_duplicates().shape[0] == 10

    def test_raises_when_no_files_found(self, tmp_path):
        # Empty directory — should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            load_ecdc_data(str(tmp_path))

    def test_partial_files_still_loads(self, tmp_path):
        # Only write 3 combinations — should load those 3, warn about the rest
        from src.data.loaders import DATASET_MAP
        keys = list(DATASET_MAP.keys())[:3]
        for key in keys:
            n_rows = [_make_fake_ecdc_row("DE", "Germany", 2020, 5.0)]
            p_rows = [_make_fake_ecdc_row("DE", "Germany", 2020, 10.0)]
            _write_fake_csv(tmp_path, f"{key}_N.csv",   n_rows)
            _write_fake_csv(tmp_path, f"{key}_pct.csv", p_rows)

        result = load_ecdc_data(str(tmp_path))
        assert result[["Pathogen", "Antibiotic"]].drop_duplicates().shape[0] == 3
