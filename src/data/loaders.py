import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Each entry maps to (Pathogen, Antibiotic)
# We expect two files per combination: _N.csv and _pct.csv
DATASET_MAP = {
    "KLPN_Carbapenems":      ("Klebsiella pneumoniae", "Carbapenems"),
    "KLPN_Fluoroquinolones": ("Klebsiella pneumoniae", "Fluoroquinolones"),
    "KLPN_3GenCeph":         ("Klebsiella pneumoniae", "3rd gen cephalosporins"),
    "ECOL_Carbapenems":      ("Escherichia coli", "Carbapenems"),
    "ECOL_Fluoroquinolones": ("Escherichia coli", "Fluoroquinolones"),
    "ECOL_3GenCeph":         ("Escherichia coli", "3rd gen cephalosporins"),
    "SAUR_Meticillin":       ("Staphylococcus aureus", "Meticillin"),
    "EFAM_Vancomycin":       ("Enterococcus faecium", "Vancomycin"),
    "PAER_Carbapenems":      ("Pseudomonas aeruginosa", "Carbapenems"),
    "ACIN_Carbapenems":      ("Acinetobacter spp.", "Carbapenems"),
}


def _parse_ecdc_file(filepath: Path) -> pd.DataFrame:
    """
    Parse a single ECDC CSV into a clean, minimal DataFrame.
    Returns: Year, CountryCode, Country, Value
    """
    df = pd.read_csv(filepath)
    df = df[["Time", "RegionCode", "RegionName", "NumValue"]].copy()
    df = df.dropna(subset=["NumValue"])
    df = df.rename(columns={
        "Time": "Year",
        "RegionCode": "CountryCode",
        "RegionName": "Country",
        "NumValue": "Value",
    })
    df["Year"] = df["Year"].astype(int)
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    return df

def _load_combination(raw_dir: Path, key: str, pathogen: str, antibiotic: str):
    """
    Load the _N and _pct files for one pathogen+antibiotic combination, merge them, and derive NumIsolates.
    Returns None if either file is missing.
    """
    path_n = raw_dir / f"{key}_N.csv"
    path_pct = raw_dir / f"{key}_pct.csv"

    missing = [p.name for p in [path_n, path_pct] if not p.exists()]
    if missing:
        logger.warning(f"Skipping {key} — missing files: {missing}")
        return None

    df_n = _parse_ecdc_file(path_n).rename(columns={"Value": "NumResistant"})
    df_pct = _parse_ecdc_file(path_pct).rename(columns={"Value": "PctResistant"})

    # merge on the three identity columns
    df = pd.merge(df_n, df_pct, on=["Year", "CountryCode", "Country"], how="inner")

    # derive total isolates from the 2 values we have
    # guard against division by 0 - if PctResistant is 0, NumIsolates is undefined
    df["NumIsolates"] = df.apply(
        lambda row: round(row["NumResistant"] / row["PctResistant"] * 100)
        if row["PctResistant"] > 0 else None,
        axis=1
    )

    df["Pathogen"] = pathogen
    df["Antibiotic"] = antibiotic

    return df[[
        "Year", "CountryCode", "Country",
        "Pathogen", "Antibiotic",
        "PctResistant", "NumResistant", "NumIsolates",
    ]]


def load_ecdc_data(raw_path: str) -> pd.DataFrame:
    """
    Load and merge all ECDC CSV pairs from raw_path into one clean DataFrame.
    Returns one row per (Year, Country, Pathogen, Antibiotic) combination.
    """
    raw_dir = Path(raw_path)
    frames = []

    for key, (pathogen, antibiotic) in DATASET_MAP.items():
        df = _load_combination(raw_dir, key, pathogen, antibiotic)
        if df is None:
            continue
        logger.info(
            f"Loaded {key}: {len(df)} rows | "
            f"years {df['Year'].min()}–{df['Year'].max()} | "
            f"{df['Country'].nunique()} countries"
        )
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No data loaded from {raw_dir}. "
            f"Expected files like KLPN_Carbapenems_N.csv and KLPN_Carbapenems_pct.csv"
        )

    combined = pd.concat(frames, ignore_index=True)

    logger.info(
        f"\nFinal merged dataset:"
        f"\n  Rows      : {len(combined)}"
        f"\n  Countries : {combined['Country'].nunique()}"
        f"\n  Years     : {combined['Year'].min()}–{combined['Year'].max()}"
        f"\n  Combos    : {combined[['Pathogen','Antibiotic']].drop_duplicates().shape[0]}"
        f"\n  Missing % : PctResistant={combined['PctResistant'].isna().sum()}, "
        f"NumIsolates={combined['NumIsolates'].isna().sum()}"
    )

    return combined
