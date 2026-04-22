import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check
import logging

logger = logging.getLogger(__name__)

# The exact set of pathogen+antibiotic combinations we expect.
# If a new combination appears or one goes missing, the validator catches it.
EXPECTED_PATHOGENS = {
    "Klebsiella pneumoniae",
    "Escherichia coli",
    "Staphylococcus aureus",
    "Enterococcus faecium",
    "Pseudomonas aeruginosa",
    "Acinetobacter spp.",
}

EXPECTED_COMBINATIONS = {
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
}

# Pandera schema — defines the contract for what valid ECDC data looks like.
# This runs against the merged DataFrame from load_ecdc_data().
ecdc_schema = DataFrameSchema(
    columns={
        "Year": Column(
            int,
            checks=Check.in_range(1999, 2030),
            nullable=False,
        ),
        "CountryCode": Column(
            str,
            checks=Check(
                lambda s: s.str.len().between(2, 3).all(), 
                error="CountryCode must be 2-3 characters"
            ),
            nullable=False,
        ),
        "Country": Column(
            str,
            nullable=False,
        ),
        "Pathogen": Column(
            str,
            checks=Check(
                lambda s: s.isin(EXPECTED_PATHOGENS).all(),
                error=f"Unknown pathogen found. Expected one of: {EXPECTED_PATHOGENS}"
            ),
            nullable=False,
        ),
        "Antibiotic": Column(
            str,
            nullable=False,
        ),
        "PctResistant": Column(
            float,
            checks=Check.in_range(0.0, 100.0),
            nullable=True,
        ),
        "NumResistant": Column(
            float,
            checks=Check.greater_than_or_equal_to(0),
            nullable=False,
        ),
        "NumIsolates": Column(
            float,
            checks=Check.greater_than_or_equal_to(0),
            nullable=True,   # Missing when PctResistant is 0 or NaN
        ),
    },
    checks=[
        # Row-level check: NumResistant can never exceed NumIsolates
        Check(
            lambda df: (
                df[df["NumIsolates"].notna() & df["NumResistant"].notna()]
                .apply(lambda r: r["NumResistant"] <= r["NumIsolates"], axis=1)
                .all()
            ),
            error="NumResistant exceeds NumIsolates in some rows — data integrity issue"
        ),
    ],
    name="ecdc_resistance_data",
)


def validate_ecdc_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate the merged ECDC DataFrame against the expected schema.
    Raises SchemaError if validation fails.
    Returns the validated DataFrame unchanged if all checks pass.
    """
    logger.info("Running schema validation on ECDC data...")

    # 1. Pandera schema check (column types, ranges, nullability)
    validated = ecdc_schema.validate(df, lazy=True)

    # 2. Check all expected combinations are present
    actual_combos = set(
        zip(df["Pathogen"], df["Antibiotic"])
    )
    missing_combos = EXPECTED_COMBINATIONS - actual_combos
    if missing_combos:
        logger.warning(
            f"Missing expected pathogen-antibiotic combinations: {missing_combos}"
        )
    
    # 3. Check minimum row count — if we have fewer than 500 rows something went wrong with loading
    if len(df) < 500:
        raise ValueError(
            f"Dataset has only {len(df)} rows — expected at least 500. "
            f"Check that all CSV files loaded correctly."
        )

    # 4. Log a summary so there's a clear audit trail in logs
    logger.info(
        f"Validation passed:"
        f"\n  Rows            : {len(validated)}"
        f"\n  Countries       : {validated['Country'].nunique()}"
        f"\n  Year range      : {validated['Year'].min()}–{validated['Year'].max()}"
        f"\n  Combinations    : {len(actual_combos)}/10 expected"
        f"\n  PctResistant NaN: {validated['PctResistant'].isna().sum()} "
        f"({validated['PctResistant'].isna().mean()*100:.1f}%)"
    )

    return validated