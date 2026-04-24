from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from pathlib import Path
import joblib
import json
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AMR Early Warning API",
    description="""
    Forecasts antibiotic resistance rates 2 years ahead using ECDC EARS-Net surveillance data and a LightGBM model 
    trained on 10 pathogen-antibiotic combinations across 31 European countries (2000-2024).
    
    Built as part of a production-grade Data Science portfolio project.
    """,
    version="1.0.0",
    docs_url="/docs",       # Swagger UI at /docs
    redoc_url="/redoc",     # ReDoc UI at /redoc
)


# ---------------------------------------------------------------------------
# Model loading — happens once at startup, not on every request
# ---------------------------------------------------------------------------

MODEL_DIR = Path("models")

def _load_model_and_metadata():
    """Load the trained model and its metadata from disk."""
    model_path = MODEL_DIR / "lgbm_model.pkl"
    metadata_path = MODEL_DIR / "model_metadata.json"

    if not model_path.exists():
        raise RuntimeError(
            f"Model file not found at {model_path}. "
            f"Run `python -m src.models.train` first."
        )
    if not metadata_path.exists():
        raise RuntimeError(
            f"Metadata file not found at {metadata_path}. "
            f"Run `python -m src.models.train` first."
        )

    model = joblib.load(model_path)
    with open(metadata_path) as f:
        metadata = json.load(f)

    logger.info(
        f"Model loaded: {metadata['model_type']} | "
        f"MAE={metadata['metrics']['mae']}pp | "
        f"R²={metadata['metrics']['r2']}"
    )
    return model, metadata

# load at module level - shared accross all requests
try:
    MODEL, METADATA = _load_model_and_metadata()
except RuntimeError as e:
    logger.error(str(e))
    MODEL, METADATA = None, None


# ---------------------------------------------------------------------------
# Valid values — used for validation and the /combinations endpoint
# ---------------------------------------------------------------------------

VALID_COMBINATIONS = [
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

VALID_COUNTRY_CODES = [
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI",
    "FR", "DE", "EL", "HU", "IS", "IE", "IT", "LV", "LT",
    "LU", "MT", "NL", "NO", "PL", "PT", "RO", "SK", "SI",
    "ES", "SE", "UK", "LI",
]


# ---------------------------------------------------------------------------
# Request and Response models
# ---------------------------------------------------------------------------

class PredictionRequest(BaseModel):
    country_code: str = Field(
        ...,
        example="DE",
        description="2-3 letter ISO country code (e.g. DE, FR, IT)"
    )
    pathogen: str = Field(
        ...,
        example="Klebsiella pneumoniae",
        description="Bacterial species name"
    )
    antibiotic: str = Field(
        ...,
        example="Carbapenems",
        description="Antibiotic group"
    )
    current_resistance_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        example=5.2,
        description="Current observed resistance percentage (0-100)"
    )
    previous_resistance_pct: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        example=4.1,
        description="Resistance percentage from 1 year ago (optional)"
    )
    num_isolates: float | None = Field(
        default=None,
        ge=0,
        example=850,
        description="Number of isolates tested (optional - improves prediction)"
    )
    regional_avg_pct: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        example=8.3,
        description="EU average resistance for this combination (optional)"
    )

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v):
        if v.upper() not in VALID_COUNTRY_CODES:
            raise ValueError(
                f"Unknown country code '{v}'. "
                f"Valid codes: {sorted(VALID_COUNTRY_CODES)}"
            )
        return v.upper()

    @field_validator("pathogen")
    @classmethod
    def validate_pathogen(cls, v):
        valid_pathogens = {p for p, _ in VALID_COMBINATIONS}
        if v not in valid_pathogens:
            raise ValueError(
                f"Unknown pathogen '{v}'. "
                f"Valid pathogens: {sorted(valid_pathogens)}"
            )
        return v

    @field_validator("antibiotic")
    @classmethod
    def validate_combination(cls, v, info):
        if "pathogen" in info.data:
            combo = (info.data["pathogen"], v)
            if combo not in VALID_COMBINATIONS:
                raise ValueError(
                    f"Invalid combination: {combo}. "
                    f"Valid combinations: {VALID_COMBINATIONS}"
                )
        return v


class PredictionResponse(BaseModel):
    country_code:              str
    pathogen:                  str
    antibiotic:                str
    current_resistance_pct:    float
    predicted_resistance_pct:  float
    forecast_horizon_years:    int
    warning_level:             str
    warning_message:           str
    model_mae_pp:              float
    model_r2:                  float


# ---------------------------------------------------------------------------
# Helper — build feature vector from request
# ---------------------------------------------------------------------------

def _build_feature_vector(request: PredictionRequest) -> pd.DataFrame:
    """
    Construct the 12-feature vector the model expects from the API request.

    For features the caller didn't provide, we fall back to the training
    median — this is the same strategy LightGBM uses internally for NaN,
    but we make it explicit so the API is transparent about it.
    """
    medians = METADATA["feature_medians"]

    current = request.current_resistance_pct
    previous = request.previous_resistance_pct

    # lag features
    lag_1yr = previous if previous is not None else medians["lag_1yr"]
    lag_2yr = medians["lag_2yr"]
    lag_3yr = medians["lag_3yr"]

    # rolling features - approximate from what we have
    if previous is not None:
        rolling_mean_3yr = (current + previous) / 2
        rolling_std_3yr = abs(current - previous) / 2
    else:
        rolling_mean_3yr = medians["rolling_mean_3yrs"]
        rolling_std_3yr = medians["rolling_std_3yrs"]

    rolling_mean_5yr = medians["rolling_mean_5yr"]
    rolling_std_5yr = medians["rolling_std_5yr"]

    # slope - approximate from current vs previous
    if previous is not None:
        slope_3yr = current - previous
        slope_5yr = current - previous
    else:
        slope_3yr = medians["slope_3yr"]
        slope_5yr = medians["slope_5yr"]

    # regional context
    regional_avg = (
        request.regional_avg_pct
        if request.regional_avg_pct is not None
        else medians["regional_avg_excl_self"]
    )

    # data quality
    log_isolates = (
        np.log1p(request.num_isolates)
        if request.num_isolates is not None
        else medians["log_num_isolates"]
    )

    years_of_data = medians["years_of_data"]

    features = {
        "lag_1yr":               lag_1yr,
        "lag_2yr":               lag_2yr,
        "lag_3yr":               lag_3yr,
        "rolling_mean_3yr":      rolling_mean_3yr,
        "rolling_mean_5yr":      rolling_mean_5yr,
        "rolling_std_3yr":       rolling_std_3yr,
        "rolling_std_5yr":       rolling_std_5yr,
        "slope_3yr":             slope_3yr,
        "slope_5yr":             slope_5yr,
        "regional_avg_excl_self": regional_avg,
        "log_num_isolates":      log_isolates,
        "years_of_data":         years_of_data,
    }

    return pd.DataFrame([features])

def _get_warning_level(predicted_pct: float) -> tuple[str, str]:
    """
    Classify predicted resistance into a warning level.
    Thresholds based on ECDC public health guidance.
    """
    if predicted_pct < 5:
        return "LOW", "Resistance remains low. Continue standard surveillance."
    elif predicted_pct < 25:
        return "MODERATE", (
            "Moderate resistance detected. Enhanced surveillance recommended."
        )
    elif predicted_pct < 50:
        return "HIGH", (
            "High resistance level. Review treatment guidelines and "
            "strengthen infection control measures."
        )
    else:
        return "CRITICAL", (
            "Critical resistance level. Immediate public health action required. "
            "Consider this antibiotic ineffective for empirical treatment."
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health_check():
    """Check if the API and model are loaded and ready."""
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run training pipeline first."
        )
    return {
        "status": "ok",
        "model_type": METADATA["model_type"],
        "model_mae": METADATA["metrics"]["mae"],
        "model_r2": METADATA["metrics"]["r2"],
        "horizon_yrs": METADATA["forecast_horizon"],
    }

@app.get("/combinations", tags=["Data"])
def get_valid_combinations():
    """List all valid pathogen-antibiotic combinations the model can predict."""
    return {
        "combinations": [
            {"pathogen": p, "antibiotic": a}
            for p, a in VALID_COMBINATIONS
        ],
        "total": len(VALID_COMBINATIONS),
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: PredictionRequest):
    """Forecast antibiotic resistance 2 years ahead for a given country + pathogen + antibiotic combination."""
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run training pipeline first."
        )

    try:
        feature_vector = _build_feature_vector(request)
        prediction     = float(MODEL.predict(feature_vector)[0])

        # Clip to valid range — model can occasionally predict slightly
        # outside [0, 100] due to gradient boosting extrapolation
        prediction = float(np.clip(prediction, 0.0, 100.0))
        prediction = round(prediction, 2)

        warning_level, warning_message = _get_warning_level(prediction)

        return PredictionResponse(
            country_code=             request.country_code,
            pathogen=                 request.pathogen,
            antibiotic=               request.antibiotic,
            current_resistance_pct=   request.current_resistance_pct,
            predicted_resistance_pct= prediction,
            forecast_horizon_years=   METADATA["forecast_horizon"],
            warning_level=            warning_level,
            warning_message=          warning_message,
            model_mae_pp=             METADATA["metrics"]["mae"],
            model_r2=                 METADATA["metrics"]["r2"],
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
