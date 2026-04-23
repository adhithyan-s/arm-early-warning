import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pathlib import Path
import logging
import yaml
import joblib
import json

from src.data.loaders import load_ecdc_data
from src.data.validators import validate_ecdc_data
from src.features.engineer import build_features
from src.models.evaluate import compute_metrics, plot_predictions, plot_residuals

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "lag_1yr",
    "lag_2yr",
    "lag_3yr",
    "rolling_mean_3yr",
    "rolling_mean_5yr",
    "rolling_std_3yr",
    "rolling_std_5yr",
    "slope_3yr",
    "slope_5yr",
    "regional_avg_excl_self",
    "log_num_isolates",
    "years_of_data",
]

TARGET_COL = "target_pct_resistant_2yr"


def temporal_split(
    df: pd.DataFrame,
    cutoff_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train and test sets by year.

    ALL data up to and including cutoff_year -> train
    ALL data after cutoff_year -> test

    Why not random split: this is a time series problem.
    A random split would put 2022 data in training and 2015 data in test — the model would be predicting the past using the future. 
    That's data leakage and produces optimistically wrong evaluation metrics.
    """
    train = df[df["Year"] <= cutoff_year].copy()
    test  = df[df["Year"] >  cutoff_year].copy()

    logger.info(
        f"Temporal split at {cutoff_year}: "
        f"train={len(train)} rows ({train['Year'].min()}–{train['Year'].max()}) | "
        f"test={len(test)} rows ({test['Year'].min()}–{test['Year'].max()})"
    )

    return train, test


def prepare_matrices(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple:
    """
    Extract feature matrices and target vectors from train/test DataFrames.
    Drops rows where the target is NaN — these are the final years of each
    series where the future doesn't exist yet.
    """
    train_valid = train.dropna(subset=[TARGET_COL])
    test_valid  = test.dropna(subset=[TARGET_COL])

    X_train = train_valid[FEATURE_COLS]
    y_train = train_valid[TARGET_COL]
    X_test  = test_valid[FEATURE_COLS]
    y_test  = test_valid[TARGET_COL]

    logger.info(
        f"Training matrix : {X_train.shape} | "
        f"target range [{y_train.min():.1f}, {y_train.max():.1f}]"
    )
    logger.info(
        f"Test matrix     : {X_test.shape} | "
        f"target range [{y_test.min():.1f}, {y_test.max():.1f}]"
    )

    return X_train, y_train, X_test, y_test


def train_baseline(
    X_train, y_train, X_test, y_test,
    figures_dir: str = "reports/figures",
) -> dict:
    """
    Train a Ridge regression baseline.

    Ridge is linear regression with L2 regularisation — it handles
    correlated features (your lag features are highly correlated)
    better than plain linear regression.

    This sets the performance floor. If LightGBM doesn't clearly
    beat this, the extra complexity isn't justified.
    """
    mlflow.set_experiment("amr-resistance-forecasting")

    with mlflow.start_run(run_name="ridge_baseline"):
        params = {"alpha": 1.0}
        mlflow.log_params(params)
        mlflow.log_param("model_type", "Ridge")
        mlflow.log_param("cutoff_year", X_train.index.max() if hasattr(X_train.index, 'max') else "unknown")
        mlflow.log_param("n_features", len(FEATURE_COLS))
        mlflow.log_param("features", FEATURE_COLS)

        # ridge needs scaled features - tree models dont, but linear does
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(**params)),
        ])

        # fill NaN with median for linear model - LightBGM handles NaN natively
        X_train_filled = X_train.fillna(X_train.median())
        X_test_filled  = X_test.fillna(X_train.median())

        model.fit(X_train_filled, y_train)
        preds = model.predict(X_test_filled)

        metrics = compute_metrics(y_test.values, preds)
        mlflow.log_metrics(metrics)

        # log the model artifact
        mlflow.sklearn.log_model(model, "model")

        # save and log plots
        plot_predictions(
            y_test.values, preds,
            title="Ridge Baseline — Predicted vs Actual",
            save_path=f"{figures_dir}/ridge_predictions.png",
        )
        plot_residuals(
            y_test.values, preds,
            title="Ridge Baseline — Residuals",
            save_path=f"{figures_dir}/ridge_residuals.png",
        )
        mlflow.log_artifact(f"{figures_dir}/ridge_predictions.png")
        mlflow.log_artifact(f"{figures_dir}/ridge_residuals.png")

        logger.info(f"Ridge baseline: {metrics}")
        return metrics
    

def train_lightgbm(
        X_train, y_train, X_test, y_test,
    params: dict = None,
    figures_dir: str = "reports/figures",
) -> dict:
    """
    Train a LightGBM gradient boosting model.

    LightGBM handles:
    - Missing values natively (no imputation needed)
    - Non-linear relationships (regime changes in carbapenem resistance)
    - Feature interactions (high regional avg + rising slope = high risk)

    The params dict is logged to MLflow so every run is reproducible.
    """
    mlflow.set_experiment("amr-resistance-forecasting")

    default_params = {
        "n_estimators":    200,
        "learning_rate":   0.05,
        "max_depth":       5,
        "num_leaves":      31,
        "min_child_samples": 20,
        "subsample":       0.8,
        "colsample_bytree": 0.8,
        "random_state":    42,
        "verbose":         -1,
    }

    if params:
        default_params.update(params)

    with mlflow.start_run(run_name="lightgbm"):

        mlflow.log_params(default_params)
        mlflow.log_param("model_type", "LightGBM")
        mlflow.log_param("n_features", len(FEATURE_COLS))
        mlflow.log_param("features", FEATURE_COLS)

        model = lgb.LGBMRegressor(**default_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(period=-1)],
        )

        preds = model.predict(X_test)

        metrics = compute_metrics(y_test.values, preds)
        mlflow.log_metrics(metrics)
        mlflow.log_metric("best_iteration", model.best_iteration_)

        # Feature importance plot
        _plot_feature_importance(
            model, save_path=f"{figures_dir}/lgbm_feature_importance.png"
        )

        mlflow.lightgbm.log_model(model, "model")
        mlflow.log_artifact(f"{figures_dir}/lgbm_feature_importance.png")

        plot_predictions(
            y_test.values, preds,
            title="LightGBM — Predicted vs Actual",
            save_path=f"{figures_dir}/lgbm_predictions.png",
        )
        plot_residuals(
            y_test.values, preds,
            title="LightGBM — Residuals",
            save_path=f"{figures_dir}/lgbm_residuals.png",
        )
        mlflow.log_artifact(f"{figures_dir}/lgbm_predictions.png")
        mlflow.log_artifact(f"{figures_dir}/lgbm_residuals.png")

        logger.info(f"LightGBM: {metrics}")

        # save model to a fixed path so the API can load it
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        joblib.dump(model, model_dir / "lgbm_model.pkl")
        logger.info(f"Model saved to models/lgbm_model.pkl")

        metadata = {
            "feature_cols": FEATURE_COLS,
            "feature_medians": X_train.median().to_dict(),
            "cutoff_year": 2020,
            "forecast_horizon": 2,
            "model_type": "LightGBM",
            "metrics": metrics,
        }
        with open(model_dir / "model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Model metadata saved to models/model_metadata.json")

        return metrics
    

def _plot_feature_importance(model, save_path: str) -> None:
    """Bar chart of LightGBM feature importances."""
    import matplotlib.pyplot as plt

    importance = pd.Series(
        model.feature_importances_,
        index=FEATURE_COLS,
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    importance.plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("LightGBM Feature Importance", fontsize=13, pad=15)
    ax.set_xlabel("Importance Score")
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved feature importance plot to {save_path}")


def run_training_pipeline(
    raw_path: str = "data/raw/",
    cutoff_year: int = 2020,
    figures_dir: str = "reports/figures",
) -> None:
    """
    End-to-end training pipeline:
    load -> validate -> engineer features -> split -> train -> evaluate
    """
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting AMR training pipeline")

    # load and validate
    df = load_ecdc_data(raw_path)
    df = validate_ecdc_data(df)

    # feature engineering
    df = build_features(df, forecast_horizon=2)

    # temporal split
    train, test = temporal_split(df, cutoff_year=cutoff_year)
    X_train, y_train, X_test, y_test = prepare_matrices(train, test)

    # train both models
    logger.info("Training Ridge baseline...")
    ridge_metrics = train_baseline(X_train, y_train, X_test, y_test, figures_dir)

    logger.info("Training LightGBM...")
    lgbm_metrics = train_lightgbm(X_train, y_train, X_test, y_test, figures_dir=figures_dir)

    # summary comparison
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    print(f"{'Metric':<10} {'Ridge':>12} {'LightGBM':>12}")
    print("-"*36)
    for metric in ["mae", "rmse", "r2"]:
        print(
            f"{metric:<10} "
            f"{ridge_metrics[metric]:>12.4f} "
            f"{lgbm_metrics[metric]:>12.4f}"
        )
    print("="*50)


if __name__ == "__main__":
    run_training_pipeline()