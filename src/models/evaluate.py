import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute regression metrics for model evaluation.

    MAE  — average absolute error in percentage points.
           Most interpretable: "on average, our forecast is X pp off"
    RMSE — penalises large errors more than MAE.
           Important here because a 20pp miss on carbapenems is
           far worse than a 2pp miss.
    R²   — proportion of variance explained.
           0 = no better than predicting the mean,
           1 = perfect predictions.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # only evaluate on rows where true value exists
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        logger.warning("No valid rows to evaluate — all targets were NaN")
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "n_samples": 0}

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    metrics = {
        "mae": round(mae,  4),
        "rmse": round(rmse, 4),
        "r2": round(r2,   4),
        "n_samples": int(len(y_true)),
    }

    logger.info(
        f"Metrics — MAE={mae:.3f}pp | RMSE={rmse:.3f}pp | "
        f"R²={r2:.3f} | n={len(y_true)}"
    )

    return metrics


def plot_predictions(
    y_true: np.array,
    y_pred: np.array,
    title: str = "Predicted vs Actual Resistance",
    save_path: str = None,
) -> None:
    """
    Scatter plot of predicted vs actual resistance rates.
    Points near the diagonal = good predictions.
    """
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = np.array(y_true)[mask]
    y_pred = np.array(y_pred)[mask]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.4, s=20, color="steelblue", edgecolors="none")

    # perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Perfect prediction")

    ax.set_xlabel("Actual % Resistant", fontsize=12)
    ax.set_ylabel("Predicted % Resistant", fontsize=12)
    ax.set_title(title, fontsize=13, pad=15)
    ax.legend()

    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    ax.text(
        0.05, 0.92,
        f"MAE={mae:.2f}pp  R²={r2:.3f}  n={len(y_true)}",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8)
    )

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved prediction plot to {save_path}")

    plt.show()
    plt.close()


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residuals",
    save_path: str = None,
) -> None:
    """
    Residual plot: predicted value vs error.
    A good model has residuals randomly scattered around 0.
    Patterns in residuals = the model is missing something systematic.
    """
    mask      = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true    = np.array(y_true)[mask]
    y_pred    = np.array(y_pred)[mask]
    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(y_pred, residuals, alpha=0.4, s=20,
               color="steelblue", edgecolors="none")
    ax.axhline(0, color="red", linewidth=1.5, linestyle="--")

    ax.set_xlabel("Predicted % Resistant", fontsize=12)
    ax.set_ylabel("Residual (Actual − Predicted)", fontsize=12)
    ax.set_title(title, fontsize=13, pad=15)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved residual plot to {save_path}")

    plt.show()
    plt.close()
