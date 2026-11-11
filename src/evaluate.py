"""
Evaluation metrics module for electricity price forecasting and negative price classification.
Implements MAE, RMSE, sMAPE, MASE, PR-AUC, and stratified temporal performance analysis.
"""

from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    auc,
    roc_auc_score,
    confusion_matrix
)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Symmetric Mean Absolute Percentage Error (sMAPE):
    sMAPE = (100 / n) * sum( 2 * |y - y_hat| / (|y| + |y_hat| + eps) )
    Handles zero and negative values stably compared to traditional MAPE.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred) + 1e-8
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denom) * 100.0)


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, seasonality: int = 24) -> float:
    """
    Mean Absolute Scaled Error (MASE) relative to naive seasonal benchmark on training data.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_train = np.asarray(y_train)
    mae_model = mean_absolute_error(y_true, y_pred)
    # Seasonal naive MAE on train
    naive_diff = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    mae_naive = np.mean(naive_diff) + 1e-8
    return float(mae_model / mae_naive)


def evaluate_regression_model(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    y_train: Optional[pd.Series] = None
) -> Dict[str, float]:
    """Compute standard regression metrics for price forecasting."""
    mask = ~(pd.isna(y_true) | pd.isna(y_pred))
    yt = y_true[mask].values
    yp = np.asarray(y_pred)[mask]

    mae = float(mean_absolute_error(yt, yp))
    rmse = float(root_mean_squared_error(yt, yp))
    sm = float(smape(yt, yp))

    metrics = {
        "model": model_name,
        "mae": mae,
        "rmse": rmse,
        "smape_pct": sm,
        "n_samples": int(len(yt))
    }
    if y_train is not None:
        metrics["mase"] = float(mase(yt, yp, y_train.dropna().values, seasonality=24))

    return metrics


def evaluate_classification_model(
    y_true: pd.Series,
    y_prob: np.ndarray,
    model_name: str,
    threshold: float = 0.5
) -> Dict:
    """Compute classification metrics for negative price detection."""
    mask = ~(pd.isna(y_true) | pd.isna(y_prob))
    yt = y_true[mask].values.astype(int)
    yp_prob = np.asarray(y_prob)[mask]
    yp_pred = (yp_prob >= threshold).astype(int)

    precision, recall, _ = precision_recall_curve(yt, yp_prob)
    pr_auc = float(auc(recall, precision))
    roc_auc = float(roc_auc_score(yt, yp_prob))

    prec = float(precision_score(yt, yp_pred, zero_division=0))
    rec = float(recall_score(yt, yp_pred, zero_division=0))
    f1 = float(f1_score(yt, yp_pred, zero_division=0))
    cm = confusion_matrix(yt, yp_pred).tolist()

    return {
        "model": model_name,
        "threshold": threshold,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "n_negative_samples": int(np.sum(yt == 1)),
        "n_total_samples": int(len(yt))
    }


def analyze_errors_by_group(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    group_col: str
) -> pd.DataFrame:
    """Group errors (MAE, RMSE, bias) by temporal group (hour, year, weekday)."""
    df = df.copy()
    df["abs_error"] = (df[y_true_col] - df[y_pred_col]).abs()
    df["sq_error"] = (df[y_true_col] - df[y_pred_col]) ** 2
    df["bias"] = df[y_pred_col] - df[y_true_col]

    grouped = df.groupby(group_col).agg(
        mae=("abs_error", "mean"),
        rmse=("sq_error", lambda s: np.sqrt(s.mean())),
        bias=("bias", "mean"),
        count=("abs_error", "count")
    ).reset_index()

    return grouped

