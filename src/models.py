"""
Forecasting models for day-ahead electricity prices and negative-price classification.
Implements Naive baselines, Ridge regression, LightGBM, and XGBoost with
strict time-series cross-validation and SHAP interpretability.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import xgboost as xgb
import shap


class NaiveLag24Model:
    """Predicts day-ahead price as price at same hour yesterday (lag 24)."""
    def fit(self, X: pd.DataFrame, y: pd.Series):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if "price_lag_24" in X.columns:
            return X["price_lag_24"].values
        raise ValueError("Feature 'price_lag_24' required for NaiveLag24Model")


class SeasonalNaiveLag168Model:
    """Predicts day-ahead price as price at same hour last week (lag 168)."""
    def fit(self, X: pd.DataFrame, y: pd.Series):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if "price_lag_168" in X.columns:
            return X["price_lag_168"].values
        raise ValueError("Feature 'price_lag_168' required for SeasonalNaiveLag168Model")


def train_ridge_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    alpha: float = 100.0
) -> Pipeline:
    """Train Ridge regression with standardized features."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha, random_state=42))
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_lgbm_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Optional[Dict] = None
) -> lgb.LGBMRegressor:
    """Train LightGBM Regressor for continuous day-ahead price forecasting."""
    default_params = {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1
    }
    if params:
        default_params.update(params)
        
    model = lgb.LGBMRegressor(**default_params)
    model.fit(X_train, y_train)
    return model


def train_xgb_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Optional[Dict] = None
) -> xgb.XGBRegressor:
    """Train XGBoost Regressor for continuous day-ahead price forecasting."""
    default_params = {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1
    }
    if params:
        default_params.update(params)
        
    model = xgb.XGBRegressor(**default_params)
    model.fit(X_train, y_train)
    return model


def train_negative_price_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    pos_weight: float = 5.0
) -> lgb.LGBMClassifier:
    """
    Train LightGBM Classifier to predict binary occurrence of negative prices (price < 0).
    Uses scale_pos_weight to address class imbalance.
    """
    clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        scale_pos_weight=pos_weight,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    return clf


def compute_shap_explanations(
    model: Any,
    X_sample: pd.DataFrame
) -> Tuple[shap.Explanation, np.ndarray]:
    """Compute SHAP values using TreeExplainer for tree models."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    return shap_values, explainer.expected_value

