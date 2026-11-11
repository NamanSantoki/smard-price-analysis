"""
Feature engineering module for day-ahead electricity price forecasting.
Ensures zero data leakage: all historical price features are lagged by at least 24h.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd


def create_forecasting_features(
    df: pd.DataFrame,
    include_realised_drivers: bool = True
) -> pd.DataFrame:
    """
    Construct leakage-free feature matrix for day-ahead price forecasting.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Hourly cleaned SMARD dataset sorted by timestamp_utc.
    include_realised_drivers : bool
        If True, includes contemporaneous generation/load as an 'upper bound'
        proxy for perfect day-ahead weather/load forecasts.
        If False, uses only lagged drivers (known prior to day-ahead auction).
    
    Returns:
    --------
    pd.DataFrame with engineered features and target columns.
    """
    data = df.sort_values("timestamp_utc").copy()
    
    # 1. Target variables
    # price_eur_mwh (continuous)
    # is_negative_price (binary classification target)
    if "is_negative_price" not in data.columns and "price_eur_mwh" in data.columns:
        data["is_negative_price"] = (data["price_eur_mwh"] < 0).astype(int)

    # 2. Strict Price Lags (>= 24h to prevent look-ahead bias)
    # Day-ahead auction occurs at 12:00 CET for delivery hours 00-24 of next day
    data["price_lag_24"] = data["price_eur_mwh"].shift(24)    # Same hour yesterday
    data["price_lag_48"] = data["price_eur_mwh"].shift(48)    # Same hour 2 days ago
    data["price_lag_168"] = data["price_eur_mwh"].shift(168)  # Same hour last week (seasonal baseline)
    
    # Short-term lag prior to auction (e.g. morning price lag 24h)
    data["price_roll_mean_24h"] = data["price_lag_24"].rolling(window=24, min_periods=12).mean()
    data["price_roll_std_24h"] = data["price_lag_24"].rolling(window=24, min_periods=12).std()
    data["price_roll_mean_7d"] = data["price_lag_24"].rolling(window=168, min_periods=48).mean()
    
    # 3. Cyclical time encodings
    data["sin_hour"] = np.sin(2 * np.pi * data["hour"] / 24.0)
    data["cos_hour"] = np.cos(2 * np.pi * data["hour"] / 24.0)
    data["sin_month"] = np.sin(2 * np.pi * data["month"] / 12.0)
    data["cos_month"] = np.cos(2 * np.pi * data["month"] / 12.0)
    data["sin_day_of_week"] = np.sin(2 * np.pi * data["day_of_week"] / 7.0)
    data["cos_day_of_week"] = np.cos(2 * np.pi * data["day_of_week"] / 7.0)

    # 4. Generation & Load features
    if include_realised_drivers:
        # Labeled as "upper bound" benchmark (substitute for day-ahead TSO forecasts)
        driver_cols = [
            "wind_total_mw", "solar_mw", "residual_load_mw", "load_mw",
            "wind_onshore_mw", "wind_offshore_mw", "gas_mw", "hard_coal_mw", "lignite_mw"
        ]
        for col in driver_cols:
            if col in data.columns:
                data[f"{col}_driver"] = data[col]
    else:
        # Purely historical lagged drivers (24h and 168h lags)
        for col in ["wind_total_mw", "solar_mw", "residual_load_mw", "load_mw"]:
            if col in data.columns:
                data[f"{col}_lag_24"] = data[col].shift(24)
                data[f"{col}_lag_168"] = data[col].shift(168)

    return data


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return list of predictor column names excluding targets and raw timestamps."""
    exclude = {
        "timestamp_utc", "timestamp_ms", "Datum von", "Datum bis",
        "price_eur_mwh", "is_negative_price", "weekday_name"
    }
    # Also exclude raw contemporaneous unlagged generation unless named _driver
    raw_unlagged = {
        "wind_onshore_mw", "wind_offshore_mw", "wind_total_mw", "solar_mw",
        "load_mw", "residual_load_mw", "biomass_mw", "nuclear_mw", "hydro_mw",
        "hard_coal_mw", "lignite_mw", "gas_mw", "pumped_storage_mw",
        "renewable_total_mw", "residual_load_calc_mw", "renewable_share_load"
    }
    feature_cols = [
        c for c in df.columns
        if c not in exclude and c not in raw_unlagged and not c.startswith("Unnamed")
    ]
    return feature_cols

