"""
Data cleaning and preprocessing module for SMARD electricity market data.
Handles European/Berlin daylight saving time transitions, missing values,
German number parsing, hourly aggregation, and feature derivation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import holidays

GERMAN_HOLIDAYS = holidays.Germany(years=range(2018, 2027))


def parse_german_number(val: object) -> float:
    """
    Convert German formatted numbers (e.g. '1.234,56' or '45,20' or '-') to float.
    Returns np.nan for missing, empty, or '-' strings.
    """
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if s in ["-", "", "n/a", "N/A", "null", "None"]:
        return np.nan
    # Remove thousands separators if present, and replace decimal comma with point
    # In Germany: '.' is thousands separator, ',' is decimal separator
    if "." in s and "," in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_smard_timestamps(
    df: pd.DataFrame,
    start_col: str = "Datum von",
    end_col: Optional[str] = "Datum bis"
) -> pd.DataFrame:
    """
    Parse timestamps in Europe/Berlin timezone, accurately handling DST transitions:
    - In March: 23 hours (clock jumps 02:00 -> 03:00)
    - In October: 25 hours (clock repeats 02:00: 2A and 2B)
    Converts and standardizes to UTC datetime index.
    """
    df = df.copy()
    
    # Check column names
    col = None
    for candidate in [start_col, "Datum von", "Date from", "Datum", "Date"]:
        if candidate in df.columns:
            col = candidate
            break
    if col is None:
        raise ValueError(f"Could not find timestamp start column in {df.columns.tolist()}")

    # Parse datetime string 'DD.MM.YYYY HH:MM'
    # We parse as timezone-naive first
    dt_naive = pd.to_datetime(df[col], format="%d.%m.%Y %H:%M", errors="coerce")
    
    # Handle ambiguous / nonexistent timestamps during DST transition in Europe/Berlin
    # 'infer' lets pandas infer DST fold based on sequential ordering
    dt_berlin = dt_naive.dt.tz_localize(
        "Europe/Berlin",
        ambiguous="infer",
        nonexistent="shift_forward"
    )
    
    # Standardize to UTC
    df["timestamp_utc"] = dt_berlin.dt.tz_convert("UTC")
    return df


def aggregate_to_hourly(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp_utc",
    value_cols: Optional[List[str]] = None,
    method: str = "mean"
) -> pd.DataFrame:
    """
    Aggregate quarter-hourly or irregular series to consistent hourly resolution.
    For power (MW) and prices (EUR/MWh), the appropriate aggregation is the mean.
    """
    if value_cols is None:
        value_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != timestamp_col]
        
    df_sorted = df.sort_values(timestamp_col).copy()
    df_hourly = df_sorted.set_index(timestamp_col)[value_cols].resample("1h").agg(method)
    df_hourly = df_hourly.reset_index()
    return df_hourly


def engineer_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived energy metrics and calendar features:
    - wind_total_mw
    - renewable_total_mw
    - residual_load_mw
    - is_negative_price
    - hour, day_of_week, weekday, month, year, is_weekend, is_holiday
    """
    df = df.copy()
    
    # 1. Total Wind
    wind_cols = [c for c in ["wind_onshore_mw", "wind_offshore_mw"] if c in df.columns]
    if wind_cols:
        df["wind_total_mw"] = df[wind_cols].sum(axis=1, min_count=1)

    # 2. Total Renewable
    renew_cols = [c for c in ["wind_onshore_mw", "wind_offshore_mw", "solar_mw", "biomass_mw", "hydro_mw"] if c in df.columns]
    if renew_cols:
        df["renewable_total_mw"] = df[renew_cols].sum(axis=1, min_count=1)

    # 3. Residual Load = load - wind - solar
    if "load_mw" in df.columns and "wind_total_mw" in df.columns and "solar_mw" in df.columns:
        df["residual_load_calc_mw"] = df["load_mw"] - df["wind_total_mw"] - df["solar_mw"]
        if "residual_load_mw" not in df.columns or df["residual_load_mw"].isna().all():
            df["residual_load_mw"] = df["residual_load_calc_mw"]

    # 4. Renewable Share of Load
    if "renewable_total_mw" in df.columns and "load_mw" in df.columns:
        df["renewable_share_load"] = (df["renewable_total_mw"] / df["load_mw"].replace(0, np.nan)).clip(0, 5)

    # 5. Negative Price Indicator
    if "price_eur_mwh" in df.columns:
        df["is_negative_price"] = (df["price_eur_mwh"] < 0).astype(int)

    # 6. Calendar & Holiday Features (based on local Berlin time for realistic German calendar)
    ts = pd.to_datetime(df["timestamp_utc"], utc=True)
    ts_berlin = ts.dt.tz_convert("Europe/Berlin")

    df["hour"] = ts_berlin.dt.hour
    df["day_of_week"] = ts_berlin.dt.dayofweek
    df["weekday_name"] = ts_berlin.dt.day_name()
    df["is_weekend"] = (ts_berlin.dt.dayofweek >= 5).astype(int)
    df["month"] = ts_berlin.dt.month
    df["quarter"] = ts_berlin.dt.quarter
    df["year"] = ts_berlin.dt.year

    # German national public holidays
    berlin_dates = ts_berlin.dt.date
    df["is_holiday"] = berlin_dates.apply(lambda d: int(d in GERMAN_HOLIDAYS))
    df["is_workday"] = ((df["is_weekend"] == 0) & (df["is_holiday"] == 0)).astype(int)

    return df

