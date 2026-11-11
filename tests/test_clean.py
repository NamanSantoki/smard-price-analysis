"""
Unit tests for timestamp handling, number parsing, aggregation,
negative-price flag, and train/test split integrity.
"""

import numpy as np
import pandas as pd
import pytest
from src.clean import (
    parse_german_number,
    parse_smard_timestamps,
    aggregate_to_hourly,
    engineer_derived_features,
)


# -----------------------------------------------------------
# 1. German Number Parsing
# -----------------------------------------------------------
class TestParseGermanNumber:
    def test_simple_decimal(self):
        assert parse_german_number("12,34") == 12.34

    def test_thousands_separator(self):
        assert parse_german_number("1.234,56") == 1234.56

    def test_negative_value(self):
        assert parse_german_number("-45,5") == -45.5

    def test_dash_is_nan(self):
        assert np.isnan(parse_german_number("-"))

    def test_empty_string_is_nan(self):
        assert np.isnan(parse_german_number(""))

    def test_na_strings(self):
        for val in ["n/a", "N/A", "null", "None"]:
            assert np.isnan(parse_german_number(val))

    def test_none_is_nan(self):
        assert np.isnan(parse_german_number(None))

    def test_float_passthrough(self):
        assert parse_german_number(42.0) == 42.0

    def test_integer_passthrough(self):
        assert parse_german_number(7) == 7.0


# -----------------------------------------------------------
# 2. Daylight Saving Time Transitions
# -----------------------------------------------------------
class TestDSTHandling:
    def test_october_repeated_hour(self):
        """October DST transition: 02:00 CEST repeated as 02:00 CET. Must map to two distinct UTC times."""
        dates = [
            "29.10.2023 00:00",
            "29.10.2023 01:00",
            "29.10.2023 02:00",  # first occurrence (CEST)
            "29.10.2023 02:00",  # second occurrence (CET)
            "29.10.2023 03:00",
        ]
        df = pd.DataFrame({"Datum von": dates})
        result = parse_smard_timestamps(df)
        utc_times = result["timestamp_utc"].tolist()
        # The two 02:00 entries must differ by exactly 1 hour in UTC
        assert utc_times[2] < utc_times[3]
        assert (utc_times[3] - utc_times[2]).total_seconds() == 3600

    def test_march_skipped_hour(self):
        """March DST transition: 02:00 does not exist (clock jumps 01:59 → 03:00)."""
        dates = [
            "26.03.2023 00:00",
            "26.03.2023 01:00",
            "26.03.2023 03:00",  # 02:00 does not exist
        ]
        df = pd.DataFrame({"Datum von": dates})
        result = parse_smard_timestamps(df)
        assert len(result) == 3
        assert not result["timestamp_utc"].isna().any()

    def test_utc_monotonically_increasing(self):
        """Any valid hourly sequence should produce strictly increasing UTC timestamps."""
        dates = [f"15.06.2023 {h:02d}:00" for h in range(24)]
        df = pd.DataFrame({"Datum von": dates})
        result = parse_smard_timestamps(df)
        diffs = result["timestamp_utc"].diff().dropna()
        assert (diffs == pd.Timedelta(hours=1)).all()


# -----------------------------------------------------------
# 3. Hourly Aggregation
# -----------------------------------------------------------
class TestAggregation:
    def test_quarterhourly_mean(self):
        """Four quarter-hourly values should aggregate to the mean for each hour."""
        times = pd.date_range("2023-01-01 00:00", periods=8, freq="15min", tz="UTC")
        values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
        df = pd.DataFrame({"timestamp_utc": times, "price_eur_mwh": values})
        result = aggregate_to_hourly(df, "timestamp_utc", ["price_eur_mwh"])
        assert len(result) == 2
        assert result.loc[0, "price_eur_mwh"] == 25.0  # mean(10,20,30,40)
        assert result.loc[1, "price_eur_mwh"] == 65.0  # mean(50,60,70,80)

    def test_already_hourly_passthrough(self):
        """Already-hourly data should pass through unchanged."""
        times = pd.date_range("2023-01-01", periods=3, freq="1h", tz="UTC")
        df = pd.DataFrame({"timestamp_utc": times, "val": [1.0, 2.0, 3.0]})
        result = aggregate_to_hourly(df, "timestamp_utc", ["val"])
        assert len(result) == 3
        assert list(result["val"]) == [1.0, 2.0, 3.0]


# -----------------------------------------------------------
# 4. Derived Feature Engineering
# -----------------------------------------------------------
class TestDerivedFeatures:
    def test_negative_price_flag(self):
        df = pd.DataFrame({
            "timestamp_utc": pd.to_datetime(["2023-01-01 00:00", "2023-01-01 01:00"], utc=True),
            "price_eur_mwh": [-15.5, 30.0],
            "wind_onshore_mw": [1000.0, 2000.0],
            "wind_offshore_mw": [500.0, 500.0],
            "solar_mw": [0.0, 100.0],
            "load_mw": [3000.0, 4000.0],
        })
        result = engineer_derived_features(df)
        assert result.loc[0, "is_negative_price"] == 1
        assert result.loc[1, "is_negative_price"] == 0

    def test_wind_total(self):
        df = pd.DataFrame({
            "timestamp_utc": pd.to_datetime(["2023-06-15 12:00"], utc=True),
            "price_eur_mwh": [50.0],
            "wind_onshore_mw": [8000.0],
            "wind_offshore_mw": [3000.0],
            "solar_mw": [20000.0],
            "load_mw": [45000.0],
        })
        result = engineer_derived_features(df)
        assert result.loc[0, "wind_total_mw"] == 11000.0

    def test_residual_load_calculation(self):
        df = pd.DataFrame({
            "timestamp_utc": pd.to_datetime(["2023-01-01 00:00"], utc=True),
            "price_eur_mwh": [50.0],
            "wind_onshore_mw": [5000.0],
            "wind_offshore_mw": [2000.0],
            "solar_mw": [1000.0],
            "load_mw": [40000.0],
        })
        result = engineer_derived_features(df)
        # residual = 40000 - 7000 - 1000 = 32000
        assert result.loc[0, "residual_load_mw"] == 32000.0

    def test_new_years_day_is_holiday(self):
        df = pd.DataFrame({
            "timestamp_utc": pd.to_datetime(["2023-01-01 12:00"], utc=True),
            "price_eur_mwh": [50.0],
            "wind_onshore_mw": [1000.0],
            "wind_offshore_mw": [500.0],
            "solar_mw": [0.0],
            "load_mw": [3000.0],
        })
        result = engineer_derived_features(df)
        assert result.loc[0, "is_holiday"] == 1
        assert result.loc[0, "is_workday"] == 0

    def test_regular_tuesday_is_workday(self):
        # 2023-01-03 is a Tuesday, not a holiday
        df = pd.DataFrame({
            "timestamp_utc": pd.to_datetime(["2023-01-03 12:00"], utc=True),
            "price_eur_mwh": [50.0],
            "wind_onshore_mw": [1000.0],
            "wind_offshore_mw": [500.0],
            "solar_mw": [0.0],
            "load_mw": [3000.0],
        })
        result = engineer_derived_features(df)
        assert result.loc[0, "is_workday"] == 1
        assert result.loc[0, "is_weekend"] == 0


# -----------------------------------------------------------
# 5. Train/Test Split Integrity (No Temporal Leakage)
# -----------------------------------------------------------
class TestTrainTestSplit:
    def test_no_temporal_overlap(self):
        """Train set must end strictly before test set begins."""
        df = pd.read_parquet("data/processed/merged_hourly.parquet")
        cutoff = df["timestamp_utc"].max() - pd.DateOffset(months=12)
        train = df[df["timestamp_utc"] < cutoff]
        test = df[df["timestamp_utc"] >= cutoff]
        assert train["timestamp_utc"].max() < test["timestamp_utc"].min()

    def test_no_shared_timestamps(self):
        """No timestamp should appear in both train and test."""
        df = pd.read_parquet("data/processed/merged_hourly.parquet")
        cutoff = df["timestamp_utc"].max() - pd.DateOffset(months=12)
        train_ts = set(df[df["timestamp_utc"] < cutoff]["timestamp_utc"])
        test_ts = set(df[df["timestamp_utc"] >= cutoff]["timestamp_utc"])
        assert len(train_ts & test_ts) == 0

    def test_train_before_test(self):
        """Every train timestamp must precede every test timestamp."""
        df = pd.read_parquet("data/processed/merged_hourly.parquet")
        cutoff = df["timestamp_utc"].max() - pd.DateOffset(months=12)
        train_max = df[df["timestamp_utc"] < cutoff]["timestamp_utc"].max()
        test_min = df[df["timestamp_utc"] >= cutoff]["timestamp_utc"].min()
        assert train_max < test_min
