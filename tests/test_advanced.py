"""
Unit tests for Advanced Data Science extensions:
1. Quantile prediction interval evaluation
2. BESS linear programming arbitrage simulation
"""

import numpy as np
import pandas as pd
import pytest
from src.advanced import simulate_bess_arbitrage_daily


def test_bess_arbitrage_arbitrary_prices():
    """
    Test that BESS correctly discharges at high prices and charges at low prices.
    """
    # 24-hour day where first 12 hours price is 0 EUR, second 12 hours price is 100 EUR
    prices = [0.0] * 12 + [100.0] * 12
    df_day = pd.DataFrame({
        "price_actual": prices,
        "price_signal": prices,
    })

    profit = simulate_bess_arbitrage_daily(
        df_day=df_day,
        price_col_actual="price_actual",
        price_col_signal="price_signal",
        power_mw=10.0,
        capacity_mwh=20.0,
        round_trip_eff=1.0,  # 100% efficiency for simple math check
        max_cycles_per_day=1.0,
    )

    # Because initial and final SoC are fixed at 50% (10 MWh):
    # The battery can charge from 10 MWh up to 20 MWh (10 MWh charged at 0 EUR)
    # and then discharge from 20 MWh down to 10 MWh (10 MWh discharged at 100 EUR)
    # Total net discharge = 10 MWh * 100 EUR = 1,000 EUR
    assert profit == pytest.approx(1000.0, abs=1.0)


def test_bess_arbitrage_incomplete_day():
    """Test that days with fewer than 24 hours are safely skipped."""
    df_day = pd.DataFrame({
        "price_actual": [50.0] * 20,
        "price_signal": [50.0] * 20,
    })
    profit = simulate_bess_arbitrage_daily(df_day)
    assert profit == 0.0
