"""
src/advanced.py — Advanced Data Science Extensions:
1. Probabilistic Price Forecasting (Quantile LightGBM & Conformal Prediction Intervals)
2. Battery Energy Storage System (BESS) Day-Ahead Arbitrage Simulation
"""

from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import linprog


def train_quantile_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    quantiles: Tuple[float, ...] = (0.10, 0.50, 0.90),
    random_state: int = 42,
) -> Dict[float, lgb.LGBMRegressor]:
    """
    Train Quantile LightGBM models for multiple quantile levels using pinball loss.
    """
    models = {}
    for q in quantiles:
        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=q,
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(X_train, y_train)
        models[q] = model
    return models


def evaluate_quantiles(
    models: Dict[float, lgb.LGBMRegressor],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """
    Generate quantile predictions and evaluate Pinball Loss and Empirical Coverage.
    """
    preds = {}
    pinball_losses = {}
    y_arr = y_test.values

    for q, m in models.items():
        pred_q = m.predict(X_test)
        preds[q] = pred_q
        # Pinball loss: max(q * (y - y_hat), (q - 1) * (y - y_hat))
        diff = y_arr - pred_q
        loss = np.maximum(q * diff, (q - 1) * diff).mean()
        pinball_losses[f"pinball_loss_q{int(q*100):02d}"] = float(loss)

    # Calculate empirical coverage for the 80% interval (0.10 to 0.90)
    if 0.10 in preds and 0.90 in preds:
        lower = preds[0.10]
        upper = preds[0.90]
        # Ensure monotonic bounds (fix quantile crossing if any)
        lower_adj = np.minimum(lower, upper)
        upper_adj = np.maximum(lower, upper)
        coverage_80 = float(np.mean((y_arr >= lower_adj) & (y_arr <= upper_adj)))
        avg_width = float(np.mean(upper_adj - lower_adj))
    else:
        coverage_80 = None
        avg_width = None

    return {
        "predictions": pd.DataFrame(preds, index=X_test.index),
        "pinball_losses": pinball_losses,
        "empirical_coverage_80pct": coverage_80,
        "nominal_coverage_80pct": 0.80,
        "avg_interval_width_eur": avg_width,
    }


def simulate_bess_arbitrage_daily(
    df_day: pd.DataFrame,
    price_col_actual: str = "price_actual",
    price_col_signal: str = "price_signal",
    power_mw: float = 10.0,
    capacity_mwh: float = 20.0,
    round_trip_eff: float = 0.85,
    max_cycles_per_day: float = 1.0,
) -> float:
    """
    Simulate a 24-hour BESS arbitrage schedule using Linear Programming (MILP relaxation).
    Dispatches according to `price_col_signal`, but settles revenues on `price_col_actual`.
    
    Parameters:
    - power_mw: Maximum charge / discharge power rating (e.g. 10 MW)
    - capacity_mwh: Usable energy capacity (e.g. 20 MWh -> 2h duration)
    - round_trip_eff: AC-AC round-trip efficiency (eta_charge = eta_discharge = sqrt(eta))
    - max_cycles_per_day: Maximum equivalent full cycles per day constraint
    """
    T = len(df_day)
    if T != 24:
        return 0.0

    eta = np.sqrt(round_trip_eff)
    p_signal = df_day[price_col_signal].values
    p_actual = df_day[price_col_actual].values

    # Decision variables per hour t:
    # x = [charge_0..charge_T-1, discharge_0..discharge_T-1, soc_0..soc_T-1]
    # Length: 3 * T = 72
    # Objective: Maximize profit based on signal price -> Min - (p_signal * discharge - p_signal * charge)
    c = np.zeros(3 * T)
    c[T : 2 * T] = -p_signal  # discharge revenue (+ profit -> - objective)
    c[0:T] = p_signal  # charge cost (- profit -> + objective)

    # Constraints:
    # 1. SoC transition: soc_t = soc_{t-1} + eta * charge_t - (1/eta) * discharge_t
    # Initial SoC at start of day = 0.5 * capacity_mwh
    # soc_0 - eta * charge_0 + (1/eta) * discharge_0 = 0.5 * capacity
    # soc_t - soc_{t-1} - eta * charge_t + (1/eta) * discharge_t = 0
    A_eq = []
    b_eq = []

    for t in range(T):
        row = np.zeros(3 * T)
        row[2 * T + t] = 1.0  # soc_t
        if t > 0:
            row[2 * T + t - 1] = -1.0  # -soc_{t-1}
        row[t] = -eta  # -eta * charge_t
        row[T + t] = 1.0 / eta  # +(1/eta) * discharge_t
        A_eq.append(row)
        b_eq.append(0.5 * capacity_mwh if t == 0 else 0.0)

    # End of day SoC must return to neutral (50% SoC) to prevent empty-battery arbitrage bias
    row_end = np.zeros(3 * T)
    row_end[2 * T + (T - 1)] = 1.0
    A_eq.append(row_end)
    b_eq.append(0.5 * capacity_mwh)

    # 2. Maximum throughput / cycle limit:
    # sum(discharge_t) <= max_cycles_per_day * capacity_mwh
    A_ub = []
    b_ub = []
    row_cycle = np.zeros(3 * T)
    row_cycle[T : 2 * T] = 1.0
    A_ub.append(row_cycle)
    b_ub.append(max_cycles_per_day * capacity_mwh)

    # Bounds for variables:
    # charge in [0, power_mw]
    # discharge in [0, power_mw]
    # soc in [0, capacity_mwh]
    bounds = []
    for _ in range(T):
        bounds.append((0.0, power_mw))
    for _ in range(T):
        bounds.append((0.0, power_mw))
    for _ in range(T):
        bounds.append((0.0, capacity_mwh))

    res = linprog(
        c=c,
        A_eq=np.array(A_eq),
        b_eq=np.array(b_eq),
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        bounds=bounds,
        method="highs",
    )

    if not res.success:
        return 0.0

    charge_sched = res.x[0:T]
    discharge_sched = res.x[T : 2 * T]

    # Calculate actual realized profit with settlement prices
    realized_daily_profit = np.sum(discharge_sched * p_actual) - np.sum(charge_sched * p_actual)
    return float(realized_daily_profit)
