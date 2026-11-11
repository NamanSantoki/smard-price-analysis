"""
scripts/run_advanced.py — Execute Advanced Data Science Extensions:
1. Probabilistic Price Forecasting (Quantile LightGBM)
2. Empirical Coverage & Pinball Loss
3. Battery Energy Storage System (BESS) 10 MW / 20 MWh Arbitrage Simulation
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.features import create_forecasting_features, get_feature_columns
from src.advanced import (
    train_quantile_lgbm,
    evaluate_quantiles,
    simulate_bess_arbitrage_daily,
)


def main():
    print("=" * 70)
    print("  RUNNING ADVANCED DATA SCIENCE EXTENSIONS")
    print("=" * 70)

    # 1. Load merged dataset
    parquet_path = ROOT_DIR / "data/processed/merged_hourly.parquet"
    if not parquet_path.exists():
        print(f"Error: {parquet_path} not found.")
        sys.exit(1)

    print("Loading processed dataset...")
    df = pd.read_parquet(parquet_path)
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    # 2. Build leakage-free forecasting features
    print("Building lag, rolling and calendar features...")
    df_feat = create_forecasting_features(df, include_realised_drivers=True)
    df_feat = df_feat.dropna().reset_index(drop=True)

    feature_cols = get_feature_columns(df_feat)
    target_col = "price_eur_mwh"

    # Strict chronological train/test split (last 365 days / 8,760 hours test)
    TEST_HOURS = 8760
    train_df = df_feat.iloc[:-TEST_HOURS].copy()
    test_df = df_feat.iloc[-TEST_HOURS:].copy()

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    print(f"Train samples: {len(X_train):,}, Test samples: {len(X_test):,}")

    # -------------------------------------------------------------
    # Extension 1: Probabilistic Forecasting (Quantile LightGBM)
    # -------------------------------------------------------------
    print("\n--- Training Quantile LightGBM (q=0.10, 0.50, 0.90) ---")
    quantiles = (0.10, 0.50, 0.90)
    q_models = train_quantile_lgbm(X_train, y_train, quantiles=quantiles)

    print("Evaluating prediction intervals & pinball loss on test set...")
    q_eval = evaluate_quantiles(q_models, X_test, y_test)
    preds_df = q_eval["predictions"]

    print(f"Empirical Coverage (Nominal 80%): {q_eval['empirical_coverage_80pct']*100:.2f}%")
    print(f"Average Interval Width: {q_eval['avg_interval_width_eur']:.2f} EUR/MWh")
    for k, v in q_eval["pinball_losses"].items():
        print(f"  {k}: {v:.3f} EUR/MWh")

    # Generate Probabilistic Forecast Visualization
    fig, ax = plt.subplots(figsize=(14, 6))
    sample_window = slice(168 * 2, 168 * 3)  # Sample week from test set
    ts_sample = test_df["timestamp_utc"].iloc[sample_window]
    y_sample = y_test.iloc[sample_window].values
    q10_sample = preds_df[0.10].iloc[sample_window].values
    q50_sample = preds_df[0.50].iloc[sample_window].values
    q90_sample = preds_df[0.90].iloc[sample_window].values

    ax.plot(ts_sample, y_sample, label="Actual Day-Ahead Price", color="black", linewidth=1.5, zorder=4)
    ax.plot(ts_sample, q50_sample, label="Median Forecast (q=0.50)", color="#1f77b4", linestyle="--", linewidth=1.2)
    ax.fill_between(
        ts_sample,
        q10_sample,
        q90_sample,
        color="#1f77b4",
        alpha=0.25,
        label=f"80% Prediction Interval (Empirical: {q_eval['empirical_coverage_80pct']*100:.1f}%)",
    )
    ax.axhline(0, color="crimson", linestyle=":", linewidth=1, label="0 EUR/MWh Threshold")
    ax.set_title(
        "Probabilistic Day-Ahead Price Forecasting: Quantile LightGBM with 80% Prediction Intervals",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Date (UTC)", fontsize=11)
    ax.set_ylabel("Electricity Price [EUR/MWh]", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()

    fig_path = ROOT_DIR / "reports/figures/05_probabilistic_prediction_intervals.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"Saved figure: {fig_path.name}")

    # Also copy to results/figures
    (ROOT_DIR / "results/figures").mkdir(exist_ok=True, parents=True)
    fig_path_res = ROOT_DIR / "results/figures/05_probabilistic_prediction_intervals.png"
    import shutil
    shutil.copy(fig_path, fig_path_res)

    # -------------------------------------------------------------
    # Extension 2: Battery Storage Arbitrage Simulation (BESS)
    # -------------------------------------------------------------
    print("\n--- Simulating 10 MW / 20 MWh Utility BESS Arbitrage ---")
    test_df_eval = test_df.copy()
    test_df_eval["price_actual"] = y_test.values
    test_df_eval["price_pred_median"] = preds_df[0.50].values
    test_df_eval["price_naive_lag24"] = test_df["price_lag_24"].values
    test_df_eval["date"] = pd.to_datetime(test_df_eval["timestamp_utc"]).dt.date

    daily_profits_perfect = []
    daily_profits_lgbm = []
    daily_profits_naive = []

    unique_days = test_df_eval["date"].unique()
    for day in unique_days:
        df_d = test_df_eval[test_df_eval["date"] == day]
        if len(df_d) != 24:
            continue
        # Perfect foresight upper bound
        p_perf = simulate_bess_arbitrage_daily(df_d, "price_actual", "price_actual")
        # ML Model-guided dispatch
        p_lgbm = simulate_bess_arbitrage_daily(df_d, "price_actual", "price_pred_median")
        # Naive 24h lag-guided dispatch
        p_naive = simulate_bess_arbitrage_daily(df_d, "price_actual", "price_naive_lag24")

        daily_profits_perfect.append(p_perf)
        daily_profits_lgbm.append(p_lgbm)
        daily_profits_naive.append(p_naive)

    tot_perf = sum(daily_profits_perfect)
    tot_lgbm = sum(daily_profits_lgbm)
    tot_naive = sum(daily_profits_naive)
    n_days = len(daily_profits_perfect)

    # Annualize revenue for a 10 MW / 20 MWh battery
    annual_factor = 365.0 / n_days
    ann_perf = tot_perf * annual_factor
    ann_lgbm = tot_lgbm * annual_factor
    ann_naive = tot_naive * annual_factor

    rev_per_mw_perf = ann_perf / 10.0
    rev_per_mw_lgbm = ann_lgbm / 10.0
    rev_per_mw_naive = ann_naive / 10.0
    capture_rate = (ann_lgbm / ann_perf) * 100.0

    print(f"Simulated {n_days} full 24-hour market days in out-of-sample period:")
    print(f"  Theoretical Perfect Foresight (LP Upper Bound): {ann_perf:,.0f} EUR/yr ({rev_per_mw_perf:,.0f} EUR/MW-yr)")
    print(f"  Quantile LGBM Guided Dispatch:                 {ann_lgbm:,.0f} EUR/yr ({rev_per_mw_lgbm:,.0f} EUR/MW-yr) [Capture: {capture_rate:.1f}%]")
    print(f"  Naive (Lag-24) Guided Dispatch:                {ann_naive:,.0f} EUR/yr ({rev_per_mw_naive:,.0f} EUR/MW-yr)")

    # BESS Comparison Figure
    fig, ax = plt.subplots(figsize=(9, 5))
    strategies = [
        "Naive (Lag-24)\nDispatch",
        "LightGBM Forecast\nGuided Dispatch",
        "Perfect Foresight\n(Upper Bound)",
    ]
    revenues_k_per_mw = [rev_per_mw_naive / 1000.0, rev_per_mw_lgbm / 1000.0, rev_per_mw_perf / 1000.0]
    colors = ["#7f7f7f", "#2ca02c", "#1f77b4"]

    bars = ax.bar(strategies, revenues_k_per_mw, color=colors, width=0.55, edgecolor="black", linewidth=1)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:.1f} k€/MW-yr",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    ax.set_title(
        "Commercial Value: 10 MW / 20 MWh Battery Arbitrage Annualized Revenue (DE Day-Ahead)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylabel("Annualized Revenue [kEUR / MW-year]", fontsize=11)
    ax.set_ylim(0, max(revenues_k_per_mw) * 1.25)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    fig_bess_path = ROOT_DIR / "reports/figures/05_bess_arbitrage_comparison.png"
    fig.savefig(fig_bess_path, dpi=300)
    plt.close(fig)
    print(f"Saved figure: {fig_bess_path.name}")

    fig_bess_path_res = ROOT_DIR / "results/figures/05_bess_arbitrage_comparison.png"
    shutil.copy(fig_bess_path, fig_bess_path_res)

    # Save summary json
    summary = {
        "probabilistic_forecasting": {
            "empirical_coverage_80pct": q_eval["empirical_coverage_80pct"],
            "nominal_coverage_80pct": q_eval["nominal_coverage_80pct"],
            "avg_interval_width_eur_mwh": q_eval["avg_interval_width_eur"],
            "pinball_losses": q_eval["pinball_losses"],
        },
        "bess_arbitrage_simulation": {
            "system_specs": "10 MW / 20 MWh (2-hour duration), 85% round-trip efficiency, max 1 cycle/day",
            "eval_days_count": n_days,
            "annualized_revenue_eur": {
                "perfect_foresight_upper_bound": ann_perf,
                "lightgbm_forecast_guided": ann_lgbm,
                "naive_lag24_guided": ann_naive,
            },
            "annualized_revenue_eur_per_mw": {
                "perfect_foresight_upper_bound": rev_per_mw_perf,
                "lightgbm_forecast_guided": rev_per_mw_lgbm,
                "naive_lag24_guided": rev_per_mw_naive,
            },
            "theoretical_capture_rate_pct": capture_rate,
        },
    }

    out_json = ROOT_DIR / "reports/advanced_extensions_results.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved results: {out_json.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
