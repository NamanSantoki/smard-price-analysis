"""
Step 2: Exploratory Data Analysis & Merit-Order Profiling.
Generates publication-quality figures with units, annotations, and empirical takeaways:
1. Price distribution (linear and symlog)
2. Negative-price trends: annual counts and month-by-hour heatmap
3. Merit-order effect: binned relationship of price vs wind, solar, and residual load
4. Diurnal price curves by year (2019-2026) showing the evolving solar "duck curve"
5. Diurnal and seasonal generation profiles of wind vs photovoltaics
6. 30-day rolling correlation of price with wind and solar over time
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.plots import (
    plot_price_distribution,
    plot_negative_price_trends,
    plot_merit_order_effect,
    plot_hourly_profiles_by_year
)

PROCESSED_DIR = ROOT_DIR / "data" / "processed"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def run_step2():
    print("=" * 75)
    print("STEP 2: EXPLORATORY DATA ANALYSIS")
    print("=" * 75)

    df = pd.read_parquet(PROCESSED_DIR / "merged_hourly.parquet")
    print(f"Loaded {len(df):,} hourly records.")

    # 1. Price Distribution (Linear + Symlog)
    print("\nGenerating Figure 1: Price Distribution...")
    fig1_path = FIGURES_DIR / "02_price_distribution.png"
    plot_price_distribution(df, fig1_path)
    print(f"Saved: {fig1_path.name}")

    # 2. Negative Price Frequency & Heatmap
    print("\nGenerating Figure 2: Negative Price Trends & Heatmap...")
    fig2_path = FIGURES_DIR / "02_negative_price_trends.png"
    plot_negative_price_trends(df, fig2_path)
    print(f"Saved: {fig2_path.name}")

    # 3. Merit Order Effect (Price vs Wind, Solar, Residual Load)
    print("\nGenerating Figure 3: Merit Order Effect (Binned Means)...")
    fig3_path = FIGURES_DIR / "02_merit_order_binned.png"
    plot_merit_order_effect(df, fig3_path)
    print(f"Saved: {fig3_path.name}")

    # 4. Hourly Diurnal Price Profiles by Year (Solar Duck Curve Evolution)
    print("\nGenerating Figure 4: Diurnal Price Curve Evolution...")
    fig4_path = FIGURES_DIR / "02_diurnal_price_by_year.png"
    plot_hourly_profiles_by_year(df, fig4_path)
    print(f"Saved: {fig4_path.name}")

    # 5. Seasonal and Diurnal Generation Profiles of Wind vs Solar
    print("\nGenerating Figure 5: Generation Profiles (Wind vs Solar)...")
    fig5, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # 5a. Hourly Diurnal Profile by Technology
    hourly_gen = df.groupby("hour")[["wind_total_mw", "solar_mw"]].mean() / 1000.0
    axes[0].plot(hourly_gen.index, hourly_gen["wind_total_mw"], label="Wind Total (Onshore + Offshore)", color="#1f77b4", marker="o", linewidth=2)
    axes[0].plot(hourly_gen.index, hourly_gen["solar_mw"], label="Solar (PV)", color="#ff7f0e", marker="s", linewidth=2)
    axes[0].set_title("Average Diurnal Generation Profile (Germany 2019-2026)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Hour of Day (Europe/Berlin) [0 - 23]")
    axes[0].set_ylabel("Average Generation [GW]")
    axes[0].set_xticks(range(0, 24, 2))
    axes[0].legend(frameon=True)
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # 5b. Monthly Seasonal Profile by Technology
    monthly_gen = df.groupby("month")[["wind_total_mw", "solar_mw"]].mean() / 1000.0
    axes[1].plot(monthly_gen.index, monthly_gen["wind_total_mw"], label="Wind Total (Winter Peaking)", color="#1f77b4", marker="o", linewidth=2)
    axes[1].plot(monthly_gen.index, monthly_gen["solar_mw"], label="Solar (Summer Peaking)", color="#ff7f0e", marker="s", linewidth=2)
    axes[1].set_title("Average Monthly Seasonal Generation Profile (Seasonal Complementarity)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Month of Year [1 - 12]")
    axes[1].set_ylabel("Average Generation [GW]")
    axes[1].set_xticks(range(1, 13))
    axes[1].legend(frameon=True)
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig5_path = FIGURES_DIR / "02_generation_profiles_seasonal.png"
    fig5.savefig(fig5_path, dpi=300, bbox_inches="tight")
    plt.close(fig5)
    print(f"Saved: {fig5_path.name}")

    # 6. 30-Day Rolling Correlations
    print("\nGenerating Figure 6: Rolling Correlations Over Time...")
    df_sorted = df.sort_values("timestamp_utc").copy()
    rolling_corr_wind = df_sorted["price_eur_mwh"].rolling(window=720, min_periods=360).corr(df_sorted["wind_total_mw"])
    rolling_corr_solar = df_sorted["price_eur_mwh"].rolling(window=720, min_periods=360).corr(df_sorted["solar_mw"])
    rolling_corr_resload = df_sorted["price_eur_mwh"].rolling(window=720, min_periods=360).corr(df_sorted["residual_load_mw"])

    fig6, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df_sorted["timestamp_utc"], rolling_corr_resload, label="Price vs Residual Load (Strong Positive)", color="#2ca02c", linewidth=1.5)
    ax.plot(df_sorted["timestamp_utc"], rolling_corr_wind, label="Price vs Total Wind (Strong Negative Merit Order)", color="#1f77b4", linewidth=1.2)
    ax.plot(df_sorted["timestamp_utc"], rolling_corr_solar, label="Price vs Solar (Seasonally Fluctuating Negative)", color="#ff7f0e", linewidth=1.2)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title("30-Day (720h) Rolling Correlation with Wholesale Day-Ahead Price (2019-2026)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Date (UTC)")
    ax.set_ylabel("Pearson Correlation Coefficient (r)")
    ax.set_ylim(-0.8, 0.9)
    ax.legend(frameon=True, loc="lower left")
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig6_path = FIGURES_DIR / "02_rolling_correlations.png"
    fig6.savefig(fig6_path, dpi=300, bbox_inches="tight")
    plt.close(fig6)
    print(f"Saved: {fig6_path.name}")

    print("\n" + "=" * 75)
    print("STEP 2 COMPLETE: ALL FIGURES GENERATED IN reports/figures/")
    print("=" * 75)


if __name__ == "__main__":
    run_step2()

