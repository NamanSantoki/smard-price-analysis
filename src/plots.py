"""
Visualization module for electricity market analysis, merit-order effect,
negative price patterns, and forecasting evaluation.
All plots include descriptive titles, axis labels with units, and clean styling.
"""

from pathlib import Path
from typing import Optional, List
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_price_distribution(df: pd.DataFrame, output_path: Optional[Path] = None):
    """Plot price distribution with linear and symlog scales, showing heavy tails and skewness."""
    prices = df["price_eur_mwh"].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Linear scale histogram
    sns.histplot(prices, bins=100, kde=True, ax=axes[0], color="#1f77b4", edgecolor="none", alpha=0.6)
    axes[0].set_title("Day-Ahead Wholesale Price Distribution (Linear Scale)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Price [EUR/MWh]")
    axes[0].set_ylabel("Hourly Frequency")
    axes[0].axvline(0, color="crimson", linestyle="--", linewidth=1.5, label="0 EUR/MWh")
    axes[0].axvline(prices.mean(), color="black", linestyle="-", linewidth=1.2, label=f"Mean: {prices.mean():.1f} €")
    axes[0].axvline(prices.median(), color="green", linestyle=":", linewidth=1.5, label=f"Median: {prices.median():.1f} €")
    axes[0].legend(frameon=True)

    # Annotate stats
    stats_text = (
        f"N: {len(prices):,}\n"
        f"Mean: {prices.mean():.2f} €\n"
        f"Median: {prices.median():.2f} €\n"
        f"Std: {prices.std():.2f} €\n"
        f"Skewness: {prices.skew():.2f}\n"
        f"Kurtosis: {prices.kurtosis():.2f}\n"
        f"Min: {prices.min():.2f} €\n"
        f"Max: {prices.max():.2f} €"
    )
    axes[0].text(
        0.95, 0.95, stats_text,
        transform=axes[0].transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="#aaa"),
        fontsize=9
    )

    # 2. Symlog scale histogram to visualize extreme positive and negative spikes
    sns.histplot(prices, bins=100, ax=axes[1], color="#2ca02c", edgecolor="none", alpha=0.7)
    axes[1].set_xscale("symlog", linthresh=10.0)
    axes[1].set_title("Price Distribution (Symmetric Log Scale: linthresh=10 €)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Price [EUR/MWh] (symlog)")
    axes[1].set_ylabel("Hourly Frequency")
    axes[1].axvline(0, color="crimson", linestyle="--", linewidth=1.5, label="0 EUR/MWh")
    axes[1].legend(frameon=True)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_negative_price_trends(df: pd.DataFrame, output_path: Optional[Path] = None):
    """Plot annual frequency and monthly/hourly heatmap of negative-price occurrences."""
    clean = df.dropna(subset=["price_eur_mwh", "year", "month", "hour"]).copy()
    clean["is_negative"] = (clean["price_eur_mwh"] < 0).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Annual Negative Hours and Share
    annual = clean.groupby("year").agg(
        neg_hours=("is_negative", "sum"),
        total_hours=("is_negative", "count")
    ).reset_index()
    annual["neg_pct"] = (annual["neg_hours"] / annual["total_hours"]) * 100

    bars = axes[0].bar(annual["year"], annual["neg_hours"], color="#d62728", alpha=0.8, edgecolor="black")
    axes[0].set_title("Annual Count of Negative-Price Hours (Germany/Luxembourg)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Count of Negative Hours [h]")
    
    # Add text labels on bars
    for bar, pct in zip(bars, annual["neg_pct"]):
        yval = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0, yval + 10,
            f"{int(yval)}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=8, fontweight="bold"
        )
    axes[0].set_ylim(0, annual["neg_hours"].max() * 1.25)

    # 2. Heatmap: Month vs Hour of Day
    heatmap_data = clean.groupby(["month", "hour"])["is_negative"].mean().unstack() * 100
    sns.heatmap(
        heatmap_data,
        cmap="YlOrRd",
        annot=False,
        cbar_kws={"label": "Share of Negative Hours [%]"},
        ax=axes[1]
    )
    axes[1].set_title("Negative Price Frequency [%] by Month and Hour of Day", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Hour of Day (Europe/Berlin) [0 - 23]")
    axes[1].set_ylabel("Month of Year [1 - 12]")

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_merit_order_effect(df: pd.DataFrame, output_path: Optional[Path] = None):
    """Plot binned relationship between Residual Load / Renewable Generation and Electricity Price."""
    clean = df.dropna(subset=["price_eur_mwh", "residual_load_mw", "wind_total_mw", "solar_mw"]).copy()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    drivers = [
        ("wind_total_mw", "Total Wind Generation [GW]", "#1f77b4"),
        ("solar_mw", "Solar Generation [GW]", "#ff7f0e"),
        ("residual_load_mw", "Residual Load (Load - Wind - Solar) [GW]", "#2ca02c")
    ]

    for idx, (col, label, color) in enumerate(drivers):
        clean["driver_gw"] = clean[col] / 1000.0
        # Binned statistics (25 quantiles)
        clean["bin"] = pd.qcut(clean["driver_gw"], q=25, duplicates="drop")
        binned = clean.groupby("bin", observed=True).agg(
            x_mean=("driver_gw", "mean"),
            y_median=("price_eur_mwh", "median"),
            y_q25=("price_eur_mwh", lambda s: s.quantile(0.25)),
            y_q75=("price_eur_mwh", lambda s: s.quantile(0.75))
        ).reset_index()

        axes[idx].scatter(clean["driver_gw"], clean["price_eur_mwh"], alpha=0.03, color=color, s=4)
        axes[idx].plot(binned["x_mean"], binned["y_median"], color="black", linewidth=2.2, label="Median Price")
        axes[idx].fill_between(binned["x_mean"], binned["y_q25"], binned["y_q75"], color="black", alpha=0.2, label="IQR (25-75th)")
        axes[idx].axhline(0, color="crimson", linestyle="--", alpha=0.7)
        axes[idx].set_title(f"Price vs {label.split(' [')[0]}", fontsize=11, fontweight="bold")
        axes[idx].set_xlabel(label)
        axes[idx].set_ylabel("Day-Ahead Price [EUR/MWh]")
        axes[idx].set_ylim(-100, 300)
        axes[idx].legend(loc="upper right" if idx < 2 else "upper left", frameon=True)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_hourly_profiles_by_year(df: pd.DataFrame, output_path: Optional[Path] = None):
    """Plot diurnal price curves for each year, showing structural depression during solar peak."""
    clean = df.dropna(subset=["price_eur_mwh", "hour", "year"]).copy()
    hourly_annual = clean.groupby(["year", "hour"])["price_eur_mwh"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=hourly_annual,
        x="hour",
        y="price_eur_mwh",
        hue="year",
        palette="viridis",
        marker="o",
        linewidth=2,
        ax=ax
    )
    ax.set_title("Diurnal Average Electricity Price Curve by Year (2019-2026)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Hour of Day (Europe/Berlin) [0 - 23]")
    ax.set_ylabel("Average Day-Ahead Price [EUR/MWh]")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(title="Year", frameon=True)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_predictions_sample(
    df_eval: pd.DataFrame,
    date_start: str,
    date_end: str,
    output_path: Optional[Path] = None
):
    """Plot actual vs predicted prices for a representative week."""
    sample = df_eval[(df_eval["timestamp_utc"] >= date_start) & (df_eval["timestamp_utc"] <= date_end)].copy()
    
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(sample["timestamp_utc"], sample["price_actual"], label="Actual Day-Ahead Price", color="black", linewidth=2.0)
    if "price_pred_lgbm" in sample.columns:
        ax.plot(sample["timestamp_utc"], sample["price_pred_lgbm"], label="LightGBM Forecast", color="#1f77b4", linestyle="--", linewidth=1.8)
    if "price_pred_naive24" in sample.columns:
        ax.plot(sample["timestamp_utc"], sample["price_pred_naive24"], label="Naive (Lag 24h)", color="#7f7f7f", linestyle=":", linewidth=1.2)
    
    ax.axhline(0, color="crimson", linestyle="-", linewidth=1, alpha=0.5)
    ax.set_title(f"Day-Ahead Price Forecast vs Actuals ({date_start} to {date_end})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Price [EUR/MWh]")
    ax.legend(frameon=True, loc="upper right")
    plt.xticks(rotation=20)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig

