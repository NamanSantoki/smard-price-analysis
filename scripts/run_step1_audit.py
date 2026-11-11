"""
Step 1: Data Audit, Cleaning, Harmonization and Merging.
Processes raw SMARD files into data/processed/merged_hourly.parquet and .csv,
and generates comprehensive data audit statistics and visualizations.
"""

from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.load import inspect_raw_file, scan_raw_directory, load_smard_csv
from src.clean import (
    parse_german_number,
    parse_smard_timestamps,
    aggregate_to_hourly,
    engineer_derived_features,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
INTERIM_DIR = ROOT_DIR / "data" / "interim"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

for d in [INTERIM_DIR, PROCESSED_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def run_step1():
    print("=" * 75)
    print("STEP 1: DATA AUDIT, CLEANING & MERGING")
    print("=" * 75)

    # 1. Inspect Raw Files
    raw_files = scan_raw_directory(RAW_DIR)
    print(f"\nFound {len(raw_files)} raw CSV files in {RAW_DIR}:")
    for rf in raw_files:
        info = inspect_raw_file(rf, n_rows=2)
        print(f"\n--- {rf.name} ---")
        print(f"  Separator: '{info['separator']}', Decimal: '{info['decimal']}'")
        print(f"  Columns ({len(info['columns'])}): {info['columns']}")

    # 2. Load and Clean Each Series
    # 2a. Wholesale Prices
    price_file = next(f for f in raw_files if "Grosshandelspreise" in f.name)
    print(f"\nLoading and parsing prices from: {price_file.name}...")
    df_price_raw = load_smard_csv(price_file)
    df_price = parse_smard_timestamps(df_price_raw, start_col="Datum von")
    # Identify price column
    price_col_raw = [c for c in df_price.columns if "Deutschland" in c or "€/MWh" in c][0]
    df_price["price_eur_mwh"] = df_price[price_col_raw].apply(parse_german_number)
    df_price_hourly = aggregate_to_hourly(
        df_price[["timestamp_utc", "price_eur_mwh"]],
        timestamp_col="timestamp_utc",
        value_cols=["price_eur_mwh"],
        method="mean"
    )

    # 2b. Generation by Source
    gen_file = next(f for f in raw_files if "Realisierte_Erzeugung" in f.name)
    print(f"\nLoading and parsing generation from: {gen_file.name}...")
    df_gen_raw = load_smard_csv(gen_file)
    df_gen = parse_smard_timestamps(df_gen_raw, start_col="Datum von")
    
    # Map raw German columns to standardized English column names
    col_map_gen = {
        "Wind Onshore": "wind_onshore_mw",
        "Wind Offshore": "wind_offshore_mw",
        "Photovoltaik": "solar_mw",
        "Biomasse": "biomass_mw",
        "Kernenergie": "nuclear_mw",
        "Wasserkraft": "hydro_mw",
        "Steinkohle": "hard_coal_mw",
        "Braunkohle": "lignite_mw",
        "Erdgas": "gas_mw",
    }
    
    gen_renamed = {}
    for raw_c in df_gen.columns:
        for pattern, std_name in col_map_gen.items():
            if pattern in raw_c:
                gen_renamed[raw_c] = std_name
                df_gen[std_name] = df_gen[raw_c].apply(parse_german_number)

    gen_cols_std = list(gen_renamed.values())
    df_gen_hourly = aggregate_to_hourly(
        df_gen[["timestamp_utc"] + gen_cols_std],
        timestamp_col="timestamp_utc",
        value_cols=gen_cols_std,
        method="mean"
    )

    # 2c. Consumption (Grid Load & Residual Load)
    load_file = next(f for f in raw_files if "Realisierter_Stromverbrauch" in f.name)
    print(f"\nLoading and parsing consumption from: {load_file.name}...")
    df_load_raw = load_smard_csv(load_file)
    df_load = parse_smard_timestamps(df_load_raw, start_col="Datum von")
    
    for raw_c in df_load.columns:
        if "Gesamt (Netzlast)" in raw_c:
            df_load["load_mw"] = df_load[raw_c].apply(parse_german_number)
        elif "Residuallast" in raw_c:
            df_load["residual_load_mw"] = df_load[raw_c].apply(parse_german_number)

    load_cols_std = [c for c in ["load_mw", "residual_load_mw"] if c in df_load.columns]
    df_load_hourly = aggregate_to_hourly(
        df_load[["timestamp_utc"] + load_cols_std],
        timestamp_col="timestamp_utc",
        value_cols=load_cols_std,
        method="mean"
    )

    # 3. Create Continuous Hourly Time Grid from 2019-01-01 to Latest Available Date
    start_utc = pd.to_datetime("2019-01-01 00:00:00", utc=True)
    end_utc = df_price_hourly["timestamp_utc"].max()
    full_grid = pd.DataFrame({"timestamp_utc": pd.date_range(start_utc, end_utc, freq="1h", tz="UTC")})
    
    print(f"\nMerging onto uniform UTC hourly grid: {start_utc} to {end_utc} (Expected hours: {len(full_grid):,})")

    # Outer/Left merge all hourly series
    merged = full_grid.merge(df_price_hourly, on="timestamp_utc", how="left")
    merged = merged.merge(df_gen_hourly, on="timestamp_utc", how="left")
    merged = merged.merge(df_load_hourly, on="timestamp_utc", how="left")

    # 4. Feature Engineering: Derived Metrics & Calendar
    print("Computing derived features (wind total, renewables, residual load, negative price flag, holidays)...")
    merged = engineer_derived_features(merged)

    # 5. Missing Value Policy & Interpolation
    # Documented rule: linearly interpolate gaps <= 2 consecutive hours. Longer gaps remain NaN.
    numeric_cols = [c for c in merged.select_dtypes(include=[np.number]).columns if not c.startswith("is_") and c not in ["hour", "month", "year", "quarter", "day_of_week"]]
    
    pre_interp_missing = merged[numeric_cols].isna().sum()
    for col in numeric_cols:
        merged[col] = merged[col].interpolate(method="linear", limit=2)
    post_interp_missing = merged[numeric_cols].isna().sum()

    # Re-evaluate is_negative_price after interpolation
    merged["is_negative_price"] = (merged["price_eur_mwh"] < 0).astype(int)

    # 6. Save Processed Datasets
    parquet_path = PROCESSED_DIR / "merged_hourly.parquet"
    csv_path = PROCESSED_DIR / "merged_hourly.csv"
    merged.to_parquet(parquet_path, index=False)
    # Save CSV version (sample or full)
    merged.to_csv(csv_path, index=False)
    print(f"\n[Saved] Merged Parquet: {parquet_path} ({len(merged):,} rows, {len(merged.columns)} columns)")
    print(f"[Saved] Merged CSV:     {csv_path}")

    # 7. Comprehensive Data Audit
    print("\n" + "=" * 75)
    print("DATA AUDIT REPORT")
    print("=" * 75)
    print(f"Total Hourly Records: {len(merged):,}")
    print(f"Date Range (UTC):     {merged['timestamp_utc'].min()} to {merged['timestamp_utc'].max()}")
    print(f"Duplicate Timestamps: {merged['timestamp_utc'].duplicated().sum()}")
    
    # Check for gaps in index
    time_diffs = merged["timestamp_utc"].diff()
    gaps = time_diffs[time_diffs > pd.Timedelta(hours=1)]
    print(f"Unexpected Gaps in Hourly Index: {len(gaps)}")

    print("\nMissing Values Audit (Before & After Interpolation of Short Gaps <= 2h):")
    audit_missing_df = pd.DataFrame({
        "Missing (Raw)": pre_interp_missing,
        "Missing Rate % (Raw)": (pre_interp_missing / len(merged) * 100).round(3),
        "Missing (Post-Interp)": post_interp_missing,
        "Missing Rate % (Post)": (post_interp_missing / len(merged) * 100).round(3)
    })
    print(audit_missing_df.to_string())

    print("\nMissing Values by Year (Post-Interpolation):")
    missing_by_year = merged.groupby("year")[numeric_cols].apply(lambda g: g.isna().sum())
    print(missing_by_year.to_string())

    print("\nDescriptive Summary of Key Energy & Price Variables:")
    summary_cols = ["price_eur_mwh", "wind_total_mw", "solar_mw", "load_mw", "residual_load_mw"]
    desc_df = merged[summary_cols].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    desc_df["skewness"] = merged[summary_cols].skew()
    desc_df["kurtosis"] = merged[summary_cols].kurtosis()
    print(desc_df[["count", "mean", "std", "min", "1%", "50%", "99%", "max", "skewness", "kurtosis"]].to_string())

    print(f"\nNegative-Price Statistics:")
    neg_count = merged["is_negative_price"].sum()
    neg_pct = (neg_count / len(merged.dropna(subset=['price_eur_mwh']))) * 100
    print(f"Total Negative-Price Hours: {neg_count:,} ({neg_pct:.2f}% of all valid price hours)")

    annual_neg = merged.groupby("year").agg(
        total_hours=("price_eur_mwh", "count"),
        negative_hours=("is_negative_price", "sum")
    )
    annual_neg["negative_pct"] = (annual_neg["negative_hours"] / annual_neg["total_hours"] * 100).round(2)
    print("\nNegative-Price Hours by Year:")
    print(annual_neg.to_string())

    # 8. Save Audit Summary JSON and Audit Figures
    audit_dict = {
        "total_records": int(len(merged)),
        "date_start": str(merged["timestamp_utc"].min()),
        "date_end": str(merged["timestamp_utc"].max()),
        "duplicate_timestamps": int(merged["timestamp_utc"].duplicated().sum()),
        "gaps_in_hourly_grid": int(len(gaps)),
        "total_negative_hours": int(neg_count),
        "negative_share_pct": float(round(neg_pct, 3)),
        "annual_negative_hours": annual_neg.to_dict(orient="index")
    }
    with open(PROCESSED_DIR / "data_audit_summary.json", "w") as f:
        json.dump(audit_dict, f, indent=2)

    # Plot 1: Missing values heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    missing_pct_year = merged.groupby("year")[numeric_cols].apply(lambda g: (g.isna().mean() * 100))
    sns.heatmap(missing_pct_year.T, annot=True, fmt=".2f", cmap="Reds", cbar_kws={"label": "Missing Rate [%]"}, ax=ax)
    ax.set_title("Data Completeness Audit: Missing Rate [%] by Variable and Year (2019-2026)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Variable")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "01_data_audit_missing_rate.png", dpi=300)
    plt.close(fig)

    # Plot 2: Price summary boxplot by year
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.boxplot(
        data=merged, x="year", y="price_eur_mwh", hue="year",
        showfliers=False, ax=ax, palette="Blues_d", legend=False
    )
    ax.axhline(0, color="crimson", linestyle="--", linewidth=1.2, label="0 EUR/MWh")
    ax.set_title("Day-Ahead Wholesale Price Range by Year (Without Outliers)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Price [EUR/MWh]")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "01_price_distribution_by_year.png", dpi=300)
    plt.close(fig)

    print("\nData Audit figures saved to reports/figures/.")


if __name__ == "__main__":
    run_step1()
