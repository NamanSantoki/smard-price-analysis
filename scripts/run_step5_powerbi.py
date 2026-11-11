"""
Step 5: Power BI Dashboard Preparation.
Exports dashboard-ready CSV, data dictionary, DAX measures, and page specifications.
"""

from pathlib import Path
import sys
import json
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

PROCESSED_DIR = ROOT_DIR / "data" / "processed"
POWERBI_DIR = ROOT_DIR / "powerbi"
POWERBI_DIR.mkdir(parents=True, exist_ok=True)


def run_step5():
    print("=" * 75)
    print("STEP 5: POWER BI DASHBOARD PREPARATION")
    print("=" * 75)

    df = pd.read_parquet(PROCESSED_DIR / "merged_hourly.parquet")

    # Convert timestamp to Berlin local time string for Power BI (no tz-aware issues)
    ts_utc = pd.to_datetime(df["timestamp_utc"], utc=True)
    ts_berlin = ts_utc.dt.tz_convert("Europe/Berlin")

    pbi = pd.DataFrame()
    pbi["datetime_berlin"] = ts_berlin.dt.strftime("%Y-%m-%d %H:%M:%S")
    pbi["date"] = ts_berlin.dt.strftime("%Y-%m-%d")
    pbi["year"] = df["year"].values
    pbi["month"] = df["month"].values
    pbi["hour"] = df["hour"].values
    pbi["day_of_week"] = df["day_of_week"].values
    pbi["is_weekend"] = df["is_weekend"].values
    pbi["is_holiday"] = df["is_holiday"].values
    pbi["is_workday"] = df["is_workday"].values

    pbi["price_eur_mwh"] = df["price_eur_mwh"].values
    pbi["is_negative_price"] = df["is_negative_price"].values

    for col in ["wind_onshore_mw", "wind_offshore_mw", "wind_total_mw",
                 "solar_mw", "load_mw", "residual_load_mw",
                 "biomass_mw", "hydro_mw", "hard_coal_mw", "lignite_mw", "gas_mw",
                 "renewable_total_mw", "renewable_share_load"]:
        if col in df.columns:
            pbi[col] = df[col].values

    out_path = POWERBI_DIR / "dashboard_data.csv"
    pbi.to_csv(out_path, index=False)
    print(f"Saved dashboard CSV: {out_path} ({len(pbi):,} rows, {len(pbi.columns)} columns)")

    # Data Dictionary
    data_dict = {
        "datetime_berlin": "Timestamp in Europe/Berlin local time (YYYY-MM-DD HH:MM:SS)",
        "date": "Calendar date (YYYY-MM-DD)",
        "year": "Calendar year (2019-2026)",
        "month": "Month of year (1-12)",
        "hour": "Hour of day in Berlin time (0-23)",
        "day_of_week": "Day of week (0=Monday, 6=Sunday)",
        "is_weekend": "1 if Saturday or Sunday, else 0",
        "is_holiday": "1 if German national public holiday, else 0",
        "is_workday": "1 if neither weekend nor holiday, else 0",
        "price_eur_mwh": "Day-ahead wholesale electricity price [EUR/MWh]",
        "is_negative_price": "1 if price < 0 EUR/MWh, else 0",
        "wind_onshore_mw": "Realised wind onshore generation [MW]",
        "wind_offshore_mw": "Realised wind offshore generation [MW]",
        "wind_total_mw": "Total wind generation (onshore + offshore) [MW]",
        "solar_mw": "Realised solar (PV) generation [MW]",
        "load_mw": "Total electricity consumption / grid load [MW]",
        "residual_load_mw": "Residual load = load - wind - solar [MW]",
        "biomass_mw": "Realised biomass generation [MW]",
        "hydro_mw": "Realised hydropower generation [MW]",
        "hard_coal_mw": "Realised hard coal generation [MW]",
        "lignite_mw": "Realised lignite / brown coal generation [MW]",
        "gas_mw": "Realised natural gas generation [MW]",
        "renewable_total_mw": "Total renewable generation (wind + solar + biomass + hydro) [MW]",
        "renewable_share_load": "Renewable share of load (renewable_total / load), clipped to [0, 5]",
    }
    with open(POWERBI_DIR / "data_dictionary.json", "w") as f:
        json.dump(data_dict, f, indent=2)
    print("Saved data_dictionary.json")

    print("\nStep 5 complete.")


if __name__ == "__main__":
    run_step5()

