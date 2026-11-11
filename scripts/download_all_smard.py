"""
Comprehensive SMARD.de Data Downloader & Formatter.
Downloads official Bundesnetzagentur SMARD electricity market data
(Prices, Generation by fuel source, Grid load & Residual load)
from 2019-01-01 to present, and saves authentic SMARD-style CSV files
in data/raw/ with semicolon delimiters, German date formats in Europe/Berlin,
and German decimal commas.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure root in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.download import download_series, SERIES_CONFIG

RAW_DIR = ROOT_DIR / "data" / "raw"
CACHE_DIR = ROOT_DIR / "data" / "interim" / "cache"
RAW_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

START_TS_MS = 1546300800000  # 2019-01-01 00:00:00 UTC (01:00:00 Berlin time)


def format_to_smard_csv(
    df_merged: pd.DataFrame,
    value_cols: dict,
    output_path: Path
):
    """
    Format DataFrame to authentic SMARD CSV:
    - Datum von: 'DD.MM.YYYY HH:MM' in Europe/Berlin
    - Datum bis: 'DD.MM.YYYY HH:MM' in Europe/Berlin (+1 hour)
    - Delimiter: ';'
    - Decimal: ','
    - Missing: '-'
    """
    # Convert timestamp_ms to Europe/Berlin
    dt_utc = pd.to_datetime(df_merged["timestamp_ms"], unit="ms", utc=True)
    dt_berlin = dt_utc.dt.tz_convert("Europe/Berlin")
    dt_berlin_end = dt_berlin + pd.Timedelta(hours=1)

    out_df = pd.DataFrame()
    out_df["Datum von"] = dt_berlin.dt.strftime("%d.%m.%Y %H:%M")
    out_df["Datum bis"] = dt_berlin_end.dt.strftime("%d.%m.%Y %H:%M")

    for col, display_name in value_cols.items():
        if col in df_merged.columns:
            def format_num(val):
                if pd.isna(val):
                    return "-"
                return f"{val:.2f}".replace(".", ",")
            out_df[display_name] = df_merged[col].apply(format_num)

    out_df.to_csv(output_path, sep=";", index=False, encoding="utf-8-sig")
    print(f"Saved authentic SMARD CSV: {output_path} ({len(out_df)} rows)")


def download_and_save_all():
    print("=" * 70, flush=True)
    print("STARTING OFFICIAL SMARD.DE DATA DOWNLOAD (2019 - PRESENT)", flush=True)
    print("=" * 70, flush=True)

    # 1. Download Wholesale Electricity Prices
    print("\n[1/3] Downloading Day-Ahead Wholesale Prices (DE-LU)...", flush=True)
    price_cfg = SERIES_CONFIG["price_day_ahead"]
    df_price = download_series(
        price_cfg["filter"], price_cfg["region"], price_cfg["resolution"],
        START_TS_MS, cache_path=CACHE_DIR / "price_day_ahead.parquet"
    )
    df_price = df_price.rename(columns={"value": "price_eur_mwh"})

    price_out = RAW_DIR / "Grosshandelspreise_20190101_2026.csv"
    format_to_smard_csv(
        df_price,
        {"price_eur_mwh": "Deutschland/Luxemburg [€/MWh] Originalauflösungen"},
        price_out
    )

    # 2. Download Realised Generation by Source
    print("\n[2/3] Downloading Realised Electricity Generation by Source...", flush=True)
    gen_sources = [
        ("wind_onshore", "Wind Onshore [MWh] Berechnete Auflösungen"),
        ("wind_offshore", "Wind Offshore [MWh] Berechnete Auflösungen"),
        ("solar", "Photovoltaik [MWh] Berechnete Auflösungen"),
        ("biomass", "Biomasse [MWh] Berechnete Auflösungen"),
        ("nuclear", "Kernenergie [MWh] Berechnete Auflösungen"),
        ("hydro", "Wasserkraft [MWh] Berechnete Auflösungen"),
        ("hard_coal", "Steinkohle [MWh] Berechnete Auflösungen"),
        ("lignite", "Braunkohle [MWh] Berechnete Auflösungen"),
        ("natural_gas", "Erdgas [MWh] Berechnete Auflösungen"),
    ]

    df_gen = pd.DataFrame({"timestamp_ms": df_price["timestamp_ms"]})
    gen_value_cols = {}

    for key, header_name in gen_sources:
        cfg = SERIES_CONFIG[key]
        print(f"  Downloading {cfg['description']}...", flush=True)
        df_s = download_series(
            cfg["filter"], cfg["region"], cfg["resolution"],
            START_TS_MS, cache_path=CACHE_DIR / f"{key}.parquet"
        )
        df_gen = df_gen.merge(
            df_s.rename(columns={"value": cfg["col_name"]}),
            on="timestamp_ms",
            how="left"
        )
        gen_value_cols[cfg["col_name"]] = header_name

    gen_out = RAW_DIR / "Realisierte_Erzeugung_20190101_2026.csv"
    format_to_smard_csv(df_gen, gen_value_cols, gen_out)

    # 3. Download Electricity Consumption (Load) & Residual Load
    print("\n[3/3] Downloading Electricity Consumption (Grid Load & Residual Load)...", flush=True)
    load_cfg = SERIES_CONFIG["load"]
    df_load_raw = download_series(
        load_cfg["filter"], load_cfg["region"], load_cfg["resolution"],
        START_TS_MS, cache_path=CACHE_DIR / "load.parquet"
    )
    df_load_merged = pd.DataFrame({"timestamp_ms": df_price["timestamp_ms"]})
    df_load_merged = df_load_merged.merge(
        df_load_raw.rename(columns={"value": "load_mw"}),
        on="timestamp_ms",
        how="left"
    )

    # Residual Load
    print("  Downloading Residual Load (Filter 4359)...", flush=True)
    df_res_raw = download_series(
        4359, "DE", "hour",
        START_TS_MS, cache_path=CACHE_DIR / "residual_load.parquet"
    )
    df_load_merged = df_load_merged.merge(
        df_res_raw.rename(columns={"value": "residual_load_mw"}),
        on="timestamp_ms",
        how="left"
    )

    load_out = RAW_DIR / "Realisierter_Stromverbrauch_20190101_2026.csv"
    format_to_smard_csv(
        df_load_merged,
        {
            "load_mw": "Gesamt (Netzlast) [MWh] Berechnete Auflösungen",
            "residual_load_mw": "Residuallast [MWh] Berechnete Auflösungen"
        },
        load_out
    )

    print("\n" + "=" * 70, flush=True)
    print("ALL SMARD DATA DOWNLOADED AND STORED IN data/raw/", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    download_and_save_all()
