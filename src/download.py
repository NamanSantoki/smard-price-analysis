"""
Official SMARD.de API data downloader.
Downloads electricity market data from Bundesnetzagentur SMARD platform:
- Day-ahead wholesale electricity price (DE-LU, filter 4169)
- Realised generation: Wind Onshore (filter 4067)
- Realised generation: Wind Offshore (filter 1225)
- Realised generation: Photovoltaics / Solar (filter 4068)
- Realised electricity consumption / Grid load (filter 410)
- Additional generation sources: Nuclear (1224), Hydro (1226), Biomass (1223),
  Lignite/Brown Coal (4070), Hard Coal (4069), Natural Gas (4071)
"""

import json
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed


SERIES_CONFIG = {
    "price_day_ahead": {
        "filter": 4169,
        "region": "DE-LU",
        "resolution": "hour",
        "col_name": "price_eur_mwh",
        "description": "Day-ahead wholesale electricity price (EUR/MWh)"
    },
    "wind_onshore": {
        "filter": 4067,
        "region": "DE",
        "resolution": "hour",
        "col_name": "wind_onshore_mw",
        "description": "Realised generation: Wind Onshore (MW)"
    },
    "wind_offshore": {
        "filter": 1225,
        "region": "DE",
        "resolution": "hour",
        "col_name": "wind_offshore_mw",
        "description": "Realised generation: Wind Offshore (MW)"
    },
    "solar": {
        "filter": 4068,
        "region": "DE",
        "resolution": "hour",
        "col_name": "solar_mw",
        "description": "Realised generation: Photovoltaics (MW)"
    },
    "load": {
        "filter": 410,
        "region": "DE",
        "resolution": "hour",
        "col_name": "load_mw",
        "description": "Realised electricity consumption / Grid Load (MW)"
    },
    "residual_load": {
        "filter": 4359,
        "region": "DE",
        "resolution": "hour",
        "col_name": "residual_load_mw",
        "description": "Realised Residual Load (MW)"
    },
    "biomass": {
        "filter": 1223,
        "region": "DE",
        "resolution": "hour",
        "col_name": "biomass_mw",
        "description": "Realised generation: Biomass (MW)"
    },
    "nuclear": {
        "filter": 1224,
        "region": "DE",
        "resolution": "hour",
        "col_name": "nuclear_mw",
        "description": "Realised generation: Nuclear (MW)"
    },
    "hydro": {
        "filter": 1226,
        "region": "DE",
        "resolution": "hour",
        "col_name": "hydro_mw",
        "description": "Realised generation: Hydro (MW)"
    },
    "lignite": {
        "filter": 4070,
        "region": "DE",
        "resolution": "hour",
        "col_name": "lignite_mw",
        "description": "Realised generation: Lignite / Brown Coal (MW)"
    },
    "hard_coal": {
        "filter": 4069,
        "region": "DE",
        "resolution": "hour",
        "col_name": "hard_coal_mw",
        "description": "Realised generation: Hard Coal (MW)"
    },
    "natural_gas": {
        "filter": 4071,
        "region": "DE",
        "resolution": "hour",
        "col_name": "gas_mw",
        "description": "Realised generation: Natural Gas (MW)"
    }
}


def fetch_timestamps(filter_id: int, region: str, resolution: str = "hour") -> List[int]:
    """Fetch index of timestamp chunks for a given filter and region."""
    url = f"https://www.smard.de/app/chart_data/{filter_id}/{region}/index_{resolution}.json"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
        return data.get("timestamps", [])
    except Exception as e:
        print(f"Warning: could not fetch timestamps for {filter_id} {region}: {e}")
        return []


def fetch_chunk(filter_id: int, region: str, resolution: str, timestamp: int) -> List[List]:
    """Fetch a single weekly time series chunk."""
    url = f"https://www.smard.de/app/chart_data/{filter_id}/{region}/{filter_id}_{region}_{resolution}_{timestamp}.json"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            return data.get("series", [])
    except Exception:
        return []


def download_series(
    filter_id: int,
    region: str,
    resolution: str,
    start_ts_ms: int = 1546300800000, # 2019-01-01 00:00:00 UTC
    max_workers: int = 16,
    cache_path: Optional[Path] = None
) -> pd.DataFrame:
    """Download full series with optional parquet disk cache."""
    if cache_path and cache_path.exists():
        print(f"  [Cache hit] Loading cached {cache_path.name}")
        return pd.read_parquet(cache_path)

    all_timestamps = fetch_timestamps(filter_id, region, resolution)
    if not all_timestamps:
        return pd.DataFrame(columns=["timestamp_ms", "value"])

    relevant_ts = [ts for ts in all_timestamps if ts >= start_ts_ms - 7 * 86400 * 1000]
    
    records = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ts = {
            executor.submit(fetch_chunk, filter_id, region, resolution, ts): ts
            for ts in relevant_ts
        }
        for future in as_completed(future_to_ts):
            chunk_data = future.result()
            records.extend(chunk_data)
            
    if not records:
        return pd.DataFrame(columns=["timestamp_ms", "value"])
        
    df = pd.DataFrame(records, columns=["timestamp_ms", "value"])
    df = df.dropna(subset=["timestamp_ms"])
    df = df.drop_duplicates(subset=["timestamp_ms"]).sort_values("timestamp_ms")
    df = df[df["timestamp_ms"] >= start_ts_ms].reset_index(drop=True)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        
    return df
