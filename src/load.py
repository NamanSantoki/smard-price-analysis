"""
Data loading module for SMARD electricity market CSV files.
Detects delimiters, German numeric formats, and file metadata.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd


def detect_csv_dialect(file_path: Path) -> Tuple[str, str]:
    """
    Detect separator and decimal format in SMARD CSVs.
    SMARD commonly uses ';' as separator and ',' as decimal separator.
    """
    with open(file_path, "r", encoding="utf-8-sig", errors="replace") as f:
        sample_lines = [f.readline() for _ in range(10)]
    
    text = "".join(sample_lines)
    sep = ";" if text.count(";") > text.count(",") else ","
    # Check if numbers contain German decimal comma like "12,5"
    dec = "," if ";" in text and any("," in token for token in text.split(";") if any(c.isdigit() for c in token)) else "."
    return sep, dec


def scan_raw_directory(raw_dir: Path) -> List[Path]:
    """
    Scan raw directory for CSV files, returning deduplicated sorted paths.
    Uses case-insensitive glob merge to avoid duplicates on case-insensitive filesystems.
    """
    seen: set = set()
    result: List[Path] = []
    for p in sorted(raw_dir.glob("*")):
        if p.suffix.lower() == ".csv" and p.name not in seen:
            seen.add(p.name)
            result.append(p)
    return result


def inspect_raw_file(file_path: Path, n_rows: int = 5) -> Dict:
    """
    Inspect raw file headers, dtypes, and sample rows before loading.
    """
    sep, dec = detect_csv_dialect(file_path)
    df_sample = pd.read_csv(
        file_path,
        sep=sep,
        decimal=dec,
        na_values=["-", "", "n/a", "N/A", "null"],
        nrows=n_rows,
        encoding="utf-8-sig",
    )
    return {
        "file_name": file_path.name,
        "separator": sep,
        "decimal": dec,
        "columns": df_sample.columns.tolist(),
        "sample_head": df_sample,
    }


def load_smard_csv(file_path: Path) -> pd.DataFrame:
    """
    Load a SMARD CSV file into a DataFrame, parsing German number formatting
    and treating '-' as NaN.
    """
    sep, dec = detect_csv_dialect(file_path)
    
    # Read as string first or with explicit na_values to avoid conversion warnings
    df = pd.read_csv(
        file_path,
        sep=sep,
        decimal=dec,
        na_values=["-", "", " ", "n/a", "N/A", "null"],
        encoding="utf-8-sig",
        low_memory=False,
    )
    return df

