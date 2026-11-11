#!/usr/bin/env python
"""
run_all.py — Reproduce every step of the SMARD Price Analysis project
from raw data to figures, models and dashboard export.

Usage:
    python run_all.py            # run all steps (assumes data/raw/ is populated)
    python run_all.py --download # also download raw data from SMARD API first
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"


def run(script_name: str) -> None:
    path = SCRIPTS / script_name
    print(f"\n{'=' * 70}")
    print(f"  RUNNING: {script_name}")
    print(f"{'=' * 70}\n")
    result = subprocess.run([sys.executable, str(path)], cwd=str(ROOT))
    if result.returncode != 0:
        print(f"\n*** {script_name} failed with exit code {result.returncode} ***")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Reproduce SMARD Price Analysis end-to-end.")
    parser.add_argument("--download", action="store_true",
                        help="Download raw SMARD data from the API before processing.")
    args = parser.parse_args()

    if args.download:
        run("download_all_smard.py")

    run("run_step1_audit.py")
    run("run_step2_eda.py")
    run("run_step3_stats.py")
    run("run_step4_forecast.py")
    run("run_step5_powerbi.py")

    # Run unit tests
    print(f"\n{'=' * 70}")
    print("  RUNNING TESTS")
    print(f"{'=' * 70}\n")
    result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"], cwd=str(ROOT))
    if result.returncode != 0:
        print("\n*** Tests failed ***")

    print(f"\n{'=' * 70}")
    print("  ALL STEPS COMPLETE")
    print(f"{'=' * 70}")
    print(f"Processed data:  data/processed/merged_hourly.parquet")
    print(f"Figures:         reports/figures/")
    print(f"Statistical tests: reports/statistical_tests_results.json")
    print(f"Forecasting:     reports/forecasting_results.json")
    print(f"Power BI export: powerbi/dashboard_data.csv")


if __name__ == "__main__":
    main()

