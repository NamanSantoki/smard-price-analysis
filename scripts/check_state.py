"""Quick project state checker."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Check notebook execution status
print("=== NOTEBOOK STATUS ===")
for nb in ['01_data_audit', '02_eda', '03_statistical_tests', '04_forecasting']:
    p = ROOT / f'notebooks/{nb}.ipynb'
    if p.exists():
        data = json.loads(p.read_text(encoding='utf-8'))
        cells_with_output = sum(1 for c in data['cells'] if c['cell_type'] == 'code' and c.get('outputs'))
        total_code = sum(1 for c in data['cells'] if c['cell_type'] == 'code')
        print(f"  {nb}: {cells_with_output}/{total_code} code cells have outputs")
    else:
        print(f"  {nb}: MISSING")

# Check figures
figs = sorted((ROOT / 'reports/figures').glob('*.png'))
print(f"\n=== FIGURES ({len(figs)}) ===")
for f in figs:
    print(f"  {f.name}")

# Check key reports
print("\n=== REPORTS ===")
for r in [
    'reports/statistical_tests_results.json',
    'reports/forecasting_results.json',
    'reports/model_comparison_regression.csv',
    'powerbi/dashboard_data.csv',
    'data/processed/merged_hourly.parquet',
]:
    p = ROOT / r
    status = f"EXISTS ({p.stat().st_size // 1024} KB)" if p.exists() else "MISSING"
    print(f"  {r}: {status}")

