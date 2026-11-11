# Makefile — SMARD Price Analysis Project
# Requires: GNU make (or nmake on Windows)
# Usage: make help / make all / make data / make test

.PHONY: help all download data eda stats forecast powerbi test lint clean

## Show this help message
help:
	@echo.
	@echo  SMARD Price Analysis - Available targets:
	@echo  ------------------------------------------
	@echo  make download    Download raw SMARD data via API
	@echo  make data        Step 1 - Clean, merge, audit
	@echo  make eda         Step 2 - EDA figures
	@echo  make stats       Step 3 - Statistical tests
	@echo  make forecast    Step 4 - ML models + SHAP
	@echo  make powerbi     Step 5 - Power BI export
	@echo  make test        Run all 22 unit tests
	@echo  make lint        Ruff code quality check
	@echo  make all         Run everything (Steps 1-5 + tests)
	@echo  make clean       Delete generated outputs (keeps raw data)
	@echo.

## Download raw SMARD data from official API
download:
	python scripts/download_all_smard.py

## Step 1: Clean, merge and audit
data:
	python scripts/run_step1_audit.py

## Step 2: EDA figures
eda:
	python scripts/run_step2_eda.py

## Step 3: Statistical hypothesis tests
stats:
	python scripts/run_step3_stats.py

## Step 4: Forecasting models + SHAP
forecast:
	python scripts/run_step4_forecast.py

## Step 5: Power BI export
powerbi:
	python scripts/run_step5_powerbi.py

## Run all pipeline steps (no download)
all:
	python run_all.py

## Run with fresh API download
all-fresh:
	python run_all.py --download

## Run unit tests
test:
	pytest tests/ -v --tb=short

## Lint all Python source files
lint:
	ruff check src/ scripts/ tests/ run_all.py

## Execute Jupyter notebooks (requires pre-built processed data)
notebooks:
	jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=180 notebooks/01_data_audit.ipynb
	jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=180 notebooks/02_eda.ipynb
	jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 notebooks/03_statistical_tests.ipynb
	jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 notebooks/04_forecasting.ipynb

## Remove all generated outputs (keeps raw data and cached downloads)
clean:
	python -c "import shutil; [shutil.rmtree(d, ignore_errors=True) for d in ['data/processed','data/interim/cache','reports/figures']]; print('Cleaned generated outputs.')"

