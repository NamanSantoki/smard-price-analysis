# Wind, Solar and the German Day-Ahead Electricity Price: Statistical Analysis, Negative-Price Hours and Price Forecasting

## 3-Line Summary

German day-ahead electricity prices fell significantly during hours of high wind and solar generation (median price difference: **−40 €/MWh for high wind vs low wind**, **Cliff's δ = −0.38**). The share of negative-price hours rose from 2.4 % in 2019 to **6.5 % in 2025**, driven disproportionately by midday solar peaks — high-solar hours carry an **odds ratio of 9.3** (95 % CI: 8.4 – 10.2) for negative prices compared to low-solar hours. A LightGBM model using realised generation drivers as an upper-bound benchmark achieved **MAE = 28.5 €/MWh** and **RMSE = 42.5 €/MWh** on the held-out 12-month test period.

---

## 1. Research Question

> *How do wind and solar generation affect German day-ahead electricity prices and the frequency of negative-price hours, and how well can next-day prices be forecast?*

---

## 2. Data Source

| Field | Detail |
|-------|--------|
| Provider | Bundesnetzagentur – [SMARD.de](https://www.smard.de/en/downloadcenter/data-download) |
| Download date | 2026-11-11 |
| Price zone | Germany / Luxembourg (DE-LU) — applies from 2018-10-01 |
| Period | 2019-01-01 to 2026-11-15 |
| Resolution | Hourly |
| Series downloaded | Day-ahead wholesale price, Wind onshore, Wind offshore, Solar (PV), Grid load, Residual load, Biomass, Hydro, Hard coal, Lignite, Natural gas |

**How to reproduce download:** `python run_all.py --download`

Data is fetched via the official SMARD public JSON API (no account required). Raw files are stored read-only in `data/raw/`.

---

## 3. Method Overview

### 3.1 Cleaning & Harmonisation (`src/load.py`, `src/clean.py`)

- Detected semicolon separators and German decimal commas (`12,34` → `12.34`) automatically.
- Timestamps parsed in `Europe/Berlin` with explicit handling of DST transitions:
  - **October** (25-hour day): repeated 02:00 hour mapped to two distinct UTC instants using `ambiguous="infer"`.
  - **March** (23-hour day): non-existent 02:00 shifted forward with `nonexistent="shift_forward"`.
- Stored in UTC for all downstream computation.
- Uniform hourly grid: 69,023 rows with **0 duplicate timestamps** and **0 index gaps**.
- Short gaps ≤ 2 consecutive hours: linearly interpolated. Longer gaps: left as `NaN`.
- **Nuclear missing data (35 % overall):** correctly reflects the statutory phaseout (Atomausstieg) completed April 2023.

### 3.2 Exploratory Analysis (`src/plots.py`, Step 2)

- Price distribution is strongly **right-skewed** (skewness = 2.36, excess kurtosis = 8.74), with a minimum of **−500 €/MWh** and a maximum of **936.28 €/MWh**.
- Average price: **97.49 €/MWh**; median: **79.78 €/MWh** — the gap reflects the influence of extreme spikes.
- Diurnal price curves show a deepening midday "dip" in later years as solar capacity grew (solar duck curve).
- Seasonal profiles confirm wind peaks in winter; solar peaks in summer — partial complementarity.

### 3.3 Statistical Tests (`src/stats_tests.py`, Step 3)

| Test | Purpose | Key Result |
|------|---------|-----------|
| Mann-Whitney U + Bootstrap CI | High vs low wind regime | Median diff = −40.1 €/MWh (95 % CI: −41.6, −38.4), p ≈ 0, Cliff's δ = −0.38 |
| Mann-Whitney U + Bootstrap CI | High vs low residual load | Median diff = +73.7 €/MWh (95 % CI: +71.6, +76.3), Cliff's δ = +0.62 |
| Stratified hourly MWU + BH FDR | Wind effect per hour | Significant at every hour of day after FDR correction |
| χ² + Fisher's Exact + Odds Ratio | High solar → negative prices | OR = 9.25 (95 % CI: 8.36 – 10.24), p ≈ 0 |
| χ² + Fisher's Exact + Odds Ratio | High wind → negative prices | OR = 3.06 (95 % CI: 2.82 – 3.31), p ≈ 0 |
| OLS with Newey-West HAC (maxlags=24) | Merit-order regression | Wind coeff = **−2.86 €/MWh per GW** (p < 0.001); Solar = **−2.35 €/MWh per GW** (p < 0.001); Load = **+1.37 €/MWh per GW** |

**Limitations:** Residuals are strongly autocorrelated (Ljung-Box p ≈ 0 at lags 1, 12, 24) — expected for hourly price series. HAC standard errors correct inference but do not remove endogeneity. Gas prices and cross-border flows are unobserved confounders. Correlation ≠ causation.

### 3.4 Forecasting (`src/models.py`, `src/evaluate.py`, Step 4)

> **Important:** Generation and load features are **realised** (actual) values, not day-ahead forecasts. This means model accuracy represents an **upper bound** of what a true forecasting system would achieve with actual day-ahead driver predictions.

| Model | MAE (€/MWh) | RMSE (€/MWh) | sMAPE (%) |
|-------|-------------|--------------|-----------|
| Naive (Lag 24h) | 31.57 | 49.45 | 43.56 % |
| Seasonal Naive (Lag 168h) | 41.59 | 64.21 | 51.33 % |
| **Ridge (α=100)** | **22.46** | **34.29** | **33.89 %** |
| LightGBM | 28.51 | 42.52 | 40.72 % |
| XGBoost | 29.94 | 43.57 | 42.73 % |

**Negative-price classification** (LightGBM, threshold optimised for F1):

| Metric | Value |
|--------|-------|
| Precision | 0.692 |
| Recall | 0.660 |
| F1 | 0.676 |
| PR-AUC | 0.719 |
| ROC-AUC | 0.980 |

SHAP analysis confirms **price_lag_24**, **residual_load_mw_driver**, and **wind_total_mw_driver** as the top three predictors by mean absolute SHAP value.

---

## 4. Key Figures

| Figure | Path |
|--------|------|
| Price Distribution (Linear + Symlog) | `reports/figures/02_price_distribution.png` |
| Negative-Price Annual Trend + Month×Hour Heatmap | `reports/figures/02_negative_price_trends.png` |
| Merit-Order Binned Scatter (Wind, Solar, Residual Load) | `reports/figures/02_merit_order_binned.png` |
| Diurnal Price Curves by Year | `reports/figures/02_diurnal_price_by_year.png` |
| Seasonal Generation Profiles (Wind vs Solar) | `reports/figures/02_generation_profiles_seasonal.png` |
| Rolling Correlations Over Time | `reports/figures/02_rolling_correlations.png` |
| LightGBM SHAP Summary | `reports/figures/04_shap_summary.png` |
| Precision-Recall Curve (Negative Price) | `reports/figures/04_pr_curve_negative_price.png` |
| Forecast vs Actuals: Summer Week | `reports/figures/04_forecast_summer_week.png` |
| Forecast vs Actuals: Winter Week | `reports/figures/04_forecast_winter_week.png` |

---

## 5. Limitations and Next Steps

**Limitations:**
1. All generation features are **realised** values — results represent the best achievable with perfect driver knowledge, not a deployable real-time forecast.
2. No gas price index or carbon (ETS) price was included; both are significant merit-order confounders, especially in 2021–2023.
3. Cross-border electricity flows (imports from France, Norway, etc.) are absent, which affects residual-load precision.
4. Autocorrelation in prices violates OLS i.i.d. assumptions even with HAC correction; a full VAR or ARIMAX model would be more appropriate for causal inference.
5. Nuclear generation data is unreliable from 2023 (Atomausstieg); this series was left with `NaN` and excluded from models.

**What I would do next:**
- Add ENTSO-E day-ahead generation *forecasts* (freely available) to build a truly leakage-free production model.
- Include natural gas (TTF) and carbon (EUA) price indices as features.
- Fit a Temporal Fusion Transformer (TFT) for multi-horizon probabilistic price forecasting.
- Extend the analysis to quantile forecasting to assess prediction intervals around negative-price events.
- Run a formal Granger causality test between renewable capacity additions (annual) and negative-price frequency.

---

## 6. How to Reproduce

### Environment Setup
```bash
# Python 3.11+ required
pip install -r requirements.txt
```

### Download data and run everything
```bash
python run_all.py --download
```

### Run steps individually
```bash
python scripts/download_all_smard.py      # Step 0: Fetch SMARD data via API
python scripts/run_step1_audit.py          # Step 1: Clean, merge, audit
python scripts/run_step2_eda.py            # Step 2: EDA figures
python scripts/run_step3_stats.py          # Step 3: Hypothesis tests
python scripts/run_step4_forecast.py       # Step 4: ML models + SHAP
python scripts/run_step5_powerbi.py        # Step 5: Power BI export
pytest tests/ -v                           # Step 6: Unit tests
```

### Power BI Dashboard
1. Open Power BI Desktop.
2. Import `powerbi/dashboard_data.csv`.
3. Follow `powerbi/dashboard_spec.md` for page layout, DAX measures, and colour palette.

---

## 7. Project Structure

```
smard-price-analysis/
├── data/
│   ├── raw/                    # Read-only raw SMARD CSV files
│   ├── interim/cache/          # Parquet download cache (auto-generated)
│   └── processed/              # merged_hourly.parquet + .csv + audit JSON
├── notebooks/                  # Jupyter notebooks (01–04, mirrors scripts)
├── src/
│   ├── download.py             # SMARD API client
│   ├── load.py                 # CSV inspection and loading
│   ├── clean.py                # Parsing, DST handling, feature derivation
│   ├── features.py             # Leakage-free forecasting features
│   ├── stats_tests.py          # Non-parametric tests, regression, bootstrap
│   ├── models.py               # Baselines, Ridge, LightGBM, XGBoost, SHAP
│   ├── evaluate.py             # MAE, RMSE, sMAPE, MASE, PR-AUC
│   └── plots.py                # All publication-quality figures
├── scripts/
│   ├── download_all_smard.py
│   ├── run_step1_audit.py
│   ├── run_step2_eda.py
│   ├── run_step3_stats.py
│   ├── run_step4_forecast.py
│   └── run_step5_powerbi.py
├── reports/
│   ├── figures/                # All .png plots (300 dpi)
│   ├── statistical_tests_results.json
│   └── forecasting_results.json
├── powerbi/
│   ├── dashboard_data.csv      # 69,023 rows × 24 columns
│   ├── data_dictionary.json
│   └── dashboard_spec.md       # Page specs, DAX measures, colour palette
├── tests/
│   └── test_clean.py           # 22 pytest tests (all pass)
├── requirements.txt
├── pyproject.toml
├── run_all.py                  # End-to-end pipeline runner
└── README.md
```

---

## 8. Five Likely Interview / Admissions Questions

**Q1: Why did you use Mann-Whitney U instead of a t-test?**
Electricity prices are heavily right-skewed with extreme spikes (skewness = 2.36, kurtosis = 8.74). The t-test assumes approximate normality of group means — valid asymptotically but fragile here because the extreme values heavily influence the mean. Mann-Whitney U is a non-parametric rank-based test that is robust to non-normality and outliers. I additionally reported Cliff's delta as an interpretable, distribution-free effect-size measure.

**Q2: How did you avoid data leakage in the forecasting model?**
Two-level protection: (1) Price lags are always ≥ 24 hours (lag 24, lag 48, lag 168), because the day-ahead auction closes approximately 12 hours before delivery — same-day prices are never known at forecast time. (2) The train/test split is strictly chronological — the test set is always the last 12 months. Random shuffling was never used. I included three automated pytest assertions that verify no timestamp overlap and that every training observation precedes every test observation.

**Q3: Why is your R² only 0.21 in the OLS regression?**
This OLS is an *explanatory* specification, not a prediction model. It includes only wind (GW), solar (GW), load (GW) and calendar fixed effects — deliberately sparse to isolate the merit-order coefficients. The remaining unexplained variance reflects gas prices, carbon costs, cross-border flows, storage dispatch, and pure spot-market liquidity effects that are not in this dataset. The coefficients are still precisely estimated with HAC standard errors (p < 0.001 for all three main drivers), which is what matters for causal inference.

**Q4: Why did the LightGBM model underperform Ridge regression?**
Likely because the feature set is dominated by continuous linear relationships: price is closely correlated with its own 24-hour lag and with residual load. Ridge is well-suited to this structure. LightGBM would likely outperform Ridge if more non-linear features (gas prices, temperature, hour×season interactions) were added. Gradient-boosted trees typically excel with more signals; with only 29 features, Ridge's L2 penalty controls overfitting more efficiently.

**Q5: What do the negative-price results mean for German energy policy?**
Negative prices signal that inflexible baseload generators (lignite, run-of-river hydro) cannot reduce output fast enough when wind and solar temporarily exceed demand, forcing them to pay buyers to absorb electricity. The escalating trend — from 2.4 % of hours in 2019 to 6.5 % in 2025 — shows that the gap between renewable capacity and grid flexibility (storage, demand response, interconnection) is widening. This creates incentives for utility-scale batteries, hydrogen electrolysis and smart demand-response tariffs. It also illustrates why the energy transition requires not just generation investment but system flexibility investment.

