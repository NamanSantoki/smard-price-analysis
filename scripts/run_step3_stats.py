"""
Step 3: Statistical Hypothesis Testing & Econometric Merit-Order Modeling.
Executes non-parametric tests, contingency tables, multiple comparisons corrections,
and Newey-West HAC autoregressive regression.
"""

from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.stats_tests import (
    test_generation_regimes,
    stratified_hourly_tests,
    test_negative_price_odds,
    fit_hac_merit_order_regression,
)

PROCESSED_DIR = ROOT_DIR / "data" / "processed"
REPORTS_DIR = ROOT_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def run_step3():
    print("=" * 75)
    print("STEP 3: STATISTICAL TESTING & ECONOMETRIC MODELING")
    print("=" * 75)

    df = pd.read_parquet(PROCESSED_DIR / "merged_hourly.parquet")
    print(f"Loaded {len(df):,} hourly records.")

    all_results = {}

    # -------------------------------------------------------------
    # 1. Non-Parametric Tests: High vs Low Generation Regimes
    # -------------------------------------------------------------
    print("\n--- 1. Non-Parametric Tests (Mann-Whitney U & Cliff's Delta) ---")
    regimes = [
        ("wind_total_mw", "Total Wind (Onshore + Offshore)"),
        ("solar_mw", "Solar (PV)"),
        ("residual_load_mw", "Residual Load")
    ]

    regime_results = []
    for col, desc in regimes:
        res = test_generation_regimes(df, driver_col=col, price_col="price_eur_mwh", n_boot=2000)
        regime_results.append(res)
        all_results[f"regime_{col}"] = res
        print(f"\nDriver: {desc} ({col})")
        print(f"  Sample sizes: N_high={res['n_high']:,}, N_low={res['n_low']:,}")
        print(f"  Median price: High={res['median_high']:.2f} €/MWh, Low={res['median_low']:.2f} €/MWh")
        print(f"  Observed Median Diff (High - Low): {res['median_diff']:.2f} €/MWh (95% CI: [{res['median_diff_ci_95'][0]:.2f}, {res['median_diff_ci_95'][1]:.2f}])")
        print(f"  Observed Mean Diff (High - Low):   {res['mean_diff']:.2f} €/MWh (95% CI: [{res['mean_diff_ci_95'][0]:.2f}, {res['mean_diff_ci_95'][1]:.2f}])")
        print(f"  Mann-Whitney U: {res['mann_whitney_u']:.2e}, p-value: {res['p_value']:.2e}")
        print(f"  Cliff's Delta:  {res['cliffs_delta']:.4f}")

    # -------------------------------------------------------------
    # 2. Hour-Stratified Tests with FDR Correction (Controlling for Diurnal Demand)
    # -------------------------------------------------------------
    print("\n--- 2. Hour-Stratified Tests (Wind Effect Controlling for Hour of Day) ---")
    strat_wind = stratified_hourly_tests(df, driver_col="wind_total_mw", price_col="price_eur_mwh")
    print(strat_wind[["hour", "med_low", "med_high", "med_diff", "cliffs_delta", "p_value_fdr", "significant_fdr"]].to_string(index=False))
    all_results["stratified_wind"] = strat_wind.to_dict(orient="records")

    # -------------------------------------------------------------
    # 3. Contingency Table Analysis for Negative-Price Hours
    # -------------------------------------------------------------
    print("\n--- 3. Negative-Price Odds Ratio & Contingency Analysis ---")
    contingency_drivers = [
        ("solar_mw", "High Solar Generation (Top 25%)"),
        ("wind_total_mw", "High Wind Generation (Top 25%)")
    ]
    for col, desc in contingency_drivers:
        # For solar, restrict to daylight hours (08:00 - 18:00) where solar actually generates
        sub_df = df if col != "solar_mw" else df[(df["hour"] >= 8) & (df["hour"] <= 18)]
        odds_res = test_negative_price_odds(sub_df, driver_col=col, quantile_cutoff=0.75)
        all_results[f"negative_odds_{col}"] = odds_res
        print(f"\nCondition: {desc}")
        print(f"  Contingency Table [[Neg_High, Pos_High], [Neg_Low, Pos_Low]]: {odds_res['contingency_table']}")
        print(f"  Negative rate in High regime: {odds_res['neg_rate_high_gen'] * 100:.2f}%")
        print(f"  Negative rate in Low regime:  {odds_res['neg_rate_low_gen'] * 100:.2f}%")
        print(f"  Odds Ratio: {odds_res['odds_ratio']:.3f} (95% CI: [{odds_res['odds_ratio_ci_95'][0]:.3f}, {odds_res['odds_ratio_ci_95'][1]:.3f}])")
        print(f"  Chi-Square Stat: {odds_res['chi2_stat']:.2f}, p-value: {odds_res['p_chi2']:.2e}")

    # -------------------------------------------------------------
    # 4. Econometric Regression with Newey-West HAC Robust Standard Errors
    # -------------------------------------------------------------
    print("\n--- 4. Econometric Merit-Order Regression with HAC Standard Errors (maxlags=24) ---")
    reg_model = fit_hac_merit_order_regression(df, max_lags=24)
    print(reg_model.summary().tables[0])
    print("\nKey Econometric Estimates (EUR per MWh per GW added):")
    coef_table = pd.DataFrame({
        "Coefficient": reg_model.params,
        "HAC Std Error": reg_model.bse,
        "t-Statistic": reg_model.tvalues,
        "p-value": reg_model.pvalues,
        "95% CI Lower": reg_model.conf_int()[0],
        "95% CI Upper": reg_model.conf_int()[1]
    })
    key_vars = ["wind_total_gw", "solar_gw", "load_gw"]
    print(coef_table.loc[key_vars].round(4).to_string())

    # Residual Autocorrelation Test (Ljung-Box)
    residuals = reg_model.resid
    lb_test = acorr_ljungbox(residuals, lags=[1, 12, 24], return_df=True)
    print("\nResidual Diagnostics: Ljung-Box Test for Autocorrelation:")
    print(lb_test.to_string())

    # Save summary dictionary
    all_results["regression"] = {
        "r_squared": float(reg_model.rsquared),
        "r_squared_adj": float(reg_model.rsquared_adj),
        "n_obs": int(reg_model.nobs),
        "coefficients": coef_table.loc[key_vars].to_dict(orient="index"),
        "ljung_box_pvalues": lb_test["lb_pvalue"].to_dict()
    }

    with open(REPORTS_DIR / "statistical_tests_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll statistical test outputs saved to: {REPORTS_DIR / 'statistical_tests_results.json'}")


if __name__ == "__main__":
    run_step3()

