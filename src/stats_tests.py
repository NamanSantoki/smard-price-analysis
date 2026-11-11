"""
Statistical hypothesis testing module for German electricity market dynamics.
Implements non-parametric tests (Mann-Whitney U, Cliff's delta, Bootstrap CIs),
contingency tests (Chi-squared, Fisher's exact, Odds Ratio), multiple testing corrections,
and HAC-robust autoregressive econometrics.
"""

from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


def bootstrap_ci_diff(
    x: np.ndarray,
    y: np.ndarray,
    stat_func=np.median,
    n_boot: int = 2000,
    ci: float = 0.95,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for difference in statistics (stat(x) - stat(y)).
    Returns (observed_diff, ci_lower, ci_upper).
    """
    rng = np.random.default_rng(random_state)
    obs_diff = float(stat_func(x) - stat_func(y))
    boot_diffs = np.empty(n_boot)
    
    n_x, n_y = len(x), len(y)
    for i in range(n_boot):
        sample_x = rng.choice(x, size=n_x, replace=True)
        sample_y = rng.choice(y, size=n_y, replace=True)
        boot_diffs[i] = stat_func(sample_x) - stat_func(sample_y)
        
    alpha = 1.0 - ci
    ci_lower = float(np.percentile(boot_diffs, 100 * (alpha / 2)))
    ci_upper = float(np.percentile(boot_diffs, 100 * (1 - alpha / 2)))
    return obs_diff, ci_lower, ci_upper


def cliffs_delta(x: np.ndarray, y: np.ndarray, u_stat: float) -> float:
    """
    Compute Cliff's delta effect size from Mann-Whitney U statistic:
    delta = (2 * U) / (n1 * n2) - 1
    """
    n1, n2 = len(x), len(y)
    return float((2.0 * u_stat) / (n1 * n2) - 1.0)


def test_generation_regimes(
    df: pd.DataFrame,
    driver_col: str = "wind_total_mw",
    price_col: str = "price_eur_mwh",
    q_low: float = 0.25,
    q_high: float = 0.75,
    n_boot: int = 1500
) -> Dict:
    """
    Non-parametric comparison of electricity prices in high vs low generation regimes.
    Uses Mann-Whitney U test, Cliff's delta effect size, and bootstrap CIs for mean & median.
    """
    clean_df = df[[driver_col, price_col]].dropna()
    low_cutoff = clean_df[driver_col].quantile(q_low)
    high_cutoff = clean_df[driver_col].quantile(q_high)

    low_group = clean_df.loc[clean_df[driver_col] <= low_cutoff, price_col].values
    high_group = clean_df.loc[clean_df[driver_col] >= high_cutoff, price_col].values

    # Mann-Whitney U (high vs low)
    # H0: P(High > Low) = P(Low > High)
    u_stat, p_val = stats.mannwhitneyu(high_group, low_group, alternative="two-sided")
    delta = cliffs_delta(high_group, low_group, u_stat)

    # Bootstrap CIs
    obs_med_diff, med_ci_l, med_ci_u = bootstrap_ci_diff(high_group, low_group, stat_func=np.median, n_boot=n_boot)
    obs_mean_diff, mean_ci_l, mean_ci_u = bootstrap_ci_diff(high_group, low_group, stat_func=np.mean, n_boot=n_boot)

    return {
        "driver": driver_col,
        "n_high": len(high_group),
        "n_low": len(low_group),
        "median_high": float(np.median(high_group)),
        "median_low": float(np.median(low_group)),
        "mean_high": float(np.mean(high_group)),
        "mean_low": float(np.mean(low_group)),
        "mann_whitney_u": float(u_stat),
        "p_value": float(p_val),
        "cliffs_delta": float(delta),
        "median_diff": obs_med_diff,
        "median_diff_ci_95": (med_ci_l, med_ci_u),
        "mean_diff": obs_mean_diff,
        "mean_diff_ci_95": (mean_ci_l, mean_ci_u),
    }


def stratified_hourly_tests(
    df: pd.DataFrame,
    driver_col: str = "wind_total_mw",
    price_col: str = "price_eur_mwh",
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Control for diurnal demand cycles by running Mann-Whitney U tests
    stratified by hour of day (0..23), followed by Benjamini-Hochberg FDR correction.
    """
    results = []
    clean_df = df[["hour", driver_col, price_col]].dropna()

    for h in range(24):
        sub = clean_df[clean_df["hour"] == h]
        q25 = sub[driver_col].quantile(0.25)
        q75 = sub[driver_col].quantile(0.75)
        low = sub.loc[sub[driver_col] <= q25, price_col].values
        high = sub.loc[sub[driver_col] >= q75, price_col].values

        if len(low) > 10 and len(high) > 10:
            u_stat, p_val = stats.mannwhitneyu(high, low, alternative="two-sided")
            delta = cliffs_delta(high, low, u_stat)
            results.append({
                "hour": h,
                "n_low": len(low),
                "n_high": len(high),
                "med_low": float(np.median(low)),
                "med_high": float(np.median(high)),
                "med_diff": float(np.median(high) - np.median(low)),
                "u_stat": float(u_stat),
                "p_value": float(p_val),
                "cliffs_delta": float(delta),
            })

    res_df = pd.DataFrame(results)
    # Apply Benjamini-Hochberg FDR correction
    reject, pvals_corrected, _, _ = multipletests(res_df["p_value"], alpha=alpha, method="fdr_bh")
    res_df["p_value_fdr"] = pvals_corrected
    res_df["significant_fdr"] = reject
    return res_df


def test_negative_price_odds(
    df: pd.DataFrame,
    driver_col: str = "solar_mw",
    quantile_cutoff: float = 0.75
) -> Dict:
    """
    Contingency table analysis: test whether the probability of negative-price hours
    is significantly higher during top-quartile generation periods.
    Computes Chi-squared test, Fisher's exact test, and Odds Ratio with 95% CI.
    """
    clean_df = df[[driver_col, "is_negative_price"]].dropna().copy()
    cutoff = clean_df[driver_col].quantile(quantile_cutoff)
    clean_df["is_high_gen"] = (clean_df[driver_col] >= cutoff).astype(int)

    # Contingency Table:
    #                 Negative Price (1) | Positive Price (0)
    # High Gen (1)  |         a         |         b
    # Low Gen (0)   |         c         |         d
    contingency = pd.crosstab(clean_df["is_high_gen"], clean_df["is_negative_price"])
    a = contingency.loc[1, 1] if (1 in contingency.index and 1 in contingency.columns) else 0
    b = contingency.loc[1, 0] if (1 in contingency.index and 0 in contingency.columns) else 0
    c = contingency.loc[0, 1] if (0 in contingency.index and 1 in contingency.columns) else 0
    d = contingency.loc[0, 0] if (0 in contingency.index and 0 in contingency.columns) else 0

    table_2x2 = np.array([[a, b], [c, d]])
    chi2, p_chi2, dof, _ = stats.chi2_contingency(table_2x2)
    odds_ratio, p_fisher = stats.fisher_exact(table_2x2)

    # Standard error of log odds ratio: SE = sqrt(1/a + 1/b + 1/c + 1/d)
    if all(v > 0 for v in [a, b, c, d]):
        se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
        log_or = np.log(odds_ratio)
        ci_lower = np.exp(log_or - 1.96 * se_log_or)
        ci_upper = np.exp(log_or + 1.96 * se_log_or)
    else:
        ci_lower, ci_upper = np.nan, np.nan

    rate_high = a / (a + b) if (a + b) > 0 else 0
    rate_low = c / (c + d) if (c + d) > 0 else 0

    return {
        "driver": driver_col,
        "contingency_table": table_2x2.tolist(),
        "neg_rate_high_gen": float(rate_high),
        "neg_rate_low_gen": float(rate_low),
        "odds_ratio": float(odds_ratio),
        "odds_ratio_ci_95": (float(ci_lower), float(ci_upper)),
        "chi2_stat": float(chi2),
        "p_chi2": float(p_chi2),
        "p_fisher": float(p_fisher),
    }


def fit_hac_merit_order_regression(
    df: pd.DataFrame,
    max_lags: int = 24
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Econometric OLS regression of day-ahead price on renewable generation and load,
    with calendar controls and Newey-West HAC standard errors (accounting for autocorrelation).
    
    Model:
    price ~ wind_total_gw + solar_gw + load_gw + C(hour) + C(month) + C(day_of_week) + is_holiday
    (Variables converted to GW for interpretable EUR/(MWh*GW) coefficients).
    """
    reg_df = df.dropna(subset=["price_eur_mwh", "wind_total_mw", "solar_mw", "load_mw"]).copy()
    
    # Scale to Gigawatts for clean readable coefficients
    reg_df["wind_total_gw"] = reg_df["wind_total_mw"] / 1000.0
    reg_df["solar_gw"] = reg_df["solar_mw"] / 1000.0
    reg_df["load_gw"] = reg_df["load_mw"] / 1000.0

    formula = (
        "price_eur_mwh ~ wind_total_gw + solar_gw + load_gw "
        "+ C(hour) + C(month) + C(day_of_week) + is_holiday"
    )
    
    model = smf.ols(formula, data=reg_df)
    results_hac = model.fit(cov_type="HAC", cov_kwds={"maxlags": max_lags})
    return results_hac

