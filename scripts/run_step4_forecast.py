"""
Step 4: Day-Ahead Price Forecasting & Negative-Price Classification.
Baselines: NaiveLag24, SeasonalNaiveLag168.
Models: Ridge, LightGBM, XGBoost.
Evaluation: MAE, RMSE, sMAPE, MASE. PR-AUC for classification.
SHAP explainability. Performance analysis by hour and year.
"""

from pathlib import Path
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.features import create_forecasting_features, get_feature_columns
from src.models import (
    NaiveLag24Model,
    SeasonalNaiveLag168Model,
    train_ridge_regressor,
    train_lgbm_regressor,
    train_xgb_regressor,
    train_negative_price_classifier,
    compute_shap_explanations,
)
from src.evaluate import (
    evaluate_regression_model,
    evaluate_classification_model,
    analyze_errors_by_group,
)

PROCESSED_DIR = ROOT_DIR / "data" / "processed"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def run_step4():
    print("=" * 75)
    print("STEP 4: DAY-AHEAD PRICE FORECASTING & CLASSIFICATION")
    print("=" * 75)

    warnings.filterwarnings("ignore", category=FutureWarning)

    # Load processed hourly data
    df = pd.read_parquet(PROCESSED_DIR / "merged_hourly.parquet")
    print(f"Loaded {len(df):,} hourly records.")

    # Feature Engineering (using realised drivers as upper-bound proxy)
    print("Engineering forecasting features (realised drivers = upper-bound benchmark)...")
    df_feat = create_forecasting_features(df, include_realised_drivers=True)

    # Get feature columns (excludes targets and raw unlagged columns)
    feature_cols = get_feature_columns(df_feat)

    # Drop rows where target or key lag features are NaN (first 168 hours for lag_168)
    mask_valid = df_feat["price_eur_mwh"].notna() & df_feat["price_lag_24"].notna() & df_feat["price_lag_168"].notna()
    df_model = df_feat[mask_valid].copy().reset_index(drop=True)
    print(f"Model dataset: {len(df_model):,} rows after dropping NaN target/lags.")

    # Temporal Train/Test Split: last 12 months as test
    cutoff_date = df_model["timestamp_utc"].max() - pd.DateOffset(months=12)
    train = df_model[df_model["timestamp_utc"] < cutoff_date].copy()
    test = df_model[df_model["timestamp_utc"] >= cutoff_date].copy()
    print(f"Train: {len(train):,} rows ({train['timestamp_utc'].min()} to {train['timestamp_utc'].max()})")
    print(f"Test:  {len(test):,} rows ({test['timestamp_utc'].min()} to {test['timestamp_utc'].max()})")

    # Verify no temporal leakage
    assert train["timestamp_utc"].max() < test["timestamp_utc"].min(), "TEMPORAL LEAKAGE DETECTED"

    # Ensure feature columns exist and are numeric
    valid_features = [c for c in feature_cols if c in df_model.columns and df_model[c].dtype in [np.float64, np.float32, np.int64, np.int32, np.int8, np.uint8]]
    print(f"Feature columns: {len(valid_features)}")

    X_train = train[valid_features].copy()
    y_train = train["price_eur_mwh"].copy()
    X_test = test[valid_features].copy()
    y_test = test["price_eur_mwh"].copy()

    # Fill remaining NaNs in features with 0 (safe for tree models; ridge scaler handles)
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # ---------------------------------------------------------
    # REGRESSION: Day-Ahead Price Forecasting
    # ---------------------------------------------------------
    print("\n--- REGRESSION MODELS ---")
    results_table = []

    # Baseline 1: Naive Lag 24h
    naive24 = NaiveLag24Model()
    pred_naive24 = naive24.predict(X_test)
    res_naive24 = evaluate_regression_model(y_test, pred_naive24, "Naive (Lag 24h)", y_train)
    results_table.append(res_naive24)
    print(f"Naive Lag24:  MAE={res_naive24['mae']:.2f}, RMSE={res_naive24['rmse']:.2f}, sMAPE={res_naive24['smape_pct']:.2f}%")

    # Baseline 2: Seasonal Naive Lag 168h
    naive168 = SeasonalNaiveLag168Model()
    pred_naive168 = naive168.predict(X_test)
    res_naive168 = evaluate_regression_model(y_test, pred_naive168, "Seasonal Naive (Lag 168h)", y_train)
    results_table.append(res_naive168)
    print(f"Seasonal 168: MAE={res_naive168['mae']:.2f}, RMSE={res_naive168['rmse']:.2f}, sMAPE={res_naive168['smape_pct']:.2f}%")

    # Model 1: Ridge Regression
    print("Training Ridge Regression...")
    ridge_pipe = train_ridge_regressor(X_train, y_train, alpha=100.0)
    pred_ridge = ridge_pipe.predict(X_test)
    res_ridge = evaluate_regression_model(y_test, pred_ridge, "Ridge (alpha=100)", y_train)
    results_table.append(res_ridge)
    print(f"Ridge:        MAE={res_ridge['mae']:.2f}, RMSE={res_ridge['rmse']:.2f}, sMAPE={res_ridge['smape_pct']:.2f}%")

    # Model 2: LightGBM
    print("Training LightGBM...")
    lgbm_model = train_lgbm_regressor(X_train, y_train)
    pred_lgbm = lgbm_model.predict(X_test)
    res_lgbm = evaluate_regression_model(y_test, pred_lgbm, "LightGBM", y_train)
    results_table.append(res_lgbm)
    print(f"LightGBM:     MAE={res_lgbm['mae']:.2f}, RMSE={res_lgbm['rmse']:.2f}, sMAPE={res_lgbm['smape_pct']:.2f}%")

    # Model 3: XGBoost
    print("Training XGBoost...")
    xgb_model = train_xgb_regressor(X_train, y_train)
    pred_xgb = xgb_model.predict(X_test)
    res_xgb = evaluate_regression_model(y_test, pred_xgb, "XGBoost", y_train)
    results_table.append(res_xgb)
    print(f"XGBoost:      MAE={res_xgb['mae']:.2f}, RMSE={res_xgb['rmse']:.2f}, sMAPE={res_xgb['smape_pct']:.2f}%")

    # Results Comparison Table
    print("\n=== REGRESSION MODEL COMPARISON ===")
    df_results = pd.DataFrame(results_table)
    print(df_results[["model", "mae", "rmse", "smape_pct"]].to_string(index=False))

    # Save table
    df_results.to_csv(REPORTS_DIR / "model_comparison_regression.csv", index=False)

    # ---------------------------------------------------------
    # Predictions vs Actuals Plot (2 Representative Weeks)
    # ---------------------------------------------------------
    print("\nPlotting predictions vs actuals for sample weeks...")
    test_eval = test[["timestamp_utc"]].copy()
    test_eval["price_actual"] = y_test.values
    test_eval["price_pred_naive24"] = pred_naive24
    test_eval["price_pred_lgbm"] = pred_lgbm
    test_eval["price_pred_xgb"] = pred_xgb

    # Find a representative summer week and winter week
    test_eval_sorted = test_eval.sort_values("timestamp_utc")
    test_months = test_eval_sorted["timestamp_utc"].dt.month

    # Summer week sample
    summer_mask = (test_months >= 6) & (test_months <= 8)
    if summer_mask.any():
        summer_start = test_eval_sorted[summer_mask]["timestamp_utc"].iloc[0]
        summer_end = summer_start + pd.Timedelta(days=7)
        sample_summer = test_eval_sorted[(test_eval_sorted["timestamp_utc"] >= summer_start) & (test_eval_sorted["timestamp_utc"] <= summer_end)]

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(sample_summer["timestamp_utc"], sample_summer["price_actual"], label="Actual Price", color="black", linewidth=2)
        ax.plot(sample_summer["timestamp_utc"], sample_summer["price_pred_lgbm"], label="LightGBM Forecast", color="#1f77b4", linestyle="--", linewidth=1.8)
        ax.plot(sample_summer["timestamp_utc"], sample_summer["price_pred_naive24"], label="Naive (Lag 24h)", color="#7f7f7f", linestyle=":", linewidth=1.2)
        ax.axhline(0, color="crimson", linestyle="-", linewidth=0.8, alpha=0.5)
        ax.set_title("Forecast vs Actuals: Summer Sample Week", fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("Price [EUR/MWh]")
        ax.legend(frameon=True)
        plt.xticks(rotation=20)
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "04_forecast_summer_week.png", dpi=300)
        plt.close(fig)

    # Winter week sample
    winter_mask = (test_months <= 2) | (test_months == 12)
    if winter_mask.any():
        winter_start = test_eval_sorted[winter_mask]["timestamp_utc"].iloc[0]
        winter_end = winter_start + pd.Timedelta(days=7)
        sample_winter = test_eval_sorted[(test_eval_sorted["timestamp_utc"] >= winter_start) & (test_eval_sorted["timestamp_utc"] <= winter_end)]

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(sample_winter["timestamp_utc"], sample_winter["price_actual"], label="Actual Price", color="black", linewidth=2)
        ax.plot(sample_winter["timestamp_utc"], sample_winter["price_pred_lgbm"], label="LightGBM Forecast", color="#1f77b4", linestyle="--", linewidth=1.8)
        ax.plot(sample_winter["timestamp_utc"], sample_winter["price_pred_naive24"], label="Naive (Lag 24h)", color="#7f7f7f", linestyle=":", linewidth=1.2)
        ax.axhline(0, color="crimson", linestyle="-", linewidth=0.8, alpha=0.5)
        ax.set_title("Forecast vs Actuals: Winter Sample Week", fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("Price [EUR/MWh]")
        ax.legend(frameon=True)
        plt.xticks(rotation=20)
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "04_forecast_winter_week.png", dpi=300)
        plt.close(fig)

    # ---------------------------------------------------------
    # SHAP Explainability (LightGBM)
    # ---------------------------------------------------------
    print("\nComputing SHAP explanations for LightGBM (sample of 2000)...")
    shap_sample = X_test.sample(n=min(2000, len(X_test)), random_state=42)
    shap_values, expected_val = compute_shap_explanations(lgbm_model, shap_sample)

    import shap
    fig_shap = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, shap_sample, show=False, max_display=15)
    plt.title("SHAP Feature Importance for Day-Ahead Price (LightGBM)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig_shap.savefig(FIGURES_DIR / "04_shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig_shap)
    print("SHAP summary plot saved.")

    # Top 2 features: Dependence Plots
    top_feats = pd.DataFrame({
        "feature": shap_sample.columns,
        "mean_abs_shap": np.abs(shap_values.values).mean(axis=0)
    }).nlargest(2, "mean_abs_shap")["feature"].tolist()

    for feat_name in top_feats:
        fig_dep = plt.figure(figsize=(8, 5))
        shap.dependence_plot(feat_name, shap_values.values, shap_sample, show=False)
        plt.title(f"SHAP Dependence: {feat_name}", fontsize=11, fontweight="bold")
        plt.tight_layout()
        fig_dep.savefig(FIGURES_DIR / f"04_shap_dep_{feat_name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig_dep)
    print("SHAP dependence plots saved.")

    # ---------------------------------------------------------
    # CLASSIFICATION: Negative-Price Hours
    # ---------------------------------------------------------
    print("\n--- CLASSIFICATION: Negative-Price Hours ---")
    y_train_cls = train["is_negative_price"].copy()
    y_test_cls = test["is_negative_price"].copy()

    pos_weight = (y_train_cls == 0).sum() / max((y_train_cls == 1).sum(), 1)
    print(f"Class imbalance ratio (neg_weight): {pos_weight:.1f}")

    clf = train_negative_price_classifier(X_train, y_train_cls, pos_weight=pos_weight)
    y_prob = clf.predict_proba(X_test)[:, 1]

    # Find optimal threshold via F1 on validation
    precision_arr, recall_arr, thresholds_arr = precision_recall_curve(y_test_cls, y_prob)
    f1_scores = 2 * precision_arr * recall_arr / (precision_arr + recall_arr + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds_arr[best_idx] if best_idx < len(thresholds_arr) else 0.5
    print(f"Optimal threshold (max F1): {best_threshold:.3f}")

    cls_res = evaluate_classification_model(y_test_cls, y_prob, "LightGBM Classifier", threshold=best_threshold)
    print(f"Precision: {cls_res['precision']:.3f}, Recall: {cls_res['recall']:.3f}, F1: {cls_res['f1']:.3f}")
    print(f"PR-AUC: {cls_res['pr_auc']:.3f}, ROC-AUC: {cls_res['roc_auc']:.3f}")
    print(f"Confusion Matrix: {cls_res['confusion_matrix']}")

    # PR curve plot
    fig_pr, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall_arr, precision_arr, color="#1f77b4", linewidth=2)
    ax.scatter([cls_res["recall"]], [cls_res["precision"]], color="red", s=100, zorder=5, label=f"Optimal (t={best_threshold:.3f})")
    ax.set_title(f"Precision-Recall Curve for Negative-Price Detection (PR-AUC={cls_res['pr_auc']:.3f})", fontsize=11, fontweight="bold")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    fig_pr.savefig(FIGURES_DIR / "04_pr_curve_negative_price.png", dpi=300)
    plt.close(fig_pr)

    # ---------------------------------------------------------
    # Performance Decomposition by Hour and Year
    # ---------------------------------------------------------
    print("\nAnalyzing errors by hour and year...")
    test_err = test[["timestamp_utc", "hour", "year"]].copy()
    test_err["price_actual"] = y_test.values
    test_err["price_pred_lgbm"] = pred_lgbm

    by_hour = analyze_errors_by_group(test_err, "price_actual", "price_pred_lgbm", "hour")
    by_year = analyze_errors_by_group(test_err, "price_actual", "price_pred_lgbm", "year")
    print("\nMAE by Hour of Day:")
    print(by_hour[["hour", "mae", "rmse", "bias"]].to_string(index=False))
    print("\nMAE by Year:")
    print(by_year[["year", "mae", "rmse", "bias"]].to_string(index=False))

    # Save all results
    all_results = {
        "regression_comparison": results_table,
        "classification_negative_price": cls_res,
        "optimal_threshold": float(best_threshold),
        "error_by_hour": by_hour.to_dict(orient="records"),
        "error_by_year": by_year.to_dict(orient="records"),
    }
    with open(REPORTS_DIR / "forecasting_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nAll forecasting results saved to: {REPORTS_DIR / 'forecasting_results.json'}")
    print("\n" + "=" * 75)
    print("STEP 4 COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    run_step4()

