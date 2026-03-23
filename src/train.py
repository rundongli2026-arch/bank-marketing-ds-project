from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from business_metrics import (
    BusinessAssumptions,
    build_targeting_recommendation,
    evaluate_thresholds,
    evaluate_topk,
    simulate_business,
)
from data_utils import load_dataset, split_xy
from evaluate import (
    build_models,
    evaluate_binary_classifier,
    run_cv_robustness,
    summarize_leakage,
)
from plot_utils import (
    ensure_dir,
    save_cumulative_gains,
    save_precision_recall_curve,
    save_roc_curve,
    save_threshold_tradeoff,
)


RANDOM_STATE = 42
REPORT_DIR = Path("reports")
RESULT_DIR = Path("results")
FIGURE_DIR = RESULT_DIR / "figures"

for p in [REPORT_DIR, RESULT_DIR, FIGURE_DIR]:
    ensure_dir(p)


def save_table(df: pd.DataFrame, file_name: str) -> None:
    """Save a table to both reports/ and results/ for analysis + portfolio display."""
    df.to_csv(REPORT_DIR / file_name, index=False)
    df.to_csv(RESULT_DIR / file_name, index=False)


def save_text(content: str, file_name: str) -> None:
    (REPORT_DIR / file_name).write_text(content, encoding="utf-8")
    (RESULT_DIR / file_name).write_text(content, encoding="utf-8")


def run_training() -> None:
    df = load_dataset()
    X_all, y = split_xy(df)

    train_idx, test_idx = train_test_split(
        df.index,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    feature_sets = {
        "with_duration": X_all.columns.tolist(),
        "without_duration": [c for c in X_all.columns if c != "duration"],
    }

    metric_rows = []
    confusion_rows = []
    fitted_models = {}

    for fs_name, cols in feature_sets.items():
        X_train = X_all.loc[train_idx, cols]
        X_test = X_all.loc[test_idx, cols]
        y_train = y.loc[train_idx]
        y_test = y.loc[test_idx]

        for model_name, model in build_models(X_train).items():
            metric = evaluate_binary_classifier(model, X_train, y_train, X_test, y_test)
            fitted_models[(fs_name, model_name)] = model

            metric_rows.append(
                {
                    "feature_set": fs_name,
                    "model": model_name,
                    "roc_auc": metric["roc_auc"],
                    "f1": metric["f1"],
                    "precision": metric["precision"],
                    "recall": metric["recall"],
                }
            )

            cm = metric["confusion_matrix"]
            confusion_rows.append(
                {
                    "feature_set": fs_name,
                    "model": model_name,
                    "tn": int(cm[0, 0]),
                    "fp": int(cm[0, 1]),
                    "fn": int(cm[1, 0]),
                    "tp": int(cm[1, 1]),
                }
            )

    results_df = pd.DataFrame(metric_rows).sort_values(
        ["feature_set", "roc_auc"], ascending=[True, False]
    )
    confusion_df = pd.DataFrame(confusion_rows)

    save_table(results_df, "model_metrics.csv")
    save_table(confusion_df, "confusion_matrices.csv")

    leakage_df = summarize_leakage(results_df)
    save_table(leakage_df, "leakage_summary.csv")

    deployable = results_df[results_df["feature_set"] == "without_duration"].sort_values(
        "roc_auc", ascending=False
    )
    best_deploy_model_name = deployable.iloc[0]["model"]
    best_deploy_model = fitted_models[("without_duration", best_deploy_model_name)]

    X_test_deploy = X_all.loc[test_idx, feature_sets["without_duration"]]
    y_test_deploy = y.loc[test_idx].to_numpy()
    deploy_scores = best_deploy_model.predict_proba(X_test_deploy)[:, 1]

    topk_extended = evaluate_topk(y_test_deploy, deploy_scores, top_fracs=[0.05, 0.10, 0.20, 0.30, 0.50])
    save_table(topk_extended, "deployable_lift_table.csv")

    topk_targeting = evaluate_topk(y_test_deploy, deploy_scores, top_fracs=[0.05, 0.10, 0.20, 0.30])
    save_table(topk_targeting, "topk_metrics.csv")

    threshold_metrics = evaluate_thresholds(
        y_test_deploy,
        deploy_scores,
        thresholds=np.linspace(0.10, 0.90, 17),
    )
    save_table(threshold_metrics, "threshold_metrics.csv")

    assumptions = BusinessAssumptions(
        cost_per_contact=float(os.getenv("COST_PER_CONTACT", "2.0")),
        revenue_per_conversion=float(os.getenv("REVENUE_PER_CONVERSION", "120.0")),
    )

    business_threshold = simulate_business(
        threshold_metrics,
        assumptions,
        strategy_label="threshold",
        value_col="threshold",
    )
    business_topk = simulate_business(
        topk_targeting,
        assumptions,
        strategy_label="top_fraction",
        value_col="top_fraction",
    )

    business_sim = pd.concat([business_threshold, business_topk], ignore_index=True, sort=False)
    keep_cols = [
        "strategy_type",
        "strategy_value",
        "contacted_n",
        "contacted_share",
        "precision",
        "recall",
        "expected_conversions",
        "expected_revenue",
        "contact_cost",
        "expected_net_gain",
        "roi_proxy",
    ]
    business_sim = business_sim[keep_cols].sort_values("expected_net_gain", ascending=False)
    save_table(business_sim, "business_simulation.csv")

    recommendation_text = build_targeting_recommendation(
        threshold_df=threshold_metrics,
        topk_df=topk_targeting,
        business_df=business_sim,
        baseline_conversion=float(y_test_deploy.mean()),
        assumptions=assumptions,
    )
    save_text(recommendation_text, "targeting_recommendation.md")

    X_deploy_full = X_all[feature_sets["without_duration"]]
    deploy_model_dict = build_models(X_deploy_full)
    cv_df = run_cv_robustness(deploy_model_dict, X_deploy_full, y)
    save_table(cv_df, "cv_robustness.csv")

    save_roc_curve(
        y_test_deploy,
        deploy_scores,
        FIGURE_DIR / "roc_curve_deployable.png",
        title=f"ROC Curve ({best_deploy_model_name}, Without Duration)",
    )
    save_precision_recall_curve(
        y_test_deploy,
        deploy_scores,
        FIGURE_DIR / "precision_recall_curve_deployable.png",
        title=f"Precision-Recall ({best_deploy_model_name}, Without Duration)",
    )
    save_threshold_tradeoff(
        threshold_metrics,
        FIGURE_DIR / "threshold_tradeoff.png",
    )
    save_cumulative_gains(
        y_test_deploy,
        deploy_scores,
        FIGURE_DIR / "cumulative_gains_curve.png",
    )

    print("Saved tables to reports/ and results/")
    print("- model_metrics.csv")
    print("- confusion_matrices.csv")
    print("- leakage_summary.csv")
    print("- deployable_lift_table.csv")
    print("- topk_metrics.csv")
    print("- threshold_metrics.csv")
    print("- business_simulation.csv")
    print("- cv_robustness.csv")
    print("- targeting_recommendation.md")
    print("\nSaved figures to results/figures/")
    print("- roc_curve_deployable.png")
    print("- precision_recall_curve_deployable.png")
    print("- threshold_tradeoff.png")
    print("- cumulative_gains_curve.png")

    print("\nBest deployable model:", best_deploy_model_name)
    print(deployable.to_string(index=False))


if __name__ == "__main__":
    run_training()
