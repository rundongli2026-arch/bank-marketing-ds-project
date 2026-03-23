from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from data_utils import load_dataset, make_preprocessor, split_xy


RANDOM_STATE = 42
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model(pipe, X_train, y_train, X_test, y_test):
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y_test, proba),
        "f1": f1_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "confusion_matrix": confusion_matrix(y_test, pred),
        "proba": proba,
    }


def build_models(X_train):
    lr = Pipeline(
        steps=[
            ("preprocess", make_preprocessor(X_train, scale_numeric=True)),
            (
                "model",
                LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            ),
        ]
    )

    gb = Pipeline(
        steps=[
            ("preprocess", make_preprocessor(X_train, scale_numeric=False)),
            ("model", GradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]
    )

    return {
        "Logistic Regression": lr,
        "Gradient Boosting": gb,
    }


def run_training():
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

    rows = []
    conf_rows = []
    fitted = {}

    for fs_name, cols in feature_sets.items():
        X_train = X_all.loc[train_idx, cols]
        X_test = X_all.loc[test_idx, cols]
        y_train = y.loc[train_idx]
        y_test = y.loc[test_idx]

        for model_name, model in build_models(X_train).items():
            metric = evaluate_model(model, X_train, y_train, X_test, y_test)
            fitted[(fs_name, model_name)] = model
            rows.append(
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
            conf_rows.append(
                {
                    "feature_set": fs_name,
                    "model": model_name,
                    "tn": int(cm[0, 0]),
                    "fp": int(cm[0, 1]),
                    "fn": int(cm[1, 0]),
                    "tp": int(cm[1, 1]),
                }
            )

    results = pd.DataFrame(rows).sort_values(
        ["feature_set", "roc_auc"], ascending=[True, False]
    )
    results.to_csv(REPORT_DIR / "model_metrics.csv", index=False)
    pd.DataFrame(conf_rows).to_csv(REPORT_DIR / "confusion_matrices.csv", index=False)

    deploy = results[results["feature_set"] == "without_duration"].sort_values(
        "roc_auc", ascending=False
    )
    best_name = deploy.iloc[0]["model"]
    best_model = fitted[("without_duration", best_name)]

    X_test_deploy = X_all.loc[test_idx, feature_sets["without_duration"]]
    y_test_deploy = y.loc[test_idx].reset_index(drop=True)
    score = best_model.predict_proba(X_test_deploy)[:, 1]

    ranking = pd.DataFrame({"y_true": y_test_deploy, "score": score})
    lift_rows = []
    for pct in [0.1, 0.2, 0.3, 0.4, 0.5]:
        thr = ranking["score"].quantile(1 - pct)
        seg = ranking[ranking["score"] >= thr]
        lift_rows.append(
            {
                "top_fraction": pct,
                "n_customers": len(seg),
                "conversion_rate": seg["y_true"].mean(),
                "lift_vs_baseline": seg["y_true"].mean() / ranking["y_true"].mean(),
            }
        )

    lift_df = pd.DataFrame(lift_rows)
    lift_df.to_csv(REPORT_DIR / "deployable_lift_table.csv", index=False)

    X_deploy, y_deploy = split_xy(df, drop_cols=[])
    X_deploy = X_deploy.drop(columns=["duration"], errors="ignore")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_rows = []
    for mname, pipe in build_models(X_deploy).items():
        auc_scores = cross_val_score(pipe, X_deploy, y_deploy, cv=cv, scoring="roc_auc", n_jobs=-1)
        cv_rows.append(
            {
                "model": mname,
                "cv_auc_mean": auc_scores.mean(),
                "cv_auc_std": auc_scores.std(),
            }
        )

    cv_df = pd.DataFrame(cv_rows).sort_values("cv_auc_mean", ascending=False)
    cv_df.to_csv(REPORT_DIR / "cv_robustness.csv", index=False)

    print("Saved reports:")
    print("-", REPORT_DIR / "model_metrics.csv")
    print("-", REPORT_DIR / "confusion_matrices.csv")
    print("-", REPORT_DIR / "deployable_lift_table.csv")
    print("-", REPORT_DIR / "cv_robustness.csv")
    print("\nBest deployable model:", best_name)
    print(results.to_string(index=False))


if __name__ == "__main__":
    run_training()
