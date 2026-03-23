from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from data_utils import make_preprocessor


RANDOM_STATE = 42


def build_models(X_train: pd.DataFrame) -> Dict[str, Pipeline]:
    """Build baseline and benchmark models with appropriate preprocessing."""
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


def evaluate_binary_classifier(
    pipe: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> dict:
    """Fit model and return standard out-of-sample classification metrics."""
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)

    return {
        "roc_auc": roc_auc_score(y_test, proba),
        "f1": f1_score(y_test, pred, zero_division=0),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, pred),
        "proba": proba,
        "pred": pred,
    }


def run_cv_robustness(
    model_dict: Dict[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Cross-validated AUC for deployable setting robustness checks."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for model_name, model in model_dict.items():
        auc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        rows.append(
            {
                "model": model_name,
                "cv_auc_mean": auc_scores.mean(),
                "cv_auc_std": auc_scores.std(),
            }
        )

    return pd.DataFrame(rows).sort_values("cv_auc_mean", ascending=False)


def summarize_leakage(results_df: pd.DataFrame) -> pd.DataFrame:
    """Create leakage summary table with deployability annotation."""
    out = results_df.copy()
    out["deployable"] = out["feature_set"].eq("without_duration")
    out["setting_note"] = np.where(
        out["deployable"],
        "Pre-contact features only; suitable for production scoring.",
        "Includes call duration (post-contact); leakage risk for targeting.",
    )
    return out
