from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, precision_score, recall_score


@dataclass
class BusinessAssumptions:
    cost_per_contact: float = 2.0
    revenue_per_conversion: float = 120.0


def _metrics_at_threshold(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> dict:
    pred = (y_proba >= threshold).astype(int)

    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    contacted_n = int(pred.sum())

    return {
        "threshold": float(threshold),
        "contacted_n": contacted_n,
        "contacted_share": contacted_n / len(y_true),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "conversion_rate_targeted": tp / contacted_n if contacted_n > 0 else 0.0,
    }


def evaluate_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Evaluate classification and targeting metrics across thresholds."""
    if thresholds is None:
        thresholds = np.linspace(0.10, 0.90, 17)

    rows = [_metrics_at_threshold(y_true, y_proba, thr) for thr in thresholds]
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def evaluate_topk(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    top_fracs: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Evaluate targeting quality for top-K ranked customer slices."""
    if top_fracs is None:
        top_fracs = [0.05, 0.10, 0.20, 0.30]

    ranking = pd.DataFrame({"y_true": y_true, "score": y_proba}).sort_values(
        "score", ascending=False
    )

    rows: List[dict] = []
    baseline_rate = ranking["y_true"].mean()
    total_positives = int(ranking["y_true"].sum())

    for frac in top_fracs:
        n_take = max(1, int(np.ceil(len(ranking) * frac)))
        seg = ranking.head(n_take)
        tp = int(seg["y_true"].sum())
        rows.append(
            {
                "top_fraction": float(frac),
                "n_targeted": int(n_take),
                "contacted_share": float(frac),
                "conversion_rate": seg["y_true"].mean(),
                "lift_vs_baseline": seg["y_true"].mean() / baseline_rate,
                "expected_conversions": tp,
                "recall_capture": tp / total_positives if total_positives > 0 else 0.0,
                "threshold_at_cutoff": seg["score"].min(),
            }
        )

    return pd.DataFrame(rows).sort_values("top_fraction").reset_index(drop=True)


def simulate_business(
    strategy_df: pd.DataFrame,
    assumptions: BusinessAssumptions,
    strategy_label: str,
    value_col: str,
) -> pd.DataFrame:
    """Apply simple profit simulation to threshold or top-K strategy table."""
    out = strategy_df.copy()
    out["strategy_type"] = strategy_label
    out["strategy_value"] = out[value_col]

    if "tp" in out.columns:
        conversions = out["tp"]
        contacts = out["contacted_n"]
        precision_vals = out["precision"]
        recall_vals = out["recall"]
    else:
        conversions = out["expected_conversions"]
        contacts = out["n_targeted"]
        precision_vals = out["conversion_rate"]
        recall_vals = out["recall_capture"]
        if "contacted_share" not in out.columns:
            out["contacted_share"] = out["top_fraction"]

    out["contacted_n"] = contacts
    out["expected_conversions"] = conversions
    out["precision"] = precision_vals
    out["recall"] = recall_vals

    out["expected_revenue"] = conversions * assumptions.revenue_per_conversion
    out["contact_cost"] = contacts * assumptions.cost_per_contact
    out["expected_net_gain"] = out["expected_revenue"] - out["contact_cost"]
    out["roi_proxy"] = np.where(
        out["contact_cost"] > 0,
        out["expected_net_gain"] / out["contact_cost"],
        np.nan,
    )

    return out


def build_targeting_recommendation(
    threshold_df: pd.DataFrame,
    topk_df: pd.DataFrame,
    business_df: pd.DataFrame,
    baseline_conversion: float,
    assumptions: BusinessAssumptions,
    max_contact_share_for_profit: float = 0.50,
) -> str:
    """Generate concise targeting recommendation text for business stakeholders."""
    constrained_pool = business_df[business_df["contacted_share"] <= max_contact_share_for_profit]
    if constrained_pool.empty:
        constrained_pool = business_df.copy()
    best_profit_row = constrained_pool.sort_values("expected_net_gain", ascending=False).iloc[0]

    limited_budget = topk_df.iloc[(topk_df["top_fraction"] - 0.10).abs().argmin()]

    recall_candidate = threshold_df[
        (threshold_df["precision"] >= baseline_conversion)
        & (threshold_df["contacted_share"] <= 0.60)
    ]
    if recall_candidate.empty:
        recall_candidate = threshold_df[threshold_df["contacted_share"] <= 0.60]
    if recall_candidate.empty:
        recall_candidate = threshold_df.copy()
    recall_row = recall_candidate.sort_values("recall", ascending=False).iloc[0]

    lines = [
        "# Targeting Recommendation",
        "",
        "## Assumptions Used",
        f"- Cost per contact: {assumptions.cost_per_contact:.2f}",
        f"- Revenue per conversion: {assumptions.revenue_per_conversion:.2f}",
        f"- Test-set baseline conversion: {baseline_conversion:.3f}",
        "",
        "## Recommended Strategies",
        (
            "1. Budget-limited campaign: prioritize top 10% scored customers "
            f"(conversion={limited_budget['conversion_rate']:.3f}, "
            f"lift={limited_budget['lift_vs_baseline']:.2f}x)."
        ),
        (
            "2. Profit-oriented strategy under current assumptions: "
            f"{best_profit_row['strategy_type']}={best_profit_row['strategy_value']:.3f} "
            f"(contacted share={best_profit_row['contacted_share']:.3f}, "
            f"expected net gain={best_profit_row['expected_net_gain']:.2f}, "
            f"ROI proxy={best_profit_row['roi_proxy']:.2f})."
        ),
        (
            "3. If recall is prioritized while keeping precision above baseline: "
            f"use threshold={recall_row['threshold']:.2f} "
            f"(precision={recall_row['precision']:.3f}, recall={recall_row['recall']:.3f}, "
            f"contacted share={recall_row['contacted_share']:.3f})."
        ),
        "",
        "## Notes",
        "- These profit figures are simulation outputs under stated assumptions, not observed production outcomes.",
        (
            f"- Profit recommendation is constrained to contacted share <= "
            f"{max_contact_share_for_profit:.2f} to reflect finite campaign capacity."
        ),
        "- Final threshold should be tuned jointly with outreach budget, capacity, and compliance constraints.",
    ]

    return "\n".join(lines)
