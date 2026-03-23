from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import auc, precision_recall_curve, roc_curve


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_roc_curve(y_true, y_proba, out_path: Path, title: str = "ROC Curve") -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_precision_recall_curve(
    y_true, y_proba, out_path: Path, title: str = "Precision-Recall Curve"
) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_threshold_tradeoff(
    threshold_df: pd.DataFrame,
    out_path: Path,
    title: str = "Threshold Trade-off (Precision / Recall / Contact Share)",
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(threshold_df["threshold"], threshold_df["precision"], label="Precision", linewidth=2)
    plt.plot(threshold_df["threshold"], threshold_df["recall"], label="Recall", linewidth=2)
    plt.plot(
        threshold_df["threshold"],
        threshold_df["contacted_share"],
        label="Contacted Share",
        linewidth=2,
    )
    plt.xlabel("Score Threshold")
    plt.ylabel("Metric Value")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_cumulative_gains(
    y_true,
    y_proba,
    out_path: Path,
    title: str = "Cumulative Gains Curve",
) -> None:
    df = pd.DataFrame({"y_true": y_true, "score": y_proba}).sort_values("score", ascending=False)
    df["cum_contacts"] = np.arange(1, len(df) + 1) / len(df)
    total_positives = max(1, df["y_true"].sum())
    df["cum_positives"] = df["y_true"].cumsum() / total_positives

    plt.figure(figsize=(7, 5))
    plt.plot(df["cum_contacts"], df["cum_positives"], linewidth=2, label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Random")
    plt.xlabel("Share of Contacted Customers")
    plt.ylabel("Share of Captured Conversions")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_segment_conversion_bar(
    cluster_summary: pd.DataFrame,
    out_path: Path,
    title: str = "Conversion Rate by Segment",
) -> None:
    plot_df = cluster_summary.reset_index().copy()

    plt.figure(figsize=(7, 5))
    plt.bar(plot_df["cluster"].astype(str), plot_df["deposit_yes_rate"], color="#4C78A8")
    plt.xlabel("Cluster")
    plt.ylabel("Deposit Conversion Rate")
    plt.title(title)

    for _, row in plot_df.iterrows():
        plt.text(
            x=float(row["cluster"]),
            y=float(row["deposit_yes_rate"]) + 0.01,
            s=f"{row['deposit_yes_rate']:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.ylim(0, min(1.0, plot_df["deposit_yes_rate"].max() + 0.1))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
