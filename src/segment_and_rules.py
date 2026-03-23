from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from data_utils import load_dataset
from plot_utils import ensure_dir, save_segment_conversion_bar


RANDOM_STATE = 42
RESULT_DIR = Path("results")
FIGURE_DIR = RESULT_DIR / "figures"

for p in [RESULT_DIR, FIGURE_DIR]:
    ensure_dir(p)


def save_table(df: pd.DataFrame, file_name: str) -> None:
    df.to_csv(RESULT_DIR / file_name, index=False)


def _segment_label(row: pd.Series, baseline_rate: float) -> str:
    if row["campaign_z"] > 1.0 and row["deposit_yes_rate"] < baseline_rate:
        return "Over-contacted low responders"
    if row["duration_z"] > 1.0 and row["deposit_yes_rate"] > baseline_rate:
        return "High-call-engagement responders"
    if row["pdays_z"] > 1.0 and row["previous_z"] > 1.0:
        return "Previously-contacted relationship segment"
    return "Mainstream customer base"


def _strategy_note(label: str) -> str:
    if label == "Over-contacted low responders":
        return "Reduce repeated outreach and retest with alternative channels or messaging."
    if label == "High-call-engagement responders":
        return "Prioritize in high-intent campaigns; allocate better agents and faster follow-up."
    if label == "Previously-contacted relationship segment":
        return "Use relationship-based offers and retention-oriented messaging."
    return "Use model scores to prioritize within this large base segment."


def run_segmentation_and_rules() -> None:
    df = load_dataset()

    cluster_features = ["age", "balance", "duration", "campaign", "pdays", "previous"]
    X = df[cluster_features].copy()
    X_scaled = StandardScaler().fit_transform(X)

    sil_rows = []
    for k in range(2, 11):
        labels = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit_predict(X_scaled)
        sil_rows.append({"k": k, "silhouette": silhouette_score(X_scaled, labels)})

    sil_df = pd.DataFrame(sil_rows).sort_values("silhouette", ascending=False)
    save_table(sil_df, "silhouette_by_k.csv")

    k_final = 4
    km = KMeans(n_clusters=k_final, random_state=RANDOM_STATE, n_init=10)
    df["cluster"] = km.fit_predict(X_scaled)

    baseline_rate = (df["deposit"] == "yes").mean()

    cluster_summary = pd.concat(
        [
            df["cluster"].value_counts().sort_index().rename("n_customers"),
            df.groupby("cluster")["deposit"].apply(lambda s: (s == "yes").mean()).rename("deposit_yes_rate"),
            df.groupby("cluster")[cluster_features].mean(),
        ],
        axis=1,
    ).reset_index()

    save_table(cluster_summary, "cluster_summary.csv")

    z_cols = cluster_features
    z_df = cluster_summary[z_cols].copy()
    z_df = (z_df - z_df.mean()) / z_df.std(ddof=0)
    z_df.columns = [f"{c}_z" for c in z_cols]

    cluster_profile = pd.concat([cluster_summary, z_df], axis=1)
    cluster_profile["segment_label"] = cluster_profile.apply(
        lambda r: _segment_label(r, baseline_rate), axis=1
    )
    cluster_profile["recommended_action"] = cluster_profile["segment_label"].apply(_strategy_note)

    ordered_cols = [
        "cluster",
        "segment_label",
        "n_customers",
        "deposit_yes_rate",
        "age",
        "balance",
        "duration",
        "campaign",
        "pdays",
        "previous",
        "age_z",
        "balance_z",
        "duration_z",
        "campaign_z",
        "pdays_z",
        "previous_z",
        "recommended_action",
    ]

    save_table(cluster_profile[ordered_cols], "cluster_profile_actionable.csv")

    save_segment_conversion_bar(
        cluster_summary.set_index("cluster"),
        FIGURE_DIR / "segment_conversion_bar.png",
        title="Deposit Conversion by Customer Segment",
    )

    rule_cols = [
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "poutcome",
        "deposit",
    ]
    encoded = pd.get_dummies(df[rule_cols], dtype=bool)

    min_support = 0.05
    min_confidence = 0.60
    min_lift = 2.0

    t0 = time.time()
    fi_ap = apriori(encoded, min_support=min_support, use_colnames=True)
    rules_ap = association_rules(fi_ap, metric="lift", min_threshold=min_lift)
    rules_ap = rules_ap[rules_ap["confidence"] >= min_confidence].copy()
    ap_time = time.time() - t0

    t1 = time.time()
    fi_fp = fpgrowth(encoded, min_support=min_support, use_colnames=True)
    rules_fp = association_rules(fi_fp, metric="lift", min_threshold=min_lift)
    rules_fp = rules_fp[rules_fp["confidence"] >= min_confidence].copy()
    fp_time = time.time() - t1

    runtime_df = pd.DataFrame(
        {
            "algorithm": ["Apriori", "FP-Growth"],
            "n_itemsets": [len(fi_ap), len(fi_fp)],
            "n_rules": [len(rules_ap), len(rules_fp)],
            "runtime_seconds": [ap_time, fp_time],
        }
    ).sort_values("runtime_seconds")
    save_table(runtime_df, "rule_runtime_comparison.csv")

    rules = rules_fp.copy()
    rules["actionable"] = rules.apply(
        lambda r: (
            (not any(item.startswith("deposit_") for item in r["antecedents"]))
            and ("deposit_yes" in r["consequents"])
        ),
        axis=1,
    )

    actionable = rules[rules["actionable"]].copy().sort_values(
        ["lift", "confidence", "support"], ascending=False
    )
    actionable_cols = ["antecedents", "consequents", "support", "confidence", "lift"]
    save_table(actionable[actionable_cols], "actionable_rules.csv")

    exploratory_top = rules.sort_values(["lift", "confidence", "support"], ascending=False).head(10)
    save_table(exploratory_top[actionable_cols], "exploratory_rules_top10.csv")

    print("Saved tables to results/")
    print("- silhouette_by_k.csv")
    print("- cluster_summary.csv")
    print("- cluster_profile_actionable.csv")
    print("- rule_runtime_comparison.csv")
    print("- actionable_rules.csv")
    print("- exploratory_rules_top10.csv")
    print("\nSaved figure:")
    print("- results/figures/segment_conversion_bar.png")
    print("\nActionable rules count:", len(actionable))


if __name__ == "__main__":
    run_segmentation_and_rules()
