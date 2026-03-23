from pathlib import Path
import time

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

from data_utils import load_dataset


RANDOM_STATE = 42
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def run_segmentation_and_rules():
    df = load_dataset()

    cluster_features = ["age", "balance", "duration", "campaign", "pdays", "previous"]
    X = df[cluster_features].copy()
    X_scaled = StandardScaler().fit_transform(X)

    sil_rows = []
    for k in range(2, 11):
        labels = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit_predict(X_scaled)
        sil_rows.append({"k": k, "silhouette": silhouette_score(X_scaled, labels)})

    sil_df = pd.DataFrame(sil_rows).sort_values("silhouette", ascending=False)
    sil_df.to_csv(REPORT_DIR / "silhouette_by_k.csv", index=False)

    k_final = 4
    km = KMeans(n_clusters=k_final, random_state=RANDOM_STATE, n_init=10)
    df["cluster"] = km.fit_predict(X_scaled)

    cluster_summary = pd.concat(
        [
            df["cluster"].value_counts().sort_index().rename("n_customers"),
            df.groupby("cluster")["deposit"].apply(lambda s: (s == "yes").mean()).rename("deposit_yes_rate"),
            df.groupby("cluster")[cluster_features].mean(),
        ],
        axis=1,
    )
    cluster_summary.to_csv(REPORT_DIR / "cluster_summary.csv")

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
    )
    runtime_df.to_csv(REPORT_DIR / "rule_runtime_comparison.csv", index=False)

    rules = rules_fp.copy()
    rules["actionable"] = rules.apply(
        lambda r: (
            (not any(item.startswith("deposit_") for item in r["antecedents"]))
            and (r["consequents"] == frozenset({"deposit_yes"}))
        ),
        axis=1,
    )

    actionable = rules[rules["actionable"]].copy().sort_values(
        ["lift", "confidence", "support"], ascending=False
    )

    cols = ["antecedents", "consequents", "support", "confidence", "lift"]
    actionable[cols].to_csv(REPORT_DIR / "actionable_rules.csv", index=False)

    print("Saved reports:")
    print("-", REPORT_DIR / "silhouette_by_k.csv")
    print("-", REPORT_DIR / "cluster_summary.csv")
    print("-", REPORT_DIR / "rule_runtime_comparison.csv")
    print("-", REPORT_DIR / "actionable_rules.csv")
    print("\nActionable rules count:", len(actionable))


if __name__ == "__main__":
    run_segmentation_and_rules()
