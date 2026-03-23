# Bank Marketing Optimization (Interview-Ready DS Project)

This repository presents an end-to-end data science solution for improving bank term-deposit campaign performance.

## Business Objective
Increase campaign conversion and ROI by combining:
- Customer segmentation (KMeans)
- Predictive targeting (supervised learning)
- Leakage analysis (`duration` with vs without)
- Actionable association rules
- A/B testing design for production validation

## Dataset
- Source: Kaggle `janiobachmann/bank-marketing-dataset`
- Rows: 11,162
- Target: `deposit` (`yes`/`no`)

## Repository Structure
- `notebooks/bank_marketing_case_study.ipynb`: full narrative analysis
- `src/data_utils.py`: dataset loading and preprocessing utilities
- `src/train.py`: supervised modeling, leakage comparison, model metrics, lift, CV robustness
- `src/segment_and_rules.py`: clustering and actionable rule mining
- `data/README.md`: data download and licensing note
- `requirements.txt`: reproducible dependencies

## Quick Start
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Run supervised modeling:
```bash
python src/train.py
```

Run segmentation + rule mining:
```bash
python src/segment_and_rules.py
```

Run notebook:
```bash
jupyter notebook notebooks/bank_marketing_case_study.ipynb
```

## Key Technical Highlights
- Out-of-sample evaluation with ROC-AUC, F1, Precision, Recall, Confusion Matrix
- Leakage-aware comparison (`with_duration` vs `without_duration`)
- Interpretable baseline (Logistic Regression) + non-linear benchmark (Gradient Boosting)
- Lift analysis for top-score targeting bands
- Rule mining benchmark: Apriori vs FP-Growth runtime and overlap
- Actionability filter for rule deployment relevance

## Current Results Snapshot
- Best deployable model (`without_duration`): **Gradient Boosting**
  - ROC-AUC: **0.790**
  - F1: **0.695**
  - Precision: **0.796**
  - Recall: **0.616**
- Leakage check:
  - Best `with_duration` ROC-AUC: **0.922**
  - Best `without_duration` ROC-AUC: **0.790**
- Targeting lift (deployable model):
  - Top 10% scored customers: **90.0%** conversion (**1.90x** baseline lift)
  - Top 20% scored customers: **85.9%** conversion (**1.81x** baseline lift)
- Segmentation:
  - Four clusters with conversion rates from **21.3%** to **85.4%**
- Rule mining:
  - Apriori and FP-Growth produce identical rule sets under project thresholds
  - FP-Growth is faster in this run
  - Strict actionability filter returns 0 `deposit_yes` rules at current thresholds

## Business Validation Plan
A/B test design is included in the notebook:
- Control: current/random targeting
- Treatment: model-based targeting
- Metrics: conversion rate, CPA, ROI

## Notes
- Raw data is not committed to this repository.
- Scripts auto-download the dataset via `kagglehub` and include local CSV fallback.
