"""
Fathom Labs Researcher Retention — predictive model.

Fits a logistic regression predicting `left_company` from the structural
and psychological variables in the dataset, then saves the standardized
coefficients so the Streamlit app can simulate how shifting organizational
variables (escalation response, workload, equity structure) would move
predicted attrition, without needing to refit on every slider move.

Usage:
    python model.py
Produces:
    model_coefficients.json
"""

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

FEATURES = [
    "moral_injury_index",
    "escalation_confidence",
    "pace_vs_safety_tension",
    "org_justice_perception",
    "manager_support",
    "workload_hours_week",
    "equity_vested_pct",
]


def main():
    df = pd.read_csv("../data/fathom_labs_researchers_clean.csv")

    X = df[FEATURES].values
    y = df["left_company"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)

    train_acc = model.score(X_scaled, y)
    probs = model.predict_proba(X_scaled)[:, 1]
    auc = roc_auc_score(y, probs)
    print(f"Training accuracy: {train_acc:.1%}")
    print(f"Baseline (always predict 'stayed'): {1 - y.mean():.1%}")
    print(f"ROC AUC: {auc:.3f}  (accuracy alone is misleading at a ~23% base rate)")

    coefficients = dict(zip(FEATURES, model.coef_[0].tolist()))
    ranked = sorted(coefficients.items(), key=lambda kv: abs(kv[1]), reverse=True)
    print("\nFeature importance (standardized logistic coefficients):")
    for name, coef in ranked:
        direction = "raises" if coef > 0 else "lowers"
        print(f"  {name:28s} {coef:+.3f}  ({direction} attrition risk)")

    output = {
        "features": FEATURES,
        "coefficients": coefficients,
        "intercept": float(model.intercept_[0]),
        "feature_means": {f: float(df[f].mean()) for f in FEATURES},
        "feature_stds": {f: float(df[f].std()) for f in FEATURES},
        "train_accuracy": train_acc,
        "auc": auc,
        "base_attrition_rate": float(y.mean()),
    }

    with open("model_coefficients.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\nSaved model_coefficients.json")


if __name__ == "__main__":
    main()
