"""
Fathom Labs Researcher Retention — synthetic dataset generator.

Fathom Labs is a fictional frontier AI lab. This script generates a synthetic
workforce dataset of safety-adjacent researchers, built to carry the kind of
noise a real HR/people-analytics export has (missing values, duplicate rows,
inconsistent casing) and to encode a testable hypothesis:

    Researchers who feel they can't be heard internally (low escalation
    confidence) AND believe their work carries real moral weight (high moral
    injury) are far more likely to leave than researchers who are simply
    overworked.

This mirrors the real-world pattern behind a September 2026 resignation at
an AI lab, used here only as the inspiration for the hypothesis — all names,
IDs and figures below are synthetic.

Usage:
    python generate_data.py
Produces:
    fathom_labs_researchers_raw.csv    (messy, as a real export would look)
    fathom_labs_researchers_clean.csv  (cleaned, analysis-ready)
"""

import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()
Faker.seed(42)
np.random.seed(42)

N = 850  # synthetic researcher population

ROLES = [
    "Alignment Research", "Interpretability", "Safety Evaluations",
    "Policy & Governance", "Infrastructure", "Applied Research",
]
ROLE_WEIGHTS = [0.22, 0.16, 0.18, 0.10, 0.20, 0.14]

DEPARTMENTS = {
    "Alignment Research": "Research",
    "Interpretability": "Research",
    "Safety Evaluations": "Research",
    "Policy & Governance": "Policy",
    "Infrastructure": "Engineering",
    "Applied Research": "Research",
}


def clip(arr, lo, hi):
    return np.clip(arr, lo, hi)


def likert(mean, sd, n):
    """1-5 Likert-style score with realistic rounding and noise."""
    return clip(np.round(np.random.normal(mean, sd, n)), 1, 5).astype(int)


def main():
    rows = []

    for i in range(N):
        researcher_id = f"FL-{10000 + i}"
        role = np.random.choice(ROLES, p=ROLE_WEIGHTS)
        department = DEPARTMENTS[role]

        # Tenure: most researchers are early-career at the company (AI labs
        # are young and growing fast), right-skewed.
        tenure_months = int(clip(np.random.exponential(scale=14), 1, 60))

        # --- Psychological / structural variables -----------------------

        # Moral injury: belief that the pace of the work itself causes harm,
        # independent of personal conduct. Skews higher in safety-facing roles.
        base_moral_injury = 3.2 if role in (
            "Alignment Research", "Interpretability", "Safety Evaluations"
        ) else 2.3
        moral_injury_index = likert(base_moral_injury, 1.0, 1)[0]

        # Escalation confidence: belief that raising a concern internally
        # leads to real action (psychological safety / voice efficacy).
        escalation_confidence = likert(3.1, 1.1, 1)[0]

        # Pace-vs-safety tension: felt conflict between speed and rigor
        # (Job Demands-Resources: demand side).
        pace_vs_safety_tension = likert(3.4, 1.0, 1)[0]

        # Organizational justice: perceived fairness of how decisions get made.
        org_justice = likert(3.0, 1.0, 1)[0]

        # Manager support: day-to-day felt support from direct manager.
        manager_support = likert(3.3, 1.0, 1)[0]

        # Weekly workload hours.
        workload_hours = int(clip(np.random.normal(52, 9), 35, 85))

        # Equity: standard 6-month cliff, linear vesting to 48 months.
        if tenure_months < 6:
            equity_vested_pct = 0.0
        else:
            equity_vested_pct = round(min(100.0, (tenure_months / 48) * 100), 1)

        # --- Outcome: latent attrition score ------------------------------
        # Moral injury and low escalation confidence are the dominant
        # drivers, by hypothesis. Workload matters but less. Low equity
        # vested slightly *raises* the cost of leaving (sunk-cost effect),
        # so it's protective up to a point.
        INTERCEPT = -1.6  # calibrated so baseline attrition lands near real-world rates (~20%)
        latent = (
            INTERCEPT
            + 0.55 * (moral_injury_index - 3)
            - 0.60 * (escalation_confidence - 3)
            + 0.25 * (pace_vs_safety_tension - 3)
            - 0.20 * (org_justice - 3)
            - 0.15 * (manager_support - 3)
            + 0.012 * (workload_hours - 52)
            - 0.01 * (equity_vested_pct - 50) * 0.15  # weak protective effect
            + np.random.normal(0, 0.9)
        )
        prob_leave = 1 / (1 + np.exp(-latent))
        left_company = int(np.random.random() < prob_leave)

        rows.append({
            "researcher_id": researcher_id,
            "name": fake.name(),
            "role": role,
            "department": department,
            "tenure_months": tenure_months,
            "hire_date": fake.date_between(start_date="-5y", end_date="-1M"),
            "moral_injury_index": moral_injury_index,
            "escalation_confidence": escalation_confidence,
            "pace_vs_safety_tension": pace_vs_safety_tension,
            "org_justice_perception": org_justice,
            "manager_support": manager_support,
            "workload_hours_week": workload_hours,
            "equity_vested_pct": equity_vested_pct,
            "left_company": left_company,
        })

    df = pd.DataFrame(rows)

    # --- Write a clean version ------------------------------------------
    df.to_csv("fathom_labs_researchers_clean.csv", index=False)

    # --- Write a messy "raw export" version, matching the convention of
    # the other case studies (duplicates, nulls, inconsistent casing) ----
    raw = df.copy()
    raw["role"] = raw["role"].apply(
        lambda r: r.upper() if np.random.random() < 0.15 else r
    )
    dupe_rows = raw.sample(frac=0.03, random_state=1)
    raw = pd.concat([raw, dupe_rows], ignore_index=True)
    for col in ["manager_support", "org_justice_perception", "workload_hours_week"]:
        null_idx = raw.sample(frac=0.04, random_state=2).index
        raw.loc[null_idx, col] = np.nan
    raw = raw.sample(frac=1, random_state=3).reset_index(drop=True)
    raw.to_csv("fathom_labs_researchers_raw.csv", index=False)

    print(f"Generated {len(df)} clean rows, {len(raw)} raw rows.")
    print(f"Attrition rate: {df['left_company'].mean():.1%}")


if __name__ == "__main__":
    main()
