# Fathom Labs — Researcher Retention

**Why do AI safety researchers leave, and is it the workload or something else?**

[Live dashboard](https://fathom-labs-retention-lbhqdkluiwzuuihijufbcv.streamlit.app/) · 

## Context

In September 2026, an Anthropic researcher resigned publicly, citing fear that the pace of the AI race left too little room for safety work. It wasn't an isolated incident — several AI lab employees have made similar moves this year, and internal surveys across the industry increasingly surface the same tension: people who believe the work matters enormously, working inside organizations racing to ship faster than they can verify.

That event is the starting point for this case study, not its subject. I'm not diagnosing any specific person's reasons for leaving — a news article isn't enough to know what someone was actually going through, and I don't think it's my place to guess. What I wanted to test instead was the underlying *pattern*: using organizational psychology frameworks, what structurally predicts a safety-minded researcher leaving a high-pressure AI lab — and is it actually the hours, or something else?

**Fathom Labs** is a fictional AI lab built to test that question on synthetic data.

## The hypothesis

Three frameworks shaped the variables in this dataset:

- **Job Demands-Resources model** — burnout and exit aren't just about workload (demand); they're about whether people have enough perceived control and support (resources) to meet that demand.
- **Moral injury** (originally clinical/military psychology, now used in healthcare and tech) — distress from believing you're complicit in harm, even without doing anything wrong personally. A researcher who believes the industry's pace itself is dangerous experiences this differently than a researcher who's just tired.
- **Psychological safety / voice efficacy** (Amy Edmondson) — whether people believe raising a concern internally will actually lead somewhere, or whether silence (or going public) feels like the only option.

The working hypothesis: **escalation confidence and moral injury predict attrition far more than raw workload does.**

## Ask → Prepare → Analyze → Share → Act

**Ask:** What organizational factors predict a safety researcher leaving — and can an organization act on them before it gets to a public resignation?

**Prepare:** `data/generate_data.py` builds an 850-researcher synthetic population with Python Faker, encoding tenure, role, workload, and five psychological measures (moral injury, escalation confidence, pace-vs-safety tension, organizational justice, manager support), plus a standard 6-month equity cliff. A raw export version is generated separately with the duplicates, inconsistent casing, and missing values a real HR export would have.

**Analyze:** `analysis/clean.sql` deduplicates and imputes missing values at the role level. `analysis/model.py` fits a standardized logistic regression predicting `left_company`, so the result is a ranked, quantified answer rather than an opinion.

**Share:** The [live dashboard](https://fathom-labs-retention-lbhqdkluiwzuuihijufbcv.streamlit.app/) visualizes the ranked factors and lets you simulate organizational changes — improve escalation response, reduce moral injury, shorten the equity cliff — and watch modeled attrition move in response, scored by the same fitted model.

**Act:** See [Findings](#findings) below.

## Findings

On this synthetic population (22.6% baseline attrition, ROC AUC 0.72):

| Factor | Standardized effect | Direction |
|---|---|---|
| Escalation confidence | −0.64 | Strongest protective factor |
| Moral injury | +0.47 | Strongest risk factor |
| Manager support | −0.32 | Protective |
| Pace-vs-safety tension | +0.11 | Mild risk |
| Organizational justice | −0.10 | Mild protective |
| Equity vested % | −0.04 | Weak, as expected (sunk-cost effect) |
| Workload hours/week | +0.02 | Negligible |

The clearest signal: **raw hours worked barely moved the model at all.** What mattered was whether people believed raising a concern changed anything, and whether they believed the work itself was causing harm. That's a meaningfully different problem than "reduce workload" — and a much harder one to solve with a wellness stipend.

## What this suggests an organization do

Grounded in the model, not asserted from opinion:

1. **Build an escalation path with visible follow-through**, not just an open-door policy. The gap between "I can raise a concern" and "I've seen a concern lead to a changed decision" is what the model is picking up as escalation confidence.
2. **Treat moral injury as distinct from burnout.** A wellness program doesn't address it; it requires actually giving people more say over release pace or scope, not just lighter workloads.
3. **Don't rely on equity vesting as a retention strategy for this population.** The model shows it's a weak lever — someone who believes the work is causing harm is not meaningfully retained by sunk-cost economics, which matches real cases of researchers forfeiting unvested equity to leave anyway.

## Run it locally

Prefer not to clone anything first? Try the [live dashboard](https://fathom-labs-retention-lbhqdkluiwzuuihijufbcv.streamlit.app/) directly.

```bash
git clone https://github.com/professorpasqua-byte/fathom-labs-retention.git
cd fathom-labs-retention
pip install -r requirements.txt
streamlit run app.py
```

To regenerate the dataset or refit the model from scratch:

```bash
cd data && python generate_data.py
cd ../analysis && python model.py
```

## Repo structure

```
fathom-labs-retention/
├── data/
│   ├── generate_data.py                      # Faker-based synthetic data generator
│   ├── fathom_labs_researchers_raw.csv        # messy export, as generated
│   └── fathom_labs_researchers_clean.csv      # cleaned, analysis-ready
├── analysis/
│   ├── clean.sql                              # dedup + impute pipeline
│   ├── model.py                               # logistic regression, saves coefficients
│   └── model_coefficients.json                # fitted model, loaded by the app
├── app.py                                     # Streamlit app
├── requirements.txt
└── README.md
```

## A note on the data

Fathom Labs, every researcher in this dataset, and every figure above are synthetic, generated to carry the kind of noise a real workforce export has. No real individual's data, survey responses, or employment history appears anywhere in this project.

---

**Nwafor Martins** — People Analytics Engineer & I-O Psychology Consultant
[Email](mailto:nwaforchubby@gmail.com)
