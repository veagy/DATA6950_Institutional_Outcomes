# Predicting Institutional Outcomes Through Efficiency Modeling and Machine Learning
### DATA 6950 — Graduate Capstone Project, Spring 2026

---

## Overview

This project builds a multi-method framework for analyzing the performance and efficiency of U.S. higher education institutions. It combines two complementary analytical approaches:

1. **Efficiency Measurement** — Data Envelopment Analysis (DEA, CCR and BCC models) and Stochastic Frontier Analysis (SFA, Cobb-Douglas specification)
2. **Predictive Modeling** — Random Forest, XGBoost, and Elastic Net regression with SHAP-based interpretability, partial dependence plots, and a permutation hypothesis test

Data sources span 2015–2023 and include the Integrated Postsecondary Education Data System (IPEDS) and the U.S. Department of Education College Scorecard, covering 2,579 unique institutions across 21,875 institution-year observations.

---

## Research Questions

1. Can a multidimensional efficiency framework identify meaningful performance differences across U.S. colleges and universities?
2. Which institutional characteristics most strongly predict graduation rates, earnings, loan repayment, and overall efficiency?
3. Do DEA efficiency rankings and machine learning predictions agree — and where they diverge, what does that reveal?

---

## Key Findings

- **DEA sample:** 1,559 four-year institutions; BCC mean score = 0.782
- **Pell grant share** is the #1 SHAP predictor for all four modeled outcomes, reflecting structural mission differences between access-oriented and selective institutions
- **Multidimensional models significantly outperform cost-only baselines** for all four outcomes (p < 0.002, 500-permutation test); R² improvement ranges from +0.203 (earnings) to +0.530 (DEA efficiency)
- **DEA and ML agree substantially** (Spearman ρ = 0.884, Cohen's κ = 0.600); 20 maximally divergent institutions split into two interpretable patterns
- **SFA convergence failure** — all Cobb-Douglas scores ≈ 0.999 due to model degeneracy; DEA is the sole efficiency measure used

---

## Repository Structure

```
Capstone/
├── code/               # All R and Python analysis scripts + slide visuals
├── data/
│   ├── raw/            # Stacked IPEDS CSVs and College Scorecard file
│   └── processed/      # Pipeline output: aligned panel, cleaned data, efficiency scores
├── docs/
│   ├── Flags/          # Data quality flag files for human review
│   └── data_dictionary.csv
└── output/
    ├── efficiency/     # DEA/SFA scores, tier profiles, diagnostic plots
    ├── models/         # Trained model files (.joblib), metrics, predictions
    ├── shap/           # SHAP importance plots, beeswarms, waterfalls, heatmap
    ├── pdp/            # Partial dependence plots, ICE plots, hypothesis test results
    ├── agreement/      # DEA-ML agreement statistics, confusion matrix, heatmap
    └── slides/         # Presentation-ready PNG figures
```

---

## Pipeline

Scripts run in order. R scripts (00–03) build the data. Python scripts (04–07) run the ML pipeline.

| Script | Language | Purpose |
|---|---|---|
| `00_stack_raw.R` | R | Stack annual IPEDS files into single CSVs per component |
| `01_integrate.R` | R | Merge all sources into a longitudinal panel with lagged inputs |
| `02_clean.R` | R | Impute, winsorize, standardize, engineer composite features |
| `03_efficiency.R` | R | Run DEA (CCR + BCC) and SFA; classify efficiency tiers |
| `04_ml_models.py` | Python | Train RF, XGBoost, Elastic Net for four outcomes |
| `05_shap_interpretability.py` | Python | Compute and visualize SHAP values |
| `06_pdp_hypothesis.py` | Python | PDPs, ICE plots, 500-permutation hypothesis test |
| `07_dea_ml_agreement.py` | Python | DEA-ML agreement statistics and divergent institution profiles |

---

## Requirements

**R packages:** `tidyverse`, `Benchmarking`, `frontier`, `mice`, `janitor`

**Python packages:** `pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`, `matplotlib`, `scipy`, `joblib`

**Python version:** 3.11+

---

## Data Sources

- **IPEDS:** [nces.ed.gov/ipeds](https://nces.ed.gov/ipeds/) — annual survey data for all Title IV institutions
- **College Scorecard:** [collegescorecard.ed.gov](https://collegescorecard.ed.gov/) — earnings and loan repayment outcomes

Raw data files are not included in this repository due to size. See `data/README.md` for download instructions.

---

## Reproducibility

1. Download raw IPEDS and Scorecard files into `data/raw/` per the instructions in `data/README.md`
2. Run scripts 00–07 in order from the project root
3. All random operations use seed `2026`
4. All paths are relative to the project root — no `setwd()` or absolute paths required
