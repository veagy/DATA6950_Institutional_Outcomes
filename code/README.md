# code/

This folder contains all analysis scripts and presentation visual scripts for the capstone project. Scripts are numbered and intended to be run in order from the **project root directory** (`Capstone/`), not from inside `code/`.

---

## Analysis Pipeline

### `00_stack_raw.R`
**Language:** R

Stacks annual IPEDS survey files (one file per year) into a single CSV per IPEDS component, and stacks all College Scorecard `MERGED*.csv` files into one file. Run this once whenever new raw data is added.

- **Input:** `data/raw/ipeds/` (annual subdirectories), `data/raw/college scorecard/`
- **Output:** One stacked CSV per component in `data/raw/` (e.g., `ipeds_hd.csv`, `ipeds_gr.csv`, `college_scorecard.csv`)

---

### `01_integrate.R`
**Language:** R

Merges all IPEDS components and College Scorecard into a single institution-year longitudinal panel. Applies institution-type-specific input-to-outcome lags (4-year lag for 4-yr institutions, 2-year lag for 2-yr institutions). Writes data quality flag files for human review.

**Critical implementation note:** Graduation rate (`grad_rate_150_4yr`) is computed using IPEDS grtype 3 (all completers within 150% of normal time) divided by grtype 2 (bachelor's-seeking cohort). grtype 4 (completers excluding transfer-outs) must not be used — it produces artificially low rates (~25% of the correct value).

- **Input:** Stacked CSVs from `data/raw/`
- **Output:** `data/processed/master_aligned_raw.csv`, flag files in `docs/Flags/`

---

### `02_clean.R`
**Language:** R

Cleans the aligned panel:
- Rescales rates to 0–1 proportions
- Winsorizes outliers (99th percentile for most variables; tighter for `stud_serv_exp_per_fte` due to Strayer University anomalies)
- MICE imputation (pmm, 10 iterations) for features with < 20% missing
- Median imputation + binary `_missing` indicator for features with ≥ 20% missing
- Log-transforms financial variables
- Z-score standardizes all predictors
- Engineers five composite features: `selectivity_composite`, `value_added_proxy`, `student_support_intensity`, `resource_concentration_idx`, `financial_health_idx`

- **Input:** `data/processed/master_aligned_raw.csv`
- **Output:** `data/processed/analysis_ready.csv` (21,875 rows × 79 columns), `docs/data_dictionary.csv`

---

### `03_efficiency.R`
**Language:** R

Runs Data Envelopment Analysis (DEA) and Stochastic Frontier Analysis (SFA):

- **DEA:** Input-oriented CCR and BCC models
  - Inputs: `instr_exp_per_fte`, `stud_serv_exp_per_fte`, `stud_fac_ratio`
  - Outputs: `grad_rate_150_4yr`, `median_earnings_6yr`, `loan_repayment_3yr`
- **SFA:** Cobb-Douglas specification (note: all scores converged to ~0.999 due to model degeneracy — DEA is the primary efficiency measure)
- Classifies institutions into High/Mid/Low efficiency tiers using tertile cutoffs on `avg_efficiency_score`
- Runs a sensitivity analysis excluding for-profit institutions

- **Input:** `data/processed/analysis_ready.csv`
- **Output:** `data/processed/efficiency_scores_full.csv`, `data/processed/efficiency_scores_no_forprofit.csv`, plots and tier profiles in `output/efficiency/`

---

### `04_ml_models.py`
**Language:** Python

Trains three models (Random Forest, XGBoost, Elastic Net) for four outcomes:
- `grad_rate_150_4yr`, `median_earnings_6yr`, `loan_repayment_3yr`, `avg_efficiency_score`

Uses a 70/15/15 stratified train/validation/test split and 5-fold cross-validation on the training set. Elastic Net is wrapped in a scikit-learn Pipeline with StandardScaler to prevent data leakage. `value_added_proxy` is excluded from the graduation rate model (circular feature — it is derived from the graduation rate).

- **Input:** `data/processed/analysis_ready.csv`, `data/processed/efficiency_scores_full.csv`
- **Output:** Trained models as `.joblib` files, `output/models/metrics.csv`, `output/models/predictions_<outcome>.csv`

---

### `05_shap_interpretability.py`
**Language:** Python

Computes SHAP values for all trained models:
- TreeExplainer for Random Forest and XGBoost
- LinearExplainer for Elastic Net

Generates global feature importance bar charts, beeswarm plots, waterfall plots for the highest and lowest predicted institutions, a cross-outcome heatmap, and a Spearman correlation validation between SHAP and permutation importance rankings.

- **Input:** `output/models/`, `data/processed/analysis_ready.csv`
- **Output:** `output/shap/`
- **Requires:** `04_ml_models.py` completed first

---

### `06_pdp_hypothesis.py`
**Language:** Python

Generates Partial Dependence Plots (PDPs) and Individual Conditional Expectation (ICE) plots for the top SHAP features per outcome, varying each from the 5th to 95th percentile. Also produces two-way PDPs for the three highest-importance feature pairs per outcome.

Runs the central hypothesis test: full multidimensional model vs. cost-only baseline (inputs only: `instr_exp_per_fte`, `stud_serv_exp_per_fte`, `stud_fac_ratio`) using a 500-permutation test.

- **Input:** `output/models/`, `output/shap/`, `data/processed/analysis_ready.csv`
- **Output:** `output/pdp/`, `output/pdp/hypothesis/`
- **Requires:** `04_ml_models.py` and `05_shap_interpretability.py` completed first

---

### `07_dea_ml_agreement.py`
**Language:** Python

Computes agreement between DEA tier classifications and ML-predicted efficiency score quintile tiers. Calculates Spearman rank correlation (continuous scores) and Cohen's kappa (tier-level classification). Profiles the 20 institutions where the two methods diverge by two full tiers.

- **Input:** `data/processed/efficiency_scores_full.csv`, `output/models/predictions_avg_efficiency_score.csv`, `data/processed/analysis_ready.csv`
- **Output:** `output/agreement/`
- **Requires:** `03_efficiency.R` and `04_ml_models.py` completed first

---

## Presentation Visual Scripts

These scripts generate the PNG figures used in the capstone presentation. They read from pipeline outputs and do not modify any data files. Run from the project root after completing scripts 03–07.

| Script | Output | Description |
|---|---|---|
| `slide08_efficiency.py` | `output/slides/slide08_efficiency.png` | DEA-BCC score distribution by tier + tier profile bar chart |
| `slide10_model_performance.py` | `output/slides/slide10_model_performance.png` | Test-set R² grouped horizontal bar chart, all outcomes × models |
| `slide11_shap_table.py` | `output/slides/slide11_shap_table.png` | Top 3 SHAP predictors per outcome, ranked table |
| `slide12_pdp_nonlinear.py` | `output/slides/slide12_pdp_nonlinear.png` | Partial dependence plots: Pell%, instructional spending, selectivity |
| `slide14_quadrant.py` | `output/slides/slide14_quadrant.png` | DEA vs. ML 2×2 quadrant diagram with labeled institutions |

---

## Conventions

- All paths are relative to the project root — run scripts from `Capstone/`, not from `code/`
- Random seed: `2026` for all random operations (R and Python)
- R style: tidyverse, native pipe `|>`
- Python style: pandas + scikit-learn, no deprecated APIs
- Column names: `snake_case` throughout
