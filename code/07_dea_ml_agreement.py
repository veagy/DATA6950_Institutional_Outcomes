# 07_dea_ml_agreement.py
# Week 8 analysis: DEA efficiency tier vs ML-predicted efficiency score agreement.
#
# Computes:
#   - Spearman rank correlation between DEA avg_efficiency_score and ML predictions
#   - Cohen's kappa between DEA efficiency tiers and ML score quintile bins
#   - Divergence profile: institutions where DEA tier and ML quintile differ by >= 2
#
# Join logic:
#   efficiency_scores_full.csv has one row per institution (unitid unique).
#   predictions_avg_efficiency_score.csv contains test-set rows indexed by
#   position in analysis_ready.csv. We aggregate ML predictions to one value
#   per institution (mean across years if multiple test rows exist), then join
#   on unitid.
#
# Input:  data/processed/efficiency_scores_full.csv
#         output/models/predictions_avg_efficiency_score.csv
#         data/processed/analysis_ready.csv
# Output: output/agreement/agreement_summary.csv
#         output/agreement/divergent_institutions.csv
#         output/agreement/confusion_matrix.csv
#         output/agreement/tier_vs_quintile_heatmap.png
#
# Author: Audrey Nichols | DATA 6950 | Spring 2026

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, confusion_matrix

# -- Paths --------------------------------------------------------------------
PROC_DIR    = Path("data/processed")
MODEL_DIR   = Path("output/models")
OUT_DIR     = Path("output/agreement")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Ordered tier labels (low to high) for numeric encoding
TIER_ORDER  = ["Low-Efficiency", "Mid-Efficiency", "High-Efficiency"]
TIER_NUM    = {t: i for i, t in enumerate(TIER_ORDER)}   # 0, 1, 2

# =============================================================================
# 1. Load and join
# =============================================================================

print("Loading data...")

eff = pd.read_csv(PROC_DIR / "efficiency_scores_full.csv")
eff["unitid"] = eff["unitid"].astype(str)

ar = pd.read_csv(PROC_DIR / "analysis_ready.csv", low_memory=False, encoding="latin-1")
ar["unitid"] = ar["unitid"].astype(str)

preds = pd.read_csv(MODEL_DIR / "predictions_avg_efficiency_score.csv")

# Map test-set rows back to unitid via analysis_ready index
test_rows = ar.iloc[preds["test_index"].values][["unitid", "input_year"]].copy()
test_rows["rf_pred"]  = preds["random_forest"].values
test_rows["xgb_pred"] = preds["xgboost"].values

# One prediction per institution: mean across years if an institution appears
# in multiple test rows (rare but possible with multi-year panel)
ml_agg = (
    test_rows
    .groupby("unitid")[["rf_pred", "xgb_pred"]]
    .mean()
    .reset_index()
)

# Join DEA scores with ML predictions on unitid
df = eff.merge(ml_agg, on="unitid", how="inner")
print(f"  {len(df):,} institutions with both DEA tiers and ML test predictions")

# =============================================================================
# 2. Encode tiers and bin ML predictions into quintiles
# =============================================================================

df["tier_num"] = df["efficiency_tier"].map(TIER_NUM)

# Quintile-bin each ML prediction (1–5), then collapse to 3 bins to match tiers
# Quintile mapping: Q1-Q2 -> Low (0), Q3 -> Mid (1), Q4-Q5 -> High (2)
for model_col, label in [("rf_pred", "rf"), ("xgb_pred", "xgb")]:
    quintile = pd.qcut(df[model_col], q=5, labels=False)   # 0–4
    df[f"{label}_quintile"]  = quintile
    df[f"{label}_tier_num"]  = quintile.map({0: 0, 1: 0, 2: 1, 3: 2, 4: 2})

# =============================================================================
# 3. Spearman correlation: DEA score vs ML predicted score
# =============================================================================

print("\nSpearman rank correlation (DEA avg_efficiency_score vs ML prediction):")
summary_rows = []

for model_col, label, full_name in [
    ("rf_pred",  "rf",  "Random Forest"),
    ("xgb_pred", "xgb", "XGBoost"),
]:
    rho, pval = spearmanr(df["avg_efficiency_score"], df[model_col])
    print(f"  {full_name}: rho = {rho:.4f}  p = {pval:.4e}")
    summary_rows.append({
        "model":              full_name,
        "spearman_rho":       round(rho,  4),
        "spearman_p":         round(pval, 6),
        "n":                  len(df),
    })

# =============================================================================
# 4. Cohen's kappa: DEA tier vs ML quintile-derived tier
# =============================================================================

print("\nCohen's kappa (DEA efficiency tier vs ML-derived tier):")

for model_col, label, full_name in [
    ("rf_tier_num",  "rf",  "Random Forest"),
    ("xgb_tier_num", "xgb", "XGBoost"),
]:
    kappa = cohen_kappa_score(df["tier_num"], df[model_col])
    print(f"  {full_name}: kappa = {kappa:.4f}")
    for row in summary_rows:
        if row["model"] == full_name:
            row["cohen_kappa"] = round(kappa, 4)

# Save summary
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_DIR / "agreement_summary.csv", index=False)
print(f"\nSummary saved -> {OUT_DIR / 'agreement_summary.csv'}")

# =============================================================================
# 5. Confusion matrix: DEA tier vs ML-derived tier (RF)
# =============================================================================

cm = confusion_matrix(df["tier_num"], df["rf_tier_num"], labels=[0, 1, 2])
cm_df = pd.DataFrame(
    cm,
    index  =[f"DEA_{t}" for t in TIER_ORDER],
    columns=[f"ML_{t}"  for t in TIER_ORDER],
)
cm_df.to_csv(OUT_DIR / "confusion_matrix.csv")
print(f"Confusion matrix saved -> {OUT_DIR / 'confusion_matrix.csv'}")
print("\nConfusion matrix (rows=DEA tier, cols=ML-derived tier):")
print(cm_df.to_string())

# =============================================================================
# 6. Heatmap: DEA tier vs ML quintile (finer grain than confusion matrix)
# =============================================================================

hm_data = (
    df.groupby(["efficiency_tier", "rf_quintile"])
    .size()
    .unstack(fill_value=0)
    .reindex(TIER_ORDER)
)

fig, ax = plt.subplots(figsize=(8, 4))
im = ax.imshow(hm_data.values, cmap="Blues", aspect="auto")
plt.colorbar(im, ax=ax, label="Institution count")
ax.set_xticks(range(hm_data.shape[1]))
ax.set_xticklabels([f"Q{i+1}" for i in range(hm_data.shape[1])])
ax.set_yticks(range(3))
ax.set_yticklabels(TIER_ORDER)
ax.set_xlabel("ML Predicted Quintile (Q1=lowest, Q5=highest)")
ax.set_ylabel("DEA Efficiency Tier")
ax.set_title("DEA Tier vs ML-Predicted Quintile Agreement\n(Random Forest)")

# Annotate cells
for i in range(hm_data.shape[0]):
    for j in range(hm_data.shape[1]):
        ax.text(j, i, str(hm_data.values[i, j]),
                ha="center", va="center", fontsize=9,
                color="white" if hm_data.values[i, j] > hm_data.values.max() * 0.5 else "black")

plt.tight_layout()
fig.savefig(OUT_DIR / "tier_vs_quintile_heatmap.png", dpi=120)
plt.close(fig)
print(f"Heatmap saved -> {OUT_DIR / 'tier_vs_quintile_heatmap.png'}")

# =============================================================================
# 7. Divergent institutions: DEA tier vs ML-derived tier differ by >= 2 levels
# =============================================================================

df["tier_gap_rf"]  = (df["tier_num"] - df["rf_tier_num"]).abs()
df["tier_gap_xgb"] = (df["tier_num"] - df["xgb_tier_num"]).abs()

# Flag if either model diverges by >= 2 (i.e. one says High, the other says Low)
divergent = df[
    (df["tier_gap_rf"] >= 2) | (df["tier_gap_xgb"] >= 2)
].copy()

divergent["dea_tier"]    = divergent["efficiency_tier"]
divergent["ml_rf_tier"]  = divergent["rf_tier_num"].map({v: k for k, v in TIER_NUM.items()})
divergent["ml_xgb_tier"] = divergent["xgb_tier_num"].map({v: k for k, v in TIER_NUM.items()})

profile_cols = [
    "unitid", "instnm", "stabbr", "sector",
    "dea_tier", "ml_rf_tier", "ml_xgb_tier",
    "avg_efficiency_score", "dea_bcc_score", "dea_ccr_score",
    "rf_pred", "xgb_pred",
    "tier_gap_rf", "tier_gap_xgb",
    "pell_pct", "instr_exp_per_fte", "stud_fac_ratio", "grad_rate_150_4yr",
]
profile_cols = [c for c in profile_cols if c in divergent.columns]

divergent_out = divergent[profile_cols].sort_values("tier_gap_rf", ascending=False)
divergent_out.to_csv(OUT_DIR / "divergent_institutions.csv", index=False)

print(f"\nDivergent institutions (|DEA - ML| >= 2 tiers): {len(divergent_out)}")
print(f"  Saved -> {OUT_DIR / 'divergent_institutions.csv'}")

if len(divergent_out) > 0:
    print("\nTop divergent institutions:")
    show_cols = ["instnm", "stabbr", "sector", "dea_tier", "ml_rf_tier",
                 "avg_efficiency_score", "rf_pred", "pell_pct"]
    show_cols = [c for c in show_cols if c in divergent_out.columns]
    print(divergent_out[show_cols].head(15).to_string(index=False))

# =============================================================================
# 8. Summary statistics on divergent institutions
# =============================================================================

if len(divergent_out) > 0:
    print("\nDivergent institution characteristics vs full sample:")
    char_cols = ["pell_pct", "instr_exp_per_fte", "stud_fac_ratio",
                 "avg_efficiency_score", "rf_pred"]
    char_cols = [c for c in char_cols if c in df.columns]

    compare = pd.DataFrame({
        "divergent_mean": divergent[char_cols].mean().round(4),
        "all_mean":       df[char_cols].mean().round(4),
    })
    print(compare.to_string())

print(f"\n{'='*55}")
print("DEA-ML Agreement Analysis complete.")
print(f"  Outputs saved to {OUT_DIR}")
print(f"{'='*55}")
