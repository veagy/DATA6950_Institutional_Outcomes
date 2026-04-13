"""
Slide 11 — SHAP Feature Importance Table
Top 3 SHAP predictors per outcome (Random Forest).
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Data ──────────────────────────────────────────────────────────────────────
OUTCOMES = [
    ("grad_rate_150_4yr",    "Graduation Rate\n(6-Year)"),
    ("median_earnings_6yr",  "Median Earnings\n(6-Year)"),
    ("loan_repayment_3yr",   "Loan Repayment\n(3-Year)"),
    ("avg_efficiency_score", "DEA Efficiency\nScore"),
]

FEATURE_LABELS = {
    "pell_pct":              "Pell Grant Share",
    "instr_exp_per_fte":     "Instructional Spending / FTE",
    "selectivity_composite": "Selectivity Composite",
    "total_sal_per_fte":     "Total Salary / FTE",
    "total_completions":     "Total Completions",
    "sal_instr_avg_9mo":     "Avg. Instructional Salary",
    "sal_noninstr_avg":      "Avg. Non-Instructional Salary",
    "value_added_proxy":     "Value-Added Proxy",
    "stud_fac_ratio":        "Student-Faculty Ratio",
}

TOP_N = 3

shap_data = {}
for key, _ in OUTCOMES:
    df = pd.read_csv(f"output/shap/{key}/permutation_vs_shap_random_forest.csv")
    df = df.sort_values("shap_mean_abs", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    shap_data[key] = df.head(TOP_N)

# ── Style ─────────────────────────────────────────────────────────────────────
RANK_BG   = ["#1A6B3C", "#2C6FAC", "#5B9BD5"]   # #1 green, #2 blue, #3 lighter blue
RANK_FG   = ["white",   "white",   "white"]
PELL_ROW  = "#EAF7EC"
ALT_ROW   = "#F7F9FB"
WHITE_ROW = "#FFFFFF"
HDR_BG    = "#2C3E50"

plt.rcParams.update({"font.family": "sans-serif", "font.size": 11,
                     "figure.facecolor": "white"})

n_outcomes = len(OUTCOMES)
fig_w = 13
fig_h = 5.2
fig, axes = plt.subplots(1, n_outcomes, figsize=(fig_w, fig_h))
fig.suptitle("SHAP Feature Importance — Top 3 Predictors by Outcome (Random Forest)",
             fontsize=13, fontweight="bold", y=1.02)

for ax, (key, label) in zip(axes, OUTCOMES):
    df = shap_data[key]
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, TOP_N + 0.9)
    ax.axis("off")

    # Column header
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, TOP_N + 0.05), 1, 0.75,
        boxstyle="round,pad=0.04", fc=HDR_BG, ec="none"
    ))
    ax.text(0.5, TOP_N + 0.42, label, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="white")

    for i, row in df.iterrows():
        rank  = int(row["rank"])
        feat  = row["feature"]
        fname = FEATURE_LABELS.get(feat, feat)
        shap  = row["shap_mean_abs"]
        y     = TOP_N - rank

        is_pell = feat == "pell_pct"
        bg = PELL_ROW if is_pell else (ALT_ROW if rank % 2 == 0 else WHITE_ROW)

        # Row background
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.01, y + 0.03), 0.98, 0.88,
            boxstyle="round,pad=0.03", fc=bg, ec="none"
        ))

        # Rank badge
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.04, y + 0.15), 0.14, 0.62,
            boxstyle="round,pad=0.04",
            fc=RANK_BG[rank - 1], ec="none"
        ))
        ax.text(0.11, y + 0.47, f"#{rank}",
                ha="center", va="center",
                fontsize=11, fontweight="bold", color=RANK_FG[rank - 1])

        # Feature name
        ax.text(0.22, y + 0.60, fname,
                ha="left", va="center",
                fontsize=9.5,
                fontweight="bold" if is_pell else "normal",
                color="#1A6B3C" if is_pell else "#222222")

        # SHAP value
        ax.text(0.22, y + 0.22, f"mean |SHAP| = {shap:.4f}",
                ha="left", va="center",
                fontsize=8.5, color="#666666")

# Footer
fig.text(0.5, -0.03,
         "★ Pell Grant Share ranks #1 in 3 of 4 outcomes — reflects structural mission differences, "
         "not institutional mismanagement.\n"
         "SHAP validated against permutation importance (Spearman ρ = 0.89–1.00, all p < 0.001).",
         ha="center", fontsize=9, color="#555555", style="italic")

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = Path("output/slides/slide11_shap_table.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"Saved -> {out_path}")
plt.close()
