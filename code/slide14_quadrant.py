"""
Slide 14 — DEA vs. ML Agreement Quadrant Diagram
2×2 conceptual quadrant with all institutions plotted faintly,
two labeled divergent institutions highlighted, and quadrant annotations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── Data ──────────────────────────────────────────────────────────────────────
ar    = pd.read_csv("data/processed/analysis_ready.csv",
                    encoding="latin-1", low_memory=False)
eff   = pd.read_csv("data/processed/efficiency_scores_full.csv")
pred  = pd.read_csv("output/models/predictions_avg_efficiency_score.csv")
div   = pd.read_csv("output/agreement/divergent_institutions.csv")

# Map test_index -> unitid
pred["unitid"] = ar.loc[pred["test_index"].values, "unitid"].values

# Merge efficiency scores into predictions
merged = pred.merge(
    eff[["unitid", "dea_bcc_score", "avg_efficiency_score", "efficiency_tier",
         "instnm", "pell_pct"]],
    on="unitid", how="inner"
)

# x = DEA-BCC score, y = RF prediction (ML-predicted performance)
x_all = merged["dea_bcc_score"].values
y_all = merged["random_forest"].values

# Midpoints for quadrant lines
x_mid = np.median(x_all)
y_mid = np.median(y_all)

# Specific institutions (confirmed present in merged, confirmed in correct quadrants)
# Low DEA / High ML → bottom-left of axes but top-left quadrant text area
akron_uid   = 200800   # University of Akron OH:  DEA=0.765(Low), ML=0.891(High)
goddard_uid = 230889   # Goddard College VT:       DEA=1.000(High), ML=0.793(Low)
wit_uid     = 168227   # Wentworth Institute of Technology: DEA=0.753(Low), ML~0.81(Low)

akron   = merged[merged["unitid"] == akron_uid].iloc[0]
goddard = merged[merged["unitid"] == goddard_uid].iloc[0]
wit_rows = merged[merged["unitid"] == wit_uid]
wit_ml   = wit_rows["random_forest"].mean()   # average across two test rows

# ── Style ─────────────────────────────────────────────────────────────────────
Q_COLORS = {
    "TL": "#EAF4FB",   # low DEA / high ML  — light blue
    "TR": "#EAF7EC",   # high DEA / high ML — light green
    "BL": "#FDF2F2",   # low DEA  / low ML  — light red
    "BR": "#FEF9EC",   # high DEA / low ML  — light amber
}
SCATTER_COLOR = "#8EB4D8"
GRID_COLOR    = "#CCCCCC"
LINE_COLOR    = "#888888"

plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        11,
    "axes.facecolor":   "white",
    "figure.facecolor": "white",
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

fig, ax = plt.subplots(figsize=(10, 8))

# ── Quadrant shading ──────────────────────────────────────────────────────────
x_lo, x_hi = x_all.min() - 0.01, x_all.max() + 0.01
y_lo, y_hi = y_all.min() - 0.005, y_all.max() + 0.005

ax.fill_between([x_lo, x_mid], [y_mid, y_mid], [y_hi, y_hi],
                color=Q_COLORS["TL"], zorder=0)   # top-left:  low DEA / high ML
ax.fill_between([x_mid, x_hi], [y_mid, y_mid], [y_hi, y_hi],
                color=Q_COLORS["TR"], zorder=0)   # top-right: high DEA / high ML
ax.fill_between([x_lo, x_mid], [y_lo, y_lo], [y_mid, y_mid],
                color=Q_COLORS["BL"], zorder=0)   # bottom-left:  low DEA / low ML
ax.fill_between([x_mid, x_hi], [y_lo, y_lo], [y_mid, y_mid],
                color=Q_COLORS["BR"], zorder=0)   # bottom-right: high DEA / low ML

# Quadrant dividers
ax.axvline(x_mid, color=LINE_COLOR, lw=1.2, ls="--", zorder=1)
ax.axhline(y_mid, color=LINE_COLOR, lw=1.2, ls="--", zorder=1)

# ── Quadrant labels ───────────────────────────────────────────────────────────
label_kw = dict(fontsize=10, ha="center", va="center", color="#555555",
                fontstyle="italic", zorder=2)
pad_x = (x_hi - x_lo) * 0.14
pad_y = (y_hi - y_lo) * 0.10

ax.text(x_lo + pad_x, y_mid + pad_y * 1.5,
        "Low DEA /\nHigh ML Prediction\n\n"
        "Scale-inefficient;\n"
        "strong observable outputs",
        **label_kw)
ax.text(x_mid + pad_x, y_mid + pad_y * 1.5,
        "High DEA /\nHigh ML Prediction\n\n"
        "Strong agreement:\n"
        "genuinely high-performing",
        **label_kw)
ax.text(x_lo + pad_x, y_mid - pad_y * 1.5,
        "Low DEA /\nLow ML Prediction\n\n"
        "Strong agreement:\n"
        "lower-performing",
        **label_kw)
ax.text(x_mid + pad_x, y_mid - pad_y * 1.5,
        "High DEA /\nLow ML Prediction\n\n"
        "Efficient relative to peers;\n"
        "lower absolute outcomes",
        **label_kw)

# ── Background scatter: all institutions ──────────────────────────────────────
colors_scatter = []
for xi, yi in zip(x_all, y_all):
    if xi >= x_mid and yi >= y_mid:
        colors_scatter.append("#3A8F5E")   # TR green
    elif xi < x_mid and yi >= y_mid:
        colors_scatter.append("#2C6FAC")   # TL blue
    elif xi < x_mid and yi < y_mid:
        colors_scatter.append("#D45F5F")   # BL red
    else:
        colors_scatter.append("#E07B2A")   # BR orange

ax.scatter(x_all, y_all, c=colors_scatter, alpha=0.25, s=18, zorder=3,
           edgecolors="none")

# ── Two labeled institutions ──────────────────────────────────────────────────
# University of Akron OH — LOW DEA (0.765) / HIGH ML (0.891) → top-left quadrant
ax.scatter(akron["dea_bcc_score"], akron["random_forest"],
           color="#2C6FAC", s=160, zorder=6, edgecolors="white", linewidth=2)

ax.annotate(
    "Univ. of Akron (OH)\n\"Large public — scale-inefficient;\nstrong observable outcomes\"",
    xy=(akron["dea_bcc_score"], akron["random_forest"]),
    xytext=(0.40, 0.915),
    fontsize=10, color="#1A4F80", fontweight="bold",
    ha="left", va="center",
    bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#2C6FAC", alpha=0.95, lw=1.5),
    arrowprops=dict(arrowstyle="-|>", color="#2C6FAC", lw=1.6,
                    connectionstyle="arc3,rad=0.20"),
    zorder=7
)

# Goddard College VT — HIGH DEA (1.000) / LOW ML (0.793) → bottom-right quadrant
ax.scatter(goddard["dea_bcc_score"], goddard["random_forest"],
           color="#E07B2A", s=160, zorder=6, edgecolors="white", linewidth=2)

ax.annotate(
    "Goddard College (VT)\n\"DEA score = 1.0; small, low-resource;\nlower absolute outcomes\"",
    xy=(goddard["dea_bcc_score"], goddard["random_forest"]),
    xytext=(0.75, 0.725),
    fontsize=10, color="#8B4A0A", fontweight="bold",
    ha="left", va="center",
    bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#E07B2A", alpha=0.95, lw=1.5),
    arrowprops=dict(arrowstyle="-|>", color="#E07B2A", lw=1.6,
                    connectionstyle="arc3,rad=-0.25"),
    zorder=7
)

# ── Wentworth Institute of Technology ────────────────────────────────────────
if not wit_rows.empty:
    wit_dea = float(wit_rows["dea_bcc_score"].iloc[0])
    ax.scatter(wit_dea, wit_ml,
               color="#9B59B6", s=160, zorder=6, edgecolors="white", linewidth=2,
               marker="D")   # diamond marker to distinguish from the other two
    ax.annotate(
        "Wentworth Inst. of Tech. (MA)\n\"Mid-efficiency; below-median\non both DEA and ML\"",
        xy=(wit_dea, wit_ml),
        xytext=(0.45, 0.760),
        fontsize=10, color="#6C3483", fontweight="bold",
        ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#9B59B6",
                  alpha=0.95, lw=1.5),
        arrowprops=dict(arrowstyle="-|>", color="#9B59B6", lw=1.6,
                        connectionstyle="arc3,rad=0.15"),
        zorder=7
    )

# ── Axis labels and titles ────────────────────────────────────────────────────
ax.set_xlabel("DEA-BCC Efficiency Score  (Low ← → High)", fontsize=12, labelpad=8)
ax.set_ylabel("ML-Predicted Efficiency Score  (Low ← → High)", fontsize=12, labelpad=8)
ax.set_title(
    "Slide 14 — DEA Efficiency vs. ML-Predicted Performance\n"
    f"Agreement and Divergence Across {len(merged):,} Four-Year Institutions",
    fontsize=13, fontweight="bold", pad=14
)

ax.set_xlim(x_lo, x_hi)
ax.set_ylim(y_lo, y_hi)

# Median reference labels on axes
ax.text(x_mid, y_lo - 0.003, f"Median DEA\n({x_mid:.3f})",
        ha="center", va="top", fontsize=9, color=LINE_COLOR)
ax.text(x_lo - 0.003, y_mid, f"Median ML\n({y_mid:.3f})",
        ha="right", va="center", fontsize=9, color=LINE_COLOR, rotation=90)

# Stats footer
n_agree = ((x_all >= x_mid) & (y_all >= y_mid)).sum() + \
          ((x_all < x_mid)  & (y_all < y_mid)).sum()
n_total = len(x_all)
fig.text(
    0.5, -0.03,
    f"n = {n_total:,} institutions in joint DEA-ML analysis  |  "
    f"Spearman ρ = 0.884  |  Cohen's κ = 0.600  |  "
    f"On-diagonal (agreement) quadrants: {n_agree} of {n_total} ({n_agree/n_total*100:.0f}%)",
    ha="center", fontsize=9.5, color="#555555", style="italic"
)

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = Path("output/slides/slide14_quadrant.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"Saved -> {out_path}")
plt.close()
