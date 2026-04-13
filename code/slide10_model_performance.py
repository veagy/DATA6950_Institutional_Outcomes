"""
Slide 10 — Model Performance
Grouped horizontal bar chart: R² by model and outcome (test set).
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Data ──────────────────────────────────────────────────────────────────────
metrics = pd.read_csv("output/models/metrics.csv")
test = metrics[metrics["split"] == "test"].copy()

OUTCOMES = [
    ("grad_rate_150_4yr",    "Graduation Rate\n(6-year)"),
    ("median_earnings_6yr",  "Median Earnings\n(6 yr post-entry)"),
    ("loan_repayment_3yr",   "Loan Repayment\n(3-year rate)"),
    ("avg_efficiency_score", "DEA Efficiency\nScore"),
]
MODELS = [
    ("random_forest", "Random Forest"),
    ("xgboost",       "XGBoost"),
    ("elastic_net",   "Elastic Net"),
]

COLORS = {
    "random_forest": "#4A90D9",   # steel blue
    "xgboost":       "#5BAD72",   # green
    "elastic_net":   "#F5A623",   # amber
}
EN_OUTLIER_COLOR = "#D45F5F"      # red — flags the dramatic EN underperformance on efficiency

# ── Layout ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.spines.left": False,
    "axes.facecolor":   "#FAFAFA",
    "figure.facecolor": "white",
})

n_outcomes = len(OUTCOMES)
n_models   = len(MODELS)
bar_h      = 0.22
group_gap  = 0.30        # vertical space between outcome groups
group_h    = n_models * bar_h + group_gap

fig_h = n_outcomes * group_h + 0.8
fig, ax = plt.subplots(figsize=(11, fig_h))

# Build y-positions (top to bottom: first outcome at top)
ytick_positions = []
ytick_labels    = []

for g_idx, (outcome_key, outcome_label) in enumerate(OUTCOMES):
    # group center from top
    group_center = -(g_idx * group_h)

    # sub-rows within group (RF top, XGB middle, EN bottom)
    offsets = [bar_h, 0, -bar_h]

    for m_idx, ((model_key, model_label), offset) in enumerate(
            zip(MODELS, offsets)):

        row = test[(test["outcome"] == outcome_key) &
                   (test["model"]   == model_key)]
        if row.empty:
            continue
        r2  = row["r2"].iloc[0]
        y   = group_center + offset

        # Flag EN on efficiency score in red
        is_outlier = (model_key == "elastic_net" and
                      outcome_key == "avg_efficiency_score")
        color = EN_OUTLIER_COLOR if is_outlier else COLORS[model_key]
        edge  = "#8B0000"        if is_outlier else color

        bar = ax.barh(y, r2, height=bar_h * 0.82,
                      color=color, edgecolor=edge, linewidth=1.2 if is_outlier else 0,
                      zorder=3)

        # R² label to the right of bar
        label_x = r2 + 0.012
        ax.text(label_x, y, f"{r2:.3f}",
                va="center", ha="left", fontsize=10,
                fontweight="bold" if is_outlier else "normal",
                color="#8B0000" if is_outlier else "#333333")

    # Y-tick at group center
    ytick_positions.append(group_center)
    ytick_labels.append(outcome_label)

    # Light horizontal separator between groups (except after last)
    if g_idx < n_outcomes - 1:
        sep_y = group_center - group_h / 2
        ax.axhline(sep_y, color="#CCCCCC", lw=0.8, zorder=1)

# ── Axes formatting ───────────────────────────────────────────────────────────
ax.set_yticks(ytick_positions)
ax.set_yticklabels(ytick_labels, fontsize=12)
ax.set_xlabel("Test-Set R²", fontsize=12, labelpad=8)
ax.set_xlim(0, 1.08)
ax.set_title("Predictive Model Performance (Test Set R²)",
             fontsize=13, fontweight="bold", pad=14)

# Vertical reference lines
for x_ref in [0.25, 0.50, 0.75]:
    ax.axvline(x_ref, color="#DDDDDD", lw=0.9, zorder=0)
ax.axvline(0, color="#AAAAAA", lw=0.8, zorder=0)

ax.tick_params(axis="x", labelsize=10)
ax.tick_params(axis="y", length=0, pad=10)
ax.spines["bottom"].set_color("#AAAAAA")

# ── Legend ────────────────────────────────────────────────────────────────────
patches = [mpatches.Patch(color=COLORS[k], label=label)
           for k, label in MODELS]
patches.append(mpatches.Patch(color=EN_OUTLIER_COLOR,
               label="Elastic Net — DEA"))
ax.legend(handles=patches, frameon=True, framealpha=0.95,
          edgecolor="#CCCCCC", fontsize=10,
          loc="lower right", bbox_to_anchor=(1.0, 0.0))

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = Path("output/slides/slide10_model_performance.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"Saved -> {out_path}")
plt.close()
