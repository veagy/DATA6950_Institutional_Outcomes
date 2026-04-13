"""
Slide 08 — Efficiency Results
Two visuals:
  Left:  KDE density of DEA-BCC scores, color-coded by tier, mean line
  Right: Grouped bar chart comparing High vs. Low tier on 3 metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde
from pathlib import Path

# ── Data ──────────────────────────────────────────────────────────────────────
eff = pd.read_csv("data/processed/efficiency_scores_full.csv")
profiles = pd.read_csv("output/efficiency/tier_profiles_full.csv")

# Tier boundaries (tertile cutoffs used in 03_efficiency.R)
low  = eff[eff["efficiency_tier"] == "Low-Efficiency"]["dea_bcc_score"]
mid  = eff[eff["efficiency_tier"] == "Mid-Efficiency"]["dea_bcc_score"]
high = eff[eff["efficiency_tier"] == "High-Efficiency"]["dea_bcc_score"]
mean_bcc = eff["dea_bcc_score"].mean()           # 0.811

# Tier cut points (boundary between Low/Mid and Mid/High)
cut_low_mid  = low.max()
cut_mid_high = mid.max()

# ── Style ─────────────────────────────────────────────────────────────────────
COLORS = {
    "low":  "#D45F5F",   # muted red
    "mid":  "#F5A623",   # amber
    "high": "#4A90D9",   # steel blue
}
GRAY   = "#555555"
BG     = "#FAFAFA"

plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.facecolor":   BG,
    "figure.facecolor": "white",
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("Slide 08 — Efficiency Analysis Results", fontsize=14,
             fontweight="bold", y=1.01)

# ── LEFT: KDE density with tier shading ───────────────────────────────────────
all_scores = eff["dea_bcc_score"].values
x_grid = np.linspace(all_scores.min() - 0.02, 1.02, 500)

def tier_kde(scores, x):
    kde = gaussian_kde(scores, bw_method=0.12)
    return kde(x)

y_low  = tier_kde(low.values,  x_grid)
y_mid  = tier_kde(mid.values,  x_grid)
y_high = tier_kde(high.values, x_grid)

# Scale each tier KDE by its proportion so areas sum to total
n = len(eff)
y_low  *= len(low)  / n
y_mid  *= len(mid)  / n
y_high *= len(high) / n

ax1.fill_between(x_grid, y_low,  alpha=0.55, color=COLORS["low"],  label="Low-Efficiency  (n=349)")
ax1.fill_between(x_grid, y_mid,  alpha=0.55, color=COLORS["mid"],  label="Mid-Efficiency  (n=349)")
ax1.fill_between(x_grid, y_high, alpha=0.55, color=COLORS["high"], label="High-Efficiency (n=350)")

ax1.plot(x_grid, y_low,  color=COLORS["low"],  lw=1.5, alpha=0.9)
ax1.plot(x_grid, y_mid,  color=COLORS["mid"],  lw=1.5, alpha=0.9)
ax1.plot(x_grid, y_high, color=COLORS["high"], lw=1.5, alpha=0.9)

# Tier boundary lines
for cut, label in [(cut_low_mid, ""), (cut_mid_high, "")]:
    ax1.axvline(cut, color=GRAY, lw=1, ls="--", alpha=0.5)

# Mean line
ymax_approx = max(y_low.max(), y_mid.max(), y_high.max())
ax1.axvline(mean_bcc, color="black", lw=2, ls="--", zorder=5)
ax1.text(mean_bcc + 0.007, ymax_approx * 0.92,
         f"Mean = {mean_bcc:.3f}", fontsize=10, color="black", va="top")

ax1.set_xlabel("DEA-BCC Efficiency Score", fontsize=12)
ax1.set_ylabel("Density (weighted by tier size)", fontsize=11)
ax1.set_title("Distribution of DEA-BCC Efficiency Scores\n1,048 Four-Year Institutions",
              fontsize=12, fontweight="bold")
ax1.set_xlim(0.43, 1.04)
ax1.legend(frameon=False, fontsize=10, loc="upper left")
ax1.tick_params(labelsize=10)

# ── RIGHT: Grouped bar chart ──────────────────────────────────────────────────
# Extract High and Low rows from profiles
prof_high = profiles[profiles["efficiency_tier"] == "High-Efficiency"].iloc[0]
prof_low  = profiles[profiles["efficiency_tier"] == "Low-Efficiency"].iloc[0]

metrics = [
    ("Pell Grant\nShare (%)",
     prof_high["pell_pct"] * 100,
     prof_low["pell_pct"]  * 100,
     "%", 1),
    ("Instructional\nSpending / FTE ($k)",
     prof_high["instr_exp_per_fte"] / 1000,
     prof_low["instr_exp_per_fte"]  / 1000,
     "k", 1),
    ("Student Services\nSpending / FTE ($k)",
     prof_high["stud_serv_exp_per_fte"] / 1000,
     prof_low["stud_serv_exp_per_fte"]  / 1000,
     "k", 1),
]

labels      = [m[0] for m in metrics]
high_vals   = [m[1] for m in metrics]
low_vals    = [m[2] for m in metrics]
suffixes    = [m[3] for m in metrics]

x      = np.arange(len(labels))
width  = 0.32
gap    = 0.04

bars_high = ax2.bar(x - width/2 - gap/2, high_vals, width,
                    color=COLORS["high"], label="High-Efficiency", zorder=3)
bars_low  = ax2.bar(x + width/2 + gap/2, low_vals,  width,
                    color=COLORS["low"],  label="Low-Efficiency",  zorder=3)

# Value labels on bars (single pass with correct format per metric)
for ax_bars, vals, sfxs in [(bars_high, high_vals, suffixes),
                              (bars_low,  low_vals,  suffixes)]:
    for bar, v, sfx in zip(ax_bars, vals, sfxs):
        txt = f"{v:.1f}%" if sfx == "%" else f"${v:.1f}k"
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.15,
                 txt, ha="center", va="bottom", fontsize=10,
                 fontweight="bold", color="#222222")

ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=11)
ax2.set_ylabel("Value", fontsize=11)
ax2.set_title("Tier Profiles: High-Efficiency vs. Low-Efficiency\n(three key institutional metrics)",
              fontsize=12, fontweight="bold")
ax2.legend(frameon=False, fontsize=10, loc="upper right")
ax2.set_ylim(0, max(high_vals + low_vals) * 1.25)
ax2.tick_params(labelsize=10)
ax2.yaxis.set_visible(False)
ax2.spines["left"].set_visible(False)
ax2.grid(axis="y", color="white", lw=0)   # no y-grid; values on bars

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = Path("output/slides/slide08_efficiency.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"Saved -> {out_path}")
plt.close()
