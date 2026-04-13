"""
Non-Linear Threshold Effects (PDP Analysis)
Three clean PDP panels (corrected graduation rate data):
  1. pell_pct               — strong negative gradient (dominant predictor, ~32.4 pp range)
  2. instr_exp_per_fte      — positive, largest gains below $10k/FTE (~17.0 pp range)
  3. selectivity_composite  — more selective -> higher grad rate (~9.7 pp range)

Uses the saved RF model for grad_rate_150_4yr (post value_added_proxy removal).
Applies box-car smoothing for presentation clarity. Raw PDP plotted faintly behind.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import joblib
from sklearn.inspection import partial_dependence
from pathlib import Path
from scipy.ndimage import uniform_filter1d

# ── Load model and data ───────────────────────────────────────────────────────
ar = pd.read_csv("data/processed/analysis_ready.csv",
                 encoding="latin-1", low_memory=False)
model    = joblib.load("output/models/random_forest_grad_rate_150_4yr.joblib")
features = model.feature_names_in_.tolist()
outcome  = "grad_rate_150_4yr"

subset = ar[features + [outcome]].dropna(subset=[outcome])
X = subset[features].copy()
X = X.fillna(X.median(numeric_only=True))
X = X.astype(float)

# ── Compute PDPs ──────────────────────────────────────────────────────────────
def get_pdp(feat, grid_resolution=80):
    idx = features.index(feat)
    result = partial_dependence(model, X, [idx], kind="average",
                                grid_resolution=grid_resolution,
                                percentiles=(0.05, 0.95))
    return result["grid_values"][0], result["average"][0]

def smooth(y, window=7):
    return uniform_filter1d(y.astype(float), size=window)

pell_x,  pell_y  = get_pdp("pell_pct", 80)
instr_x, instr_y = get_pdp("instr_exp_per_fte", 80)
sel_x,   sel_y   = get_pdp("selectivity_composite", 80)

# ── Style ─────────────────────────────────────────────────────────────────────
BLUE   = "#2C6FAC"
ORANGE = "#E07B2A"
GREEN  = "#3A8F5E"
ANNOT  = "#444444"
BG     = "#FAFAFA"

plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.facecolor":   BG,
    "figure.facecolor": "white",
})

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
fig.suptitle("Non-Linear Relationships in the Graduation Rate Model\n"
             "(Partial Dependence Plots, Random Forest)",
             fontsize=13, fontweight="bold", y=1.02)

# ─────────────────────────────────────────────────────────────────────────────
# Panel 1: pell_pct
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[0]
pell_ys = smooth(pell_y, window=7)

ax.plot(pell_x, pell_y, color=GREEN, alpha=0.20, lw=1.2)
ax.plot(pell_x, pell_ys, color=GREEN, lw=2.5, zorder=3)

# Shade high-Pell zone
plateau_idx = np.searchsorted(pell_x, 0.55)
plateau_x   = float(pell_x[plateau_idx])
plateau_y   = float(pell_ys[plateau_idx])
ax.axvspan(plateau_x, pell_x[-1], color="#D45F5F", alpha=0.07)
ax.axvline(plateau_x, color="#C0392B", lw=1.5, ls="--", alpha=0.8)
ax.text(plateau_x + 0.01, plateau_y + 0.003,
        f"≥{int(plateau_x*100)}% Pell:\nsteeper decline",
        fontsize=9, color="#C0392B", va="bottom")

# Effect-size arrow on right edge
y_lo = float(pell_ys.min())
y_hi = float(pell_ys.max())
ax.annotate("", xy=(pell_x[-1], y_lo), xytext=(pell_x[-1], y_hi),
            arrowprops=dict(arrowstyle="<->", color=ANNOT, lw=1.3))
ax.text(pell_x[-1] + 0.015, (y_hi + y_lo) / 2,
        f"{(y_hi-y_lo)*100:.1f} pp",
        fontsize=9, va="center", color=ANNOT)

ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax.set_xlabel("Pell Grant Recipient Share", fontsize=11)
ax.set_ylabel("Predicted 6-Year Graduation Rate", fontsize=11)
ax.set_title("Pell Grant Concentration\n(dominant predictor, ~32.4 pp range)",
             fontsize=11, fontweight="bold")
ax.set_xlim(pell_x[0] - 0.01, pell_x[-1] + 0.07)

# ─────────────────────────────────────────────────────────────────────────────
# Panel 2: instr_exp_per_fte
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[1]
instr_ys = smooth(instr_y, window=7)

ax.plot(instr_x, instr_y, color=ORANGE, alpha=0.20, lw=1.2)
ax.plot(instr_x, instr_ys, color=ORANGE, lw=2.5, zorder=3)

# Plateau at ~$10k
plateau_spend = 10000
plateau_idx   = np.searchsorted(instr_x, plateau_spend)
plateau_x     = float(instr_x[plateau_idx])
plateau_y     = float(instr_ys[plateau_idx])
low_y         = float(instr_ys[0])

# Shade steep zone and add plateau line
ax.axvspan(instr_x[0], plateau_x, color=ORANGE, alpha=0.10)
ax.axvline(plateau_x, color="#A05A10", lw=1.5, ls="--", alpha=0.8)
ax.text(plateau_x + 300, instr_ys.min() + 0.005,
        f"Plateau\n≥$10k",
        fontsize=9, color="#A05A10", va="bottom")

# Effect-size arrow on right edge
y_lo = float(instr_ys.min())
y_hi = float(instr_ys.max())
ax.annotate("", xy=(instr_x[-1], y_lo), xytext=(instr_x[-1], y_hi),
            arrowprops=dict(arrowstyle="<->", color=ANNOT, lw=1.3))
ax.text(instr_x[-1] + 400, (y_hi + y_lo) / 2,
        f"{(y_hi-y_lo)*100:.1f} pp",
        fontsize=9, va="center", color=ANNOT)

# Median reference line only
instr_median = 5013
ax.axvline(instr_median, color="#AAAAAA", lw=1, ls=":", alpha=0.9)
ax.text(instr_median - 200, y_hi - 0.003,
        f"Median\n($5k)",
        fontsize=8.5, color="#888888", va="top", ha="right")

ax.set_xlabel("Instructional Spending per FTE", fontsize=11)
ax.set_ylabel("Predicted 6-Year Graduation Rate", fontsize=11)
ax.set_title("Instructional Spending / FTE\n(largest gains below $10k, ~17.0 pp range)",
             fontsize=11, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax.xaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f"${int(x/1000)}k" if x >= 1000 else f"${int(x)}"))
ax.set_xlim(instr_x[0], instr_x[-1] + 1500)

# ─────────────────────────────────────────────────────────────────────────────
# Panel 3: selectivity_composite
# x-axis inverted so right = more selective (lower z), matching intuition
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[2]
sel_ys = smooth(sel_y, window=5)

ax.plot(sel_x, sel_y, color=BLUE, alpha=0.20, lw=1.2)
ax.plot(sel_x, sel_ys, color=BLUE, lw=2.5, zorder=3)

ax.invert_xaxis()

# Average reference
ax.axvline(0, color="#AAAAAA", lw=1, ls="--")
ax.text(0.08, float(sel_ys.max()) - 0.001,
        "Avg.", fontsize=8.5, color="#888888", va="top")

# Shade the most-selective zone
ax.axvspan(-1, sel_x.min() - 0.1, color=BLUE, alpha=0.08)
ax.text(-1.1, float(sel_ys.max()) - 0.001,
        "Most\nselective", fontsize=8.5, color=BLUE, va="top", ha="right")

# Effect-size arrow — right side of the inverted axis (less selective end)
y_lo = float(sel_ys.min())
y_hi = float(sel_ys.max())
ax.annotate("", xy=(sel_x[0], y_lo), xytext=(sel_x[0], y_hi),
            arrowprops=dict(arrowstyle="<->", color=ANNOT, lw=1.3))
ax.text(sel_x[0] + 0.2, (y_hi + y_lo) / 2,
        f"{(y_hi-y_lo)*100:.1f} pp",
        fontsize=9, va="center", color=ANNOT)

ax.set_xlabel("← Less Selective     |     More Selective →", fontsize=10)
ax.set_ylabel("Predicted 6-Year Graduation Rate", fontsize=11)
ax.set_title("Selectivity\n(more selective → higher grad rate, ~9.7 pp range)",
             fontsize=11, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax.set_xlim(sel_x[-1] + 0.4, sel_x[0] - 0.1)

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
fig.text(0.5, -0.04,
         "All PDPs computed from the Random Forest model (test-set R²=0.806) on the "
         "grad_rate_150_4yr outcome. value_added_proxy excluded from this model.\n"
         "Faint lines = raw PDP; solid lines = smoothed. Y-axis range varies across panels.",
         ha="center", fontsize=9, color="#555555", style="italic")

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = Path("output/slides/slide12_pdp_nonlinear.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"Saved -> {out_path}")
plt.close()

# ── Effect-size summary ───────────────────────────────────────────────────────
print("\nEffect-size summary:")
print(f"  pell_pct              : {(pell_y.max()-pell_y.min())*100:.1f} pp (FEATURED)")
print(f"  instr_exp_per_fte     : {(instr_y.max()-instr_y.min())*100:.1f} pp (FEATURED)")
print(f"  selectivity_composite : {(sel_y.max()-sel_y.min())*100:.1f} pp (FEATURED)")
print(f"  total_completions     : {(get_pdp('total_completions')[1].max() - get_pdp('total_completions')[1].min())*100:.1f} pp (not featured)")
print(f"  stud_fac_ratio        : {(get_pdp('stud_fac_ratio')[1].max() - get_pdp('stud_fac_ratio')[1].min())*100:.1f} pp (not featured)")
