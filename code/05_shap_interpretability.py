# 05_shap_interpretability.py
# Computes SHAP values for all trained models, generates global and local
# importance plots, validates against permutation importance, and produces
# a cross-outcome feature importance heatmap.
#
# Input:  output/models/<model>_<outcome>.joblib
#         data/processed/analysis_ready.csv
#         data/processed/efficiency_scores_full.csv
# Output: output/shap/  (one subdirectory per outcome)
#           global_importance_<outcome>.png
#           beeswarm_<outcome>.png
#           waterfall_high_<outcome>.png
#           waterfall_low_<outcome>.png
#           shap_heatmap.png
#           permutation_vs_shap_<outcome>.csv
#
# Author: Audrey Nichols | DATA 6950 | Spring 2026

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import shap
import joblib
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance

# -- Paths --------------------------------------------------------------------
PROC_DIR   = Path("data/processed")
MODEL_DIR  = Path("output/models")
OUTPUT_DIR = Path("output/shap")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 2026

OUTCOMES = [
    "grad_rate_150_4yr",
    "median_earnings_6yr",
    "loan_repayment_3yr",
    "avg_efficiency_score",
]

FEATURES = [
    "instr_exp_per_fte", "stud_serv_exp_per_fte", "stud_fac_ratio",
    "pell_pct", "total_fte", "total_enrollment_ipeds", "faculty_fte",
    "sal_instr_avg_9mo", "sal_noninstr_avg", "total_sal_per_fte",
    "total_exp", "total_completions", "sector", "outcome_lag",
    "value_added_proxy", "student_support_intensity",
    "resource_concentration_idx", "selectivity_composite", "financial_health_idx",
]

MODEL_NAMES = ["random_forest", "xgboost", "elastic_net"]


# =============================================================================
# 1. Load data (same logic as 04_ml_models.py)
# =============================================================================

def load_data():
    df = pd.read_csv(PROC_DIR / "analysis_ready.csv", low_memory=False, encoding="latin-1")
    eff_path = PROC_DIR / "efficiency_scores_full.csv"
    if eff_path.exists():
        eff = pd.read_csv(eff_path, usecols=["unitid", "avg_efficiency_score"])
        eff["unitid"] = eff["unitid"].astype(str)
        df["unitid"] = df["unitid"].astype(str)
        df = df.merge(eff, on="unitid", how="left")
    return df


def prepare_x(df, outcome):
    features = [f for f in FEATURES
                if not (f == "value_added_proxy" and outcome == "grad_rate_150_4yr")]
    available = [c for c in features if c in df.columns]
    subset = df[available + [outcome]].dropna(subset=[outcome])
    X = subset[available].fillna(subset[available].median(numeric_only=True))
    return X, available


# =============================================================================
# 2. SHAP computation
# =============================================================================

def compute_shap(model, X, model_name, feature_names):
    """Return shap_values array (n_samples x n_features)."""
    # Use a background sample for tree models to keep runtime reasonable
    background = shap.sample(X, min(200, len(X)), random_state=SEED)

    if model_name in ("random_forest", "xgboost"):
        explainer = shap.TreeExplainer(model, data=background)
        shap_values = explainer.shap_values(X)
    else:
        # Elastic Net is a Pipeline; extract the fitted linear model
        scaler = model.named_steps["scaler"]
        linear = model.named_steps["model"]
        X_scaled = scaler.transform(X)
        explainer = shap.LinearExplainer(linear, X_scaled)
        shap_values = explainer.shap_values(X_scaled)

    return shap_values


# =============================================================================
# 3. Plots
# =============================================================================

def plot_global_importance(shap_values, X, feature_names, outcome, model_name, out_dir):
    mean_abs = np.abs(shap_values).mean(axis=0)
    df_imp = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    df_imp = df_imp.sort_values("mean_abs_shap", ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(df_imp["feature"], df_imp["mean_abs_shap"], color="#4C72B0")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Global Feature Importance\n{model_name} | {outcome}")
    plt.tight_layout()
    fig.savefig(out_dir / f"global_importance_{model_name}.png", dpi=120)
    plt.close(fig)


def plot_beeswarm(shap_values, X, feature_names, outcome, model_name, out_dir):
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    mean_abs = shap_df.abs().mean().sort_values(ascending=False)
    top_features = mean_abs.head(15).index.tolist()

    fig, ax = plt.subplots(figsize=(8, 7))
    for i, feat in enumerate(reversed(top_features)):
        col_idx = feature_names.index(feat)
        vals = shap_values[:, col_idx]
        feat_vals = X[feat].values
        # Normalize feature values to [0, 1] for color mapping
        vmin, vmax = feat_vals.min(), feat_vals.max()
        norm = (feat_vals - vmin) / (vmax - vmin + 1e-10)
        colors = plt.cm.RdBu_r(norm)
        jitter = np.random.default_rng(SEED).uniform(-0.2, 0.2, len(vals))
        ax.scatter(vals, np.full(len(vals), i) + jitter,
                   c=colors, alpha=0.5, s=8, linewidths=0)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(list(reversed(top_features)), fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value")
    ax.set_title(f"SHAP Beeswarm Plot\n{model_name} | {outcome}")
    plt.tight_layout()
    fig.savefig(out_dir / f"beeswarm_{model_name}.png", dpi=120)
    plt.close(fig)


def plot_waterfall(shap_values, X, feature_names, outcome, model_name, out_dir,
                   idx, label):
    sv = shap_values[idx]
    base = np.mean(shap_values)  # approximate base value

    df_w = pd.DataFrame({"feature": feature_names, "shap": sv})
    df_w = df_w.reindex(df_w["shap"].abs().sort_values(ascending=False).index).head(12)
    df_w = df_w.sort_values("shap")

    colors = ["#d62728" if v > 0 else "#1f77b4" for v in df_w["shap"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df_w["feature"], df_w["shap"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value")
    ax.set_title(f"Waterfall ({label})\n{model_name} | {outcome}")
    plt.tight_layout()
    fig.savefig(out_dir / f"waterfall_{label}_{model_name}.png", dpi=120)
    plt.close(fig)


# =============================================================================
# 4. Permutation importance and Spearman agreement
# =============================================================================

def permutation_vs_shap(model, X, y, shap_values, feature_names, outcome, model_name, out_dir):
    result = permutation_importance(model, X, y, n_repeats=10, random_state=SEED, n_jobs=-1)
    perm_imp = result.importances_mean

    shap_imp = np.abs(shap_values).mean(axis=0)

    rho, pval = spearmanr(shap_imp, perm_imp)
    print(f"    Spearman rho (SHAP vs permutation): {rho:.4f}  p={pval:.4f}")

    df_comp = pd.DataFrame({
        "feature":    feature_names,
        "shap_mean_abs": shap_imp,
        "perm_importance": perm_imp,
    }).sort_values("shap_mean_abs", ascending=False)
    df_comp.to_csv(out_dir / f"permutation_vs_shap_{model_name}.csv", index=False)

    return rho


# =============================================================================
# 5. Cross-outcome heatmap
# =============================================================================

def build_heatmap(heatmap_data, all_features):
    """heatmap_data: dict of {(model, outcome): shap_importance_series}"""
    if not heatmap_data:
        return

    combined = pd.DataFrame(heatmap_data).fillna(0)

    # Keep top 15 features by max importance across all columns
    top_features = combined.max(axis=1).nlargest(15).index
    combined = combined.loc[top_features]

    fig, ax = plt.subplots(figsize=(max(8, len(combined.columns) * 1.2), 6))
    im = ax.imshow(combined.values, aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(combined.columns)))
    ax.set_xticklabels(
        [f"{m}\n{o[:12]}" for m, o in combined.columns],
        fontsize=7, rotation=45, ha="right"
    )
    ax.set_yticks(range(len(combined.index)))
    ax.set_yticklabels(combined.index, fontsize=8)
    plt.colorbar(im, ax=ax, label="Mean |SHAP|")
    ax.set_title("Cross-Outcome Feature Importance (Top 15 Features)")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "shap_heatmap.png", dpi=120)
    plt.close(fig)
    print(f"Heatmap saved -> {OUTPUT_DIR / 'shap_heatmap.png'}")


# =============================================================================
# 6. Main
# =============================================================================

if __name__ == "__main__":
    df = load_data()
    heatmap_data = {}
    spearman_results = []

    for outcome in OUTCOMES:
        if outcome not in df.columns:
            print(f"Skipping {outcome} -- column not found")
            continue

        X, feature_names = prepare_x(df, outcome)
        out_dir = OUTPUT_DIR / outcome
        out_dir.mkdir(exist_ok=True)

        # Load the test predictions to get y and the original test-set row indices
        pred_path = MODEL_DIR / f"predictions_{outcome}.csv"
        if not pred_path.exists():
            print(f"Skipping {outcome} -- no predictions file (run 04 first)")
            continue
        preds = pd.read_csv(pred_path)
        y = preds["y_true"].values
        # Slice X to the test rows so permutation_importance sees matching sizes
        if "test_index" in preds.columns:
            X_test = X.loc[preds["test_index"].values]
        else:
            # Fallback for predictions files saved before test_index was added
            X_test = X.iloc[:len(y)]

        print(f"\n{'='*50}")
        print(f"SHAP: {outcome}  ({len(X):,} institutions)")
        print(f"{'='*50}")

        for model_name in MODEL_NAMES:
            model_path = MODEL_DIR / f"{model_name}_{outcome}.joblib"
            if not model_path.exists():
                print(f"  [{model_name}] model file not found -- skipping")
                continue

            print(f"  [{model_name}]")
            model = joblib.load(model_path)

            shap_values = compute_shap(model, X, model_name, feature_names)

            # Global importance plot
            plot_global_importance(shap_values, X, feature_names, outcome, model_name, out_dir)

            # Beeswarm plot
            plot_beeswarm(shap_values, X, feature_names, outcome, model_name, out_dir)

            # Waterfall plots for one high-efficiency and one low-efficiency institution
            mean_pred = model.predict(X)
            idx_high = int(np.argmax(mean_pred))
            idx_low  = int(np.argmin(mean_pred))
            plot_waterfall(shap_values, X, feature_names, outcome, model_name, out_dir, idx_high, "high")
            plot_waterfall(shap_values, X, feature_names, outcome, model_name, out_dir, idx_low,  "low")

            # Permutation importance comparison -- must use test-set X/y (same rows)
            # test_index holds pandas label values; convert to positional offsets for numpy indexing
            if "test_index" in preds.columns:
                test_pos = X.index.get_indexer(preds["test_index"].values)
                shap_test = shap_values[test_pos]
            else:
                shap_test = shap_values[:len(y)]
            rho = permutation_vs_shap(
                model, X_test, y, shap_test, feature_names, outcome, model_name, out_dir
            )
            spearman_results.append({
                "model": model_name, "outcome": outcome, "spearman_rho": round(rho, 4)
            })

            # Store for heatmap
            heatmap_data[(model_name, outcome)] = pd.Series(
                np.abs(shap_values).mean(axis=0), index=feature_names
            )

    # Cross-outcome heatmap
    build_heatmap(heatmap_data, FEATURES)

    # Save Spearman agreement table
    if spearman_results:
        pd.DataFrame(spearman_results).to_csv(
            OUTPUT_DIR / "spearman_shap_vs_permutation.csv", index=False
        )

    print(f"\nAll SHAP outputs saved to {OUTPUT_DIR}")
