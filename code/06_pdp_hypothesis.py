# 06_pdp_hypothesis.py
# Generates partial dependence plots (PDPs) and ICE plots for the top features
# from SHAP analysis, produces two-way PDPs for the highest-importance feature
# pairs, and tests the central hypothesis by comparing multidimensional models
# against cost-only baseline models.
#
# Input:  output/models/<model>_<outcome>.joblib
#         output/shap/<outcome>/permutation_vs_shap_random_forest.csv  (for top features)
#         data/processed/analysis_ready.csv
#         data/processed/efficiency_scores_full.csv
# Output: output/pdp/
#           pdp_<feature>_<outcome>.png
#           ice_<feature>_<outcome>.png
#           pdp_2way_<feat1>_<feat2>_<outcome>.png
#         output/hypothesis/
#           hypothesis_test_results.csv
#           baseline_vs_full_comparison.png
#
# Author: Audrey Nichols | DATA 6950 | Spring 2026

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import joblib
from pathlib import Path
from scipy import stats

# -- Paths --------------------------------------------------------------------
PROC_DIR   = Path("data/processed")
MODEL_DIR  = Path("output/models")
SHAP_DIR   = Path("output/shap")
PDP_DIR    = Path("output/pdp")
HYP_DIR    = Path("output/pdp/hypothesis")
PDP_DIR.mkdir(parents=True, exist_ok=True)
HYP_DIR.mkdir(parents=True, exist_ok=True)

SEED = 2026
N_GRID = 40   # grid points per feature for PDP

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

# Cost-only features for the baseline hypothesis test
COST_FEATURES = ["instr_exp_per_fte", "stud_serv_exp_per_fte", "total_exp"]


# =============================================================================
# 1. Load data
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
    y = subset[outcome]
    return X, y, available


# =============================================================================
# 2. Get top features for an outcome from SHAP importance file
# =============================================================================

def get_top_features(outcome, feature_names, n=12):
    shap_path = SHAP_DIR / outcome / "permutation_vs_shap_random_forest.csv"
    if shap_path.exists():
        df_imp = pd.read_csv(shap_path).sort_values("shap_mean_abs", ascending=False)
        top = [f for f in df_imp["feature"].tolist() if f in feature_names][:n]
    else:
        # Fall back to all features if SHAP file not found
        top = feature_names[:n]
    return top


# =============================================================================
# 3. Partial dependence plot (single feature)
# =============================================================================

def compute_pdp(model, X, feature_names, feature, n_grid=N_GRID):
    col_idx = feature_names.index(feature)
    p5, p95 = np.percentile(X[feature].dropna(), [5, 95])
    grid = np.linspace(p5, p95, n_grid)

    pdp_vals = []
    ice_vals = []

    X_copy = X.copy()
    for val in grid:
        X_copy[feature] = val
        preds = model.predict(X_copy)
        pdp_vals.append(preds.mean())
        ice_vals.append(preds.copy())

    return grid, np.array(pdp_vals), np.array(ice_vals)


def plot_pdp_ice(grid, pdp_vals, ice_vals, feature, outcome, model_name, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # PDP
    axes[0].plot(grid, pdp_vals, color="#2E5C8A", linewidth=2)
    axes[0].set_xlabel(feature)
    axes[0].set_ylabel(f"Predicted {outcome}")
    axes[0].set_title(f"PDP: {feature}")

    # ICE (sample 100 lines for readability)
    n_sample = min(100, ice_vals.shape[1])
    sample_idx = np.random.default_rng(SEED).choice(ice_vals.shape[1], n_sample, replace=False)
    for i in sample_idx:
        axes[1].plot(grid, ice_vals[:, i], color="steelblue", alpha=0.1, linewidth=0.5)
    axes[1].plot(grid, pdp_vals, color="red", linewidth=2, label="PDP mean")
    axes[1].set_xlabel(feature)
    axes[1].set_title(f"ICE: {feature}")
    axes[1].legend(fontsize=8)

    plt.suptitle(f"{model_name} | {outcome}", fontsize=9)
    plt.tight_layout()
    fname = out_dir / f"pdp_ice_{feature}_{model_name}.png"
    fig.savefig(fname, dpi=120)
    plt.close(fig)


# =============================================================================
# 4. Two-way PDP
# =============================================================================

def compute_pdp_2way(model, X, feature_names, feat1, feat2, n_grid=20):
    p5_1, p95_1 = np.percentile(X[feat1].dropna(), [5, 95])
    p5_2, p95_2 = np.percentile(X[feat2].dropna(), [5, 95])
    grid1 = np.linspace(p5_1, p95_1, n_grid)
    grid2 = np.linspace(p5_2, p95_2, n_grid)

    Z = np.zeros((n_grid, n_grid))
    X_copy = X.copy()
    for i, v1 in enumerate(grid1):
        for j, v2 in enumerate(grid2):
            X_copy[feat1] = v1
            X_copy[feat2] = v2
            Z[i, j] = model.predict(X_copy).mean()

    return grid1, grid2, Z


def plot_pdp_2way(grid1, grid2, Z, feat1, feat2, outcome, model_name, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    c = ax.contourf(grid2, grid1, Z, levels=20, cmap="RdYlGn")
    plt.colorbar(c, ax=ax, label=f"Predicted {outcome}")
    ax.set_xlabel(feat2)
    ax.set_ylabel(feat1)
    ax.set_title(f"Two-Way PDP: {feat1} x {feat2}\n{model_name} | {outcome}")
    plt.tight_layout()
    fname = out_dir / f"pdp_2way_{feat1}_{feat2}_{model_name}.png"
    fig.savefig(fname, dpi=120)
    plt.close(fig)


# =============================================================================
# 5. Hypothesis test: full model vs cost-only baseline
# =============================================================================

def run_hypothesis_test(df):
    """
    Tests the central hypothesis: do multidimensional models predict outcomes
    significantly better than cost-only models (using only expenditure features)?

    Compares R2 on the test set between the full Random Forest model and a
    Random Forest trained with only cost features. Reports improvement in R2
    and a permutation test p-value.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score

    results = []

    for outcome in OUTCOMES:
        if outcome not in df.columns:
            continue

        available_full = [c for c in FEATURES if c in df.columns]
        available_cost = [c for c in COST_FEATURES if c in df.columns]

        if not available_cost:
            continue

        subset = df[available_full + [outcome]].dropna(subset=[outcome])
        if len(subset) < 100:
            continue

        X_full = subset[available_full].fillna(subset[available_full].median(numeric_only=True))
        X_cost = subset[available_cost].fillna(subset[available_cost].median(numeric_only=True))
        y = subset[outcome].values

        # Same train/test split for both models
        X_tr_f, X_te_f, X_tr_c, X_te_c, y_train, y_test = train_test_split(
            X_full, X_cost, y, test_size=0.20, random_state=SEED
        )

        # Full model
        rf_full = RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                                        max_features=0.5, n_jobs=-1, random_state=SEED)
        rf_full.fit(X_tr_f, y_train)
        r2_full = r2_score(y_test, rf_full.predict(X_te_f))

        # Cost-only baseline
        rf_cost = RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                                        max_features="sqrt", n_jobs=-1, random_state=SEED)
        rf_cost.fit(X_tr_c, y_train)
        r2_cost = r2_score(y_test, rf_cost.predict(X_te_c))

        r2_improvement = r2_full - r2_cost

        # Permutation test: shuffle outcome labels and re-estimate improvement
        # to get a null distribution of R2 differences
        n_perm = 500
        null_diffs = []
        rng = np.random.default_rng(SEED)
        for _ in range(n_perm):
            y_perm = rng.permutation(y_train)
            rf_f_p = RandomForestRegressor(n_estimators=50, min_samples_leaf=5,
                                           max_features=0.5, n_jobs=-1, random_state=SEED)
            rf_c_p = RandomForestRegressor(n_estimators=50, min_samples_leaf=5,
                                           max_features="sqrt", n_jobs=-1, random_state=SEED)
            rf_f_p.fit(X_tr_f, y_perm)
            rf_c_p.fit(X_tr_c, y_perm)
            null_diffs.append(
                r2_score(y_test, rf_f_p.predict(X_te_f)) -
                r2_score(y_test, rf_c_p.predict(X_te_c))
            )

        p_value = np.mean(np.array(null_diffs) >= r2_improvement)

        print(f"  {outcome}")
        print(f"    Full R2={r2_full:.4f}  Cost-only R2={r2_cost:.4f}  "
              f"Delta={r2_improvement:.4f}  p={p_value:.4f}")

        results.append({
            "outcome":        outcome,
            "r2_full":        round(r2_full, 4),
            "r2_cost_only":   round(r2_cost, 4),
            "r2_improvement": round(r2_improvement, 4),
            "p_value":        round(p_value, 4),
            "significant":    p_value < 0.05,
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(HYP_DIR / "hypothesis_test_results.csv", index=False)
    print(f"\nHypothesis test results saved -> {HYP_DIR / 'hypothesis_test_results.csv'}")

    # Bar chart of R2 improvement
    if len(results_df) > 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ["#2ca02c" if s else "#d62728" for s in results_df["significant"]]
        ax.bar(results_df["outcome"], results_df["r2_improvement"], color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("R² improvement (full vs cost-only)")
        ax.set_title("Hypothesis Test: Multidimensional vs Cost-Only Models\n"
                     "(green = p < 0.05)")
        ax.tick_params(axis="x", rotation=15)
        plt.tight_layout()
        fig.savefig(HYP_DIR / "baseline_vs_full_comparison.png", dpi=120)
        plt.close(fig)

    return results_df


# =============================================================================
# 6. Main
# =============================================================================

if __name__ == "__main__":
    df = load_data()

    # -- PDPs and ICE plots ---------------------------------------------------
    for outcome in OUTCOMES:
        if outcome not in df.columns:
            continue

        X, y, feature_names = prepare_x(df, outcome)
        model_path = MODEL_DIR / f"random_forest_{outcome}.joblib"
        if not model_path.exists():
            print(f"Skipping PDP for {outcome} -- model not found (run 04 first)")
            continue

        model = joblib.load(model_path)
        top_features = get_top_features(outcome, feature_names)
        out_dir = PDP_DIR / outcome
        out_dir.mkdir(exist_ok=True)

        print(f"\nPDPs: {outcome}  (top {len(top_features)} features)")

        for feat in top_features:
            if feat not in feature_names:
                continue
            grid, pdp_vals, ice_vals = compute_pdp(model, X, feature_names, feat)
            plot_pdp_ice(grid, pdp_vals, ice_vals, feat, outcome, "random_forest", out_dir)

        # Two-way PDPs for top 3 feature pairs
        top3 = [f for f in top_features if f in feature_names][:3]
        pairs = [(top3[i], top3[j]) for i in range(len(top3)) for j in range(i+1, len(top3))]
        for feat1, feat2 in pairs[:3]:
            print(f"  2-way PDP: {feat1} x {feat2}")
            g1, g2, Z = compute_pdp_2way(model, X, feature_names, feat1, feat2)
            plot_pdp_2way(g1, g2, Z, feat1, feat2, outcome, "random_forest", out_dir)

    # -- Hypothesis test ------------------------------------------------------
    print(f"\n{'='*50}")
    print("Hypothesis Test: Full vs Cost-Only Models")
    print(f"{'='*50}")
    run_hypothesis_test(df)

    print(f"\nAll PDP outputs saved to {PDP_DIR}")
    print(f"Hypothesis test outputs saved to {HYP_DIR}")
