# 04_ml_models.py
# Trains Random Forest, XGBoost, and Elastic Net models to predict four
# institutional outcomes. Saves trained models, test-set predictions, and a
# performance metrics table.
#
# Input:  data/processed/analysis_ready.csv
#         data/processed/efficiency_scores_full.csv  (for avg_efficiency_score)
# Output: output/models/metrics.csv
#         output/models/<model>_<outcome>.joblib  (trained model objects)
#         output/models/predictions_<outcome>.csv (test-set predictions)
#
# Author: Audrey Nichols | DATA 6950 | Spring 2026

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb

# -- Paths --------------------------------------------------------------------
PROC_DIR   = Path("data/processed")
OUTPUT_DIR = Path("output/models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 2026

# Outcomes to predict
OUTCOMES = [
    "grad_rate_150_4yr",
    "median_earnings_6yr",
    "loan_repayment_3yr",
    "avg_efficiency_score",
]

# Features: raw inputs + engineered composites
# Log and z-score versions are excluded to avoid leakage with the raw columns
FEATURES = [
    "instr_exp_per_fte",
    "stud_serv_exp_per_fte",
    "stud_fac_ratio",
    "pell_pct",
    "total_fte",
    "total_enrollment_ipeds",
    "faculty_fte",
    "sal_instr_avg_9mo",
    "sal_noninstr_avg",
    "total_sal_per_fte",
    "total_exp",
    "total_completions",
    "sector",
    "outcome_lag",
    # Engineered features (present if 02_clean.R produced them)
    "value_added_proxy",
    "student_support_intensity",
    "resource_concentration_idx",
    "selectivity_composite",
    "financial_health_idx",
]

# =============================================================================
# 1. Load and merge data
# =============================================================================

def load_data():
    df = pd.read_csv(PROC_DIR / "analysis_ready.csv", low_memory=False, encoding="latin-1")

    # Merge in efficiency score if it exists
    eff_path = PROC_DIR / "efficiency_scores_full.csv"
    if eff_path.exists():
        eff = pd.read_csv(eff_path, usecols=["unitid", "avg_efficiency_score"])
        eff["unitid"] = eff["unitid"].astype(str)
        df["unitid"] = df["unitid"].astype(str)
        df = df.merge(eff, on="unitid", how="left")

    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    return df


# =============================================================================
# 2. Prepare features and target for one outcome
# =============================================================================

def prepare_xy(df, outcome):
    features = [f for f in FEATURES
                if not (f == "value_added_proxy" and outcome == "grad_rate_150_4yr")]
    available = [c for c in features if c in df.columns]
    cols_needed = list(dict.fromkeys(available + [outcome, "sector"]))
    subset = df[cols_needed].dropna(subset=[outcome])

    X = subset[available].copy()
    y = subset[outcome].copy()
    sectors = subset["sector"].copy()

    # Fill any remaining feature NAs with column median
    X = X.fillna(X.median(numeric_only=True))

    return X, y, sectors


# =============================================================================
# 3. Stratified train / validation / test split
# =============================================================================

def split_data(X, y, sectors):
    n = len(X)
    idx = np.arange(n)
    np.random.seed(SEED)

    # Stratify by sector for proportional representation
    strat = sectors.values

    # 70 / 15 / 15 split
    from sklearn.model_selection import train_test_split
    idx_trainval, idx_test = train_test_split(
        idx, test_size=0.15, stratify=strat, random_state=SEED
    )
    strat_trainval = strat[idx_trainval]
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=0.15 / 0.85, stratify=strat_trainval, random_state=SEED
    )

    X_train = X.iloc[idx_train]
    X_val   = X.iloc[idx_val]
    X_test  = X.iloc[idx_test]
    y_train = y.iloc[idx_train]
    y_val   = y.iloc[idx_val]
    y_test  = y.iloc[idx_test]

    print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# =============================================================================
# 4. Model definitions
# =============================================================================

def get_models():
    return {
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=5,
            max_features=0.5,
            n_jobs=-1,
            random_state=SEED,
        ),
        "xgboost": xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=SEED,
            verbosity=0,
        ),
        # Elastic Net needs scaling; wrap in a pipeline
        "elastic_net": Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000, random_state=SEED)),
        ]),
    }


# =============================================================================
# 5. Performance metrics
# =============================================================================

def compute_metrics(y_true, y_pred, model_name, outcome, split):
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    # MAPE: skip zeros to avoid division error
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.nan

    return {
        "model":   model_name,
        "outcome": outcome,
        "split":   split,
        "n":       len(y_true),
        "r2":      round(r2,   4),
        "rmse":    round(rmse, 4),
        "mae":     round(mae,  4),
        "mape":    round(mape, 2),
    }


# =============================================================================
# 6. Train and evaluate
# =============================================================================

def train_outcome(df, outcome, all_metrics):
    print(f"\n{'='*55}")
    print(f"Outcome: {outcome}")
    print(f"{'='*55}")

    X, y, sectors = prepare_xy(df, outcome)
    if len(X) < 100:
        print(f"  Skipping -- only {len(X)} complete cases")
        return

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, sectors)
    models = get_models()

    predictions = {"y_true": y_test.values, "test_index": X_test.index.values}

    for name, model in models.items():
        print(f"\n  [{name}]")

        # 5-fold CV on training set
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=5, scoring="r2", n_jobs=-1
        )
        print(f"  CV R2: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

        # Fit on full training set, evaluate on validation
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_metrics = compute_metrics(y_val.values, val_pred, name, outcome, "val")
        print(f"  Val  R2={val_metrics['r2']:.4f}  RMSE={val_metrics['rmse']:.4f}")

        # Final evaluation on held-out test set
        test_pred = model.predict(X_test)
        test_metrics = compute_metrics(y_test.values, test_pred, name, outcome, "test")
        print(f"  Test R2={test_metrics['r2']:.4f}  RMSE={test_metrics['rmse']:.4f}")

        all_metrics.extend([val_metrics, test_metrics])
        predictions[name] = test_pred

        # Save model
        joblib.dump(model, OUTPUT_DIR / f"{name}_{outcome}.joblib")

    # Save test predictions for this outcome
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(OUTPUT_DIR / f"predictions_{outcome}.csv", index=False)


# =============================================================================
# 7. Main
# =============================================================================

if __name__ == "__main__":
    df = load_data()
    all_metrics = []

    for outcome in OUTCOMES:
        if outcome not in df.columns:
            print(f"\nSkipping {outcome} -- column not found")
            continue
        train_outcome(df, outcome, all_metrics)

    # Save metrics table
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(OUTPUT_DIR / "metrics.csv", index=False)

    print("\n" + "="*55)
    print("Test-set performance summary:")
    print("="*55)
    test_metrics = metrics_df[metrics_df["split"] == "test"]
    print(test_metrics[["model", "outcome", "r2", "rmse", "mae"]].to_string(index=False))
    print(f"\nAll outputs saved to {OUTPUT_DIR}")
