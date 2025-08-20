# rainfall_gfg.py
# Reference: GeeksforGeeks project "Rainfall Prediction using Machine Learning - Python"
# Dataset (auto-download fallback): https://media.geeksforgeeks.org/wp-content/uploads/20240510131249/Rainfall.csv

import os, sys, io, textwrap, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

from xgboost import XGBClassifier
import joblib

DATA_URL = "https://media.geeksforgeeks.org/wp-content/uploads/20240510131249/Rainfall.csv"
DATA_LOCAL = "Rainfall.csv"
RANDOM_SEED = 42

def safe_read_csv(path_or_url: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path_or_url)
    except Exception:
        # Try requests as a fallback to avoid SSL/redirect hiccups
        try:
            import requests
            r = requests.get(path_or_url, timeout=30)
            r.raise_for_status()
            return pd.read_csv(io.StringIO(r.text))
        except Exception as e:
            raise RuntimeError(f"Could not load CSV from {path_or_url}. Error: {e}")

def load_dataset() -> pd.DataFrame:
    if os.path.exists(DATA_LOCAL):
        print(f"[i] Found local file: {DATA_LOCAL}")
        return safe_read_csv(DATA_LOCAL)
    print("[i] Local Rainfall.csv not found. Attempting to download from GFG media…")
    df = safe_read_csv(DATA_URL)
    # Cache it locally for repeatable runs
    try:
        df.to_csv(DATA_LOCAL, index=False)
        print(f"[i] Downloaded and cached as {DATA_LOCAL}")
    except Exception:
        pass
    return df

def guess_target_column(df: pd.DataFrame) -> str:
    # Common names used across rainfall tutorials/datasets
    candidates = [
        "Rainfall", "rainfall", "RainToday", "Rain_Today", "RainfallToday",
        "WillRainToday", "will_it_rain", "Rain", "rain"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # If no explicit target, fallback: pick a column that looks binary or rainfall mm (we binarize > 0)
    # Heuristic: if any numeric column with sparse nonzeros, assume it's rainfall amount
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns to infer a target from. Please rename your target to 'Rainfall' or 'RainToday'.")
    # choose the most skewed numeric col (often rainfall amount = many zeros)
    sparsity = []
    for c in numeric_cols:
        s = (df[c] == 0).mean()
        sparsity.append((s, c))
    sparsity.sort(reverse=True)  # highest zero ratio first
    return sparsity[0][1]

def binarize_target(series: pd.Series) -> pd.Series:
    # Handle common encodings: Yes/No, True/False, 1/0, or rainfall amount (mm) -> (amount > 0)
    if series.dtype == "O":
        return series.astype(str).str.strip().str.lower().map(
            {"yes": 1, "y": 1, "true": 1, "1": 1, "no": 0, "n": 0, "false": 0, "0": 0}
        )
    if series.dtype == bool:
        return series.astype(int)
    if np.issubdtype(series.dtype, np.number):
        return (series.fillna(0) > 0).astype(int)
    # fallback: try to coerce strings to float, then >0
    try:
        vals = pd.to_numeric(series, errors="coerce").fillna(0.0)
        return (vals > 0).astype(int)
    except Exception:
        raise ValueError("Cannot binarize target column automatically. Please convert it to 0/1 or Yes/No.")

def build_model_pipelines(numeric_features, categorical_features):
    # Preprocessors
    numeric_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  pd.get_dummies)  # placeholder to keep structure (we'll not use ColumnTransformer for cat if we use get_dummies globally)
    ])
    # We will one-hot all categoricals up front with get_dummies for simplicity/robustness.
    # The pipelines below assume a purely numeric X after get_dummies.

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, n_jobs=None),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            eval_metric="logloss",
            tree_method="hist"  # fast & compatible
        ),
    }

    # Imbalanced-learn pipeline: oversample AFTER train split
    pipelines = {}
    for name, clf in models.items():
        pipelines[name] = ImbPipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),  # with_mean False for sparse safety
            ("ros", RandomOverSampler(random_state=RANDOM_SEED)),
            ("clf", clf),
        ])
    return pipelines

def evaluate_model(name, model, X_test, y_test):
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        # convert decision scores to [0,1] via rank-based min-max as fallback
        scores = model.decision_function(X_test)
        mn, mx = scores.min(), scores.max()
        y_proba = (scores - mn) / (mx - mn + 1e-9)
    else:
        # classification only with hard labels
        y_proba = None

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0)
    }
    if y_proba is not None and len(np.unique(y_test)) == 2:
        try:
            metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
        except Exception:
            metrics["roc_auc"] = np.nan
    else:
        metrics["roc_auc"] = np.nan

    print(f"\n===== {name} =====")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))
    print("Scores:", {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})
    return metrics

def main():
    print("[i] Loading dataset…")
    df = load_dataset()

    # Clean column names (strip spaces, unify case)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    # Detect target column
    target_col = guess_target_column(df)
    print(f"[i] Using target column: {target_col}")

    # Binarize target
    y = binarize_target(df[target_col])
    if y.isna().any():
        # If some values are unrecognized, drop them
        bad = y.isna()
        print(f"[!] Dropping {bad.sum()} rows with unrecognized target encodings.")
        df = df.loc[~bad].copy()
        y = y.loc[~bad].astype(int)

    # Drop target from features
    X = df.drop(columns=[target_col])

    # One-hot encode categoricals (robust to unknown columns)
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # Ensure all remaining are numeric
    for c in X.columns:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    # Impute later in pipeline; but drop columns completely NaN
    all_nan = X.columns[X.isna().all()].tolist()
    if all_nan:
        X = X.drop(columns=all_nan)

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=0.2, random_state=RANDOM_SEED, stratify=y.values
    )

    # Build pipelines
    pipelines = build_model_pipelines(
        numeric_features=None, categorical_features=None
    )

    # Train & evaluate
    results = []
    best_name, best_f1 = None, -1
    best_model = None

    for name, pipe in pipelines.items():
        try:
            print(f"\n[i] Training {name}…")
            pipe.fit(X_train, y_train)
            m = evaluate_model(name, pipe, X_test, y_test)
            results.append((name, m))
            if m["f1"] > best_f1:
                best_f1 = m["f1"]
                best_name = name
                best_model = pipe
        except Exception as e:
            print(f"[!] Skipping {name} due to error: {e}")

    # Leaderboard
    if results:
        print("\n=== Leaderboard (sorted by F1) ===")
        for name, m in sorted(results, key=lambda x: x[1]["f1"], reverse=True):
            row = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}
            print(f"{name:>18}: {row}")

        # Save best model
        if best_model is not None:
            joblib.dump(best_model, "rain_model.pkl")
            print(f"\n[i] Saved best model ({best_name}) to 'rain_model.pkl'")
    else:
        print("[!] No model trained successfully. Please check your environment/data.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERROR]", e)
        print(textwrap.dedent("""
            Tips:
            • If auto-download failed, manually place Rainfall.csv next to this script and rerun.
            • Ensure you installed: pandas numpy scikit-learn xgboost imbalanced-learn joblib
            • If your dataset has a different target name, rename it to 'RainToday' or 'Rainfall'.
        """))
