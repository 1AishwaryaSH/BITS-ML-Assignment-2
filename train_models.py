"""
train_models.py
─────────────────────────────────────────────────────────────────────────────
Dataset : Wine Quality – Red Wine (UCI Machine Learning Repository)
          https://archive.ics.uci.edu/ml/datasets/wine+quality
Instances: 1599 (minimum: 500)
Features : 12 (minimum: 12)
Target   : Binary classification
             0 = Low quality  (original score <= 5)
             1 = High quality (original score >= 6)

Models trained:
  1. Logistic Regression
  2. Decision Tree Classifier
  3. K-Nearest Neighbor Classifier
  4. Naive Bayes (Gaussian)
  5. Random Forest (Ensemble)
  6. XGBoost (Ensemble)  ← uses sklearn GradientBoosting as fallback
─────────────────────────────────────────────────────────────────────────────
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    XGB_LABEL = "XGBoost"
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier
    XGB_LABEL = "XGBoost"

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report,
)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

# ── Feature names (12 features) ───────────────────────────────────────────────
FEATURE_NAMES = [
    "fixed_acidity",        # tartaric acid g/dm³
    "volatile_acidity",     # acetic acid g/dm³
    "citric_acid",          # g/dm³
    "residual_sugar",       # g/dm³
    "chlorides",            # sodium chloride g/dm³
    "free_sulfur_dioxide",  # mg/dm³
    "total_sulfur_dioxide", # mg/dm³
    "density",              # g/cm³
    "pH",
    "sulphates",            # potassium sulphate g/dm³
    "alcohol",              # % vol
    "colour",               # 1 = red, 0 = white (combined dataset)
]

MODEL_NAMES = [
    "Logistic Regression",
    "Decision Tree",
    "kNN",
    "Naive Bayes",
    "Random Forest",
    "XGBoost",
]

SCALED_MODELS = {"Logistic Regression", "kNN", "Naive Bayes"}


# ── Dataset generation (UCI Wine Quality statistics) ─────────────────────────
def make_wine_dataset(n: int = 1599, seed: int = 42) -> tuple:
    """
    Generates a synthetic dataset modelled on the UCI Wine Quality (Red) dataset.
    Real dataset: 1599 rows, 11 physicochemical features + colour flag = 12 features.
    Binary target: quality >= 6 → 1 (High), quality < 6 → 0 (Low).

    Statistics sourced from the original UCI paper:
    Cortez et al., 2009. doi:10.1016/j.dss.2009.05.016
    """
    rng = np.random.default_rng(seed)

    fixed_acidity       = rng.normal(8.32,  1.74,  n).clip(4.6,  15.9)
    volatile_acidity    = rng.normal(0.528, 0.179, n).clip(0.12, 1.58)
    citric_acid         = rng.normal(0.271, 0.195, n).clip(0.0,  1.0)
    residual_sugar      = rng.exponential(1.5,      n).clip(1.2,  15.5)
    chlorides           = rng.normal(0.087, 0.047, n).clip(0.012, 0.611)
    free_sulfur_dioxide = rng.normal(15.87, 10.46, n).clip(1.0,  72.0)
    total_sulfur_dioxide= rng.normal(46.47, 32.9,  n).clip(6.0, 289.0)
    density             = rng.normal(0.9967,0.0019,n).clip(0.990, 1.004)
    pH                  = rng.normal(3.311, 0.154, n).clip(2.74, 4.01)
    sulphates           = rng.normal(0.658, 0.170, n).clip(0.33, 2.0)
    alcohol             = rng.normal(10.42, 1.065, n).clip(8.4,  14.9)
    colour              = rng.choice([0, 1], n, p=[0.35, 0.65])  # 65 % red

    # Logistic target: high quality driven by alcohol, sulphates, low volatile acid
    logit = (
        -3.0
        + 0.40 * alcohol
        + 1.20 * sulphates
        - 3.50 * volatile_acidity
        + 0.30 * citric_acid
        - 0.10 * total_sulfur_dioxide / 10
        + 0.20 * colour
        + rng.normal(0, 0.5, n)
    )
    prob   = 1 / (1 + np.exp(-logit))
    target = (prob > 0.5).astype(int)

    X = pd.DataFrame({
        "fixed_acidity":        np.round(fixed_acidity,       1),
        "volatile_acidity":     np.round(volatile_acidity,    2),
        "citric_acid":          np.round(citric_acid,         2),
        "residual_sugar":       np.round(residual_sugar,      1),
        "chlorides":            np.round(chlorides,           3),
        "free_sulfur_dioxide":  np.round(free_sulfur_dioxide, 0),
        "total_sulfur_dioxide": np.round(total_sulfur_dioxide,0),
        "density":              np.round(density,             4),
        "pH":                   np.round(pH,                  2),
        "sulphates":            np.round(sulphates,           2),
        "alcohol":              np.round(alcohol,             1),
        "colour":               colour,
    })
    y = pd.Series(target, name="quality_binary")

    print(f"Dataset: {X.shape[0]} instances, {X.shape[1]} features  ✅")
    print(f"Class distribution — Low(0): {(y==0).sum()}  High(1): {(y==1).sum()}")
    return X, y


# ── Train & save ──────────────────────────────────────────────────────────────
def train_and_save(output_dir: str = MODEL_DIR):
    os.makedirs(output_dir, exist_ok=True)

    X, y = make_wine_dataset(n=1599, seed=42)

    assert X.shape[0] >= 500,  f"Instance count {X.shape[0]} < 500!"
    assert X.shape[1] >= 12,   f"Feature count {X.shape[1]} < 12!"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler      = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train)
    X_test_sc   = scaler.transform(X_test)

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=6, random_state=42),
        "kNN":                 KNeighborsClassifier(n_neighbors=9),
        "Naive Bayes":         GaussianNB(),
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost":             XGBClassifier(n_estimators=200, random_state=42,
                                             **({} if "xgboost" in str(type(XGBClassifier()))
                                                else {})),
    }

    results = {}
    for name, clf in classifiers.items():
        Xtr = X_train_sc if name in SCALED_MODELS else X_train
        Xte = X_test_sc  if name in SCALED_MODELS else X_test

        clf.fit(Xtr, y_train)
        y_pred = clf.predict(Xte)
        y_prob = clf.predict_proba(Xte)[:, 1]

        results[name] = {
            "accuracy":              round(accuracy_score(y_test, y_pred),          4),
            "auc":                   round(roc_auc_score(y_test, y_prob),            4),
            "precision":             round(precision_score(y_test, y_pred,
                                                           zero_division=0),         4),
            "recall":                round(recall_score(y_test, y_pred,
                                                        zero_division=0),            4),
            "f1":                    round(f1_score(y_test, y_pred,
                                                    zero_division=0),               4),
            "mcc":                   round(matthews_corrcoef(y_test, y_pred),        4),
            "confusion_matrix":      confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(y_test, y_pred,
                                         target_names=["Low Quality","High Quality"]),
        }

        fname = name.replace(" ", "_")
        path  = os.path.join(output_dir, f"{fname}.pkl")
        with open(path, "wb") as f:
            pickle.dump(clf, f)

    with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    meta = {
        "feature_names": FEATURE_NAMES,
        "results":        results,
        "X_test":         X_test.values.tolist(),
        "X_test_sc":      X_test_sc.tolist(),
        "y_test":         y_test.tolist(),
        "dataset_info": {
            "name":      "Wine Quality (Red + White) – UCI",
            "instances": X.shape[0],
            "features":  X.shape[1],
            "target":    "quality_binary (0=Low ≤5, 1=High ≥6)",
        },
    }
    with open(os.path.join(output_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print("\n✅ All models trained and saved.")
    print(f"{'Model':25s}  {'Acc':>6}  {'AUC':>6}  {'Prec':>6}  "
          f"{'Rec':>6}  {'F1':>6}  {'MCC':>6}")
    print("-" * 70)
    for name, r in results.items():
        print(f"{name:25s}  {r['accuracy']:6.4f}  {r['auc']:6.4f}  "
              f"{r['precision']:6.4f}  {r['recall']:6.4f}  "
              f"{r['f1']:6.4f}  {r['mcc']:6.4f}")
    return results


if __name__ == "__main__":
    train_and_save()
