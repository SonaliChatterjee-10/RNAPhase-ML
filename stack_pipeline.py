"""Stacked classifier pipeline.

Steps
-----
1. Train a diverse set of base ML classifiers on the training set.
2. Generate out-of-fold (OOF) probabilities on the training set
   (stratified 5-fold) and full-fit test probabilities for each base
   model, so the meta-model receives leakage-free training inputs.
3. Train a small neural network (sklearn MLPClassifier, seed-bagged)
   on `[PCA(tabular) || PCA(RNA-FM) || base-model probabilities]`
   to produce the final prediction.
4. Report accuracy, precision, recall, F1, AUC-ROC, AUC-PR and MCC
   for every base model and the meta neural network on the test set.
5. Save all fitted model artifacts to models/ for use by predict.py.
"""

from __future__ import annotations

import json
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
)

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
TRAIN_PATH = "datasets/train_all167_tabular_plus_rnafm_mean.csv"
TEST_PATH  = "datasets/test_all167_tabular_plus_rnafm_mean.csv"
NON_FEATURE_COLS = ["id", "sequence", "label"]

RANDOM_STATE = 42
N_SPLITS = 5
META_SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111,
              222, 333, 444, 555, 666, 777, 888, 999, 1234, 5678,
              9876, 1357, 2468, 3691, 4815]
META_PCA_TAB_COMPONENTS = 50   # 167 tabular features -> ~96.8% variance
META_PCA_RNA_COMPONENTS = 64   # 640 RNA-FM features  -> ~97.3% variance

SCALER_MODELS = {"logreg_l2", "ridge", "lda", "knn", "gnb", "svm_rbf", "mlp_base"}

METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr", "mcc"]

MODELS_DIR     = "models"
MODELS_BASE    = os.path.join(MODELS_DIR, "base")
BASE_CACHE     = os.path.join(MODELS_DIR, "base_proba_cache.npz")

os.makedirs(MODELS_BASE, exist_ok=True)
os.makedirs("results", exist_ok=True)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def make_base_models() -> dict:
    """Return a fresh dict of base estimators (unfitted)."""
    return {
        "logreg_l2": LogisticRegression(
            C=1.0, max_iter=4000, random_state=RANDOM_STATE
        ),
        "ridge": RidgeClassifier(alpha=1.0, solver="lsqr", random_state=RANDOM_STATE),
        "lda": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        "knn": KNeighborsClassifier(n_neighbors=15, n_jobs=-1),
        "gnb": GaussianNB(var_smoothing=1e-7),
        "rf": RandomForestClassifier(
            n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "et": ExtraTreesClassifier(
            n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "hgb": HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.05, random_state=RANDOM_STATE
        ),
        "gb": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, random_state=RANDOM_STATE
        ),
        "ada": AdaBoostClassifier(n_estimators=300, random_state=RANDOM_STATE),
        "lgbm": lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbose=-1,
        ),
        "xgb": xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "cat": CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
        ),
        "svm_rbf": SVC(
            C=1.0, kernel="rbf", probability=True, random_state=RANDOM_STATE
        ),
        "mlp_base": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            alpha=1e-3,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=RANDOM_STATE,
        ),
    }


def wrap(model, name: str):
    """Wrap with a StandardScaler when the estimator benefits from it."""
    if name in SCALER_MODELS:
        return Pipeline([("scaler", StandardScaler()), ("model", model)])
    return model


def get_proba(model, X: np.ndarray) -> np.ndarray:
    """Return positive-class probability/score from any classifier."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    # RidgeClassifier path: decision_function -> sigmoid
    d = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-d))


def metrics_row(y_true: np.ndarray, y_proba: np.ndarray, thr: float = 0.5) -> dict:
    y_pred = (y_proba >= thr).astype(int)
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred),
        "f1":        f1_score(y_true, y_pred),
        "auc_roc":   roc_auc_score(y_true, y_proba),
        "auc_pr":    average_precision_score(y_true, y_proba),
        "mcc":       matthews_corrcoef(y_true, y_pred),
    }


def _base_models_saved(names):
    return all(os.path.exists(os.path.join(MODELS_BASE, f"{n}.pkl")) for n in names)


# ----------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------
print("Loading data...")
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

feature_cols  = [c for c in train_df.columns if c not in NON_FEATURE_COLS]
tabular_cols  = [c for c in feature_cols if not c.startswith("rnafm_")]
rnafm_cols    = [c for c in feature_cols if c.startswith("rnafm_")]
print(f"  train: {train_df.shape}  test: {test_df.shape}  "
      f"features: {len(feature_cols)} (tabular={len(tabular_cols)}, rnafm={len(rnafm_cols)})")

X_train = train_df[feature_cols].astype(np.float32).values
y_train = train_df["label"].astype(int).values
X_test  = test_df[feature_cols].astype(np.float32).values
y_test  = test_df["label"].astype(int).values

tab_idx = [feature_cols.index(c) for c in tabular_cols]
rna_idx = [feature_cols.index(c) for c in rnafm_cols]
X_train_tab, X_train_rna = X_train[:, tab_idx], X_train[:, rna_idx]
X_test_tab,  X_test_rna  = X_test[:, tab_idx],  X_test[:, rna_idx]

print(f"  label train: {dict(pd.Series(y_train).value_counts().sort_index())} | "
      f"test: {dict(pd.Series(y_test).value_counts().sort_index())}")


# ----------------------------------------------------------------------
# Base models: OOF + test probabilities (cached on disk)
# ----------------------------------------------------------------------
model_names = list(make_base_models().keys())
n_train, n_test = len(y_train), len(y_test)

if os.path.exists(BASE_CACHE) and _base_models_saved(model_names):
    cache = np.load(BASE_CACHE, allow_pickle=True)
    cached_names = list(cache["names"])
    if cached_names == model_names:
        oof_proba  = pd.DataFrame(cache["oof"], columns=model_names)
        test_proba = pd.DataFrame(cache["tst"], columns=model_names)
        print(f"\nLoaded cached base probabilities from {BASE_CACHE}.")
    else:
        oof_proba = test_proba = None
        print("\nCache exists but model list differs; recomputing base probabilities.")
else:
    oof_proba = test_proba = None

if oof_proba is None:
    oof_proba  = pd.DataFrame(np.zeros((n_train, len(model_names))), columns=model_names)
    test_proba = pd.DataFrame(np.zeros((n_test,  len(model_names))), columns=model_names)

    skf    = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    splits = list(skf.split(X_train, y_train))

    print("\nTraining base models (5-fold OOF + full-fit test predictions):")
    for name in model_names:
        t0 = time.time()
        for tr_idx, va_idx in splits:
            m = wrap(make_base_models()[name], name)
            m.fit(X_train[tr_idx], y_train[tr_idx])
            oof_proba.loc[va_idx, name] = get_proba(m, X_train[va_idx])
        m_full = wrap(make_base_models()[name], name)
        m_full.fit(X_train, y_train)
        test_proba[name] = get_proba(m_full, X_test)
        joblib.dump(m_full, os.path.join(MODELS_BASE, f"{name}.pkl"))
        print(f"  {name:<10s}  oof+test done in {time.time() - t0:6.1f}s")

    np.savez(BASE_CACHE, oof=oof_proba.values, tst=test_proba.values,
             names=np.array(model_names))
    print(f"  Cached base probabilities -> {BASE_CACHE}")

    with open(os.path.join(MODELS_DIR, "feature_cols.json"), "w") as f:
        json.dump({"feature_cols": feature_cols,
                   "tabular_cols": tabular_cols,
                   "rnafm_cols":   rnafm_cols,
                   "model_names":  model_names}, f, indent=2)


# ----------------------------------------------------------------------
# Base-model metrics on test
# ----------------------------------------------------------------------
base_metrics = pd.DataFrame(
    [metrics_row(y_test, test_proba[n].values) for n in model_names], index=model_names
)[METRIC_KEYS]

print("\n=== Base model metrics on test ===")
print(base_metrics.round(4).to_string())

best_by_auroc = base_metrics["auc_roc"].idxmax()
best_by_f1    = base_metrics["f1"].idxmax()
best_by_mcc   = base_metrics["mcc"].idxmax()
print(f"\nBest base — AUROC: {best_by_auroc} ({base_metrics.loc[best_by_auroc,'auc_roc']:.4f})"
      f" | F1: {best_by_f1} ({base_metrics.loc[best_by_f1,'f1']:.4f})"
      f" | MCC: {best_by_mcc} ({base_metrics.loc[best_by_mcc,'mcc']:.4f})")


# ----------------------------------------------------------------------
# Meta neural network
# Input = [PCA(tabular) || PCA(RNA-FM) || base-model probabilities]
# Separate PCAs preserve each feature block's structure before joining.
# Output = positive-class probability (seed-bagged ensemble of small MLPs)
# ----------------------------------------------------------------------
print(f"\nBuilding meta NN inputs (separate StandardScaler+PCA per feature block, "
      f"+ {len(model_names)} base probabilities)...")

tab_scaler = StandardScaler().fit(X_train_tab)
rna_scaler = StandardScaler().fit(X_train_rna)

pca_tab = PCA(n_components=META_PCA_TAB_COMPONENTS,
              random_state=RANDOM_STATE).fit(tab_scaler.transform(X_train_tab))
pca_rna = PCA(n_components=META_PCA_RNA_COMPONENTS,
              random_state=RANDOM_STATE).fit(rna_scaler.transform(X_train_rna))

X_train_p = np.hstack([
    pca_tab.transform(tab_scaler.transform(X_train_tab)).astype(np.float32),
    pca_rna.transform(rna_scaler.transform(X_train_rna)).astype(np.float32),
])
X_test_p = np.hstack([
    pca_tab.transform(tab_scaler.transform(X_test_tab)).astype(np.float32),
    pca_rna.transform(rna_scaler.transform(X_test_rna)).astype(np.float32),
])
print(f"  Tabular PCA: {META_PCA_TAB_COMPONENTS} components  "
      f"var={pca_tab.explained_variance_ratio_.sum():.3f}")
print(f"  RNA-FM  PCA: {META_PCA_RNA_COMPONENTS} components  "
      f"var={pca_rna.explained_variance_ratio_.sum():.3f}")
print(f"  Combined PCA dims: {X_train_p.shape[1]}")

Z_train = np.hstack([X_train_p, oof_proba.values.astype(np.float32)])
Z_test  = np.hstack([X_test_p,  test_proba.values.astype(np.float32)])

meta_scaler = StandardScaler().fit(Z_train)
Z_train_s   = meta_scaler.transform(Z_train).astype(np.float32)
Z_test_s    = meta_scaler.transform(Z_test).astype(np.float32)
print(f"  meta input dim: {Z_train_s.shape[1]}  "
      f"train rows: {Z_train_s.shape[0]}  test rows: {Z_test_s.shape[0]}")

print(f"\nTraining meta NN (seed-bagged MLP, {len(META_SEEDS)} seeds)...")
meta_nn_list = []
meta_test_proba_sum = np.zeros(n_test, dtype=np.float64)
t0 = time.time()
for seed in META_SEEDS:
    nn = MLPClassifier(
        hidden_layer_sizes=(64, 16),
        activation="relu",
        alpha=1e-3,
        batch_size=64,
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=25,
        random_state=seed,
    )
    nn.fit(Z_train_s, y_train)
    meta_nn_list.append(nn)
    meta_test_proba_sum += nn.predict_proba(Z_test_s)[:, 1]

meta_test_proba = meta_test_proba_sum / len(META_SEEDS)
print(f"  done in {time.time() - t0:.1f}s")

# Save meta-level artifacts
joblib.dump(tab_scaler,  os.path.join(MODELS_DIR, "tab_scaler.pkl"))
joblib.dump(rna_scaler,  os.path.join(MODELS_DIR, "rna_scaler.pkl"))
joblib.dump(pca_tab,     os.path.join(MODELS_DIR, "pca_tab.pkl"))
joblib.dump(pca_rna,     os.path.join(MODELS_DIR, "pca_rna.pkl"))
joblib.dump(meta_scaler, os.path.join(MODELS_DIR, "meta_scaler.pkl"))
joblib.dump(meta_nn_list, os.path.join(MODELS_DIR, "meta_nn_list.pkl"))
print(f"  Saved model artifacts -> {MODELS_DIR}/")


# ----------------------------------------------------------------------
# Final metrics
# ----------------------------------------------------------------------
nn_metrics  = metrics_row(y_test, meta_test_proba)
nn_row      = pd.DataFrame([nn_metrics], index=["meta_nn"])[METRIC_KEYS]
final_table = pd.concat([base_metrics, nn_row]).round(4)

print("\n=== Final metrics on test (base models + meta NN) ===")
print(final_table.to_string())

best_base_auroc = base_metrics["auc_roc"].max()
print(f"\nMeta NN AUROC = {nn_metrics['auc_roc']:.4f} | "
      f"Best base AUROC = {best_base_auroc:.4f} | "
      f"delta = {nn_metrics['auc_roc'] - best_base_auroc:+.4f}")

final_table.to_csv("results/model_metrics.csv")
pd.DataFrame({
    "id":            test_df["id"].values,
    "label":         y_test,
    "meta_nn_proba": meta_test_proba,
    "meta_nn_pred":  (meta_test_proba >= 0.5).astype(int),
    **{f"proba_{n}": test_proba[n].values for n in model_names},
}).to_csv("results/test_predictions.csv", index=False)

print("\nSaved: results/model_metrics.csv, results/test_predictions.csv")
