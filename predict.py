"""Predict LLPS propensity for new RNA sequences.

Usage
-----
    python predict.py <input.csv> [output.csv]

The input CSV must contain the same 807 feature columns as the training data
(167 handcrafted tabular features + 640 RNA-FM mean-pool embeddings).
Optional columns 'id' and 'sequence' are carried through to the output.
The 'label' column is ignored if present.

Run stack_pipeline.py once before using this script to train and save all
model artifacts to models/.
"""

from __future__ import annotations

import json
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

MODELS_DIR  = "models"
MODELS_BASE = "models/base"


def get_proba(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    d = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-d))


def load_artifacts():
    with open(f"{MODELS_DIR}/feature_cols.json") as f:
        col_info = json.load(f)
    base_models = {
        name: joblib.load(f"{MODELS_BASE}/{name}.pkl")
        for name in col_info["model_names"]
    }
    tab_scaler  = joblib.load(f"{MODELS_DIR}/tab_scaler.pkl")
    rna_scaler  = joblib.load(f"{MODELS_DIR}/rna_scaler.pkl")
    pca_tab     = joblib.load(f"{MODELS_DIR}/pca_tab.pkl")
    pca_rna     = joblib.load(f"{MODELS_DIR}/pca_rna.pkl")
    meta_scaler = joblib.load(f"{MODELS_DIR}/meta_scaler.pkl")
    meta_nns    = joblib.load(f"{MODELS_DIR}/meta_nn_list.pkl")
    return col_info, base_models, tab_scaler, rna_scaler, pca_tab, pca_rna, meta_scaler, meta_nns


def predict(input_path: str, output_path: str | None = None):
    print(f"Loading model artifacts from {MODELS_DIR}/ ...")
    col_info, base_models, tab_scaler, rna_scaler, pca_tab, pca_rna, meta_scaler, meta_nns = load_artifacts()

    tabular_cols = col_info["tabular_cols"]
    rnafm_cols   = col_info["rnafm_cols"]
    feature_cols = col_info["feature_cols"]
    model_names  = col_info["model_names"]

    print(f"Reading input: {input_path}")
    df = pd.read_csv(input_path)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{len(missing)} feature columns missing from input CSV. "
            f"First few: {missing[:5]}"
        )

    X_full = df[feature_cols].astype(np.float32).values
    X_tab  = df[tabular_cols].astype(np.float32).values
    X_rna  = df[rnafm_cols].astype(np.float32).values

    print(f"  {len(df)} sequences  |  {X_full.shape[1]} features")

    # Base model probabilities
    print("Running base models ...")
    base_proba = np.column_stack([
        get_proba(base_models[name], X_full) for name in model_names
    ])

    # Meta input: PCA(tabular) || PCA(RNA-FM) || base probabilities
    pca_features = np.hstack([
        pca_tab.transform(tab_scaler.transform(X_tab)).astype(np.float32),
        pca_rna.transform(rna_scaler.transform(X_rna)).astype(np.float32),
    ])
    Z   = np.hstack([pca_features, base_proba])
    Z_s = meta_scaler.transform(Z).astype(np.float32)

    # Seed-bagged meta NN
    print(f"Running meta NN ({len(meta_nns)} seeds) ...")
    proba_sum = sum(nn.predict_proba(Z_s)[:, 1] for nn in meta_nns)
    meta_proba = proba_sum / len(meta_nns)
    meta_pred  = (meta_proba >= 0.5).astype(int)

    # Build output
    out = pd.DataFrame()
    if "id" in df.columns:
        out["id"] = df["id"].values
    if "sequence" in df.columns:
        out["sequence"] = df["sequence"].values
    out["meta_nn_proba"] = meta_proba
    out["meta_nn_pred"]  = meta_pred
    for i, name in enumerate(model_names):
        out[f"proba_{name}"] = base_proba[:, i]

    if output_path is None:
        output_path = input_path.replace(".csv", "_predictions.csv")
    out.to_csv(output_path, index=False)
    print(f"Saved predictions -> {output_path}")
    print(f"  LLPS-positive: {meta_pred.sum()} / {len(meta_pred)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <input.csv> [output.csv]")
        sys.exit(1)
    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    predict(inp, out)
