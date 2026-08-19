# RNA Sequence Stacked Classifier

A reproducible pipeline that trains 15 diverse machine-learning classifiers
on 807-dimensional RNA sequence features (167 handcrafted tabular features
+ 640-dim RNA-FM mean embedding) and then stacks them with a small
seed-bagged neural network to produce a final binary LLPS prediction.

**Headline result (test n = 183, threshold 0.5):**
meta-NN achieves AUC-ROC = **0.935**, F1 = **0.880**, MCC = **0.782** —
better than the strongest individual base learner on all primary metrics.

## Repository layout

```
RNA-LLPS/
├── datasets/
│   ├── train_all167_tabular_plus_rnafm_mean.csv   # 732 rows × 810 cols
│   └── test_all167_tabular_plus_rnafm_mean.csv    # 183 rows × 810 cols
├── results/
│   ├── model_metrics.csv                          # per-model test metrics
│   └── test_predictions.csv                       # per-sample predictions
├── models/                                        # created after first run
│   ├── base/                                      # 15 fitted base models
│   ├── pca_tab.pkl, pca_rna.pkl                   # PCA transformers
│   ├── tab_scaler.pkl, rna_scaler.pkl             # feature scalers
│   ├── meta_scaler.pkl                            # meta-level scaler
│   ├── meta_nn_list.pkl                           # 25 seed-bagged MLPs
│   ├── feature_cols.json                          # column name registry
│   └── base_proba_cache.npz                       # OOF probability cache
├── stack_pipeline.py                              # trains and saves all models
├── predict.py                                     # inference on new sequences
├── requirements.txt
├── LICENSE
└── README.md
```

Each CSV row is one RNA sequence with columns:
`id, sequence, label, <807 numeric features>`. The classifier
ignores `id` and `sequence` and predicts `label`.

## Quick reproduction

```bash
# 1. Install pinned dependencies
pip install -r requirements.txt

# 2. Train all models, evaluate on test set, save model artifacts
python stack_pipeline.py
#   -> results/model_metrics.csv
#   -> results/test_predictions.csv
#   -> models/  (all fitted model files)
```

> **Exact reproducibility requires Python 3.9 and the versions in `requirements.txt`.**
> Results are bit-identical across runs on the same machine; minor floating-point
> differences may occur on machines with a different CPU architecture.

## Predicting on new sequences

Once `stack_pipeline.py` has been run (models are saved), you can predict
on any new feature CSV with the same 807-column format:

```bash
python predict.py <your_features.csv> [output.csv]
```

If no output path is given, predictions are saved to
`<your_features.csv>_predictions.csv`.

## Pipeline at a glance

1. **Pre-process** — 732 training sequences and 183 test sequences; features
   split into 167 handcrafted tabular descriptors and 640-dim RNA-FM embeddings.
2. **Base layer** (15 classifiers across linear, kernel, neighbours,
   Bayesian, tree-ensemble, gradient-boosting and shallow-NN families)
   produces leakage-free out-of-fold (OOF) probabilities on the train set
   via stratified 5-fold CV, plus a single full-fit prediction on the test set.
3. **Meta NN** — tabular features are standardised and projected onto
   **50 principal components**; RNA-FM features onto **64 principal
   components** (separate PCA per block). The two PCA outputs (114 dims)
   are concatenated with the 15 base-model probabilities to form a
   **129-dim** input, which is fed to a small MLP `(64, 16)` with ReLU
   activations, L2 regularisation and early stopping, seed-bagged across
   25 random seeds.

## Reproducibility notes

- Global seed `RANDOM_STATE = 42` controls all stochastic components.
- The 25 meta-NN seeds are listed explicitly in `META_SEEDS` and produce
  bit-identical predictions on the same machine.
- The cache file `models/base_proba_cache.npz` is created automatically
  after the first run; subsequent runs skip the ~12-minute base-model
  retraining.
- Class balance: 53 % / 47 % (negative / positive) in both train and
  test — no resampling or class-weighting is applied.

## Outputs explained

- **`results/model_metrics.csv`** — one row per model with columns
  `accuracy, precision, recall, f1, auc_roc, auc_pr, mcc`.
- **`results/test_predictions.csv`** — 183 rows; for each test sequence:
  `id`, `label`, `meta_nn_proba`, `meta_nn_pred`, and `proba_<model>`
  for each of the 15 base learners.

## Dependencies

`Python 3.9`, `scikit-learn 1.6.1`, `lightgbm 4.4.0`, `xgboost 2.1.4`,
`catboost 1.2.10`, `pandas 2.2.3`, `numpy 1.23.0`, `matplotlib 3.5.2`.
See `requirements.txt` for pinned versions.

## License

MIT — see [LICENSE](LICENSE).
