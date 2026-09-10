# RNA Sequence Stacked Classifier(RNAPhase_ML)

## Associated manuscript
**Sequence architecture of RNA reveals key molecular signatures driving RNA phase separation**

This repository contains the reproducible analysis pipeline associated with the manuscript above.

A reproducible machine-learning pipeline for identifying sequence signatures
associated with LLPS-associated RNAs. The framework integrates 167 handcrafted
RNA sequence descriptors with 640-dimensional mean-pooled RNA-FM embeddings,
yielding an 807-dimensional representation for each RNA.

Fifteen heterogeneous base classifiers are combined through cross-validated
probability stacking with a neural-network meta-learner to distinguish
experimentally supported LLPS-associated RNAs from LLPS-unannotated background
RNAs.

**Independent test-set performance (n = 183; threshold = 0.5):**
the final stacked Meta-NN achieves accuracy = **0.891**, F1 = **0.880**,
AUC-ROC = **0.935**, AUC-PR = **0.944**, and MCC = **0.782**.
The Meta-NN provides the strongest overall performance among the evaluated
models, ranking first on six of seven reported metrics.
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

Once `stack_pipeline.py` has been run and the fitted models have been saved,
predictions can be generated for new RNAs represented using the same
807-feature format (167 handcrafted descriptors + 640 mean-pooled RNA-FM
embedding dimensions):

```bash
python predict.py <your_features.csv> [output.csv]
```

If no output path is given, predictions are saved to
`<your_features.csv>_predictions.csv`.

## Pipeline at a glance

1. **Input representation** — The benchmark contains 732 training RNAs and
   183 independent test RNAs. Each RNA is represented by 167 handcrafted
   sequence descriptors and 640 mean-pooled RNA-FM embedding dimensions.

2. **Base layer** — Fifteen heterogeneous classifiers generate stratified
   five-fold out-of-fold (OOF) probability predictions for the training set.
   The base models are subsequently refitted on the complete training set to
   generate predictions for the independent test set.

3. **Meta-learner** — The handcrafted feature block is standardized and
   reduced to 50 principal components, while the RNA-FM block is separately
   standardized and reduced to 64 principal components. These 114 components
   are concatenated with the 15 base-model probabilities to produce a
   129-dimensional meta-model input.

4. **Final prediction** — A two-hidden-layer MLP (64 and 16 neurons), trained
   with Adam optimization, L2 regularization and early stopping, integrates
   these complementary information sources. Predictions are aggregated across
   the seed-bagged Meta-NN ensemble.
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
