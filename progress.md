# IoT Botnet Detection — Progress & Lessons Learned

**Project:** BoTNeTIoT-L01 IoT intrusion detection (binary and multiclass experiments)  
**Team context:** CMU Neural Networks course project (README overview)

This document summarizes what we implemented, which artifacts exist, and what we learned from data and tooling—not final paper conclusions.

---

## What we set out to do

- Detect **attack vs normal** IoT traffic from **23 flow-statistics features** (10-second window, L0.1 decay).
- Extend to **multiclass** questions: botnet family (Mirai vs Gafgyt vs Normal) and (optionally) attack subtype.
- Study **feature importance**, **per-device** behavior of a global model, and **latency vs accuracy** signals for a deployment-style discussion.

---

## Completed work (by phase)

### Phase 1 — Definitions, splits, and baselines

- **Locked:** Binary labels `0` = attack, `1` = normal; `random_state=42`; stratified **70% / 15% / 15%** train/validation/test; primary imbalance handling via **`class_weight="balanced"`** on sklearn models.
- **Stratified sampling:** Because the CSV rows are **not shuffled**, reading only the first *N* rows can yield a **single class**. We added **`load_binary_stratified_sample`** (chunked read until both classes exist, then stratified subsample).
- **Script:** `scripts/phase1_verify.py` — balanced logistic regression on preprocessed features (after Phase 2 alignment; see below).

### Phase 2 — Preprocessing pipeline

- **Pipeline steps (fit on training data only):** optional **log1p** on columns with high **|skew|** on the train fold; optional **drop** of redundant columns with **|correlation| > threshold**; then **Standard / Robust / MinMax** scaling.
- **Persistence:** `save_preprocessor` / `load_preprocessor` via **joblib** (default path under `artifacts/`).
- **Script:** `scripts/phase2_verify.py` saves the pipeline and checks that **reload + transform** matches the in-memory transform.
- **Alignment:** `phase1_verify.py` was updated to use the **same** preprocessing API and config defaults as Phase 2.

### Phase 3 — Model comparison

- **Script:** `scripts/phase3_train.py` fits the preprocessing pipeline, then trains selected models on transformed features.
- **Models:** Logistic regression, Random Forest, **HistGradientBoostingClassifier** (default strong tabular baseline), **MLPClassifier**, optional **XGBoost** (`xgb` only if the library loads).
- **Outputs:** Training and inference **timings**, validation/test metrics → **`results/phase3_metrics.json`**.
- **Note:** On some **macOS** setups, **XGBoost** fails to load without **OpenMP** (`libomp`); the default stack uses **HGB** so experiments still run without XGBoost.

### Phase 4 — Research-style experiments

- **Multiclass (v2):** `scripts/phase4_multiclass.py` — targets `Attack` (family) or `Attack_subType`.
  - We fixed an important data issue: early file chunks are often **attack-heavy**, so a naive stratified sample could **omit “Normal”** entirely. The loader now **waits until mirai, gafgyt, and Normal** all appear before subsampling (for `attack_family`).
  - Outputs: **`results/phase4_multiclass.json`** (includes macro F1 and multiclass ROC-AUC when well-defined).
- **Feature importance:** `scripts/phase4_feature_importance.py` — Random Forest on preprocessed binary data; ranked importances → **`results/phase4_feature_importance_rf.csv`**.
- **Per-device:** `scripts/phase4_per_device.py` — one **global** `HistGradientBoostingClassifier` (no device ID in features), metrics on the **held-out test set** **grouped by `Device_Name`**. Devices with **only one class** in the test slice skip ROC-AUC (undefined).
- **Summary bundle:** `scripts/phase4_summary.py` aggregates **`phase3_metrics.json`**, Phase 4 JSONs, and the feature-importance CSV path into **`results/phase4_summary.json`** for reporting and deployment notes.

### Code layout

- **`iot_ids/`** — `config`, `data`, `splits`, `splits_with_extras`, `metrics`, `preprocessing`, `training`, and loaders for stratified / multiclass / device metadata.
- **`scripts/`** — `phase1_verify.py`, `phase2_verify.py`, `phase3_train.py`, `phase4_*.py`.
- **`results/`** — JSON/CSV outputs from training and Phase 4 (regenerate by running scripts).
- **`requirements.txt`** — pandas, numpy, scikit-learn, joblib, xgboost (optional at runtime).

---

## What we learned (so far)

### Data and sampling

1. **Row order matters.** A random “first N rows” sample is **not** representative; stratified **chunked** loading was necessary for both binary and multiclass v2 experiments.
2. **Multiclass family labels** need **all three** classes (Mirai, Gafgyt, Normal) in the pool; otherwise models collapse to **binary** `predict_proba` shapes and multiclass ROC metrics break or mislead.
3. **Per-device test slices** can contain **only attacks or only normal** traffic; ROC-AUC is **not defined** there—report accuracy / precision / recall with care, or skip ROC for that slice.

### Modeling and metrics

4. **Accuracy is misleading** under heavy imbalance; we consistently report **precision, recall, F1, ROC-AUC / AP** (binary) and **macro F1 / OVR ROC** (multiclass) where valid.
5. **Class imbalance:** `class_weight="balanced"` and **HGB’s** `class_weight="balanced"` are practical at full scale; resampling the entire millions-of-rows dataset was avoided.
6. **Preprocessing:** Log1p on **skewed** columns materially changes which features dominate tree splits; keeping **one shared pipeline** across Phase 1–3 avoided train/test leakage and kept experiments comparable.

### Feature importance (empirical snapshot)

From one RF run on preprocessed data, top drivers included **`H_L0.1_weight`**, **`MI_dir_L0.1_weight`**, **`HpHp_L0.1_weight`**, and **`HH_L0.1_magnitude`** — consistent with **volume / direction / host–host statistics** mattering for separating flows (see `results/phase4_feature_importance_rf.csv`).

### Engineering

7. **Sklearn / NumPy versions:** e.g. `MLPClassifier` may not accept **`sample_weight`** on older versions; we fall back to unweighted `fit`.
8. **Reproducibility:** fixed `random_state`, stratified splits, and **joblib**-saved preprocessors support consistent inference-style evaluation.

---

## Suggested next steps (not yet done as a single “final report”)

- Write the **course report** tying metrics to research questions (binary vs multiclass tradeoffs, deployment choice using `phase3_metrics.json` timings).
- Optional: **SHAP** on a subsample for interpretability beyond RF importances.
- Optional: **confusion matrices** and **per-class calibration** plots for the report.
- If using **XGBoost** on macOS: install **libomp** (e.g. Homebrew) and add `xgb` back to comparison runs.

---

## How to reproduce the main pipeline

```bash
cd IOT_Intrusion_Detection_Systems
pip install -r requirements.txt   # adjust for your Python environment

python3 scripts/phase3_train.py --nrows 150000
python3 scripts/phase4_multiclass.py --nrows 120000
python3 scripts/phase4_feature_importance.py --nrows 150000
python3 scripts/phase4_per_device.py --nrows 200000
python3 scripts/phase4_summary.py
```

---

*Last updated to reflect Phases 1–4 implementation and observed dataset behavior.*
