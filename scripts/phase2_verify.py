#!/usr/bin/env python3
"""
Phase 2: fit preprocessing pipeline on training data, transform splits, persist with joblib,
reload and verify bitwise consistency.

Usage (from project root IOT_Intrusion_Detection_Systems):
  python scripts/phase2_verify.py --nrows 50000
  python scripts/phase2_verify.py --scaler robust --drop-correlated --artifact artifacts/my.joblib
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

from iot_ids.config import (
    PREPROCESS_ARTIFACT_PATH,
    PRIMARY_CLASS_WEIGHT,
    RANDOM_STATE,
)
from iot_ids.data import load_binary, load_binary_stratified_sample
from iot_ids.preprocessing import (
    fit_preprocessor,
    load_preprocessor,
    save_preprocessor,
)
from iot_ids.splits import stratified_train_val_test


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2: preprocessing pipeline + joblib persistence")
    parser.add_argument("--dataset", choices=("noduplicates", "v2"), default="noduplicates")
    parser.add_argument("--nrows", type=int, default=80_000, help="Stratified sample size (if not --full)")
    parser.add_argument("--full", action="store_true", help="Use full CSV")
    parser.add_argument(
        "--scaler",
        choices=("standard", "robust", "minmax"),
        default="standard",
        help="Scaling policy after optional log/correlation steps",
    )
    parser.add_argument(
        "--skew-threshold",
        type=float,
        default=1.0,
        help="Apply log1p to columns with |skew| > this on train; use negative to disable (e.g. -1)",
    )
    parser.add_argument(
        "--drop-correlated",
        action="store_true",
        help="Drop redundant columns with |corr| > threshold (fit on train)",
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.99,
        help="Used with --drop-correlated",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=PREPROCESS_ARTIFACT_PATH,
        help="Where to save/load the fitted Pipeline",
    )
    parser.add_argument(
        "--quick-model",
        action="store_true",
        help="Train a balanced logistic regression on transformed features (sanity check)",
    )
    args = parser.parse_args()

    skew_threshold: float | None = args.skew_threshold
    if args.skew_threshold < 0:
        skew_threshold = None

    if args.full:
        X, y = load_binary(args.dataset, nrows=None)
    else:
        X, y = load_binary_stratified_sample(args.nrows, dataset=args.dataset, random_state=RANDOM_STATE)
    y = np.asarray(y)

    X_train, X_val, X_test, y_train, y_val, y_test = stratified_train_val_test(X, y)

    print("Phase 2 — preprocessing")
    print(f"  scaler={args.scaler}, skew_threshold={skew_threshold}, drop_correlated={args.drop_correlated}")
    print(f"  train/val/test shapes: {X_train.shape}, {X_val.shape}, {X_test.shape}")

    pipe = fit_preprocessor(
        X_train,
        skew_threshold=skew_threshold,
        drop_correlated=args.drop_correlated,
        correlation_threshold=args.correlation_threshold,
        scaler=args.scaler,
    )

    X_train_t = pipe.transform(X_train)
    X_val_t = pipe.transform(X_val)
    X_test_t = pipe.transform(X_test)

    path = save_preprocessor(pipe, args.artifact)
    print(f"  saved pipeline -> {path}")

    pipe_loaded = load_preprocessor(path)
    assert np.allclose(pipe_loaded.transform(X_test), X_test_t, rtol=1e-6, atol=1e-9)
    print("  load/transform check: OK (test set matches after reload)")

    log_step = pipe.named_steps.get("log1p_skew")
    if log_step is not None:
        n_log = len(getattr(log_step, "log_indices_", []))
        print(f"  log1p applied to {n_log} / {log_step.n_features_in_} columns (train skew)")

    if args.quick_model:
        clf = LogisticRegression(
            max_iter=2000,
            class_weight=PRIMARY_CLASS_WEIGHT,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        clf.fit(X_train_t, y_train)
        y_pred = clf.predict(X_test_t)
        proba = clf.predict_proba(X_test_t)[:, 1]
        print("\nQuick model (balanced logistic on transformed test set):")
        print(classification_report(y_test, y_pred, digits=4, zero_division=0))
        print("ROC-AUC (P(normal)):", f"{roc_auc_score(y_test, proba):.4f}")

    print("\nPhase 2 verification complete.")


if __name__ == "__main__":
    main()
