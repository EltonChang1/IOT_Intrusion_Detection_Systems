#!/usr/bin/env python3
"""
Phase 1 verification: stratified splits, metrics, primary imbalance strategy (class weights).

Uses the same preprocessing Pipeline as Phase 2 (optional log1p on skewed columns, optional
correlation dropping, then scaling — fit on train only), then LogisticRegression (balanced).

Usage (from project root IOT_Intrusion_Detection_Systems):
  python scripts/phase1_verify.py
  python scripts/phase1_verify.py --nrows 200000 --dataset noduplicates
  python scripts/phase1_verify.py --scaler robust --skew-threshold -1
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

from iot_ids.config import (
    PREPROCESS_CORRELATION_THRESHOLD,
    PREPROCESS_SKEW_THRESHOLD,
    PREPROCESS_SCALER,
    PRIMARY_CLASS_WEIGHT,
    RANDOM_STATE,
)
from iot_ids.data import load_binary, load_binary_stratified_sample
from iot_ids.metrics import binary_metrics
from iot_ids.preprocessing import fit_preprocessor
from iot_ids.splits import stratified_train_val_test


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: splits + metrics + weighted logistic baseline")
    parser.add_argument(
        "--dataset",
        choices=("noduplicates", "v2"),
        default="noduplicates",
        help="Which CSV to use for binary classification",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=100_000,
        help="Number of rows after stratified sampling (ignored if --full)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Load entire CSV (slow, memory-heavy; no stratified subsample)",
    )
    parser.add_argument(
        "--scaler",
        choices=("standard", "robust", "minmax"),
        default=PREPROCESS_SCALER,
        help="Same as Phase 2: scaling after optional log/correlation steps",
    )
    parser.add_argument(
        "--skew-threshold",
        type=float,
        default=PREPROCESS_SKEW_THRESHOLD,
        help="Same as Phase 2: log1p columns with |skew| > this on train; negative disables (e.g. -1)",
    )
    parser.add_argument(
        "--drop-correlated",
        action="store_true",
        help="Same as Phase 2: drop redundant columns (|corr| > threshold on train)",
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=PREPROCESS_CORRELATION_THRESHOLD,
        help="Used with --drop-correlated",
    )
    args = parser.parse_args()

    skew_threshold: float | None = args.skew_threshold
    if args.skew_threshold < 0:
        skew_threshold = None

    print("Phase 1 — definitions, stratified split, metrics, class_weight=", repr(PRIMARY_CLASS_WEIGHT))
    if args.full:
        print(f"Dataset: {args.dataset}, mode=FULL FILE")
    else:
        print(f"Dataset: {args.dataset}, stratified_sample_n={args.nrows}")
    print(f"random_state={RANDOM_STATE}, train/val/test=70/15/15")
    print(
        "Preprocessing (aligned with Phase 2): "
        f"skew_threshold={skew_threshold}, scaler={args.scaler!r}, "
        f"drop_correlated={args.drop_correlated}, correlation_threshold={args.correlation_threshold}\n"
    )

    if args.full:
        X, y = load_binary(args.dataset, nrows=None)
    else:
        X, y = load_binary_stratified_sample(args.nrows, dataset=args.dataset, random_state=RANDOM_STATE)
    y = np.asarray(y)

    X_train, X_val, X_test, y_train, y_val, y_test = stratified_train_val_test(X, y)

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

    clf = LogisticRegression(
        max_iter=2000,
        class_weight=PRIMARY_CLASS_WEIGHT,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(X_train_t, y_train)

    for name, Xs, yt in (
        ("validation", X_val_t, y_val),
        ("test", X_test_t, y_test),
    ):
        y_pred = clf.predict(Xs)
        proba_normal = clf.predict_proba(Xs)[:, 1]

        print("=" * 60)
        print(f"{name.upper()} SET")
        print("=" * 60)
        print("Accuracy (reference only; imbalanced):", f"{accuracy_score(yt, y_pred):.4f}")
        print("\nClassification report (0=attack, 1=normal):")
        print(classification_report(yt, y_pred, digits=4, zero_division=0))
        print("Confusion matrix [rows=true, cols=pred] order [0, 1]:")
        print(confusion_matrix(yt, y_pred, labels=[0, 1]))
        print("ROC-AUC (score = P(normal)):", f"{roc_auc_score(yt, proba_normal):.4f}")
        proba_attack = clf.predict_proba(Xs)[:, 0]
        bm_norm = binary_metrics(yt, y_pred, proba_normal, pos_label=1)
        bm_att = binary_metrics(yt, y_pred, proba_attack, pos_label=0)
        print("Binary metrics (normal positive):", bm_norm)
        print("Binary metrics (attack positive): ", {k: bm_att[k] for k in ("precision", "recall", "f1", "roc_auc", "average_precision")})

    print("\nPhase 1 verification complete.")


if __name__ == "__main__":
    main()
