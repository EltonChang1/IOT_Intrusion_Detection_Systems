#!/usr/bin/env python3
"""
Phase 4 — multiclass: Mirai vs Gafgyt vs Normal (`Attack`) or attack subtypes (`Attack_subType`).

Uses Phase-2 preprocessing fit on train, stratified splits, and RF / HGB / multinomial LR.
Writes results/phase4_multiclass.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from iot_ids.config import (
    PREPROCESS_CORRELATION_THRESHOLD,
    PREPROCESS_SKEW_THRESHOLD,
    PREPROCESS_SCALER,
    RANDOM_STATE,
)
from iot_ids.data import load_multiclass_stratified_sample
from iot_ids.metrics import evaluate_multiclass
from iot_ids.preprocessing import fit_preprocessor
from iot_ids.splits import stratified_train_val_test


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 multiclass (v2)")
    parser.add_argument(
        "--target",
        choices=("attack_family", "attack_subtype"),
        default="attack_family",
        help="Attack = mirai/gafgyt/Normal; Attack_subType = UDP/TCP/...",
    )
    parser.add_argument("--nrows", type=int, default=120_000)
    parser.add_argument(
        "--scaler",
        choices=("standard", "robust", "minmax"),
        default=PREPROCESS_SCALER,
    )
    parser.add_argument("--skew-threshold", type=float, default=PREPROCESS_SKEW_THRESHOLD)
    parser.add_argument("--drop-correlated", action="store_true")
    parser.add_argument("--correlation-threshold", type=float, default=PREPROCESS_CORRELATION_THRESHOLD)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "phase4_multiclass.json")
    args = parser.parse_args()

    skew_threshold: float | None = args.skew_threshold
    if args.skew_threshold < 0:
        skew_threshold = None

    X, y_str = load_multiclass_stratified_sample(
        args.nrows, args.target, random_state=RANDOM_STATE
    )
    y_str = y_str.astype(str)

    X_train, X_val, X_test, y_train_s, y_val_s, y_test_s = stratified_train_val_test(
        X, y_str
    )

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_s)
    y_val = le.transform(y_val_s)
    y_test = le.transform(y_test_s)
    classes = le.classes_.tolist()

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

    models: dict[str, object] = {
        "logistic_regression": LogisticRegression(
            max_iter=3000,
            multi_class="multinomial",
            class_weight="balanced",
            solver="lbfgs",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_depth=8,
            learning_rate=0.1,
            max_iter=300,
            class_weight="balanced",
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=RANDOM_STATE,
        ),
    }

    results: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "target": args.target,
        "n_classes": len(classes),
        "classes": classes,
        "nrows_requested": args.nrows,
        "shapes": {
            "train": list(X_train_t.shape),
            "val": list(X_val_t.shape),
            "test": list(X_test_t.shape),
        },
        "preprocessing": {
            "skew_threshold": skew_threshold,
            "scaler": args.scaler,
            "drop_correlated": args.drop_correlated,
        },
        "models": {},
    }

    label_list = list(range(len(classes)))

    for name, est in models.items():
        est.fit(X_train_t, y_train)
        y_val_pred = est.predict(X_val_t)
        y_test_pred = est.predict(X_test_t)
        proba_val = est.predict_proba(X_val_t)
        proba_test = est.predict_proba(X_test_t)

        results["models"][name] = {
            "validation": evaluate_multiclass(
                y_val, y_val_pred, proba_val, labels=label_list
            ),
            "test": evaluate_multiclass(
                y_test, y_test_pred, proba_test, labels=label_list
            ),
        }

        print(name)
        v = results["models"][name]["validation"]
        t = results["models"][name]["test"]
        print(
            f"  val  acc={v['accuracy']:.4f} f1_macro={v['f1_macro']:.4f} "
            f"roc_auc_ovr={v.get('roc_auc_ovr_macro')}"
        )
        print(
            f"  test acc={t['accuracy']:.4f} f1_macro={t['f1_macro']:.4f} "
            f"roc_auc_ovr={t.get('roc_auc_ovr_macro')}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
