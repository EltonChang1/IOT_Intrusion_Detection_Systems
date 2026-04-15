#!/usr/bin/env python3
"""
Phase 4 — per-device evaluation: global model (no device in features), metrics by Device_Name on test.

Uses v2 stratified sample with device labels preserved.
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
from sklearn.ensemble import HistGradientBoostingClassifier

from iot_ids.config import (
    PREPROCESS_CORRELATION_THRESHOLD,
    PREPROCESS_SKEW_THRESHOLD,
    PREPROCESS_SCALER,
    RANDOM_STATE,
)
from iot_ids.data import load_binary_v2_stratified_with_device
from iot_ids.metrics import binary_metrics
from sklearn.metrics import precision_score, recall_score, f1_score
from iot_ids.preprocessing import fit_preprocessor
from iot_ids.splits import stratified_train_val_test_with_extras


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 per-device breakdown")
    parser.add_argument("--nrows", type=int, default=200_000)
    parser.add_argument("--min-test-per-device", type=int, default=80)
    parser.add_argument(
        "--scaler",
        choices=("standard", "robust", "minmax"),
        default=PREPROCESS_SCALER,
    )
    parser.add_argument("--skew-threshold", type=float, default=PREPROCESS_SKEW_THRESHOLD)
    parser.add_argument("--drop-correlated", action="store_true")
    parser.add_argument("--correlation-threshold", type=float, default=PREPROCESS_CORRELATION_THRESHOLD)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "phase4_per_device.json",
    )
    args = parser.parse_args()

    skew_threshold: float | None = args.skew_threshold
    if args.skew_threshold < 0:
        skew_threshold = None

    X, y, device = load_binary_v2_stratified_with_device(
        args.nrows, random_state=RANDOM_STATE
    )
    y = np.asarray(y).ravel()
    device = device.reset_index(drop=True)

    split = stratified_train_val_test_with_extras(X, y, device)
    X_train, X_val, X_test = split[0], split[1], split[2]
    y_train, y_val, y_test = split[3], split[4], split[5]
    dev_train, dev_val, dev_test = split[6], split[7], split[8]

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

    clf = HistGradientBoostingClassifier(
        max_depth=8,
        learning_rate=0.1,
        max_iter=400,
        class_weight="balanced",
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train_t, y_train)

    y_test_pred = clf.predict(X_test_t)
    proba_normal = clf.predict_proba(X_test_t)[:, 1]
    proba_attack = clf.predict_proba(X_test_t)[:, 0]

    global_test = {
        "accuracy": float((y_test_pred == y_test).mean()),
        "binary_metrics_normal": binary_metrics(y_test, y_test_pred, proba_normal, pos_label=1),
        "binary_metrics_attack": binary_metrics(y_test, y_test_pred, proba_attack, pos_label=0),
    }

    per_device: dict[str, dict] = {}
    dev_test_np = dev_test.to_numpy() if hasattr(dev_test, "to_numpy") else np.asarray(dev_test)

    for d in np.unique(dev_test_np):
        m = dev_test_np == d
        if m.sum() < args.min_test_per_device:
            continue
        yt = y_test[m]
        yp = y_test_pred[m]
        pn = proba_normal[m]
        pa = proba_attack[m]
        entry: dict = {
            "n_test": int(m.sum()),
            "accuracy": float((yp == yt).mean()),
            "attack_rate": float((yt == 0).mean()),
        }
        if np.unique(yt).size < 2:
            entry["note"] = "single_class_in_test_slice_roc_skipped"
            entry["precision_normal"] = float(
                precision_score(yt, yp, pos_label=1, zero_division=0)
            )
            entry["recall_normal"] = float(recall_score(yt, yp, pos_label=1, zero_division=0))
            entry["f1_normal"] = float(f1_score(yt, yp, pos_label=1, zero_division=0))
        else:
            entry["binary_metrics_normal"] = {
                k: float(v) if isinstance(v, (float, np.floating)) else v
                for k, v in binary_metrics(yt, yp, pn, pos_label=1).items()
            }
            entry["binary_metrics_attack"] = {
                k: float(v) if isinstance(v, (float, np.floating)) else v
                for k, v in binary_metrics(yt, yp, pa, pos_label=0).items()
            }
        per_device[str(d)] = entry

    out = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "nrows_requested": args.nrows,
        "min_test_per_device": args.min_test_per_device,
        "global_test": global_test,
        "per_device": per_device,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)

    print(f"Global test accuracy: {global_test['accuracy']:.4f}")
    print(f"Devices reported (n_test >= {args.min_test_per_device}): {len(per_device)}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
