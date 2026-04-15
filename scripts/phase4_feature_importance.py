#!/usr/bin/env python3
"""
Phase 4 — feature importance: fit RandomForest on preprocessed binary data (train only),
export ranked importances to CSV (and optional JSON sidecar).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from iot_ids.config import (
    PREPROCESS_CORRELATION_THRESHOLD,
    PREPROCESS_SKEW_THRESHOLD,
    PREPROCESS_SCALER,
    PRIMARY_CLASS_WEIGHT,
    RANDOM_STATE,
)
from iot_ids.data import load_binary_stratified_sample
from iot_ids.preprocessing import fit_preprocessor
from iot_ids.splits import stratified_train_val_test


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 RF feature importance")
    parser.add_argument("--nrows", type=int, default=150_000)
    parser.add_argument(
        "--scaler",
        choices=("standard", "robust", "minmax"),
        default=PREPROCESS_SCALER,
    )
    parser.add_argument("--skew-threshold", type=float, default=PREPROCESS_SKEW_THRESHOLD)
    parser.add_argument("--drop-correlated", action="store_true")
    parser.add_argument("--correlation-threshold", type=float, default=PREPROCESS_CORRELATION_THRESHOLD)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "results" / "phase4_feature_importance_rf.csv",
    )
    args = parser.parse_args()

    skew_threshold: float | None = args.skew_threshold
    if args.skew_threshold < 0:
        skew_threshold = None

    X, y = load_binary_stratified_sample(args.nrows, dataset="noduplicates", random_state=RANDOM_STATE)
    y = np.asarray(y).ravel()
    feature_names = X.columns.tolist()

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

    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight=PRIMARY_CLASS_WEIGHT,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train_t, y_train)

    n_feat = X_train_t.shape[1]
    if pipe.named_steps.get("drop_correlated") is not None:
        keep = pipe.named_steps["drop_correlated"].keep_indices_
        feature_names = [feature_names[i] for i in keep]
    if len(feature_names) != n_feat:
        feature_names = [f"f{i}" for i in range(n_feat)]

    imp = rf.feature_importances_
    df = pd.DataFrame({"feature": list(feature_names)[: len(imp)], "importance": imp})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    meta = {
        "nrows": args.nrows,
        "n_features": int(n_feat),
        "oob_not_used": True,
        "validation_accuracy": float((rf.predict(X_val_t) == y_val).mean()),
    }
    meta_path = args.output_csv.with_suffix(".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {args.output_csv}")
    print(f"Wrote {meta_path}")
    print("\nTop 10 features:")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
