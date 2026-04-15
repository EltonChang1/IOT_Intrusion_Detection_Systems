#!/usr/bin/env python3
"""
Phase 3: train and compare classifiers on Phase-2-preprocessed features.

Default models: logistic regression (lr), Random Forest (rf), sklearn
HistGradientBoostingClassifier (hgb), MLP (mlp). Optional: xgboost (xgb) if the
library loads (may require OpenMP on macOS).

Preprocessing is fit on train only; validation and test metrics plus timings are printed
and written to results/phase3_metrics.json.

Usage (from project root IOT_Intrusion_Detection_Systems):
  python scripts/phase3_train.py --nrows 100000
  python scripts/phase3_train.py --models lr rf hgb --nrows 50000
  python scripts/phase3_train.py --models lr rf xgb mlp --save-preprocessor
  python scripts/phase3_train.py --full --models hgb
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
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from iot_ids.config import (
    PREPROCESS_CORRELATION_THRESHOLD,
    PREPROCESS_SKEW_THRESHOLD,
    PREPROCESS_SCALER,
    PREPROCESS_ARTIFACT_PATH,
    PRIMARY_CLASS_WEIGHT,
    RANDOM_STATE,
)
from iot_ids.data import load_binary, load_binary_stratified_sample
from iot_ids.preprocessing import fit_preprocessor, save_preprocessor
from iot_ids.splits import stratified_train_val_test
from iot_ids.training import evaluate_binary, mlp_sample_weights, time_call, xgboost_scale_pos_weight


def _parse_models(raw: list[str]) -> list[str]:
    aliases = {
        "logistic": "lr",
        "logreg": "lr",
        "random_forest": "rf",
        "xgboost": "xgb",
        "hist_gbm": "hgb",
        "mlp": "mlp",
    }
    out: list[str] = []
    for x in raw:
        k = x.strip().lower()
        k = aliases.get(k, k)
        if k not in ("lr", "rf", "xgb", "hgb", "mlp"):
            raise ValueError(f"Unknown model: {x!r}; use lr, rf, xgb, hgb, mlp")
        out.append(k)
    return out


def _make_estimators(
    y_train: np.ndarray,
    model_keys: list[str],
    *,
    random_state: int,
) -> dict[str, tuple[str, object]]:
    """Short key -> (artifact name, unfitted estimator)."""
    spw = xgboost_scale_pos_weight(y_train)
    estimators: dict[str, tuple[str, object]] = {}

    if "lr" in model_keys:
        estimators["lr"] = (
            "logistic_regression",
            LogisticRegression(
                max_iter=2000,
                class_weight=PRIMARY_CLASS_WEIGHT,
                random_state=random_state,
                n_jobs=-1,
            ),
        )

    if "rf" in model_keys:
        estimators["rf"] = (
            "random_forest",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                class_weight=PRIMARY_CLASS_WEIGHT,
                random_state=random_state,
                n_jobs=-1,
            ),
        )

    if "xgb" in model_keys:
        try:
            from xgboost import XGBClassifier
        except ImportError as e:
            raise ImportError(
                "Model 'xgb' requires XGBoost. Install: pip install xgboost "
                "(on macOS you may need: brew install libomp)"
            ) from e

        estimators["xgb"] = (
            "xgboost",
            XGBClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=1,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=-1,
                tree_method="hist",
                scale_pos_weight=spw,
            ),
        )

    # sklearn-native boosting (no libomp); good default if XGBoost fails to load
    if "hgb" in model_keys:
        from sklearn.ensemble import HistGradientBoostingClassifier

        estimators["hgb"] = (
            "hist_gradient_boosting",
            HistGradientBoostingClassifier(
                max_depth=8,
                learning_rate=0.1,
                max_iter=300,
                l2_regularization=1e-3,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=15,
                class_weight="balanced",
                random_state=random_state,
            ),
        )

    if "mlp" in model_keys:
        estimators["mlp"] = (
            "mlp",
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size=256,
                learning_rate_init=1e-3,
                max_iter=400,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=15,
                random_state=random_state,
            ),
        )

    return estimators


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: baseline + RF + XGBoost + MLP")
    parser.add_argument("--dataset", choices=("noduplicates", "v2"), default="noduplicates")
    parser.add_argument("--nrows", type=int, default=150_000, help="Stratified sample size if not --full")
    parser.add_argument("--full", action="store_true", help="Load entire CSV")
    parser.add_argument(
        "--scaler",
        choices=("standard", "robust", "minmax"),
        default=PREPROCESS_SCALER,
    )
    parser.add_argument("--skew-threshold", type=float, default=PREPROCESS_SKEW_THRESHOLD)
    parser.add_argument("--drop-correlated", action="store_true")
    parser.add_argument("--correlation-threshold", type=float, default=PREPROCESS_CORRELATION_THRESHOLD)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lr", "rf", "hgb", "mlp"],
        help="One or more of: lr rf xgb hgb mlp (default uses hgb instead of xgb; use xgb if installed)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results",
        help="Directory for phase3_metrics.json",
    )
    parser.add_argument(
        "--save-preprocessor",
        action="store_true",
        help=f"Save fitted pipeline (default path: {PREPROCESS_ARTIFACT_PATH})",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=PREPROCESS_ARTIFACT_PATH,
        help="Path for preprocessor when --save-preprocessor is set",
    )
    args = parser.parse_args()

    model_keys = _parse_models(args.models)
    skew_threshold: float | None = args.skew_threshold
    if args.skew_threshold < 0:
        skew_threshold = None

    if args.full:
        X, y = load_binary(args.dataset, nrows=None)
    else:
        X, y = load_binary_stratified_sample(
            args.nrows, dataset=args.dataset, random_state=RANDOM_STATE
        )
    y = np.asarray(y).ravel()

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

    if args.save_preprocessor:
        saved = save_preprocessor(pipe, args.artifact)
        print(f"Saved preprocessor -> {saved}")

    all_estimators = _make_estimators(y_train, model_keys, random_state=RANDOM_STATE)
    results: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "nrows_requested": None if args.full else args.nrows,
        "full_csv": args.full,
        "shapes": {
            "train": list(X_train_t.shape),
            "val": list(X_val_t.shape),
            "test": list(X_test_t.shape),
        },
        "preprocessing": {
            "skew_threshold": skew_threshold,
            "scaler": args.scaler,
            "drop_correlated": args.drop_correlated,
            "correlation_threshold": args.correlation_threshold,
        },
        "random_state": RANDOM_STATE,
        "models": {},
    }

    print("Phase 3 — training")
    print(f"  shapes: train={X_train_t.shape}, val={X_val_t.shape}, test={X_test_t.shape}")
    print(f"  models: {model_keys}\n")

    for key in model_keys:
        name, est = all_estimators[key]
        sw = mlp_sample_weights(y_train) if key == "mlp" else None

        def fit_fn():
            if sw is not None:
                try:
                    est.fit(X_train_t, y_train, sample_weight=sw)
                except TypeError:
                    # Older sklearn: MLPClassifier may not support sample_weight
                    est.fit(X_train_t, y_train)
            else:
                est.fit(X_train_t, y_train)
            return est

        _, train_s = time_call(fit_fn)

        def predict_val():
            return est.predict(X_val_t), est.predict_proba(X_val_t)[:, 1]

        (y_val_pred, y_val_proba), infer_val_s = time_call(predict_val)

        def predict_test():
            return est.predict(X_test_t), est.predict_proba(X_test_t)[:, 1]

        (y_test_pred, y_test_proba), infer_test_s = time_call(predict_test)

        n_val = X_val_t.shape[0]
        n_test = X_test_t.shape[0]

        results["models"][name] = {
            "train_seconds": train_s,
            "infer_val_seconds": infer_val_s,
            "infer_test_seconds": infer_test_s,
            "infer_test_seconds_per_sample": infer_test_s / max(n_test, 1),
            "validation": evaluate_binary(y_val, y_val_pred, y_val_proba),
            "test": evaluate_binary(y_test, y_test_pred, y_test_proba),
        }

        print("-" * 60)
        print(name)
        print(f"  train: {train_s:.2f}s | infer val: {infer_val_s:.4f}s | infer test: {infer_test_s:.4f}s ({infer_test_s / max(n_test, 1) * 1e6:.2f} µs/sample)")
        v = results["models"][name]["validation"]
        t = results["models"][name]["test"]
        print(
            f"  val  — acc={v['accuracy']:.4f} f1_macro={v['f1_macro']:.4f} roc_auc={v['roc_auc']:.4f}"
        )
        print(
            f"  test — acc={t['accuracy']:.4f} f1_macro={t['f1_macro']:.4f} roc_auc={t['roc_auc']:.4f}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "phase3_metrics.json"
    # JSON: sklearn report has float keys — convert
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out_path}")
    print("Phase 3 complete.")


if __name__ == "__main__":
    main()
