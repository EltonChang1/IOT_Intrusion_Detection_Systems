#!/usr/bin/env python3
"""
Phase 4 — aggregate Phase 3 timings/metrics and Phase 4 artifacts into one JSON
for deployment / report discussion (latency vs accuracy, experiment coverage).
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


def _load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 summary bundle")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "phase4_summary.json",
    )
    args = parser.parse_args()

    rd = args.results_dir
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase3_metrics": _load_json(rd / "phase3_metrics.json"),
        "phase4_multiclass": _load_json(rd / "phase4_multiclass.json"),
        "phase4_per_device": _load_json(rd / "phase4_per_device.json"),
        "feature_importance_csv": str(rd / "phase4_feature_importance_rf.csv")
        if (rd / "phase4_feature_importance_rf.csv").is_file()
        else None,
        "notes": {
            "deployment": (
                "Compare infer_test_seconds_per_sample from phase3_metrics with accuracy/F1; "
                "RF is typically slower per sample than linear/MLP; HGB is often a good tradeoff."
            ),
            "multiclass": "phase4_multiclass uses v2 Attack or Attack_subType labels.",
            "per_device": "phase4_per_device evaluates one global HGB model by Device_Name on held-out test rows.",
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
