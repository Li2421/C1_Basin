"""Analyze independent replication or gradient-validation outcomes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scipy.stats import fisher_exact

from diagnostics.rrisk_true_chunk.analyze_true_chunk import (
    atomic_json, clopper_pearson, holm_adjust,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("replication", "gradient_validation"),
                        required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    complete = args.records.parent / "complete.json"
    if not complete.exists() or not json.loads(complete.read_text())["complete"]:
        raise RuntimeError("validation input is incomplete")
    rows = json.loads(args.records.read_text())
    for row in rows:
        row["Q_D_ci95"] = clopper_pearson(row["deadlocks"], row["continuations"])
    baseline = next(row for row in rows if row["arm_id"] == "baseline")
    tests = []
    for row in rows:
        if row is baseline:
            continue
        table = [[baseline["deadlocks"], baseline["continuations"]-baseline["deadlocks"]],
                 [row["deadlocks"], row["continuations"]-row["deadlocks"]]]
        tests.append(dict(
            arm=row["arm_id"], baseline_Q=baseline["Q_D"], arm_Q=row["Q_D"],
            delta_Q=row["Q_D"]-baseline["Q_D"],
            p_raw=float(fisher_exact(table).pvalue)))
    holm_adjust(tests)
    for test in tests:
        test["significant"] = bool(test["p_holm"] <= .05 and
                                   abs(test["delta_Q"]) >= .25)
    lower = [test for test in tests
             if test["delta_Q"] <= -.25 and test["p_holm"] <= .05]
    result = dict(
        schema="rrisk_true_chunk_validation_analysis_v1", stage=args.stage,
        arms=[{key: value for key, value in row.items() if key != "outcomes"}
              for row in rows],
        baseline_arm="baseline", comparisons=tests,
        true_descent_replicated=bool(lower),
        best_validated_arm=(min(lower, key=lambda row: row["arm_Q"])["arm"]
                            if lower else None),
    )
    if args.stage == "gradient_validation":
        by_id = {row["arm_id"]: row for row in rows}
        minus = by_id["minus_g_TRUE"]["Q_D"]
        plus = by_id["plus_g_TRUE"]["Q_D"]
        base = baseline["Q_D"]
        result["desired_order_observed"] = bool(minus < base < plus)
        result["descent_below_baseline"] = bool(
            any(test["arm"] == "minus_g_TRUE" and test["delta_Q"] < 0 and
                test["significant"] for test in tests))
        result["directional_order"] = dict(
            minus_g_TRUE=minus, baseline=base, plus_g_TRUE=plus)
    atomic_json(args.out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
