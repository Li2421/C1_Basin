"""Combine initial matched branches and plan only statistically ambiguous extensions."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import norm


HERE = Path(__file__).resolve().parent


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    z = float(norm.ppf(0.975)); p = k / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return float(center - half), float(center + half)


def paired_bootstrap_ci(diff: np.ndarray, seed: int) -> tuple[float, float]:
    if len(diff) == 0:
        return float("nan"), float("nan")
    if np.all(diff == diff[0]):
        return float(diff[0]), float(diff[0])
    rng = np.random.default_rng(seed)
    # Chunked to keep memory small and deterministic.
    values = []
    for _ in range(20):
        indices = rng.integers(0, len(diff), size=(1000, len(diff)))
        values.append(diff[indices].mean(axis=1))
    boot = np.concatenate(values)
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def classify(delta: float, lo: float, hi: float, diffs: np.ndarray) -> str:
    if lo > 0:
        return "NOW_INTERVENTION_NECESSARY"
    # No arbitrary equivalence margin: this label is used only when all
    # observed matched success indicators show exactly zero effect, or when
    # intervention is significantly harmful rather than beneficial.
    if np.all(diffs == 0) or hi < 0:
        return "NOW_INTERVENTION_NOT_NECESSARY"
    return "ONE_STEP_EFFECT_AMBIGUOUS"


def main() -> None:
    plan = json.loads((HERE / "audit_plan.json").read_text())
    state_meta = {row["state_id"]: row for row in plan["states"]}
    manifests = []
    records = []
    for shard in range(3):
        directory = HERE / "raw" / f"initial64_shard{shard}"
        manifests.append(json.loads((directory / "manifest.json").read_text()))
        records.extend(json.loads(line) for line in (directory / "records.jsonl").read_text().splitlines())
    expected = len(state_meta) * 2 * 64
    keys = {(row["state_id"], row["branch"], int(row["seed"])) for row in records}
    if len(records) != expected or len(keys) != expected:
        raise AssertionError((len(records), len(keys), expected))
    if any(row.get("execution_error") is not None for row in records):
        raise AssertionError("execution errors present")

    comparison = []
    for state_index, (state_id, meta) in enumerate(state_meta.items()):
        by_branch = {branch: {int(row["seed"]): row for row in records if row["state_id"] == state_id and row["branch"] == branch} for branch in ("N", "I")}
        seeds = sorted(set(by_branch["N"]) & set(by_branch["I"]))
        if len(seeds) != 64:
            raise AssertionError((state_id, len(seeds)))
        n_success = np.asarray([by_branch["N"][seed]["success"] for seed in seeds], dtype=np.int8)
        i_success = np.asarray([by_branch["I"][seed]["success"] for seed in seeds], dtype=np.int8)
        diff = i_success - n_success
        qn = float(n_success.mean()); qi = float(i_success.mean()); delta = float(diff.mean())
        lo, hi = paired_bootstrap_ci(diff, 41000 + state_index)
        qn_lo, qn_hi = wilson(int(n_success.sum()), len(seeds)); qi_lo, qi_hi = wilson(int(i_success.sum()), len(seeds))
        future_n = np.asarray([by_branch["N"][seed]["J_future"] for seed in seeds])
        future_i = np.asarray([by_branch["I"][seed]["J_future"] for seed in seeds])
        classification = classify(delta, lo, hi, diff)
        comparison.append({
            "state_id": state_id, "set_role": meta["set_role"], "category": meta["category"],
            "source_group": meta["source_group"], "y_long": meta["y_long"], "eta_best": json.dumps(meta["eta_best"]),
            "n": len(seeds), "N_successes": int(n_success.sum()), "I_successes": int(i_success.sum()),
            "Q_N": qn, "Q_N_wilson95_lower": qn_lo, "Q_N_wilson95_upper": qn_hi,
            "Q_I": qi, "Q_I_wilson95_lower": qi_lo, "Q_I_wilson95_upper": qi_hi,
            "Delta_Q_I_minus_N": delta, "paired_bootstrap95_lower": lo, "paired_bootstrap95_upper": hi,
            "I_success_N_fail": int(np.sum((i_success == 1) & (n_success == 0))),
            "N_success_I_fail": int(np.sum((n_success == 1) & (i_success == 0))),
            "matched_success_indicator_agreement": float(np.mean(i_success == n_success)),
            "mean_J_future_N": float(future_n.mean()), "mean_J_future_I": float(future_i.mean()),
            "mean_future_deformation_N_minus_I": float(np.mean(future_n - future_i)),
            "one_step_necessity": classification,
            "N_outcomes": json.dumps(dict(Counter(by_branch["N"][seed]["outcome"] for seed in seeds)), sort_keys=True),
            "I_outcomes": json.dumps(dict(Counter(by_branch["I"][seed]["outcome"] for seed in seeds)), sort_keys=True),
        })

    fieldnames = list(comparison[0])
    with (HERE / "initial_qn_qi_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames); writer.writeheader(); writer.writerows(comparison)
    ambiguous = [row["state_id"] for row in comparison if row["one_step_necessity"] == "ONE_STEP_EFFECT_AMBIGUOUS"]
    # Extension is frozen solely from initial paired-CI ambiguity, not by desired conclusion.
    extension = {
        "selection_rule": "all and only initial64 ONE_STEP_EFFECT_AMBIGUOUS states",
        "state_ids": ambiguous,
        "next_total_n": 128 if ambiguous else 64,
        "new_seeds": list(range(95910101, 95910165)) if ambiguous else [],
    }
    (HERE / "adaptive_extension_plan.json").write_text(json.dumps(extension, indent=2) + "\n")
    summary = {
        "states": len(comparison), "records": len(records),
        "new_rollouts": sum(item["new_rollouts"] for item in manifests),
        "reused_rollouts": sum(item["reused_rollouts"] for item in manifests),
        "classifications": dict(Counter(row["one_step_necessity"] for row in comparison)),
        "difficult_classifications": dict(Counter(row["one_step_necessity"] for row in comparison if row["set_role"] == "DIFFICULT_STABLE")),
        "y_long1_difficult_not_necessary": sum(row["set_role"] == "DIFFICULT_STABLE" and int(row["y_long"]) == 1 and row["one_step_necessity"] == "NOW_INTERVENTION_NOT_NECESSARY" for row in comparison),
        "ambiguous_extension_states": ambiguous,
    }
    (HERE / "initial_analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
