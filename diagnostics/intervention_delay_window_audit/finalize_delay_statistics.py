"""Finalize matched-seed intervention-delay statistics without rerunning rollouts.

The state list, delay set, and seed list are frozen in audit_plan.json.  This
script reads completed raw records only.  It never creates rollout tuples.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
Z95 = 1.959963984540054
BOOTSTRAP_REPLICATES = 50_000


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def wilson(successes: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return math.nan, math.nan
    p = successes / n
    z2 = Z95 * Z95
    denominator = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denominator
    radius = Z95 * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q, method="linear"))


def paired_bootstrap_ci(diff: np.ndarray, seed: int) -> tuple[float, float]:
    """Percentile paired bootstrap, simulated through empirical diff counts."""
    n = len(diff)
    values, counts = np.unique(diff.astype(np.int8), return_counts=True)
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(n, counts / n, size=BOOTSTRAP_REPLICATES)
    means = (draws @ values.astype(np.float64)) / n
    return percentile(means, 0.025), percentile(means, 0.975)


def state_bootstrap_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    batch = 1000
    for start in range(0, BOOTSTRAP_REPLICATES, batch):
        stop = min(BOOTSTRAP_REPLICATES, start + batch)
        indices = rng.integers(0, n, size=(stop - start, n))
        means[start:stop] = values[indices].mean(axis=1)
    return percentile(means, 0.025), percentile(means, 0.975)


def stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def effect_status(diff: np.ndarray, lo: float, hi: float) -> str:
    """No arbitrary effect threshold: nonzero inconclusive effects stay ambiguous."""
    if lo > 0:
        return "SUPPORTED_RECOVERABILITY_LOSS"
    if hi < 0:
        return "SUPPORTED_RECOVERABILITY_IMPROVEMENT"
    if np.all(diff == 0):
        return "EXACT_NO_OBSERVED_SUCCESS_EFFECT"
    return "DELAY_EFFECT_AMBIGUOUS"


def strict_tolerated(status: str) -> bool:
    return status in {
        "EXACT_NO_OBSERVED_SUCCESS_EFFECT",
        "SUPPORTED_RECOVERABILITY_IMPROVEMENT",
    }


def delay_category(statuses: dict[int, str]) -> str:
    if statuses[1] == "SUPPORTED_RECOVERABILITY_LOSS":
        return "IMMEDIATE_INTERVENTION_REQUIRED"
    if not strict_tolerated(statuses[1]):
        return "DELAY_EFFECT_AMBIGUOUS"
    if any(statuses[d] == "SUPPORTED_RECOVERABILITY_LOSS" for d in (2, 4)):
        return "SHORT_DELAY_TOLERATED"
    if all(strict_tolerated(statuses[d]) for d in (2, 4)):
        return "MULTI_STEP_DELAY_TOLERATED"
    return "DELAY_EFFECT_AMBIGUOUS"


def load_records(plan: dict) -> tuple[list[dict], list[dict], list[str]]:
    plan_hash = sha(HERE / "audit_plan.json")
    manifests, warnings = [], []
    unique: dict[tuple[str, int, int], dict] = {}
    for path in sorted((HERE / "raw").glob("*/manifest.json")):
        manifest = json.loads(path.read_text())
        if manifest.get("stage", "").startswith("smoke"):
            continue
        if manifest.get("audit_plan_sha256") != plan_hash:
            warnings.append(f"ignored plan-hash mismatch: {path}")
            continue
        records_path = path.with_name("records.jsonl")
        if not records_path.exists() or sha(records_path) != manifest.get("records_sha256"):
            raise AssertionError(f"record hash failure: {records_path}")
        manifests.append(manifest)
        for line in records_path.read_text().splitlines():
            row = json.loads(line)
            key = (row["state_id"], int(row["delay_steps"]), int(row["seed"]))
            if key in unique:
                prior = unique[key]
                comparable = ("outcome", "success", "steps", "terminal_step", "J_total")
                if any(prior.get(k) != row.get(k) for k in comparable):
                    raise AssertionError(f"conflicting duplicate tuple: {key}")
                continue
            unique[key] = row
    if not manifests:
        raise RuntimeError("no non-smoke completed raw manifests")
    planned_states = {state["state_id"] for state in plan["states"]}
    planned_delays = set(plan["delays"])
    if any(key[0] not in planned_states or key[1] not in planned_delays for key in unique):
        raise AssertionError("raw tuple outside frozen plan")
    if any(row.get("execution_error") is not None for row in unique.values()):
        raise AssertionError("execution error present in accepted raw tuple")
    return list(unique.values()), manifests, warnings


def main() -> None:
    plan_path = HERE / "audit_plan.json"
    plan = json.loads(plan_path.read_text())
    states = {row["state_id"]: row for row in plan["states"]}
    delays = [int(value) for value in plan["delays"]]
    if delays != [0, 1, 2, 4]:
        raise AssertionError(delays)
    records, raw_manifests, warnings = load_records(plan)
    by_tuple = {
        (row["state_id"], int(row["delay_steps"]), int(row["seed"])): row
        for row in records
    }

    per_state_delay, paired_rows, failure_rows = [], [], []
    state_effects: dict[str, dict[int, dict]] = defaultdict(dict)
    matched_seed_inventory = {}
    for state_index, (state_id, meta) in enumerate(states.items()):
        seed_sets = {
            delay: {seed for sid, d, seed in by_tuple if sid == state_id and d == delay}
            for delay in delays
        }
        common = sorted(set.intersection(*(seed_sets[d] for d in delays)))
        if len(common) < 64:
            raise AssertionError(f"{state_id}: only {len(common)} four-way matched seeds")
        matched_seed_inventory[state_id] = {
            "n_common": len(common),
            "minimum_seed": min(common),
            "maximum_seed": max(common),
            "per_delay_available": {str(d): len(seed_sets[d]) for d in delays},
        }
        arrays = {}
        for delay in delays:
            subset = [by_tuple[(state_id, delay, seed)] for seed in common]
            success = np.asarray([bool(row["success"]) for row in subset], dtype=np.int8)
            steps = np.asarray([int(row["steps"]) for row in subset], dtype=np.float64)
            deformation = np.asarray([float(row["J_total"]) for row in subset], dtype=np.float64)
            after = np.asarray([float(row.get("J_after_switch", row["J_total"])) for row in subset], dtype=np.float64)
            outcomes = Counter(row["outcome"] for row in subset)
            successes = int(success.sum())
            q_lo, q_hi = wilson(successes, len(common))
            arrays[delay] = {"success": success, "steps": steps, "rows": subset}
            per_state_delay.append({
                "state_id": state_id,
                "category": meta["category"],
                "source_group": meta["source_group"],
                "y_long": int(meta["y_long"]),
                "is_generic_cohort": bool(meta["is_generic_cohort"]),
                "is_hard13_anchor": bool(meta["is_hard13_anchor"]),
                "delay_steps": delay,
                "n": len(common),
                "successes": successes,
                "Q_hat": successes / len(common),
                "Q_wilson95_lower": q_lo,
                "Q_wilson95_upper": q_hi,
                "deadlocks": outcomes.get("deadlock", 0),
                "deadlock_rate": outcomes.get("deadlock", 0) / len(common),
                "timeouts": outcomes.get("timeout", 0),
                "timeout_rate": outcomes.get("timeout", 0) / len(common),
                "collisions": outcomes.get("collision", 0),
                "collision_rate": outcomes.get("collision", 0) / len(common),
                "other_failures": len(common) - successes - outcomes.get("deadlock", 0) - outcomes.get("timeout", 0) - outcomes.get("collision", 0),
                "mean_episode_steps": float(steps.mean()),
                "median_episode_steps": float(np.median(steps)),
                "p95_episode_steps": percentile(steps, 0.95),
                "mean_episode_seconds": float(steps.mean() * plan["environment"]["dt"]),
                "mean_J_total": float(deformation.mean()),
                "mean_J_after_switch": float(after.mean()),
                "reused_rollouts": sum(bool(row.get("reused")) for row in subset),
            })
        base = arrays[0]["success"]
        for delay in (1, 2, 4):
            delayed = arrays[delay]["success"]
            diff = base - delayed
            delta = float(diff.mean())
            lo, hi = paired_bootstrap_ci(diff, stable_seed(f"{state_id}:{delay}"))
            status = effect_status(diff, lo, hi)
            row = {
                "state_id": state_id,
                "category": meta["category"],
                "source_group": meta["source_group"],
                "y_long": int(meta["y_long"]),
                "is_generic_cohort": bool(meta["is_generic_cohort"]),
                "is_hard13_anchor": bool(meta["is_hard13_anchor"]),
                "delay_steps": delay,
                "n_matched": len(common),
                "Q0": float(base.mean()),
                "Q_delay": float(delayed.mean()),
                "Delta_Q0_minus_Qd": delta,
                "paired_bootstrap95_lower": lo,
                "paired_bootstrap95_upper": hi,
                "Q0_success_Qd_fail": int(np.sum((base == 1) & (delayed == 0))),
                "Qd_success_Q0_fail": int(np.sum((base == 0) & (delayed == 1))),
                "paired_outcome_agreement": float(np.mean(base == delayed)),
                "effect_status": status,
            }
            paired_rows.append(row)
            state_effects[state_id][delay] = row

    # Aggregate failure and duration table.  Pooling is explicitly descriptive;
    # state-macro uncertainty is reported separately below.
    scopes = [("ALL", lambda m: True), ("GENERIC", lambda m: bool(m["is_generic_cohort"])),
              ("HARD13", lambda m: bool(m["is_hard13_anchor"]))]
    for category in sorted({m["category"] for m in states.values()}):
        scopes.append((f"CATEGORY:{category}", lambda m, c=category: m["category"] == c))
    for label in (0, 1):
        scopes.append((f"Y_LONG:{label}", lambda m, y=label: int(m["y_long"]) == y))
    for scope_name, keep in scopes:
        ids = [sid for sid, meta in states.items() if keep(meta)]
        for delay in delays:
            rows = [by_tuple[(sid, delay, seed)] for sid in ids
                    for seed in sorted(set.intersection(*(
                        {s for ss, d, s in by_tuple if ss == sid and d == dd} for dd in delays
                    )))]
            counts = Counter(row["outcome"] for row in rows)
            steps = np.asarray([int(row["steps"]) for row in rows], dtype=np.float64)
            for outcome_name in ("success", "deadlock", "timeout", "collision", "other_failure"):
                if outcome_name == "other_failure":
                    count = len(rows) - sum(counts.get(x, 0) for x in ("success", "deadlock", "timeout", "collision"))
                else:
                    count = counts.get(outcome_name, 0)
                lo, hi = wilson(count, len(rows))
                failure_rows.append({
                    "scope": scope_name,
                    "state_count": len(ids),
                    "delay_steps": delay,
                    "outcome": outcome_name,
                    "count": count,
                    "rollout_count": len(rows),
                    "rate": count / len(rows),
                    "wilson95_lower": lo,
                    "wilson95_upper": hi,
                    "mean_episode_steps_all_outcomes": float(steps.mean()),
                    "median_episode_steps_all_outcomes": float(np.median(steps)),
                    "p95_episode_steps_all_outcomes": percentile(steps, 0.95),
                })

    urgency_rows, hard_rows, comparison_rows = [], [], []
    categories = Counter()
    for state_id, meta in states.items():
        effects = state_effects[state_id]
        statuses = {d: effects[d]["effect_status"] for d in (1, 2, 4)}
        category = delay_category(statuses)
        categories[category] += 1
        no_supported_loss = [0] + [d for d in (1, 2, 4) if statuses[d] != "SUPPORTED_RECOVERABILITY_LOSS"]
        exact_or_improved = [0] + [d for d in (1, 2, 4) if strict_tolerated(statuses[d])]
        urgency = {
            "state_id": state_id,
            "category": meta["category"],
            "source_group": meta["source_group"],
            "y_long": int(meta["y_long"]),
            "is_generic_cohort": bool(meta["is_generic_cohort"]),
            "is_hard13_anchor": bool(meta["is_hard13_anchor"]),
            "delay_category": category,
            "d_safe_max_no_supported_degradation": max(no_supported_loss),
            "d_max_exact_no_observed_loss_or_supported_improvement": max(exact_or_improved),
            "d_safe_max_caveat": "CI overlap is not equivalence; ambiguous nonzero effects are not called harmless",
            "d1_effect_status": statuses[1],
            "d2_effect_status": statuses[2],
            "d4_effect_status": statuses[4],
        }
        urgency_rows.append(urgency)
        comparison = {
            **urgency,
            "current_step_necessity": (
                "NOW_INTERVENTION_NECESSARY" if statuses[1] == "SUPPORTED_RECOVERABILITY_LOSS"
                else "NOW_INTERVENTION_NOT_DEMONSTRATED" if strict_tolerated(statuses[1])
                else "CURRENT_STEP_EFFECT_AMBIGUOUS"
            ),
            "y_long1_but_d1_strictly_tolerated": int(meta["y_long"]) == 1 and strict_tolerated(statuses[1]),
            "y_long1_but_d2_strictly_tolerated": int(meta["y_long"]) == 1 and strict_tolerated(statuses[2]),
            "y_long1_but_d4_strictly_tolerated": int(meta["y_long"]) == 1 and strict_tolerated(statuses[4]),
            "y_long0_but_d1_supported_loss": int(meta["y_long"]) == 0 and statuses[1] == "SUPPORTED_RECOVERABILITY_LOSS",
        }
        comparison_rows.append(comparison)
        if bool(meta["is_hard13_anchor"]):
            delay_lookup = {(row["state_id"], row["delay_steps"]): row for row in per_state_delay}
            hard = {
                "state_id": state_id,
                "source_group": meta["source_group"],
                "y_long": int(meta["y_long"]),
            }
            for d in delays:
                row = delay_lookup[(state_id, d)]
                hard.update({f"Q{d}": row["Q_hat"], f"Q{d}_wilson95_lower": row["Q_wilson95_lower"], f"Q{d}_wilson95_upper": row["Q_wilson95_upper"]})
            for d in (1, 2, 4):
                row = effects[d]
                hard.update({f"Delta_{d}": row["Delta_Q0_minus_Qd"], f"Delta_{d}_ci_lower": row["paired_bootstrap95_lower"],
                             f"Delta_{d}_ci_upper": row["paired_bootstrap95_upper"], f"delay_{d}_status": row["effect_status"]})
            hard.update({"delay_category": category, "d_safe_max_no_supported_degradation": max(no_supported_loss)})
            hard_rows.append(hard)

    # Macro-state and pooled-rollout aggregate summaries.
    aggregate_rows = []
    for scope_name, keep in scopes:
        ids = [sid for sid, meta in states.items() if keep(meta)]
        q0_by_state = {
            row["state_id"]: float(row["Q_hat"])
            for row in per_state_delay
            if row["state_id"] in ids and int(row["delay_steps"]) == 0
        }
        for delay in delays:
            rows = [row for row in per_state_delay if row["state_id"] in ids and int(row["delay_steps"]) == delay]
            q_values = np.asarray([float(row["Q_hat"]) for row in rows])
            lo, hi = state_bootstrap_ci(q_values, stable_seed(f"aggregate:{scope_name}:{delay}"))
            delta_values = np.asarray([q0_by_state[row["state_id"]] - float(row["Q_hat"]) for row in rows])
            delta_lo, delta_hi = state_bootstrap_ci(
                delta_values, stable_seed(f"aggregate-delta:{scope_name}:{delay}")
            )
            total_successes = sum(int(row["successes"]) for row in rows)
            total_n = sum(int(row["n"]) for row in rows)
            pooled_lo, pooled_hi = wilson(total_successes, total_n)
            aggregate_rows.append({
                "scope": scope_name,
                "state_count": len(rows),
                "delay_steps": delay,
                "state_macro_mean_Q": float(q_values.mean()),
                "state_bootstrap95_lower": lo,
                "state_bootstrap95_upper": hi,
                "state_macro_mean_Delta_Q0_minus_Qd": float(delta_values.mean()),
                "state_delta_bootstrap95_lower": delta_lo,
                "state_delta_bootstrap95_upper": delta_hi,
                "pooled_successes": total_successes,
                "pooled_rollouts": total_n,
                "pooled_Q_descriptive": total_successes / total_n,
                "pooled_wilson95_lower_descriptive": pooled_lo,
                "pooled_wilson95_upper_descriptive": pooled_hi,
            })

    write_csv(HERE / "per_state_delay_results.csv", per_state_delay)
    write_csv(HERE / "paired_success_differences.csv", paired_rows)
    write_csv(HERE / "failure_type_by_delay.csv", failure_rows)
    write_csv(HERE / "hard13_delay_results.csv", hard_rows)
    write_csv(HERE / "old_vs_step_label_comparison.csv", comparison_rows)
    write_csv(HERE / "urgency_summary.csv", urgency_rows)
    write_csv(HERE / "aggregate_delay_summary.csv", aggregate_rows)

    generic = [row for row in comparison_rows if row["is_generic_cohort"]]
    y1 = [row for row in generic if int(row["y_long"]) == 1]
    hard = [row for row in comparison_rows if row["is_hard13_anchor"]]
    summary = {
        "audit_plan_sha256": sha(plan_path),
        "state_count": len(states),
        "generic_state_count": len(generic),
        "hard13_count": len(hard),
        "initial_seed_count": len(plan["seeds_initial"]),
        "matched_seed_inventory": matched_seed_inventory,
        "category_counts_all59": dict(categories),
        "category_counts_generic50": dict(Counter(row["delay_category"] for row in generic)),
        "y_long_counts_all59": dict(Counter(str(row["y_long"]) for row in comparison_rows)),
        "generic_y_long1_count": len(y1),
        "generic_y_long1_immediate_required": sum(row["current_step_necessity"] == "NOW_INTERVENTION_NECESSARY" for row in y1),
        "generic_y_long1_d1_strictly_tolerated": sum(row["y_long1_but_d1_strictly_tolerated"] for row in y1),
        "generic_y_long1_d2_strictly_tolerated": sum(row["y_long1_but_d2_strictly_tolerated"] for row in y1),
        "generic_y_long1_d4_strictly_tolerated": sum(row["y_long1_but_d4_strictly_tolerated"] for row in y1),
        "hard13_category_counts": dict(Counter(row["delay_category"] for row in hard)),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "raw_manifest_count": len(raw_manifests),
        "warnings": warnings,
        "classification_note": "Final scientific classification is intentionally made after inspecting these frozen summaries, not by an arbitrary Delta_Q threshold.",
    }
    (HERE / "statistical_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "matched_seed_inventory"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
