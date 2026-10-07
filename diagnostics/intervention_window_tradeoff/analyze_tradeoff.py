"""Read-only paired statistics for the frozen intervention-window audit.

This script never launches rollouts.  It accepts only completed raw shards
whose manifests bind to audit_plan.json, deduplicates exact logical tuples,
and writes the requested Q/J_def/transition summaries.
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
BOOTSTRAPS = 20_000


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


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
    den = 1 + z2 / n
    center = (p + z2 / (2 * n)) / den
    radius = Z95 * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / den
    return max(0.0, center - radius), min(1.0, center + radius)


def paired_bootstrap(values: np.ndarray, seed: int) -> tuple[float, float]:
    """Deterministic paired percentile bootstrap of the sample mean."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return math.nan, math.nan
    if np.all(values == values[0]):
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    means = np.empty(BOOTSTRAPS, dtype=np.float64)
    batch = 500
    for start in range(0, BOOTSTRAPS, batch):
        stop = min(BOOTSTRAPS, start + batch)
        idx = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[idx].mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))


def state_bootstrap(values: np.ndarray, seed: int) -> tuple[float, float]:
    return paired_bootstrap(values, seed)


def success_status(diff: np.ndarray, lo: float, hi: float) -> str:
    # No arbitrary practical-effect threshold is introduced.
    if lo > 0:
        return "SUPPORTED_RECOVERABILITY_DEGRADATION"
    if hi < 0:
        return "SUPPORTED_RECOVERABILITY_IMPROVEMENT"
    if np.all(diff == 0):
        return "EXACT_NO_OBSERVED_SUCCESS_EFFECT"
    return "UNRESOLVED_NONZERO_CI_CROSSES_ZERO"


def deformation_status(diff: np.ndarray, lo: float, hi: float) -> str:
    if lo > 0:
        return "SUPPORTED_J_DEF_INCREASE"
    if hi < 0:
        return "SUPPORTED_J_DEF_DECREASE"
    if np.allclose(diff, 0.0, rtol=0.0, atol=1e-14):
        return "EXACT_NO_OBSERVED_J_DEF_EFFECT"
    return "J_DEF_EFFECT_UNRESOLVED"


def strict_success_compatible(status: str) -> bool:
    return status in {
        "EXACT_NO_OBSERVED_SUCCESS_EFFECT",
        "SUPPORTED_RECOVERABILITY_IMPROVEMENT",
    }


def load_records(plan_hash: str) -> tuple[dict[tuple[str, int, int], dict], list[dict], list[str]]:
    unique: dict[tuple[str, int, int], dict] = {}
    manifests, warnings = [], []
    for manifest_path in sorted((HERE / "raw").glob("**/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        bound = manifest.get("audit_plan_sha256", manifest.get("plan_sha256"))
        if bound != plan_hash:
            warnings.append(f"ignored plan-hash mismatch: {manifest_path}")
            continue
        records_path = manifest_path.with_name("records.jsonl")
        if not records_path.exists():
            raise AssertionError(f"missing records for completed manifest: {manifest_path}")
        expected = manifest.get("records_sha256")
        if expected is not None and sha(records_path) != expected:
            raise AssertionError(f"record hash mismatch: {records_path}")
        manifests.append(manifest)
        for line in records_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["state_id"], int(row["delay_steps"]), int(row["seed"]))
            if row.get("execution_error") is not None:
                raise AssertionError(f"execution error in accepted tuple: {key}")
            if key in unique:
                old = unique[key]
                compare = ("outcome", "success", "steps", "terminal_step", "J_total")
                if any(old.get(name) != row.get(name) for name in compare):
                    raise AssertionError(f"conflicting duplicate tuple: {key}")
                continue
            unique[key] = row
    if not manifests:
        raise RuntimeError("no completed plan-bound raw manifests")
    return unique, manifests, warnings


def classify_window(delays: list[int], effects: dict[int, dict]) -> tuple[str, int, int | None, str]:
    positives = sorted(d for d in delays if d > 0)
    bad = [d for d in positives if effects[d]["success_effect_status"] == "SUPPORTED_RECOVERABILITY_DEGRADATION"]
    first_bad = min(bad) if bad else None
    upper = first_bad if first_bad is not None else max(positives)
    safe = [0] + [d for d in positives if d < upper and strict_success_compatible(effects[d]["success_effect_status"])]
    if first_bad is None and strict_success_compatible(effects[max(positives)]["success_effect_status"]):
        safe.append(max(positives))
    last_safe = max(safe)
    unresolved = [d for d in positives if effects[d]["success_effect_status"].startswith("UNRESOLVED")]

    # Categories are tied transparently to the frozen physical-delay grid.
    # They add no effect-size threshold.
    if first_bad == positives[0] and last_safe == 0:
        category = "IMMEDIATE"
    elif first_bad is not None and first_bad <= 16:
        category = "SHORT_WINDOW"
    elif first_bad is not None:
        category = "MEDIUM_WINDOW"
    elif not unresolved and last_safe >= 64:
        category = "LONG_WINDOW"
    else:
        category = "UNRESOLVED"
    resolution = (
        "BOUNDED_TRANSITION" if first_bad is not None
        else "RIGHT_CENSORED_LONG_TOLERANCE" if category == "LONG_WINDOW"
        else "UNRESOLVED_WITHIN_TESTED_RANGE"
    )
    return category, last_safe, first_bad, resolution


def main() -> None:
    plan_path = HERE / "audit_plan.json"
    plan = json.loads(plan_path.read_text())
    plan_hash = sha(plan_path)
    states = {row["state_id"]: row for row in plan["states"]}
    records, manifests, warnings = load_records(plan_hash)
    if set(sid for sid, _, _ in records) - set(states):
        raise AssertionError("raw record references state outside frozen plan")
    delays = sorted({delay for _, delay, _ in records})
    if 0 not in delays:
        raise AssertionError("d=0 baseline is absent")

    per_delay: list[dict] = []
    success_rows: list[dict] = []
    deform_rows: list[dict] = []
    effect_lookup: dict[str, dict[int, dict]] = defaultdict(dict)
    delay_lookup: dict[tuple[str, int], dict] = {}

    for sid, meta in states.items():
        available = sorted({d for ss, d, _ in records if ss == sid})
        if 0 not in available:
            raise AssertionError(f"{sid}: missing d=0")
        base_seeds = {seed for ss, d, seed in records if ss == sid and d == 0}
        for delay in available:
            seeds = sorted(base_seeds & {seed for ss, d, seed in records if ss == sid and d == delay})
            if not seeds:
                raise AssertionError(f"{sid} d={delay}: no matched d0 seeds")
            rows = [records[(sid, delay, seed)] for seed in seeds]
            success = np.asarray([bool(row["success"]) for row in rows], dtype=np.int8)
            jdef = np.asarray([float(row["J_total"]) for row in rows], dtype=np.float64)
            steps = np.asarray([int(row["steps"]) for row in rows], dtype=np.float64)
            outcomes = Counter(row["outcome"] for row in rows)
            qlo, qhi = wilson(int(success.sum()), len(seeds))
            item = {
                "state_id": sid, "category": meta["category"], "source_group": meta["source_group"],
                "y_long": int(meta["y_long"]), "is_hard13_anchor": bool(meta["is_hard13_anchor"]),
                "delay_steps": delay, "delay_seconds": delay * float(plan["environment"]["dt"]),
                "n_matched_to_d0": len(seeds), "successes": int(success.sum()),
                "Q_hat": float(success.mean()), "Q_wilson95_lower": qlo, "Q_wilson95_upper": qhi,
                "mean_J_def": float(jdef.mean()), "std_J_def": float(jdef.std(ddof=1)) if len(jdef) > 1 else 0.0,
                "mean_episode_steps": float(steps.mean()), "std_episode_steps": float(steps.std(ddof=1)) if len(steps) > 1 else 0.0,
                "deadlocks": outcomes.get("deadlock", 0), "timeouts": outcomes.get("timeout", 0),
                "collisions": outcomes.get("collision", 0),
                "other_failures": len(rows) - int(success.sum()) - outcomes.get("deadlock", 0) - outcomes.get("timeout", 0) - outcomes.get("collision", 0),
                "reused_or_materialized": sum(bool(row.get("reused")) for row in rows),
            }
            per_delay.append(item)
            delay_lookup[(sid, delay)] = item

            if delay == 0:
                effect_lookup[sid][0] = {
                    "success_effect_status": "REFERENCE",
                    "J_def_effect_status": "REFERENCE",
                    "Delta_Q0_minus_Qd": 0.0,
                    "Delta_Jd_minus_J0": 0.0,
                }
                continue
            base_rows = [records[(sid, 0, seed)] for seed in seeds]
            base_success = np.asarray([bool(row["success"]) for row in base_rows], dtype=np.int8)
            base_j = np.asarray([float(row["J_total"]) for row in base_rows], dtype=np.float64)
            qdiff = base_success - success
            jdiff = jdef - base_j
            qlo_d, qhi_d = paired_bootstrap(qdiff, stable_seed(f"Q:{sid}:{delay}"))
            jlo_d, jhi_d = paired_bootstrap(jdiff, stable_seed(f"J:{sid}:{delay}"))
            qstatus = success_status(qdiff, qlo_d, qhi_d)
            jstatus = deformation_status(jdiff, jlo_d, jhi_d)
            success_item = {
                "state_id": sid, "category": meta["category"], "source_group": meta["source_group"],
                "y_long": int(meta["y_long"]), "is_hard13_anchor": bool(meta["is_hard13_anchor"]),
                "delay_steps": delay, "n_matched": len(seeds),
                "Q0": float(base_success.mean()), "Q_delay": float(success.mean()),
                "Delta_Q0_minus_Qd": float(qdiff.mean()),
                "paired_bootstrap95_lower": qlo_d, "paired_bootstrap95_upper": qhi_d,
                "Q0_success_Qd_fail": int(np.sum((base_success == 1) & (success == 0))),
                "Qd_success_Q0_fail": int(np.sum((base_success == 0) & (success == 1))),
                "success_effect_status": qstatus,
            }
            deform_item = {
                "state_id": sid, "category": meta["category"], "source_group": meta["source_group"],
                "y_long": int(meta["y_long"]), "is_hard13_anchor": bool(meta["is_hard13_anchor"]),
                "delay_steps": delay, "n_matched": len(seeds),
                "J0_mean": float(base_j.mean()), "J_delay_mean": float(jdef.mean()),
                "Delta_Jd_minus_J0": float(jdiff.mean()),
                "paired_bootstrap95_lower": jlo_d, "paired_bootstrap95_upper": jhi_d,
                "J_def_effect_status": jstatus,
            }
            success_rows.append(success_item)
            deform_rows.append(deform_item)
            effect_lookup[sid][delay] = {**success_item, **deform_item}

    transition_rows, ylong_rows = [], []
    for sid, meta in states.items():
        tested = sorted(effect_lookup[sid])
        category, last_safe, first_bad, resolution = classify_window(tested, effect_lookup[sid])
        unresolved = [d for d in tested if d > 0 and effect_lookup[sid][d]["success_effect_status"].startswith("UNRESOLVED")]
        row = {
            "state_id": sid, "category": meta["category"], "source_group": meta["source_group"],
            "y_long": int(meta["y_long"]), "is_hard13_anchor": bool(meta["is_hard13_anchor"]),
            "window_category": category, "transition_resolution": resolution,
            "d_last_safe": last_safe, "d_first_bad": "" if first_bad is None else first_bad,
            "d_last_safe_seconds": last_safe * float(plan["environment"]["dt"]),
            "d_first_bad_seconds": "" if first_bad is None else first_bad * float(plan["environment"]["dt"]),
            "maximum_tested_delay": max(tested),
            "unresolved_delay_points": ";".join(map(str, unresolved)),
            "interval_caveat": "CI overlap is not equivalence; nonzero effects whose CI crosses zero remain unresolved",
        }
        transition_rows.append(row)
        ylong_rows.append({
            **row,
            "y_long_interpretation": "eventual_intervention_needed" if int(meta["y_long"]) else "eta0_forever_feasible",
            "immediate_need_supported": first_bad == min(d for d in tested if d > 0),
            "old_label_matches_immediate_need": int(meta["y_long"]) == int(first_bad == min(d for d in tested if d > 0)),
        })

    # Exploratory point-estimate Pareto sets, restricted to strictly
    # recoverability-compatible delays; unresolved success effects are excluded.
    pareto_rows = []
    for sid in states:
        candidates = []
        for delay in sorted(effect_lookup[sid]):
            effect = effect_lookup[sid][delay]
            if delay == 0 or strict_success_compatible(effect["success_effect_status"]):
                candidates.append(delay)
        for delay in candidates:
            point = delay_lookup[(sid, delay)]
            dominated = any(
                (delay_lookup[(sid, other)]["Q_hat"] >= point["Q_hat"] and
                 delay_lookup[(sid, other)]["mean_J_def"] <= point["mean_J_def"] and
                 (delay_lookup[(sid, other)]["Q_hat"] > point["Q_hat"] or
                  delay_lookup[(sid, other)]["mean_J_def"] < point["mean_J_def"]))
                for other in candidates if other != delay
            )
            pareto_rows.append({
                "state_id": sid, "delay_steps": delay, "Q_hat": point["Q_hat"],
                "mean_J_def": point["mean_J_def"], "strict_recoverability_compatible": True,
                "point_estimate_pareto_nondominated": not dominated,
                "caveat": "exploratory point-estimate frontier; no Q/J scalarization",
            })

    # Aggregate curves.  Adaptive delays are explicitly conditional on the
    # audited subset at that delay, never presented as full-pool estimates.
    recover_rows, aggregate_deform = [], []
    for delay in sorted({row["delay_steps"] for row in per_delay}):
        rows = [row for row in per_delay if row["delay_steps"] == delay]
        q = np.asarray([row["Q_hat"] for row in rows])
        qlo, qhi = state_bootstrap(q, stable_seed(f"macroQ:{delay}"))
        dqs = np.asarray([
            delay_lookup[(row["state_id"], 0)]["Q_hat"] - row["Q_hat"] for row in rows
        ])
        dlo, dhi = state_bootstrap(dqs, stable_seed(f"macroDQ:{delay}"))
        recover_rows.append({
            "scope": "FULL_FROZEN_POOL" if len(rows) == len(states) else "ADAPTIVELY_SELECTED_SUBSET",
            "delay_steps": delay, "delay_seconds": delay * float(plan["environment"]["dt"]),
            "state_count": len(rows), "state_macro_Q": float(q.mean()),
            "state_bootstrap95_lower": qlo, "state_bootstrap95_upper": qhi,
            "state_macro_Delta_Q0_minus_Qd": float(dqs.mean()),
            "Delta_state_bootstrap95_lower": dlo, "Delta_state_bootstrap95_upper": dhi,
        })
        j = np.asarray([row["mean_J_def"] for row in rows])
        dj = np.asarray([
            row["mean_J_def"] - delay_lookup[(row["state_id"], 0)]["mean_J_def"] for row in rows
        ])
        jlo, jhi = state_bootstrap(j, stable_seed(f"macroJ:{delay}"))
        djlo, djhi = state_bootstrap(dj, stable_seed(f"macroDJ:{delay}"))
        aggregate_deform.append({
            "scope": "FULL_FROZEN_POOL" if len(rows) == len(states) else "ADAPTIVELY_SELECTED_SUBSET",
            "delay_steps": delay, "delay_seconds": delay * float(plan["environment"]["dt"]),
            "state_count": len(rows), "state_macro_mean_J_def": float(j.mean()),
            "state_bootstrap95_lower": jlo, "state_bootstrap95_upper": jhi,
            "state_macro_Delta_Jd_minus_J0": float(dj.mean()),
            "Delta_state_bootstrap95_lower": djlo, "Delta_state_bootstrap95_upper": djhi,
        })

    failure_rows = []
    for delay in sorted({row["delay_steps"] for row in per_delay}):
        rows = [row for row in per_delay if row["delay_steps"] == delay]
        total = sum(row["n_matched_to_d0"] for row in rows)
        for name in ("deadlocks", "timeouts", "collisions", "other_failures"):
            count = sum(row[name] for row in rows)
            lo, hi = wilson(count, total)
            failure_rows.append({
                "delay_steps": delay, "state_count": len(rows), "outcome": name,
                "count": count, "rollouts": total, "rate": count / total,
                "wilson95_lower": lo, "wilson95_upper": hi,
            })

    write_csv(HERE / "per_state_delay_curves.csv", per_delay)
    write_csv(HERE / "paired_success_differences.csv", success_rows)
    write_csv(HERE / "recoverability_curves.csv", recover_rows)
    write_csv(HERE / "deformation_curves.csv", deform_rows + aggregate_deform)
    write_csv(HERE / "transition_intervals.csv", transition_rows)
    write_csv(HERE / "pareto_frontier.csv", pareto_rows)
    write_csv(HERE / "ylong_vs_window.csv", ylong_rows)
    write_csv(HERE / "failure_type_by_delay.csv", failure_rows)

    summary = {
        "audit_plan_sha256": plan_hash,
        "frozen_state_count": len(states),
        "completed_manifest_count": len(manifests),
        "accepted_tuple_count": len(records),
        "tested_delays": delays,
        "window_category_counts": dict(Counter(row["window_category"] for row in transition_rows)),
        "y_long_counts": dict(Counter(str(row["y_long"]) for row in transition_rows)),
        "bootstrap_replicates": BOOTSTRAPS,
        "warnings": warnings,
    }
    (HERE / "statistical_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
