"""Stage-1 paired Q/J_def statistics and result-independent adaptive plans.

Commands
--------
coarse
    Require the complete frozen {0,4,8,16,32,64} x 64 coarse grid, write
    paired statistics, and freeze adaptive_selection.json.
final
    Require every tuple selected by adaptive_selection.json, write final
    delay/transition summaries, then freeze at most three candidate horizons
    in candidate_windows.json.  No classifier result is read by this script.

This module never launches a rollout or trains a model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
Z95 = 1.959963984540054
BOOTSTRAPS = 20_000


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"empty table: {path}")
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def wilson(k: int, n: int) -> tuple[float, float]:
    p = k / n
    z2 = Z95 * Z95
    den = 1 + z2 / n
    center = (p + z2 / (2 * n)) / den
    radius = Z95 * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / den
    return max(0.0, center - radius), min(1.0, center + radius)


def bootstrap_mean(values: np.ndarray, seed: int) -> tuple[float, float]:
    values = np.asarray(values, np.float64)
    if np.all(values == values[0]):
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    means = np.empty(BOOTSTRAPS)
    for start in range(0, BOOTSTRAPS, 500):
        stop = min(BOOTSTRAPS, start + 500)
        idx = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[idx].mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))


def q_status(diff: np.ndarray, lo: float, hi: float) -> str:
    if lo > 0:
        return "SUPPORTED_RECOVERABILITY_LOSS"
    if hi < 0:
        return "SUPPORTED_RECOVERABILITY_IMPROVEMENT"
    if np.all(diff == 0):
        return "RESOLVED_NO_LOSS_EXACT"
    return "H_AMBIGUOUS"


def j_status(diff: np.ndarray, lo: float, hi: float) -> str:
    if lo > 0:
        return "SUPPORTED_J_DEF_INCREASE"
    if hi < 0:
        return "SUPPORTED_J_DEF_DECREASE"
    if np.allclose(diff, 0.0, rtol=0.0, atol=1e-14):
        return "RESOLVED_NO_J_DEF_CHANGE_EXACT"
    return "J_DEF_AMBIGUOUS"


def resolved_zero(status: str) -> bool:
    return status in {"RESOLVED_NO_LOSS_EXACT", "SUPPORTED_RECOVERABILITY_IMPROVEMENT"}


def completed_manifests(plan_hash: str) -> tuple[dict[tuple[str, int, int], dict], dict[str, str]]:
    records: dict[tuple[str, int, int], dict] = {}
    hashes = {}
    raw = HERE / "raw"
    for manifest_path in sorted(raw.glob("**/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("audit_plan_sha256", manifest.get("plan_sha256")) != plan_hash:
            continue
        records_path = manifest_path.with_name("records.jsonl")
        if not records_path.exists():
            raise AssertionError(f"completed shard missing records: {manifest_path}")
        if manifest.get("records_sha256") and manifest["records_sha256"] != sha(records_path):
            raise AssertionError(f"records hash mismatch: {records_path}")
        rel = str(manifest_path.relative_to(HERE))
        hashes[rel] = sha(manifest_path)
        for line in records_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["state_id"], int(row["delay_steps"]), int(row["seed"]))
            if row.get("execution_error") is not None:
                raise AssertionError(f"execution error in accepted tuple {key}")
            if key in records:
                old = records[key]
                fields = ("outcome", "success", "steps", "terminal_step", "J_total")
                if any(old.get(field) != row.get(field) for field in fields):
                    raise AssertionError(f"conflicting duplicate tuple {key}")
            else:
                records[key] = row
    if not hashes:
        raise RuntimeError("no plan-bound completed rollout shards")
    return records, hashes


def require_grid(plan: dict, records: dict, delays: list[int], seed_map: dict[str, list[int]]) -> None:
    missing = []
    for state in plan["states"]:
        sid = state["state_id"]
        for delay in delays:
            for seed in seed_map[sid]:
                if (sid, delay, int(seed)) not in records:
                    missing.append((sid, delay, seed))
                    if len(missing) >= 20:
                        break
    if missing:
        raise RuntimeError(f"incomplete required grid; first missing tuples: {missing}")


def compute(plan: dict, records: dict, allowed_delays: set[int] | None = None):
    states = {row["state_id"]: row for row in plan["states"]}
    curves, qrows, jrows = [], [], []
    effects: dict[str, dict[int, dict]] = defaultdict(dict)
    for sid, meta in states.items():
        delays = sorted({d for ss, d, _ in records if ss == sid and (allowed_delays is None or d in allowed_delays)})
        if 0 not in delays:
            raise AssertionError(f"{sid}: no d0")
        base_seeds = {seed for ss, d, seed in records if ss == sid and d == 0}
        for delay in delays:
            seeds = sorted(base_seeds & {seed for ss, d, seed in records if ss == sid and d == delay})
            rows = [records[(sid, delay, seed)] for seed in seeds]
            success = np.asarray([bool(row["success"]) for row in rows], np.int8)
            jdef = np.asarray([float(row["J_total"]) for row in rows])
            outcomes = Counter(row["outcome"] for row in rows)
            lo, hi = wilson(int(success.sum()), len(success))
            curve = {
                "state_id": sid, "category": meta["category"], "source_group": meta["source_group"],
                "y_long": int(meta["y_long"]), "is_hard13_anchor": bool(meta["is_hard13_anchor"]),
                "delay_steps": delay, "delay_seconds": delay * float(plan["environment"]["dt"]),
                "n": len(seeds), "successes": int(success.sum()), "Q_hat": float(success.mean()),
                "Q_wilson95_lower": lo, "Q_wilson95_upper": hi,
                "mean_J_def": float(jdef.mean()), "std_J_def": float(jdef.std(ddof=1)) if len(jdef) > 1 else 0.0,
                "deadlocks": outcomes.get("deadlock", 0), "timeouts": outcomes.get("timeout", 0),
                "collisions": outcomes.get("collision", 0),
            }
            curves.append(curve)
            if delay == 0:
                effects[sid][0] = {"q_status": "REFERENCE", "j_status": "REFERENCE"}
                continue
            common = seeds
            base = [records[(sid, 0, seed)] for seed in common]
            s0 = np.asarray([bool(row["success"]) for row in base], np.int8)
            j0 = np.asarray([float(row["J_total"]) for row in base])
            dq = s0 - success
            dj = jdef - j0
            qlo, qhi = bootstrap_mean(dq, stable_seed(f"Q:{sid}:{delay}:{len(common)}"))
            jlo, jhi = bootstrap_mean(dj, stable_seed(f"J:{sid}:{delay}:{len(common)}"))
            qs, js = q_status(dq, qlo, qhi), j_status(dj, jlo, jhi)
            qr = {
                "state_id": sid, "category": meta["category"], "source_group": meta["source_group"],
                "y_long": int(meta["y_long"]), "is_hard13_anchor": bool(meta["is_hard13_anchor"]),
                "delay_steps": delay, "n_matched": len(common), "Q0": float(s0.mean()),
                "QH": float(success.mean()), "Delta_Q0_minus_QH": float(dq.mean()),
                "paired_bootstrap95_lower": qlo, "paired_bootstrap95_upper": qhi,
                "q_status": qs, "Q0_success_QH_fail": int(np.sum((s0 == 1) & (success == 0))),
                "QH_success_Q0_fail": int(np.sum((s0 == 0) & (success == 1))),
            }
            jr = {
                "state_id": sid, "category": meta["category"], "source_group": meta["source_group"],
                "y_long": int(meta["y_long"]), "delay_steps": delay, "n_matched": len(common),
                "J0_mean": float(j0.mean()), "JH_mean": float(jdef.mean()),
                "Delta_JH_minus_J0": float(dj.mean()),
                "paired_bootstrap95_lower": jlo, "paired_bootstrap95_upper": jhi, "j_status": js,
            }
            qrows.append(qr)
            jrows.append(jr)
            effects[sid][delay] = {**qr, **jr}
    return curves, qrows, jrows, effects


def transitions(plan: dict, effects: dict[str, dict[int, dict]]) -> list[dict]:
    state = {row["state_id"]: row for row in plan["states"]}
    rows = []
    for sid, values in effects.items():
        positive = sorted(d for d in values if d > 0)
        bad = [d for d in positive if values[d]["q_status"] == "SUPPORTED_RECOVERABILITY_LOSS"]
        first_bad = min(bad) if bad else None
        before = [d for d in positive if first_bad is None or d < first_bad]
        safe = [0] + [d for d in before if resolved_zero(values[d]["q_status"])]
        if first_bad is None:
            safe += [d for d in positive if resolved_zero(values[d]["q_status"])]
        unresolved = [d for d in positive if values[d]["q_status"] == "H_AMBIGUOUS"]
        rows.append({
            "state_id": sid, "category": state[sid]["category"], "source_group": state[sid]["source_group"],
            "y_long": int(state[sid]["y_long"]), "is_hard13_anchor": bool(state[sid]["is_hard13_anchor"]),
            "d_last_resolved_safe": max(safe), "d_first_supported_bad": "" if first_bad is None else first_bad,
            "transition_interval": f"[{max(safe)},{first_bad}]" if first_bad is not None else f"[{max(safe)},>={max(positive)}]",
            "unresolved_delays": ";".join(map(str, unresolved)),
            "right_censored": first_bad is None,
            "caveat": "nonzero paired effect with CI crossing zero is ambiguous, never called safe",
        })
    return rows


def internal_refinement_points(left: int, right: int) -> list[int]:
    if right - left <= 1:
        return []
    raw = [round(left + (right - left) * fraction) for fraction in (0.25, 0.5, 0.75)]
    return sorted({int(x) for x in raw if left < x < right and x >= 4})


def make_adaptive(plan: dict, effects: dict[str, dict[int, dict]], manifest_hashes: dict[str, str]) -> dict:
    d128, refinement = [], defaultdict(list)
    seed_escalation = defaultdict(dict)
    for state in plan["states"]:
        sid = state["state_id"]
        values = effects[sid]
        bad = sorted(d for d in values if d > 0 and values[d]["q_status"] == "SUPPORTED_RECOVERABILITY_LOSS")
        first_bad = min(bad) if bad else None
        prior_safe = [0] + [d for d in values if d > 0 and (first_bad is None or d < first_bad) and resolved_zero(values[d]["q_status"])]
        last_safe = max(prior_safe)
        if resolved_zero(values[64]["q_status"]):
            d128.append(sid)
        if first_bad is not None:
            for delay in internal_refinement_points(last_safe, first_bad):
                if delay not in values:
                    refinement[str(delay)].append(sid)
        # Escalate only the first unresolved point that obstructs locating the
        # transition after the last resolved-safe point.
        candidates = sorted(
            d for d in values if d > last_safe and values[d]["q_status"] == "H_AMBIGUOUS"
            and (first_bad is None or d <= first_bad)
        )
        if candidates:
            delay = candidates[0]
            old = [int(v) for v in state["seeds_initial"]]
            # Use a preregistered, state-namespace-derived reserve far outside
            # every legacy 95x/957/959 continuation range.  This prevents a
            # partially cached prior adaptive tuple from being silently rerun
            # as "new" while remaining independent of all observed outcomes.
            reserve_start = 100_000_000 + (int(state["rng_namespace"]) % 1_000_000) * 256
            new = list(range(reserve_start, reserve_start + 64))
            if set(new) & set(old):
                raise AssertionError((sid, "adaptive seed collision"))
            seed_escalation[str(delay)][sid] = new
    return {
        "audit_plan_sha256": sha(HERE / "audit_plan.json"),
        "coarse_manifest_hashes": manifest_hashes,
        "selection_frozen_before_adaptive_rollout": True,
        "d128_state_ids": sorted(d128),
        "refinement_by_delay": {delay: sorted(ids) for delay, ids in sorted(refinement.items(), key=lambda x: int(x[0]))},
        "seed_escalation_by_delay": {
            delay: {"state_ids": sorted(mapping), "state_seed_map": {sid: mapping[sid] for sid in sorted(mapping)}}
            for delay, mapping in sorted(seed_escalation.items(), key=lambda x: int(x[0]))
        },
        "selection_rules": {
            "d128": "d64 has resolved exact-no-loss or statistically supported improvement",
            "refinement": "three deterministic quartile-grid integer delays strictly between last resolved-safe and first supported-bad coarse point",
            "seed_escalation": "first nonzero-CI-crossing-zero delay obstructing transition after last resolved-safe point; add 64 matched state-namespace-derived reserve seeds (100000000 + rng_namespace_mod_1e6*256 + [0,63]) for d0 and that delay; reserve is result-independent and disjoint from all legacy 95x ranges",
            "no_effect_size_threshold": True,
        },
    }


def aggregate_curves(plan: dict, curves: list[dict], qrows: list[dict], jrows: list[dict]) -> tuple[list[dict], list[dict]]:
    qout, jout = [], []
    all_count = len(plan["states"])
    for delay in sorted({int(row["delay_steps"]) for row in curves}):
        rows = [row for row in curves if int(row["delay_steps"]) == delay]
        q = np.asarray([float(row["Q_hat"]) for row in rows])
        j = np.asarray([float(row["mean_J_def"]) for row in rows])
        qlo, qhi = bootstrap_mean(q, stable_seed(f"macroQ:{delay}"))
        jlo, jhi = bootstrap_mean(j, stable_seed(f"macroJ:{delay}"))
        if delay == 0:
            dq = np.zeros(len(rows), dtype=float)
            dj = np.zeros(len(rows), dtype=float)
        else:
            qmatch = [row for row in qrows if int(row["delay_steps"]) == delay]
            jmatch = [row for row in jrows if int(row["delay_steps"]) == delay]
            if len(qmatch) != len(rows) or len(jmatch) != len(rows):
                raise AssertionError((delay, len(rows), len(qmatch), len(jmatch)))
            # Use paired per-state effects.  Some adaptively escalated states
            # have more d0 seeds than other delays, so subtracting marginal
            # curve estimates would silently break matching.
            dq = np.asarray([float(row["Delta_Q0_minus_QH"]) for row in qmatch])
            dj = np.asarray([float(row["Delta_JH_minus_J0"]) for row in jmatch])
        dqlo, dqhi = bootstrap_mean(dq, stable_seed(f"macroDQ:{delay}"))
        djlo, djhi = bootstrap_mean(dj, stable_seed(f"macroDJ:{delay}"))
        scope = "FULL_FROZEN_POOL" if len(rows) == all_count else "ADAPTIVELY_SELECTED_SUBSET"
        qout.append({"scope": scope, "delay_steps": delay, "state_count": len(rows), "macro_Q": float(q.mean()), "bootstrap95_lower": qlo, "bootstrap95_upper": qhi, "macro_Delta_Q0_minus_QH": float(dq.mean()), "Delta_bootstrap95_lower": dqlo, "Delta_bootstrap95_upper": dqhi})
        jout.append({"scope": scope, "delay_steps": delay, "state_count": len(rows), "macro_mean_J_def": float(j.mean()), "bootstrap95_lower": jlo, "bootstrap95_upper": jhi, "macro_Delta_JH_minus_J0": float(dj.mean()), "Delta_bootstrap95_lower": djlo, "Delta_bootstrap95_upper": djhi})
    return qout, jout


def freeze_candidates(plan: dict, qrows: list[dict]) -> dict:
    coarse = [int(d) for d in plan["coarse_delays"] if int(d) > 0]
    summaries = []
    for delay in coarse:
        rows = [row for row in qrows if int(row["delay_steps"]) == delay]
        positive = [row for row in rows if row["q_status"] == "SUPPORTED_RECOVERABILITY_LOSS"]
        negative = [row for row in rows if resolved_zero(row["q_status"])]
        ambiguous = [row for row in rows if row["q_status"] == "H_AMBIGUOUS"]
        pos_groups = {row["source_group"] for row in positive}
        neg_groups = {row["source_group"] for row in negative}
        eventual_positive = [row for row in positive if int(row["y_long"]) == 1]
        eventual_negative = [row for row in negative if int(row["y_long"]) == 1]
        eventual_pos_groups = {row["source_group"] for row in eventual_positive}
        eventual_neg_groups = {row["source_group"] for row in eventual_negative}
        summaries.append({
            "H": delay, "resolved_positive": len(positive), "resolved_negative": len(negative),
            "ambiguous": len(ambiguous), "positive_source_groups": len(pos_groups),
            "negative_source_groups": len(neg_groups),
            "eventually_needed_resolved_positive": len(eventual_positive),
            "eventually_needed_resolved_negative": len(eventual_negative),
            "eventually_needed_positive_source_groups": len(eventual_pos_groups),
            "eventually_needed_negative_source_groups": len(eventual_neg_groups),
            "strict_logo_feasible": (
                len(pos_groups) >= 3 and len(neg_groups) >= 3
                and len(eventual_pos_groups) >= 3 and len(eventual_neg_groups) >= 3
            ),
        })
    feasible = [row for row in summaries if row["strict_logo_feasible"]]
    selected = []
    if feasible:
        first = min(row["H"] for row in feasible)
        selected = [row["H"] for row in feasible if row["H"] >= first][:3]
    return {
        "audit_plan_sha256": sha(HERE / "audit_plan.json"),
        "paired_success_statistics_sha256": sha(HERE / "paired_success_differences.csv"),
        "transition_intervals_sha256": sha(HERE / "transition_intervals.csv"),
        "frozen_before_any_stage2_training": True,
        "selection_rule": "among full-pool coarse horizons only, start at earliest H with >=3 resolved-positive and >=3 resolved-negative source groups both overall and within the y_long=1 diagnostic subset (so urgency is not reduced to old y_long identity), then take at most the first three feasible measured horizons; y_long is never the target and no classifier/test result is available or read",
        "candidate_horizons": selected,
        "all_coarse_horizon_inventory": summaries,
        "maximum_candidates": 3,
        "uses_gate_performance": False,
        "uses_y_long_as_target": False,
    }


def pareto_frontier(curves: list[dict], qrows: list[dict]) -> list[dict]:
    curve = {(row["state_id"], int(row["delay_steps"])): row for row in curves}
    status = {(row["state_id"], int(row["delay_steps"])): row["q_status"] for row in qrows}
    state_ids = sorted({row["state_id"] for row in curves})
    output = []
    for sid in state_ids:
        delays = sorted(d for ss, d in curve if ss == sid)
        eligible = [d for d in delays if d == 0 or resolved_zero(status[(sid, d)])]
        for delay in eligible:
            point = curve[(sid, delay)]
            dominated = any(
                curve[(sid, other)]["Q_hat"] >= point["Q_hat"]
                and curve[(sid, other)]["mean_J_def"] <= point["mean_J_def"]
                and (curve[(sid, other)]["Q_hat"] > point["Q_hat"] or curve[(sid, other)]["mean_J_def"] < point["mean_J_def"])
                for other in eligible if other != delay
            )
            output.append({
                "state_id": sid, "delay_steps": delay, "Q_hat": point["Q_hat"],
                "mean_J_def": point["mean_J_def"], "strict_recoverability_compatible": True,
                "point_estimate_pareto_nondominated": not dominated,
                "caveat": "point-estimate diagnostic only; no Q/J scalarization",
            })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("coarse", "final"))
    args = parser.parse_args()
    plan_path = HERE / "audit_plan.json"
    plan = json.loads(plan_path.read_text())
    plan_hash = sha(plan_path)
    records, manifest_hashes = completed_manifests(plan_hash)
    initial = {row["state_id"]: [int(v) for v in row["seeds_initial"]] for row in plan["states"]}
    require_grid(plan, records, [int(v) for v in plan["coarse_delays"]], initial)

    if args.command == "coarse":
        curves, qrows, jrows, effects = compute(plan, records, set(map(int, plan["coarse_delays"])))
        write_csv(HERE / "coarse_per_state_delay_curves.csv", curves)
        write_csv(HERE / "coarse_paired_success_differences.csv", qrows)
        write_csv(HERE / "coarse_paired_deformation_differences.csv", jrows)
        adaptive = make_adaptive(plan, effects, manifest_hashes)
        write_json(HERE / "adaptive_selection.json", adaptive)
        print(json.dumps({"adaptive_selection_sha256": sha(HERE / "adaptive_selection.json"), "d128_states": len(adaptive["d128_state_ids"]), "refinement_jobs": {k: len(v) for k, v in adaptive["refinement_by_delay"].items()}, "seed_escalation_jobs": {k: len(v["state_ids"]) for k, v in adaptive["seed_escalation_by_delay"].items()}}, indent=2))
        return

    selection_path = HERE / "adaptive_selection.json"
    if not selection_path.exists():
        raise RuntimeError("coarse must freeze adaptive_selection.json before final")
    selection = json.loads(selection_path.read_text())
    if selection["audit_plan_sha256"] != plan_hash or not selection["selection_frozen_before_adaptive_rollout"]:
        raise AssertionError("invalid adaptive selection chain")
    convergence_path = HERE / "adaptive_followup" / "convergence.json"
    if not convergence_path.exists():
        raise RuntimeError("iterative 64->128->256 adaptive follow-up has not certified convergence")
    convergence = json.loads(convergence_path.read_text())
    if convergence.get("audit_plan_sha256") != plan_hash:
        raise AssertionError("adaptive follow-up convergence/plan hash mismatch")
    for relative, expected in convergence.get("completed_manifest_hashes", {}).items():
        path = HERE / relative
        if not path.exists() or sha(path) != expected:
            raise AssertionError(f"adaptive convergence evidence changed: {relative}")
    # Integrity: each selected adaptive point must have at least the initial
    # 64 matched seeds; seed-escalated points must additionally contain every
    # preregistered new seed at d0 and at the selected delay.
    for sid in selection["d128_state_ids"]:
        require_grid(plan, records, [128], {s["state_id"]: initial[s["state_id"]] if s["state_id"] == sid else [] for s in plan["states"]})
    for delay, ids in selection["refinement_by_delay"].items():
        chosen = set(ids)
        require_grid(plan, records, [int(delay)], {s["state_id"]: initial[s["state_id"]] if s["state_id"] in chosen else [] for s in plan["states"]})
    for delay, obj in selection["seed_escalation_by_delay"].items():
        seed_map = {s["state_id"]: obj["state_seed_map"].get(s["state_id"], []) for s in plan["states"]}
        require_grid(plan, records, [0, int(delay)], seed_map)

    curves, qrows, jrows, effects = compute(plan, records)
    transition = transitions(plan, effects)
    qcurve, jcurve = aggregate_curves(plan, curves, qrows, jrows)
    write_csv(HERE / "per_state_delay_curves.csv", curves)
    write_csv(HERE / "paired_success_differences.csv", qrows)
    write_csv(HERE / "paired_deformation_differences.csv", jrows)
    write_csv(HERE / "transition_intervals.csv", transition)
    write_csv(HERE / "recoverability_curves.csv", qcurve)
    write_csv(HERE / "deformation_curves.csv", jcurve)
    write_csv(HERE / "pareto_frontier.csv", pareto_frontier(curves, qrows))
    candidates = freeze_candidates(plan, qrows)
    write_json(HERE / "candidate_windows.json", candidates)
    integrity = {
        "audit_plan_sha256": plan_hash, "adaptive_selection_sha256": sha(selection_path),
        "candidate_windows_sha256": sha(HERE / "candidate_windows.json"),
        "adaptive_followup_convergence_sha256": sha(convergence_path),
        "terminal_ambiguous_points_at_256": int(convergence.get("terminal_ambiguous_count", 0)),
        "frozen_states": len(plan["states"]), "coarse_grid_complete": True,
        "adaptive_selection_complete": True, "execution_errors": 0,
        "candidate_frozen_before_training": True,
    }
    write_json(HERE / "sanity_checks.json", integrity)
    transition_counts = Counter(
        "BOUNDED" if row["d_first_supported_bad"] != "" else "RIGHT_CENSORED_OR_UNRESOLVED"
        for row in transition
    )
    report = f"""# Stage 1 intervention-window audit

- Frozen states: **{len(plan['states'])}** across **{plan['selection']['source_group_count']}** source groups.
- Coarse delays: `{plan['coarse_delays']}`; adaptive extension: `{plan['adaptive_extension_delay']}`.
- Transition inventory: `{dict(transition_counts)}`.
- Frozen Stage-2 candidate horizons: `{candidates['candidate_horizons']}`.
- Candidate selection was completed before any gate training and used no model/test performance.

`y_H=1` is available only where paired `Q0-QH` has a bootstrap 95% CI
strictly above zero. Exact no-effect or supported improvement is resolved
negative; every nonzero effect whose CI crosses zero remains `H_AMBIGUOUS`.

The downstream controller remains the previously validated frozen-`eta_best`
approximation, not arbitrary-state receding oracle re-query. Q and J_def are
reported separately and are never scalarized.
"""
    (HERE / "stage1_report.md").write_text(report)
    print(json.dumps({**integrity, "candidate_horizons": candidates["candidate_horizons"]}, indent=2))


if __name__ == "__main__":
    main()
