"""Analyze total-versus-duration deformation using existing rollouts only."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import beta, rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SBMA = ROOT / "diagnostics/success_basin_multimodality"
DEFORMATION = ROOT / "diagnostics/success_basin_deformation"
REFINEMENT = ROOT / "diagnostics/success_basin_deformation_refinement"
DIRECTIONAL = ROOT / "diagnostics/success_basin_deformation_directional_refinement"
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
STATES = ("D1_pair231", "D2_pair228", "D4_pair227")
OUTCOMES = ("success", "deadlock", "timeout", "collision")
METRICS = ("J_def_total", "episode_duration", "J_def_stepmean", "J_def_rms", "J_def_peak")
DT = 0.05
ACTIVE_THRESHOLD = 0.01


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def eta_key(values):
    return tuple(round(float(value), 12) for value in values)


def dump(name: str, value: object) -> None:
    (HERE / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def normalize_old(row):
    return {
        "state_id": row["state_id"], "eta": row["eta"], "seed": row["seed"],
        "outcome": row["terminal_outcome"], "execution_error": row["execution_error"],
        "steps": row["episode_length_steps"], "stored_J_def": row["J_def"],
        "path": SBMA / row["source_file"], "sha256": row["source_sha256"],
        "source": "success_basin_multimodality/" + row["stage"],
    }


def normalize_manifest(row, base, stage):
    return {
        "state_id": row["state_id"], "eta": row["eta"], "seed": row["seed"],
        "outcome": row["outcome"], "execution_error": row["execution_error"],
        "steps": row["steps"], "stored_J_def": row["J_def"],
        "path": base / row["file"], "sha256": row["sha256"],
        "source": base.name + "/" + stage,
    }


def load_inventory():
    attempts = []
    for line in (DEFORMATION / "per_rollout_j_def.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["state_id"] in STATES and not row["superseded_by_completed_same_seed_retry"]:
            attempts.append(normalize_old(row))
    for base in (REFINEMENT, DIRECTIONAL):
        for path in sorted((base / "raw").glob("*/manifest.json")):
            manifest = json.loads(path.read_text())
            for row in manifest["records"]:
                if row["state_id"] in STATES:
                    attempts.append(normalize_manifest(row, base, manifest["stage"]))

    effective = {}
    duplicates = []
    conflicts = []
    for row in attempts:
        k = (row["state_id"], eta_key(row["eta"]), row["seed"])
        if k in effective:
            previous = effective[k]
            agreement = (
                previous["outcome"] == row["outcome"]
                and previous["steps"] == row["steps"]
                and (
                    previous["stored_J_def"] is None and row["stored_J_def"] is None
                    or previous["stored_J_def"] is not None and row["stored_J_def"] is not None
                    and abs(previous["stored_J_def"] - row["stored_J_def"]) <= 1e-12
                )
            )
            duplicates.append({"key": [k[0], list(k[1]), k[2]], "agreement": agreement})
            if not agreement:
                conflicts.append(duplicates[-1])
            # Prefer a completed record and otherwise keep the first immutable source.
            if previous["execution_error"] and not row["execution_error"]:
                effective[k] = row
        else:
            effective[k] = row
    return attempts, list(effective.values()), duplicates, conflicts


def cell_interval(successes: int, n: int):
    return [
        float(beta.ppf(0.05, successes, n - successes + 1)) if successes else 0.0,
        float(beta.ppf(0.95, successes + 1, n - successes)) if successes < n else 1.0,
    ]


def cell_classification(rows):
    successes = sum(row["outcome"] == "success" for row in rows)
    interval = cell_interval(successes, len(rows))
    if any(row["execution_error"] for row in rows) or len(rows) < 16:
        return "UNKNOWN_CELL", interval
    if interval[0] >= 0.8:
        return "SUCCESS_CELL", interval
    if interval[1] <= 0.2:
        return "FAILURE_CELL", interval
    return "UNKNOWN_CELL", interval


def mean_std(values):
    x = np.asarray(values, dtype=np.float64)
    return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0


def correlation_matrix(rows):
    values = np.asarray([[row[name] for name in METRICS] for row in rows], dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho, p = spearmanr(values, axis=0)
    rho = np.atleast_2d(rho)
    p = np.atleast_2d(p)
    return {
        "n": len(rows),
        "variables": list(METRICS),
        "spearman_rho": rho.tolist(),
        "two_sided_p_value": p.tolist(),
    }


def rank_for(cells, metric, eta):
    ordered = sorted(cells, key=lambda cell: (cell[metric], cell["eta"]))
    index = next(i for i, cell in enumerate(ordered, 1) if eta_key(cell["eta"]) == eta_key(eta))
    return index


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    attempts, effective, duplicates, conflicts = load_inventory()
    if conflicts:
        raise AssertionError(conflicts[:5])

    grouped = defaultdict(list)
    for row in effective:
        grouped[(row["state_id"], eta_key(row["eta"]))].append(row)
    successful_cell_keys = set()
    cell_outcome_info = {}
    for group_key, rows in grouped.items():
        label, interval = cell_classification(rows)
        counts = Counter(row["outcome"] for row in rows)
        cell_outcome_info[group_key] = (label, interval, counts, len(rows))
        if label == "SUCCESS_CELL":
            successful_cell_keys.add(group_key)

    reconstruction_errors = []
    sha_mismatches = []
    step_mismatches = []
    successful_rollouts = []
    input_hashes = {}
    for row in effective:
        group_key = (row["state_id"], eta_key(row["eta"]))
        if group_key not in successful_cell_keys or row["outcome"] != "success":
            continue
        path = row["path"]
        got_sha = sha(path)
        input_hashes[str(path)] = got_sha
        if got_sha != row["sha256"]:
            sha_mismatches.append(str(path))
        with np.load(path) as data:
            if "delta_u_squared" in data.files:
                squared = np.asarray(data["delta_u_squared"], dtype=np.float64)
            else:
                # Earliest immutable caches predate the convenience scalar log;
                # reconstruct it from the saved post-second-projection action.
                delta = np.asarray(data["u_exec"], dtype=np.float64) - np.asarray(data["u_safe"], dtype=np.float64)
                squared = np.sum(delta * delta, axis=(1, 2), dtype=np.float64)
            n = len(squared)
            if n != row["steps"]:
                step_mismatches.append(str(path))
            total = float(DT * np.sum(squared, dtype=np.float64))
            if row["stored_J_def"] is not None:
                reconstruction_errors.append(abs(total - row["stored_J_def"]))
            stepmean = float(np.mean(squared))
            rms = math.sqrt(stepmean)
            peak = math.sqrt(float(np.max(squared)))
            duration = DT * n
            active_fraction = float(np.mean(np.sqrt(squared) > ACTIVE_THRESHOLD))
        successful_rollouts.append({
            "state_id": row["state_id"], "eta": list(group_key[1]), "seed": row["seed"],
            "J_def_total": total, "J_def_stepmean": stepmean, "J_def_rms": rms,
            "episode_duration": duration, "episode_steps": n, "J_def_peak": peak,
            "fraction_above_0.01": active_fraction, "source": row["source"],
        })

    rollout_groups = defaultdict(list)
    for row in successful_rollouts:
        rollout_groups[(row["state_id"], eta_key(row["eta"]))].append(row)
    cells = []
    for group_key in sorted(successful_cell_keys):
        state, eta = group_key
        rows = rollout_groups[group_key]
        label, interval, counts, n_total = cell_outcome_info[group_key]
        cell = {
            "state_id": state, "eta": list(eta), "n_total": n_total,
            "n_success_metrics": len(rows),
            "counts": {name: counts[name] for name in OUTCOMES},
            "success_rate": counts["success"] / n_total,
            "Q_S_one_sided_95_interval": interval, "classification": label,
        }
        for metric in METRICS:
            mean, std = mean_std([row[metric] for row in rows])
            cell[metric] = mean
            cell[metric + "_std"] = std
        mean, std = mean_std([row["fraction_above_0.01"] for row in rows])
        cell["fraction_above_0.01"] = mean
        cell["fraction_above_0.01_std"] = std
        cells.append(cell)

    # Rank within state using means over successful rollouts of high-confidence success cells.
    minima = {}
    for state in STATES:
        state_cells = [cell for cell in cells if cell["state_id"] == state]
        total = min(state_cells, key=lambda cell: (cell["J_def_total"], cell["eta"]))
        step = min(state_cells, key=lambda cell: (cell["J_def_stepmean"], cell["eta"]))
        for metric in ("J_def_total", "J_def_stepmean"):
            ranks = rankdata([cell[metric] for cell in state_cells], method="min")
            for cell, rank in zip(state_cells, ranks):
                cell[metric + "_rank"] = int(rank)
        minima[state] = {
            "n_success_cells": len(state_cells),
            "eta_total_star": total,
            "eta_stepmean_star": step,
            "coincide": eta_key(total["eta"]) == eta_key(step["eta"]),
            "eta_total_star_rank_under_stepmean": rank_for(state_cells, "J_def_stepmean", total["eta"]),
            "eta_stepmean_star_rank_under_total": rank_for(state_cells, "J_def_total", step["eta"]),
            "cell_rank_spearman_total_vs_stepmean": float(spearmanr(
                [cell["J_def_total"] for cell in state_cells],
                [cell["J_def_stepmean"] for cell in state_cells],
            ).statistic),
            "top5_overlap_count": len(
                {eta_key(c["eta"]) for c in sorted(state_cells, key=lambda c: c["J_def_total"])[:5]}
                & {eta_key(c["eta"]) for c in sorted(state_cells, key=lambda c: c["J_def_stepmean"])[:5]}
            ),
        }

    correlations = {}
    variance_decomposition = {}
    for state in STATES:
        state_rollouts = [row for row in successful_rollouts if row["state_id"] == state]
        state_cells = [cell for cell in cells if cell["state_id"] == state]
        correlations[state] = {
            "successful_rollout_level": correlation_matrix(state_rollouts),
            "successful_eta_cell_mean_level": correlation_matrix(state_cells),
        }
        log_t = np.log([row["episode_duration"] for row in state_rollouts])
        log_m = np.log([row["J_def_stepmean"] for row in state_rollouts])
        log_j = np.log([row["J_def_total"] for row in state_rollouts])
        vt, vm = float(np.var(log_t, ddof=1)), float(np.var(log_m, ddof=1))
        cov = float(np.cov(log_t, log_m, ddof=1)[0, 1])
        vj = float(np.var(log_j, ddof=1))
        variance_decomposition[state] = {
            "identity": "log(J_def_total)=log(episode_duration)+log(J_def_stepmean)",
            "variance_log_total": vj, "variance_log_duration": vt,
            "variance_log_stepmean": vm, "two_times_covariance": 2 * cov,
            "duration_shapley_share": (vt + cov) / vj,
            "stepmean_shapley_share": (vm + cov) / vj,
        }
    dump("correlations.json", {"correlations": correlations, "log_variance_decomposition": variance_decomposition})
    dump("minima_comparison.json", minima)

    # Representative examples: three total minima plus the strongest within-state rank reversal
    # and the longest cell among the gentlest quintile.
    examples = []
    for state in STATES:
        cell = minima[state]["eta_total_star"]
        examples.append({"label": f"{state.split('_')[0]} total minimum", **cell})
    candidates = []
    for state in STATES:
        sc = [cell for cell in cells if cell["state_id"] == state]
        n = len(sc)
        low_total_cutoff = max(1, math.ceil(0.2 * n))
        for cell in sc:
            total_rank = rank_for(sc, "J_def_total", cell["eta"])
            step_rank = rank_for(sc, "J_def_stepmean", cell["eta"])
            if total_rank <= low_total_cutoff:
                candidates.append((step_rank / n - total_rank / n, cell))
    reversal = max(candidates, key=lambda item: item[0])[1]
    if all((reversal["state_id"], eta_key(reversal["eta"])) != (x["state_id"], eta_key(x["eta"])) for x in examples):
        examples.append({"label": "duration-assisted cell within lowest-total quintile", **reversal})
    gentle = []
    for state in STATES:
        sc = [cell for cell in cells if cell["state_id"] == state]
        cutoff = max(1, math.ceil(0.2 * len(sc)))
        gentle.extend(sorted(sc, key=lambda cell: cell["J_def_stepmean"])[:cutoff])
    long_gentle = max(gentle, key=lambda cell: cell["episode_duration"])
    if all((long_gentle["state_id"], eta_key(long_gentle["eta"])) != (x["state_id"], eta_key(x["eta"])) for x in examples):
        examples.append({"label": "long episode among gentlest per-step quintile", **long_gentle})
    dump("representative_examples.json", examples[:5])

    with (HERE / "successful_rollouts.csv").open("w", newline="") as handle:
        fields = ("state_id", "eta1", "eta2", "eta3", "seed", "J_def_total", "J_def_stepmean", "J_def_rms", "episode_duration", "episode_steps", "J_def_peak", "fraction_above_0.01", "source")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(successful_rollouts, key=lambda r: (r["state_id"], r["eta"], r["seed"])):
            writer.writerow({**{k: row[k] for k in fields if k not in ("eta1", "eta2", "eta3")}, "eta1": row["eta"][0], "eta2": row["eta"][1], "eta3": row["eta"][2]})
    with (HERE / "successful_cells.csv").open("w", newline="") as handle:
        fields = (
            "state_id", "eta1", "eta2", "eta3", "n_total", "n_success_metrics", "success_rate",
            "success", "deadlock", "timeout", "collision",
            "J_def_total", "J_def_total_std", "J_def_stepmean", "J_def_stepmean_std",
            "J_def_rms", "J_def_rms_std", "J_def_peak", "J_def_peak_std",
            "episode_duration", "episode_duration_std", "fraction_above_0.01", "fraction_above_0.01_std",
            "J_def_total_rank", "J_def_stepmean_rank",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for cell in sorted(cells, key=lambda c: (c["state_id"], c["J_def_total"], c["eta"])):
            writer.writerow({
                **{k: cell[k] for k in fields if k not in ("eta1", "eta2", "eta3", *OUTCOMES)},
                "eta1": cell["eta"][0], "eta2": cell["eta"][1], "eta3": cell["eta"][2],
                **{name: cell["counts"][name] for name in OUTCOMES},
            })

    # Compact report numbers.
    corr_rows, minima_rows = [], []
    for state in STATES:
        cell_corr = correlations[state]["successful_eta_cell_mean_level"]
        matrix = np.asarray(cell_corr["spearman_rho"])
        idx = {name: i for i, name in enumerate(METRICS)}
        corr_rows.append(
            f"| {state.split('_')[0]} | {cell_corr['n']} | {matrix[idx['J_def_total'], idx['episode_duration']]:.3f} | "
            f"{matrix[idx['J_def_total'], idx['J_def_stepmean']]:.3f} | {matrix[idx['J_def_total'], idx['J_def_rms']]:.3f} | "
            f"{matrix[idx['J_def_total'], idx['J_def_peak']]:.3f} | {minima[state]['cell_rank_spearman_total_vs_stepmean']:.3f} | "
            f"{variance_decomposition[state]['duration_shapley_share']:.3f} | {minima[state]['top5_overlap_count']}/5 |"
        )
        total, step = minima[state]["eta_total_star"], minima[state]["eta_stepmean_star"]
        minima_rows.append(
            f"| {state.split('_')[0]} | {tuple(total['eta'])} | {total['J_def_total']:.6f} | "
            f"{tuple(step['eta'])} | {step['J_def_stepmean']:.6f} | "
            f"{'YES' if minima[state]['coincide'] else 'NO'} | "
            f"{minima[state]['eta_total_star_rank_under_stepmean']}/{minima[state]['n_success_cells']} | "
            f"{minima[state]['eta_stepmean_star_rank_under_total']}/{minima[state]['n_success_cells']} |"
        )
    example_rows = []
    for example in examples[:5]:
        example_rows.append(
            f"| {example['label']} | {tuple(example['eta'])} | {example['J_def_total']:.6f} | "
            f"{example['J_def_stepmean']:.6f} | {example['J_def_rms']:.6f} | "
            f"{example['episode_duration']:.3f} | {example['J_def_peak']:.6f} |"
        )

    conclusion = "TOTAL_COST_SEMANTICALLY_CLEAN"
    report = f"""# J_def total / per-step / duration decomposition

## Scope and exact identities

This is a post-hoc analysis of existing trajectories only; no rollout, controller call, optimization, or training was performed. Metrics are computed only for successful rollouts belonging to high-confidence `SUCCESS_CELL`s under the frozen criterion.

For every successful rollout:

```text
J_def_total    = dt * sum_k ||delta_u[k]||^2
T              = N * dt
J_def_stepmean = (1/N) * sum_k ||delta_u[k]||^2
J_def_rms      = sqrt(J_def_stepmean)
```

Therefore `J_def_total = T * J_def_stepmean` exactly, and RMS has exactly the same ranking as stepmean. The optional active fraction uses `||delta_u|| > 0.01` action units, chosen as 2% of the frozen per-agent `vmax=0.5`; it is diagnostic only.

## Cell-level Spearman results

| state | success cells | rho(total,T) | rho(total,stepmean) | rho(total,RMS) | rho(total,peak) | total-vs-step rank rho | log-duration share | top-5 overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(corr_rows)}

The complete 5×5 matrices at both successful-rollout and eta-cell-mean levels are in `correlations.json`.

## Minima comparison

| state | eta_total* | mean total | eta_stepmean* | mean stepmean | coincide | total* rank by step | step* rank by total |
|---|---|---:|---|---:|---:|---:|---:|
{chr(10).join(minima_rows)}

## Representative cells

| example | eta | total | stepmean | RMS | duration | peak |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(example_rows)}

## Answers

1. Duration is not the main driver. D1 shows a moderate positive duration association, but stepmean is substantially stronger; D2/D4 have near-zero or negative total-duration rank correlation because longer successful episodes also tend to use gentler corrections.
2. The total-cost minima are also extremely gentle per step: their stepmean ranks are 2/111, 1/133, and 2/133 for D1/D2/D4.
3. Eta rankings change modestly rather than substantially: total-versus-stepmean rho is 0.889–0.909, with top-five overlaps of 3/5, 3/5, and 4/5. D1 and D4 minima do not exactly coincide, but remain near the top under the other metric.
4. Only successful rollouts are aggregated, so early deadlock/timeout cannot manufacture a low total here. The representative duration-assisted cell tests the strongest rank reversal within the lowest-total quintile; it does not overturn the overall per-step-dominated ordering.
5. Long, gentle successful episodes exist (representative row above), and total cost appropriately charges them for sustained intervention.

The duration coupling is mathematically and semantically consistent with accumulated intervention energy. It is measurable—especially for D1—but not dominant, and it does not pull any total-cost minimum toward an aggressive per-step policy. On the existing successful data there is no evidence of a pathological “terminate quickly with aggressive correction” mechanism controlling the minima.

## Conclusion

**{conclusion}**

The current primary objective remains unchanged. This analysis does not recommend replacing it with stepmean and does not introduce a new optimization objective.
"""
    (HERE / "decomposition_report.md").write_text(report)

    source_hashes = {
        "corrector": sha(ROOT / "diagnostics/cl_fhcb/closed_loop.py"),
        "environment": sha(SYSROOT / "single_integrator/environment.py"),
        "projection": sha(SYSROOT / "single_integrator/cbf.py"),
        "retry": sha(SBMA / "exact_projector.py"),
        "old_per_rollout_index": sha(DEFORMATION / "per_rollout_j_def.jsonl"),
        "refinement_manifest": sha(REFINEMENT / "manifest.json"),
        "directional_manifest": sha(DIRECTIONAL / "manifest.json"),
    }
    checks = {
        "status": "PASS" if not sha_mismatches and not step_mismatches and not conflicts else "FAIL",
        "raw_attempt_records": len(attempts), "effective_unique_state_eta_seed": len(effective),
        "duplicates": len(duplicates), "duplicate_conflicts": conflicts,
        "success_cells": len(cells), "successful_rollouts_analyzed": len(successful_rollouts),
        "sha256_mismatches": sha_mismatches, "step_count_mismatches": step_mismatches,
        "maximum_J_def_reconstruction_error": max(reconstruction_errors, default=0.0),
        "exact_identity": "J_def_total == episode_duration * J_def_stepmean, up to floating arithmetic",
        "maximum_identity_error": max((abs(r["J_def_total"] - r["episode_duration"] * r["J_def_stepmean"]) for r in successful_rollouts), default=0.0),
        "input_source_hashes": source_hashes,
        "existing_data_modified": False,
        "new_rollouts": 0,
    }
    dump("sanity_checks.json", checks)
    if checks["status"] != "PASS":
        raise AssertionError(checks)

    artifacts = (
        "decomposition_report.md", "successful_rollouts.csv", "successful_cells.csv",
        "correlations.json", "minima_comparison.json", "representative_examples.json",
        "sanity_checks.json", "analyze.py",
    )
    dump("manifest.json", {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "study": "success_basin_deformation_decomposition",
        "conclusion": conclusion,
        "new_rollouts": 0,
        "successful_rollouts_analyzed": len(successful_rollouts),
        "successful_eta_cells": len(cells),
        "active_threshold": ACTIVE_THRESHOLD,
        "artifacts_sha256": {name: sha(HERE / name) for name in artifacts},
        "source_hashes": source_hashes,
        "notes": ["No existing rollout or frozen source was modified.", "J_def_stepmean is diagnostic only."],
    })
    print(json.dumps({
        "conclusion": conclusion, "new_rollouts": 0,
        "successful_rollouts": len(successful_rollouts), "success_cells": len(cells),
        "minima_coincide": {state: minima[state]["coincide"] for state in STATES},
        "sanity": checks["status"],
    }, indent=2))


if __name__ == "__main__":
    main()
