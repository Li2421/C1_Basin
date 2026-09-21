#!/usr/bin/env python3
"""Summarize the frozen GiveWay residual-effect pilot without rerunning it."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"results/giveway_residual_effect_pilot_v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def interval(k, n, alpha=.05):
    return [0.0 if k == 0 else float(beta.ppf(alpha/2, k, n-k+1)),
            1.0 if k == n else float(beta.ppf(1-alpha/2, k+1, n-k))]


def rate(k, n):
    return {"count": k, "denominator": n, "rate": k/n,
            "clopper_pearson_95": interval(k, n)}


def main():
    report = json.loads((OUT/"report.json").read_text())
    manifest = json.loads((OUT/"manifest.json").read_text())
    rows = [json.loads(line) for line in (OUT/"records.jsonl").read_text().splitlines()]
    safety = {row["pair_id"]: row for row in rows if row["group"] == "S_safety"}
    experiment = {row["pair_id"]: row for row in rows if row["group"] == "A_phi0_diag"}
    failed = [row for row in experiment.values() if not row["completed"]]
    completed = [row for row in experiment.values() if row["completed"]]
    n = len(experiment)

    safe_success = sum(row["completed"] and row["success"] and not row["collision"]
                       for row in experiment.values())
    timeout = sum(row["completed"] and row["ordinary_timeout"]
                  for row in experiment.values())
    numerical_failure = len(failed)
    if safe_success + timeout + numerical_failure != n:
        raise RuntimeError("experimental operational outcomes do not partition assigned rollouts")

    valid = [index for index in range(n) if experiment[index]["completed"]]
    baseline_deadlock_valid = [index for index in valid if safety[index]["D_H"]]
    baseline_non_deadlock_valid = [index for index in valid if not safety[index]["D_H"]]
    rescued = [index for index in baseline_deadlock_valid
               if not experiment[index]["D_H"] and experiment[index]["success"]
               and not experiment[index]["collision"]]
    converted_timeout = [index for index in baseline_deadlock_valid
                         if not experiment[index]["D_H"] and experiment[index]["ordinary_timeout"]]
    new_deadlock = [index for index in baseline_non_deadlock_valid
                    if experiment[index]["D_H"]]
    old_success_to_timeout = [index for index in valid if safety[index]["success"]
                              and experiment[index]["ordinary_timeout"]]
    old_success_to_solver_failure = [row["pair_id"] for row in failed
                                     if safety[row["pair_id"]]["success"]]
    failed_from_deadlock = [row["pair_id"] for row in failed
                            if safety[row["pair_id"]]["D_H"]]

    total_steps = sum(row["episode_steps"] for row in experiment.values())
    total_active = sum(row["gate_activations"] for row in experiment.values())
    residual_square = sum((row["residual_norm_rms"]**2)*row["episode_steps"]
                          for row in experiment.values() if row["episode_steps"])
    correction_square = sum((row["executed_correction_norm_rms"]**2)*row["episode_steps"]
                            for row in experiment.values() if row["episode_steps"])
    failure_kinds = Counter("speed_cut_limit" if "speed_cut_limit" in row["failure"]["message"]
                            else "qp_failed" if "qp_failed" in row["failure"]["message"]
                            else "other" for row in failed)

    analysis = {
        "schema": "giveway_residual_effect_pilot_analysis_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_script": {
            "path": str(Path(__file__).resolve()),
            "sha256": digest(Path(__file__).resolve()),
        },
        "source_run": {
            "report": str(OUT/"report.json"), "report_sha256": digest(OUT/"report.json"),
            "manifest": str(OUT/"manifest.json"), "manifest_sha256": digest(OUT/"manifest.json"),
            "records": str(OUT/"records.jsonl"), "records_sha256": digest(OUT/"records.jsonl"),
            "slurm_job_id": 80,
        },
        "scope": {
            "formal_stage_3_to_5": False,
            "formal_method_verdict": "NOT APPLICABLE",
            "gradient_test": False,
            "learning_test": False,
            "conditioning": "one archived GiveWay t=0 initial state, fixed phi0_diag, 64 paired random tapes",
        },
        "safety_arm": report["groups"]["S_safety"],
        "experimental_completed_subset": report["groups"]["A_phi0_diag"],
        "experimental_assigned_operational_outcomes": {
            "safe_success": rate(safe_success, n),
            "ordinary_timeout": rate(timeout, n),
            "projection_solver_failure": rate(numerical_failure, n),
            "collision_before_completion_or_failure": rate(
                sum(row["collision"] for row in experiment.values()), n),
            "partition": "safe_success + ordinary_timeout + projection_solver_failure = 64",
        },
        "deadlock_missing_outcome_sensitivity": {
            "completed_observed_D_H": sum(row["D_H"] for row in completed),
            "completed_denominator": len(completed),
            "failed_without_terminal_D_H": len(failed),
            "assigned_D_H_count_interval_if_failures_imputed": [0, len(failed)],
            "assigned_D_H_rate_interval_if_failures_imputed": [0.0, len(failed)/n],
            "warning": "The upper endpoint is an imputation sensitivity value, not an observed event rate; solver failure is an operational failure in its own right.",
        },
        "paired": {
            "valid_pairs": len(valid), "invalid_pairs": len(failed),
            "baseline_deadlock_valid": len(baseline_deadlock_valid),
            "baseline_non_deadlock_valid": len(baseline_non_deadlock_valid),
            "deadlock_to_safe_success": {"count": len(rescued), "pair_ids": rescued},
            "deadlock_to_ordinary_timeout": {"count": len(converted_timeout), "pair_ids": converted_timeout},
            "nondeadlock_to_deadlock": {"count": len(new_deadlock), "pair_ids": new_deadlock},
            "old_success_to_timeout": {"count": len(old_success_to_timeout),
                                       "pair_ids": old_success_to_timeout},
            "old_success_to_solver_failure": {"count": len(old_success_to_solver_failure),
                                              "pair_ids": old_success_to_solver_failure},
            "old_deadlock_to_solver_failure": {"count": len(failed_from_deadlock),
                                               "pair_ids": failed_from_deadlock},
            "safe_rescue_lower_fraction_of_all_52_baseline_deadlocks": len(rescued)/52,
        },
        "exploration_and_effect": {
            "gate_activation_time_weighted": total_active/total_steps,
            "gate_activation_rollout_mean_completed": report["groups"]["A_phi0_diag"]["gate_activation_fraction"],
            "raw_residual_joint_norm_rms_time_weighted_m_per_s": float(np.sqrt(residual_square/total_steps)),
            "executed_correction_joint_norm_rms_time_weighted_m_per_s": float(np.sqrt(correction_square/total_steps)),
            "executed_correction_energy_sum_all_observed_prefixes": sum(
                row["executed_correction_energy"] for row in experiment.values()),
            "executed_correction_energy_completed_rollout_summary": report["groups"]["A_phi0_diag"]["executed_correction_energy"],
            "active_projection_alias_count": sum(row["active_projection_alias_count"]
                                                for row in experiment.values()),
        },
        "numerical_and_safety": {
            "failure_count": len(failed),
            "failure_kinds": dict(failure_kinds),
            "failures": [{"pair_id": row["pair_id"], "action_index": row["failure"]["step"],
                          "message": row["failure"]["message"],
                          "baseline_D_H": safety[row["pair_id"]]["D_H"],
                          "baseline_success": safety[row["pair_id"]]["success"]}
                         for row in failed],
            "accepted_prefix_min_linear_residual": min(
                row["min_projection_linear_residual"] for row in experiment.values()),
            "accepted_prefix_max_speed_excess": max(
                row["max_projection_speed_excess"] for row in experiment.values()),
            "observed_collision_count": sum(row["collision"] for row in experiment.values()),
            "fallback_executed": False,
        },
        "cost": report["total"] | {"slurm_wall_seconds": report["runtime"]["wall_seconds"],
                                    "trace_bytes": report["total"]["trace_bytes"]},
        "interpretation": [
            "The fixed stochastic residual materially changed outcomes in this one archived scene.",
            "Thirty-one paired baseline deadlocks became collision-free task successes; fifteen became ordinary timeouts and therefore are not counted as escapes.",
            "Eight experimental rollouts terminated at projection solver errors, so unconditional D_H is not observable for all 64 draws.",
            "This pilot contains no score gradient, parameter update, or learning test and supplies no Direction A method verdict.",
        ],
        "configuration": manifest,
    }
    json_path = OUT/"analysis.json"
    json_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")

    s = report["groups"]["S_safety"]
    a = report["groups"]["A_phi0_diag"]
    markdown = f"""# GiveWay residual-effect pilot

This is an independent exploratory diagnostic, not formal Stage 3--5 and not a method PASS/FAIL. It uses one archived `t=0` GiveWay initial state, 64 paired random tapes, and the fixed `phi0_diag` specified in `manifest.json`.

## Frozen configuration

- Flow checkpoint: `{manifest['asset_selection']['checkpoint_actual_path']}` (`{manifest['asset_selection']['checkpoint_sha256']}`)
- Archived scene: Safety rollout 52, initial positions `{manifest['asset_selection']['initial_positions_t0']}`
- `v_ref=0.5 m/s`, `mu=0`, `p=0.5`, `sigma_min=0.025`, `sigma_max=0.25`, `sigma_init=0.1 m/s`
- Root seed 20260921; continuations 0--63; horizon 850 actions; all three termination flags enabled
- Slurm job 80; GPU backend; no parameter update, tuning, sample extension, or outcome-dependent stopping

## Results

| arm | assigned | completed | D_H among completed | safe success among assigned | ordinary timeout among assigned | solver failure | collision observed |
|---|---:|---:|---:|---:|---:|---:|---:|
| Safety | 64 | 64 | 52/64 (81.25%) | 12/64 (18.75%) | 0/64 | 0/64 | 0/64 |
| phi0_diag | 64 | 56 | 0/56 (0%; 95% CP upper 6.38%) | 39/64 (60.94%) | 17/64 (26.56%) | 8/64 (12.50%) | 0/64 prefixes |

The experimental deadlock rate over all assigned draws is not identified because eight trajectories stopped at solver failure. Assigning all eight missing outcomes as deadlock gives the sensitivity range 0--8/64 (0--12.5%); this does not turn solver failures into valid deadlock labels.

Among 56 valid pairs, 31 baseline deadlocks became collision-free successes, 15 became ordinary timeouts, no baseline non-deadlock became an observed deadlock, and 10 retained the same deadlock label. Six failed experimental trajectories corresponded to baseline deadlocks; two corresponded to baseline successes. Thus at least 31/52 (59.62%) of all baseline-deadlock pairs reached safe success, while 15 merely changed deadlock into timeout.

Mean completed-rollout gate activation was {a['gate_activation_fraction']['mean']:.4f}. The time-weighted raw residual joint-norm RMS was {analysis['exploration_and_effect']['raw_residual_joint_norm_rms_time_weighted_m_per_s']:.4f} m/s; the executed correction RMS was {analysis['exploration_and_effect']['executed_correction_joint_norm_rms_time_weighted_m_per_s']:.4f} m/s. Completed experimental trajectories had mean correction energy {a['executed_correction_energy']['mean']:.4f} m^2/s. No active gate had exactly zero measured executed effect at the declared `1e-10` threshold.

Seven failures were `speed_cut_limit`; one was a rejected QP with minimum residual `-2.3343814659071427e-09`, below the configured feasibility tolerance. No fallback or replacement action was executed. Accepted experimental prefixes had minimum linear residual {analysis['numerical_and_safety']['accepted_prefix_min_linear_residual']:.3e} and maximum speed excess {analysis['numerical_and_safety']['accepted_prefix_max_speed_excess']:.3e} m/s.

The run executed {report['total']['simulator_steps']} simulator actions ({report['total']['equivalent_850_step_rollouts']:.2f} 850-step equivalents) in {report['runtime']['wall_seconds']:.1f} seconds and stored every assigned trajectory, including failed prefixes.

## Interpretation

The fixed stochastic residual has a large empirical effect in this selected scene, including 31 actual task completions from paired Safety deadlocks. The 17 ordinary timeouts are not escapes, and the 8 projection failures are operational failures. This diagnostic did not estimate a score gradient, update parameters, or test learning, so it provides no evidence that Direction A's gradient is repeatable or useful beyond showing that residual exploration can alter outcomes in this scene.

## Reproduction and files

```bash
JAX_PLATFORMS=cpu .venv-c1/bin/python scripts/pilot_giveway_residual_effect.py --out results/giveway_residual_effect_pilot_v1 --prepare-only
sbatch scripts/run_giveway_residual_effect_pilot.sbatch
.venv-c1/bin/python scripts/analyze_giveway_residual_effect.py
```

The repository root has no Git metadata. Source, checkpoint, configuration, seed-schedule, and output SHA256 values are recorded in `manifest.json`, `report.json`, and `analysis.json`.
"""
    (OUT/"AUDIT_REPORT.md").write_text(markdown)
    print(json.dumps({"analysis": str(json_path), "analysis_sha256": digest(json_path),
                      "report": str(OUT/"AUDIT_REPORT.md"),
                      "report_sha256": digest(OUT/"AUDIT_REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()
