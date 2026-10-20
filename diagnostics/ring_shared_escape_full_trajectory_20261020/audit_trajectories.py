"""Read-only coverage audit of the frozen Ring reference domain.

The reference domain freezes each episode's goals near a radius-2.5 ideal
square. A failed necessary goal-radius condition excludes *every* saved
physical state, so the script checks each complete trace's header and goals
without decompressing its multi-gigabyte position payload. It never calls
env.step, samples a controller, or solves a QP.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCES = [
    ("round27", ROOT / "diagnostics/other_controller_classes_v1/main/raw", ROOT / "diagnostics/other_controller_classes_v1/EVAL_STATES.json"),
    ("round29", ROOT / "diagnostics/nominal_safety_baselines_v1/main/raw", ROOT / "diagnostics/nominal_safety_baselines_v1/protocol.json"),
    ("cross_type_1c", ROOT / "diagnostics/crosstype1c_two_eta_schedule_v1/new_evaluation/raw", ROOT / "diagnostics/crosstype1c_two_eta_schedule_v1/NEW_EVAL_STATES.json"),
    ("cross_type_3", ROOT / "diagnostics/crosstype3_unseen_controller_schedule_v1/new_evaluation/raw", ROOT / "diagnostics/crosstype3_unseen_controller_schedule_v1/NEW_EVAL_STATES.json"),
]


def read_states(path: Path) -> dict:
    obj = json.loads(path.read_text())
    rows = obj["states"] if isinstance(obj, dict) else obj
    return {row["state_uid"]: row for row in rows}


def npy_header(z: ZipFile, name: str) -> tuple:
    with z.open(name + ".npy") as stream:
        version = np.lib.format.read_magic(stream)
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, _, _ = reader(stream)
    return shape


def inspect_trace(path: Path, goals: np.ndarray) -> tuple[int | None, str]:
    if not path.is_file():
        return None, "trace_missing"
    try:
        with ZipFile(path) as z:
            shape = npy_header(z, "positions")
            with z.open("goals.npy") as stream:
                actual_goals = np.load(stream, allow_pickle=False)
            if actual_goals.shape != (4, 2) or not np.array_equal(actual_goals, goals):
                return None, "goal_mismatch"
            if len(shape) != 3 or shape[1:] != (4, 2):
                return None, "position_shape_mismatch"
            return shape[0] - 1, "header_and_goals_verified"
    except Exception as exc:
        return None, "trace_error:" + type(exc).__name__


FIELDS = [
    "experiment", "rollout_uid", "state_uid", "controller_uid", "controller_instance",
    "controller_type", "placement", "arm", "seed_key", "eta_a", "eta_b", "switch_step",
    "outcome", "success", "collision", "timeout", "numerical_failure", "unknown",
    "trace_path", "trace_sha256_archived", "trace_validation", "physical_steps_archived",
    "physical_steps_trace", "saved_state_count", "goal_max_radial_deviation_m",
    "goal_exclusion_margin_m", "initial_in_reference_domain", "ever_in_reference_domain",
    "first_entry_step", "entry_before_terminal", "dwell_steps", "stage_exit_reached",
    "local_coordination_completed", "full_task_success_after_entry", "classification",
]


def main() -> dict:
    output = OUT / "TRAJECTORY_COVERAGE.csv"
    counts = Counter()
    by_experiment = defaultdict(Counter)
    by_type = defaultdict(Counter)
    controllers = defaultdict(set)
    controller_instances = defaultdict(set)
    states = defaultdict(set)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for experiment, raw_dir, state_path in SOURCES:
            state_map = read_states(state_path)
            for raw_path in sorted(raw_dir.glob("*.json")):
                raw = json.loads(raw_path.read_text())
                case = raw["case"]
                uid = case["rollout_uid"]
                state_uid = case["state_uid"]
                if uid != raw_path.stem or state_uid not in state_map:
                    raise ValueError(f"identity mismatch: {raw_path}")
                goals = np.asarray(state_map[state_uid]["physical"]["goals"], dtype=np.float64)
                if "goals" in raw and not np.array_equal(np.asarray(raw["goals"], dtype=np.float64), goals):
                    raise ValueError(f"raw goal mismatch: {raw_path}")
                max_radial = float(np.max(np.abs(np.linalg.norm(goals, axis=1) - 2.5)))
                # For membership, each goal must lie <.05 m from its ideal radius-2.5 position.
                excluded = max_radial >= .05
                trace_path = Path(raw.get("trace_path", ""))
                trace_steps, validation = inspect_trace(trace_path, goals)
                archived_steps = int(raw.get("physical_steps", raw.get("episode_length", -1)))
                if trace_steps is not None and trace_steps != archived_steps:
                    validation = "step_count_mismatch"
                outcome = raw.get("outcome", "unknown")
                placement = case.get("placement", case.get("chain", "TT"))
                if experiment in ("cross_type_1c", "cross_type_3"):
                    ctype = case.get("family", "unknown")
                else:
                    name = case.get("base_controller_name", case.get("controller_name", "unknown"))
                    ctype = "MPC" if name.startswith("M") else "potential" if name.startswith("P") else "Flow" if "flow" in name.lower() else "unknown"
                controller_uid = case["controller_uid"]
                label = "excluded_all_steps_by_fixed_goal_geometry" if excluded else "requires_position_scan"
                if validation != "header_and_goals_verified":
                    label = "trace_unverified:" + validation
                # No position is classified by an absent trace. With a verified trace and
                # fixed goals that fail the frozen necessary condition, every saved time
                # point is mathematically excluded without reading position samples.
                known_zero = excluded and validation == "header_and_goals_verified"
                row = dict(
                    experiment=experiment, rollout_uid=uid, state_uid=state_uid,
                    controller_uid=controller_uid,
                    controller_instance=case.get("base_controller_name", case.get("controller_name", "")),
                    controller_type=ctype, placement=placement, arm=case.get("arm", ""),
                    seed_key=case.get("seed_key", raw.get("seed_key", "")),
                    eta_a=json.dumps(case.get("eta_a"), separators=(",", ":")),
                    eta_b=json.dumps(case.get("eta_b"), separators=(",", ":")),
                    switch_step=case.get("switch_step", ""), outcome=outcome,
                    success=raw.get("success", ""), collision=raw.get("collision", ""),
                    timeout=raw.get("timeout", ""), numerical_failure=raw.get("numerical_failure", ""),
                    unknown=raw.get("unknown", int(bool(raw.get("numerical_failure", 0) or raw.get("error")))),
                    trace_path=str(trace_path.relative_to(ROOT)) if trace_path.is_relative_to(ROOT) else str(trace_path),
                    trace_sha256_archived=raw.get("trace_sha256", ""), trace_validation=validation,
                    physical_steps_archived=archived_steps,
                    physical_steps_trace=trace_steps if trace_steps is not None else "",
                    saved_state_count=trace_steps + 1 if trace_steps is not None else "",
                    goal_max_radial_deviation_m=max_radial,
                    goal_exclusion_margin_m=max_radial - .05,
                    initial_in_reference_domain=0 if known_zero else "",
                    ever_in_reference_domain=0 if known_zero else "",
                    first_entry_step="", entry_before_terminal=0 if known_zero else "",
                    dwell_steps=0 if known_zero else "", stage_exit_reached=0 if known_zero else "",
                    local_coordination_completed="",
                    full_task_success_after_entry=0 if known_zero else "", classification=label,
                )
                writer.writerow(row)
                counts["episodes"] += 1
                by_experiment[experiment]["episodes"] += 1
                by_type[(experiment, ctype, placement)]["episodes"] += 1
                controllers[experiment].add(controller_uid)
                controller_instances[experiment].add(row["controller_instance"])
                states[experiment].add(state_uid)
                if known_zero:
                    counts["certified_zero_episode_visits"] += 1
                    counts["certified_zero_state_timepoints"] += trace_steps + 1
                    by_experiment[experiment]["certified_zero_episode_visits"] += 1
                    by_experiment[experiment]["saved_state_timepoints"] += trace_steps + 1
                    by_type[(experiment, ctype, placement)]["certified_zero_episode_visits"] += 1
                else:
                    counts["unverified_or_not_excluded"] += 1
                for flag in ("success", "collision", "timeout", "numerical_failure"):
                    by_experiment[experiment][flag] += int(raw.get(flag, 0))
                    by_type[(experiment, ctype, placement)][flag] += int(raw.get(flag, 0))
    summary = {
        "schema": "ring_frozen_reference_full_trajectory_coverage_v1",
        "reference_source": "diagnostics/shared_escape_cross_controller_audit_20261019/THEOREM_AUDIT.md",
        "necessary_condition": "each fixed goal radius lies strictly within .05 m of 2.5 m",
        "counts": dict(counts),
        "by_experiment": {k: {**dict(v), "execution_profile_uids": len(controllers[k]),
                              "base_controller_instances": len(controller_instances[k]),
                              "state_uids": len(states[k])} for k, v in by_experiment.items()},
        "by_type": {"|".join(k): dict(v) for k, v in by_type.items()},
        "source_counts": {exp: len(list(raw.glob("*.json"))) for exp, raw, _ in SOURCES},
    }
    (OUT / "COVERAGE_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True))
