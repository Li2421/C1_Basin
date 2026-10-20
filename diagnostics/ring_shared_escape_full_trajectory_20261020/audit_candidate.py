"""Exploratory read-only scan of the separately frozen Ring circulation candidate."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ring_exchange.environment import RingExchangeEnv
from ring_exchange.safety import polygonal_obstacle_snapshot
from shared_control.hard_projection import HardProjectionConfig, barrier_constraints

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PANELS = {
    "cross_type_1c": ROOT / "diagnostics/crosstype1c_two_eta_schedule_v1",
    "cross_type_3": ROOT / "diagnostics/crosstype3_unseen_controller_schedule_v1",
}
SPEC = json.loads((OUT / "CANDIDATE_SPEC.json").read_text())
SPEC_SHA = hashlib.sha256((OUT / "CANDIDATE_SPEC.json").read_bytes()).hexdigest()
II, JJ = np.triu_indices(4, 1)


def state_eligible(p: np.ndarray, g: np.ndarray) -> tuple[bool, float, float, float, float]:
    radius = np.linalg.norm(p, axis=1)
    pair = np.linalg.norm(p[II] - p[JJ], axis=1)
    theta = np.arctan2(p[:, 1], p[:, 0])
    target = np.arctan2(g[:, 1], g[:, 0])
    angle_error = np.abs(np.angle(np.exp(1j * (target - theta - np.pi))))
    return bool(radius.min() >= 2.25 and radius.max() <= 2.75 and pair.min() >= 1.5 and angle_error.max() <= .2), float(radius.min()), float(radius.max()), float(pair.min()), float(angle_error.max())


def verify_initial_reference(states: list[dict]) -> dict:
    config = HardProjectionConfig()
    min_cbf = math.inf
    min_speed_margin = math.inf
    eligible = 0
    for state in states:
        p = np.asarray(state["physical"]["positions"], dtype=np.float64)
        g = np.asarray(state["physical"]["goals"], dtype=np.float64)
        valid, *_ = state_eligible(p, g)
        eligible += int(valid)
        env = RingExchangeEnv()
        env.reset(p, goals=g)
        matrix, lower, geom = barrier_constraints(polygonal_obstacle_snapshot(env, sides=48), config)
        assert matrix.shape == (390, 8) and geom["min_wall_h"] > 0
        u = .1 * np.stack((-p[:, 1], p[:, 0]), axis=1)
        for sign in (-1, 1):
            min_cbf = min(min_cbf, float(np.min(matrix @ (sign * u).reshape(-1) - lower)))
        min_speed_margin = min(min_speed_margin, float(env.config.max_speed - np.linalg.norm(u, axis=1).max()))
    return {"initial_states": len(states), "candidate_eligible_initial_states": eligible,
            "min_both_sign_reference_CBF_slack": min_cbf, "min_reference_speed_margin": min_speed_margin,
            "CBF_rows_per_state": 390}


def longest_run(mask: np.ndarray) -> int:
    if not np.any(mask):
        return 0
    padded = np.concatenate(([False], mask, [False]))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    return int(np.max(ends - starts))


def scan(experiment: str, hist: Path, uid: str, method: str, controller: str, ctype: str, state_uid: str, seed: int, initial_valid: bool) -> dict:
    raw_path = hist / "new_evaluation/raw" / (uid + ".json")
    raw = json.loads(raw_path.read_text())
    case = raw["case"]
    assert case["rollout_uid"] == uid and case["state_uid"] == state_uid and case["base_controller_name"] == controller
    assert case["seed"] == seed
    with np.load(raw["trace_path"], allow_pickle=False) as z:
        p = np.asarray(z["positions"])
        u = np.asarray(z["actions"])
        g = np.asarray(z["goals"])
        swept = np.asarray(z["swept_clearance"])
    assert len(p) == len(u) + 1 == int(raw["physical_steps"]) + 1
    if len(u) == 0:
        return dict(experiment=experiment, method=method, rollout_uid=uid, state_uid=state_uid, controller=controller,
                    controller_type=ctype, placement=case["placement"], seed=seed,
                    outcome=raw.get("outcome", "unknown"), success=int(raw.get("success", 0)),
                    physical_steps=0, initial_domain=int(initial_valid), eligible_steps=0,
                    candidate_action_steps=0, candidate_steps_full_cbf_verified=0,
                    min_candidate_full_cbf_slack="", min_candidate_speed_slack="",
                    candidate_action_fraction="", any_candidate_action=0,
                    longest_candidate_run=0, angular_stage_exit=0, first_angular_exit_step="",
                    all_steps_physically_safe=int(len(swept) == 0), min_swept_clearance="")
    radii = np.linalg.norm(p, axis=2)
    pair_min = np.linalg.norm(p[:, II, :] - p[:, JJ, :], axis=2).min(axis=1)
    domain = (radii.min(axis=1) >= 2.25) & (radii.max(axis=1) <= 2.772) & (pair_min >= 1.5)
    normal = p[:-1] / np.maximum(radii[:-1, :, None], 1e-15)
    tangent = np.stack((-normal[:, :, 1], normal[:, :, 0]), axis=2)
    tangential = np.sum(tangent * u, axis=2)
    radial = np.sum(normal * u, axis=2)
    common_cw = np.all(-tangential >= .05 - 1e-12, axis=1)
    common_ccw = np.all(tangential >= .05 - 1e-12, axis=1)
    candidate = domain[:-1] & (common_cw | common_ccw) & np.all(np.abs(radial) <= .15 + 1e-12, axis=1) & (swept >= -1e-9)
    candidate_indices = np.flatnonzero(candidate)
    min_full_cbf_slack = math.inf
    min_speed_slack = math.inf
    cbf_verified = 0
    if len(candidate_indices):
        env = RingExchangeEnv()
        env.goals = g.copy()
        config = HardProjectionConfig()
        for t in candidate_indices:
            env.positions = p[t].copy()
            matrix, lower, _ = barrier_constraints(polygonal_obstacle_snapshot(env, sides=48), config)
            cbf_slack = float(np.min(matrix @ u[t].reshape(-1) - lower))
            speed_slack = float(env.config.max_speed - np.max(np.linalg.norm(u[t], axis=1)))
            min_full_cbf_slack = min(min_full_cbf_slack, cbf_slack)
            min_speed_slack = min(min_speed_slack, speed_slack)
            cbf_verified += int(cbf_slack >= -1e-7 and speed_slack >= -1e-9)
    angles = np.unwrap(np.arctan2(p[:, :, 1], p[:, :, 0]), axis=0)
    progress = angles - angles[0]
    goal_angle = np.arctan2(g[:, 1], g[:, 0])
    residual = np.abs(np.angle(np.exp(1j * (goal_angle[None, :] - angles))))
    common_exit = (np.all(progress >= np.pi - .21, axis=1) | np.all(-progress >= np.pi - .21, axis=1)) & np.all(residual <= .21, axis=1) & domain
    first_exit = int(np.flatnonzero(common_exit)[0]) if np.any(common_exit) else None
    return dict(experiment=experiment, method=method, rollout_uid=uid, state_uid=state_uid, controller=controller,
                controller_type=ctype, placement=case["placement"], seed=seed,
                outcome=raw.get("outcome", "unknown"), success=int(raw.get("success", 0)),
                physical_steps=len(u), initial_domain=int(initial_valid), eligible_steps=int(domain[:-1].sum()),
                candidate_action_steps=int(candidate.sum()),
                candidate_steps_full_cbf_verified=cbf_verified,
                min_candidate_full_cbf_slack=min_full_cbf_slack if len(candidate_indices) else "",
                min_candidate_speed_slack=min_speed_slack if len(candidate_indices) else "",
                candidate_action_fraction=float(candidate.sum() / max(1, domain[:-1].sum())),
                any_candidate_action=int(np.any(candidate)), longest_candidate_run=longest_run(candidate),
                angular_stage_exit=int(first_exit is not None), first_angular_exit_step=first_exit if first_exit is not None else "",
                all_steps_physically_safe=int(bool(np.all(swept >= -1e-9))),
                min_swept_clearance=float(swept.min()))


def main() -> dict:
    certificates = {}
    rows = []
    for experiment, hist in PANELS.items():
        states = json.loads((hist / "NEW_EVAL_STATES.json").read_text())
        eligible = {s["state_uid"]: state_eligible(np.asarray(s["physical"]["positions"]), np.asarray(s["physical"]["goals"]))[0] for s in states}
        certificates[experiment] = verify_initial_reference(states)
        method_uids = json.loads((hist / "NEW_EVAL_METHOD_UIDS.json").read_text())
        for item in method_uids:
            for method, uid_key in (("self_schedule", "self_schedule"), ("safety_baseline", "all_safety_baseline")):
                ids = item["method_uids"].get(uid_key, [])
                for seed, uid in enumerate(ids):
                    rows.append(scan(experiment, hist, uid, method, item["controller"], item["type"], item["state_uid"], seed, eligible[item["state_uid"]]))
    output = OUT / "CANDIDATE_EXPLORATORY.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = defaultdict(Counter)
    for row in rows:
        key = f"{row['experiment']}|{row['method']}|{row['controller_type']}|{row['placement']}"
        summary[key]["episodes"] += 1
        for field in ("initial_domain", "any_candidate_action", "angular_stage_exit", "success", "all_steps_physically_safe"):
            summary[key][field] += int(row[field])
        summary[key]["candidate_action_steps"] += int(row["candidate_action_steps"])
        summary[key]["candidate_steps_full_cbf_verified"] += int(row["candidate_steps_full_cbf_verified"])
        summary[key]["eligible_steps"] += int(row["eligible_steps"])
    result = {"status": "EXPLORATORY", "candidate_spec_sha256": SPEC_SHA,
              "initial_reference_check": certificates,
              "selected_method_episodes": len(rows),
              "by_method_type": {k: dict(v) for k, v in summary.items()},
              "warnings": ["Action signature and physical swept clearance are measured from frozen traces; full per-step CBF row residuals and precorrection anchor/B are not saved.",
                           "Angular stage exit is a geometric observation, not proof that every preceding action was in the candidate action set.",
                           "Flow placement FF is excluded from TT claims."]}
    (OUT / "CANDIDATE_SUMMARY.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True))
