"""Check held-out temporal-control failures with the existing one-way expert.

Diagnostic only: these DEV states are never written as training trajectories.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from diagnostics.double_bottleneck_eta3_basin.tools.exact_projection_retry import CertifiedHardSafetyFilter
from shared_control.hard_projection import HardProjectionConfig
from shared_rollout_db.src.planner import preflight
from shared_rollout_db.src.rollout_db import eta_identity, uid

from .collect_scale_dagger import state_queue_rollout
from .collect_competence import teacher_reference
from .environment import BottleneckEnv
from .scenario import Config


def run(traces: list[Path], output: Path, *, anchor=30000):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True)
    items = []
    requests = []
    for trace in traces:
        trace = Path(trace)
        with np.load(trace, allow_pickle=False) as data:
            positions = np.asarray(data["positions"])
            goals = np.asarray(data["goals"])
            velocity = np.asarray(data["u_safe"])
            meta = json.loads(str(data["metadata_json"].item()))
        if not meta["goal_release"]:
            raise ValueError(f"not a temporal-release trace: {trace}")
        pre_release = meta["phase_transition_step"] is None or anchor <= meta["phase_transition_step"]
        if anchor < 1 or anchor >= len(positions)-1:
            raise ValueError(f"anchor not in saved trace: {trace}")
        config = Config(**meta["config"])
        mode = meta["control_mode"]
        second = np.arange(1 if mode.endswith("even_first") else 0, config.num_agents, 2)
        first = np.setdiff1d(np.arange(config.num_agents), second)
        p, v, g = positions[anchor].copy(), velocity[anchor-1].copy(), goals.copy()
        reflected = (not mode.endswith("even_first")) if pre_release else mode.endswith("even_first")
        if reflected:
            p[:, 0] *= -1
            v[:, 0] *= -1
            g[:, 0] *= -1
        items.append((trace, meta, config, p, v, g, first, second, reflected, pre_release))
        requests.append(dict(state_uid=uid("state", {
            "physical_fingerprint": config.physical_fingerprint,
            "positions": positions[anchor].tolist(),
            "velocity": velocity[anchor-1].tolist(),
            "goals": goals.tolist(),
            "source_trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
            "anchor": anchor}), eta_uid=eta_identity((0., 0., 0.))[0],
            controller_uid=uid("ctl", {"controller":
                "held_other_group_waypoint_teacher_v1" if pre_release else
                "state_only_same_direction_queue_teacher_v1",
                "safety": HardProjectionConfig().to_dict()}),
            seed_keys=[str(meta["seed"])]))
    plan = output/"planned_rollouts.json"
    plan.write_text(json.dumps({"requests": requests}, indent=2)+"\n")
    cache = preflight(plan)
    (output/"cache_preflight.json").write_text(json.dumps(cache, indent=2)+"\n")
    if cache["summary"]["ambiguous"] or cache["summary"]["genuinely_missing"] != len(items):
        raise RuntimeError("temporal feasibility preflight requires cache review")
    projector = CertifiedHardSafetyFilter(HardProjectionConfig())
    rows = []
    for trace, meta, config, p, v, g, first, second, reflected, pre_release in items:
        if pre_release:
            env = BottleneckEnv(replace(config, seed=int(meta["seed"]), split="dev"))
            env.reset(p, g)
            env.velocities = v
            clearance = float("inf")
            steps = 0
            for _ in range(config.max_steps):
                if np.all(np.linalg.norm(env.positions[first]-env.goals[first], axis=1)
                          <= config.goal_tolerance):
                    break
                reference = teacher_reference(env)
                reference[second] = 0
                safe = np.asarray(projector(env.snapshot(), reference).velocity)
                _, done, info = env.step(safe, diagnose_stalls=False)
                clearance = min(clearance, info["swept_clearance"])
                steps += 1
                if done:
                    break
            distance = np.linalg.norm(env.positions-env.goals, axis=1)
            success = bool(np.all(distance[first] <= config.goal_tolerance) and not env.collided)
            termination = "first_group_at_goal" if success else env.termination
        else:
            result = state_queue_rollout(replace(config, seed=int(meta["seed"]), split="dev"),
                                         p, v, g, second, projector)
            distance = np.linalg.norm(result["states"][-1]-g, axis=1)
            success, termination = result["success"], result["termination"]
            steps, clearance = len(result["actions"]), result["min_swept_clearance"]
        row = dict(trace=str(trace), source_mode=meta["control_mode"], anchor=anchor,
                   reflected_to_teacher_frame=reflected,
                   success_scope="first_group" if pre_release else "all_agents",
                   success=success, termination=termination,
                   steps=steps,
                   final_at_goal=int(np.sum(distance <= config.goal_tolerance)),
                   min_swept_clearance=clearance)
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = dict(schema="gap1_n50_temporal_feasibility_v1", rows=rows,
                  preflight=cache["summary"])
    (output/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchor", type=int, default=30000)
    args = parser.parse_args()
    run(args.trace, args.output, anchor=args.anchor)


if __name__ == "__main__":
    main()
