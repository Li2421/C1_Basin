"""Diagnose whether the existing all-moving one-way expert finishes Gap1 N=50.

This does not train or alter the deployed controller. Unlike the historical
serial queue demonstrator, every pending same-direction agent receives its
ordinary goal/waypoint command before the accepted hard safety projection.
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

from .collect_competence import teacher_reference
from .environment import BottleneckEnv
from .evaluate_safety_audit import control_state
from .scenario import Config


def run(dataset: Path, output: Path, *, mode: str, fresh_seed: int, count: int):
    if mode not in ("one_way", "one_way_mirror") or count < 1:
        raise ValueError("only non-opposing full one-way modes are supported")
    dataset, output = Path(dataset), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True)
    manifest = json.loads((dataset / "manifest.json").read_text())
    config = Config(**manifest["scenario_config"])
    if config.num_agents != 50:
        raise ValueError("N=50 dataset required")
    rng = np.random.default_rng(fresh_seed)
    cases, requests = [], []
    for index in range(count):
        episode_seed = int(rng.integers(0, 2**31 - 1))
        base = BottleneckEnv(replace(config, seed=episode_seed, split="dev"))
        state = {"positions": base.positions, "goals": base.goals}
        p, g = control_state(state, mode)
        cases.append((index, episode_seed, p, g))
        requests.append(dict(state_uid=uid("state", {
            "physical_fingerprint": config.physical_fingerprint,
            "positions": p.tolist(), "goals": g.tolist(),
            "episode_seed": episode_seed, "mode": mode}),
            eta_uid=eta_identity((0., 0., 0.))[0],
            controller_uid=uid("ctl", {
                "controller": "existing_allmoving_one_way_teacher_plus_accepted_safety_v1",
                "safety": HardProjectionConfig().to_dict()}),
            seed_keys=[str(episode_seed)]))
    plan = output / "planned_rollouts.json"
    plan.write_text(json.dumps({"requests": requests}, indent=2) + "\n")
    cache = preflight(plan)
    (output / "cache_preflight.json").write_text(json.dumps(cache, indent=2) + "\n")
    if cache["summary"]["ambiguous"] or cache["summary"]["genuinely_missing"] != count:
        raise RuntimeError("preflight requires cache review/reuse")
    projector = CertifiedHardSafetyFilter(HardProjectionConfig())
    rows = []
    for index, episode_seed, p, g in cases:
        env = BottleneckEnv(replace(config, seed=episode_seed, split="dev"))
        env.reset(p, g)
        positions = [env.positions.copy()]
        active_agent_steps = 0
        wall_active = pair_active = 0
        correction = 0.
        clearance = float("inf")
        for _ in range(config.max_steps):
            reference = teacher_reference(env)
            active_agent_steps += int(np.sum(np.linalg.norm(reference, axis=1) > .1))
            result = projector(env.snapshot(), reference)
            safe = np.asarray(result.velocity)
            diagnostics = result.diagnostics
            active = np.asarray(diagnostics["active_linear_constraints"], dtype=int)
            pair_count = int(diagnostics["num_pair_constraints"])
            pair_active += int(np.any(active < pair_count))
            wall_active += int(np.any(active >= pair_count))
            correction += float(np.linalg.norm(safe - reference))
            _, done, info = env.step(safe, diagnose_stalls=False)
            clearance = min(clearance, info["swept_clearance"])
            positions.append(env.positions.copy())
            if done:
                break
        states = np.asarray(positions)
        distance = np.linalg.norm(env.positions - env.goals, axis=1)
        direction = np.sign(env.goals[:, 0])
        crossed = np.any(states[:, :, 0] * direction[None] > .55, axis=0)
        row = dict(index=index, episode_seed=episode_seed, mode=mode,
                   success=env.termination == "success" and not env.collided,
                   termination=env.termination, collision=bool(env.collided),
                   steps=env.step_count, final_at_goal=int(np.sum(distance <= config.goal_tolerance)),
                   crossed_gate=int(np.sum(crossed)),
                   active_agent_steps=active_agent_steps,
                   active_agent_fraction=active_agent_steps / max(1, env.step_count * 50),
                   wall_active_fraction=wall_active / max(1, env.step_count),
                   pair_active_fraction=pair_active / max(1, env.step_count),
                   mean_projection_norm=correction / max(1, env.step_count),
                   min_swept_clearance=clearance)
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = dict(schema="gap1_n50_allmoving_one_way_expert_probe_v1",
                  dataset_manifest_sha256=hashlib.sha256((dataset / "manifest.json").read_bytes()).hexdigest(),
                  fresh_seed=fresh_seed, mode=mode, preflight=cache["summary"], rows=rows)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("one_way", "one_way_mirror"), required=True)
    parser.add_argument("--fresh-seed", type=int, required=True)
    parser.add_argument("--count", type=int, default=2)
    args = parser.parse_args()
    run(args.dataset, args.output, mode=args.mode, fresh_seed=args.fresh_seed, count=args.count)


if __name__ == "__main__":
    main()
