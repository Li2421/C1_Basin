"""Audit held-out Gap1 post-gate targets and frozen Flow actions.

This is a read-only diagnostic on non-opposing expert trajectories. It neither
changes the nominal controller nor uses opposing Gap1 initial conditions.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import jax
import numpy as np

from diagnostics.double_bottleneck_eta3_basin.tools.exact_projection_retry import CertifiedHardSafetyFilter
from new_benchmark_common.macflow import load_checkpoint, sample_bounded_actions
from shared_control.hard_projection import HardProjectionConfig
from .collect_competence import teacher_reference
from .environment import BottleneckEnv
from .observation import policy_observation_competence
from .scenario import Config


DISTANCE_BINS = {"far": (2.0, np.inf), "intermediate": (0.5, 2.0),
                 "near": (0.12, 0.5), "threshold": (0.08, 0.12)}


def _cosine(action, direction):
    norm = np.linalg.norm(action)
    return float(np.dot(action, direction) / norm) if norm > 1e-10 else 0.0


def _cases(dataset: Path, config: Config, per_bin: int, seed: int):
    rng = np.random.default_rng(seed)
    offset = config.barrier_thickness / 2 + config.agent_radius + config.wall_radius + config.wall_collision_margin + .12
    result = []
    for mode in ("solo", "full"):
        for direction in ("LR", "RL"):
            for label, (lower, upper) in DISTANCE_BINS.items():
                pool = []
                for file in sorted((dataset / "rollouts" / "dev").glob(
                        f"*s701051*_{mode}_{direction}_perm0.npz")):
                    with np.load(file, allow_pickle=False) as data:
                        initial = json.loads(str(data["initial_state_json"].item()))
                        positions = np.asarray(data["states"][:-1])
                        actions = np.asarray(data["actions"])
                        observations = np.asarray(data["observations"][:-1])
                    goals = np.asarray(initial["goals"])
                    initially_active = np.linalg.norm(goals - positions[0], axis=1) > config.goal_tolerance
                    d = np.linalg.norm(goals[None] - positions, axis=2)
                    sign = 1 if direction == "LR" else -1
                    progress = sign * (positions[:, :, 0] - config.barrier_x[0])
                    mask = initially_active[None] & (progress >= offset - .04) & (d > lower) & (d <= upper)
                    times, agents = np.nonzero(mask)
                    if not len(times):
                        continue
                    # Sample across each independent physical trajectory, avoiding
                    # thousands of adjacent near-identical frames.
                    chosen = np.linspace(0, len(times)-1, min(per_bin, len(times)), dtype=int)
                    for index in chosen:
                        t, a = int(times[index]), int(agents[index])
                        pool.append(dict(file=str(file), time=t, agent=a,
                                         positions=positions[t], goals=goals,
                                         velocity=actions[t-1] if t else np.zeros_like(actions[t]),
                                         observation=observations[t], expert_action=actions[t],
                                         bin=label, mode=mode, direction=direction,
                                         episode_seed=int(initial["episode_seed"])))
                if len(pool) > per_bin:
                    selected = rng.choice(len(pool), size=per_bin, replace=False)
                    pool = [pool[int(i)] for i in selected]
                result.extend(pool)
    return result


def run(dataset: Path, checkpoints: list[Path], output: Path, *, per_bin=12, seed=841):
    manifest = json.loads((dataset / "manifest.json").read_text())
    config = Config(**manifest["scenario_config"])
    cases = _cases(dataset, config, per_bin, seed)
    if not cases:
        raise ValueError("no held-out post-gate cases")
    projector = CertifiedHardSafetyFilter(HardProjectionConfig())
    checkpoints_loaded = []
    for path in checkpoints:
        agent, metadata = load_checkpoint(path,
            expected_environment_fingerprint=manifest["environment_fingerprint"])
        checkpoints_loaded.append((path, agent, metadata,
                                   hashlib.sha256(path.read_bytes()).hexdigest()))
    observations = np.asarray([case["observation"] for case in cases], dtype=np.float32)
    predictions = {}
    names = {str(path): path.parent.name for path, _, _, _ in checkpoints_loaded}
    if len(set(names.values())) != len(names):
        raise ValueError("checkpoint parent names must be unique for audit keys")
    for path, agent, _, _ in checkpoints_loaded:
        # Same 10-step Flow sampler, action bound, and one latent per state as
        # the accepted nominal evaluator. No rollout or model selection here.
        predictions[str(path)] = np.asarray(sample_bounded_actions(
            agent, observations, jax.random.PRNGKey(seed)), dtype=np.float64)
    rows = []
    for index, case in enumerate(cases):
        env = BottleneckEnv(replace(config, seed=case["episode_seed"], split="dev"))
        env.reset(case["positions"], case["goals"])
        env.velocities = np.asarray(case["velocity"], dtype=np.float64)
        agent_id = case["agent"]
        delta = env.goals[agent_id] - env.positions[agent_id]
        distance = float(np.linalg.norm(delta))
        goal_direction = delta / distance
        reference = teacher_reference(env)
        stored_observation = case["observation"]
        fresh_observation = policy_observation_competence(env)
        expert = case["expert_action"]
        row = {key:case[key] for key in ("file", "time", "agent", "bin", "mode", "direction")}
        row.update(goal_distance=distance,
                   observation_max_error=float(np.max(np.abs(stored_observation-fresh_observation))),
                   goal_feature_error=float(np.max(np.abs(stored_observation[:, 4:6]-
                                                      (env.goals-env.positions)))),
                   waypoint_goal_error=float(np.linalg.norm(stored_observation[agent_id, 6:8]-delta)),
                   expert_reference_norm=float(np.linalg.norm(reference[agent_id])),
                   expert_reference_cosine=_cosine(reference[agent_id],goal_direction),
                   expert_safe_norm=float(np.linalg.norm(expert[agent_id])),
                   expert_safe_cosine=_cosine(expert[agent_id],goal_direction),
                   expert_safe_goalward_speed=float(np.dot(expert[agent_id],goal_direction)),
                   flow={})
        for path, _, _, _ in checkpoints_loaded:
            flow = predictions[str(path)][index]
            result = projector(env.snapshot(), flow)
            safe = np.asarray(result.velocity)
            active = np.asarray(result.diagnostics["active_linear_constraints"], dtype=int)
            pair_count = int(result.diagnostics["num_pair_constraints"])
            row["flow"][names[str(path)]] = dict(
                flow_norm=float(np.linalg.norm(flow[agent_id])),
                flow_cosine=_cosine(flow[agent_id],goal_direction),
                flow_goalward_speed=float(np.dot(flow[agent_id],goal_direction)),
                safe_norm=float(np.linalg.norm(safe[agent_id])),
                safe_cosine=_cosine(safe[agent_id],goal_direction),
                safe_goalward_speed=float(np.dot(safe[agent_id],goal_direction)),
                correction_norm=float(np.linalg.norm(safe[agent_id]-flow[agent_id])),
                wall_active=bool(np.any(active >= pair_count)),
                pair_active=bool(np.any(active < pair_count)),
                one_step_distance_change=float(np.linalg.norm(delta-config.dt*safe[agent_id])-distance))
        rows.append(row)
    grouped = {}
    for mode in ("solo", "full"):
        for direction in ("LR", "RL"):
            for label in DISTANCE_BINS:
                subset = [r for r in rows if (r["mode"],r["direction"],r["bin"]) == (mode,direction,label)]
                if not subset:
                    continue
                key = f"{mode}_{direction}_{label}"
                grouped[key] = dict(count=len(subset),
                    expert_reference_cosine=float(np.mean([r["expert_reference_cosine"] for r in subset])),
                    expert_safe_cosine=float(np.mean([r["expert_safe_cosine"] for r in subset])),
                    expert_goalward_speed=float(np.mean([r["expert_safe_goalward_speed"] for r in subset])),
                    observation_max_error=float(max(r["observation_max_error"] for r in subset)),
                    waypoint_goal_error=float(max(r["waypoint_goal_error"] for r in subset)),
                    flow={names[str(p)]: {metric:float(np.mean([r["flow"][names[str(p)]][metric] for r in subset]))
                                    for metric in ("flow_cosine", "flow_goalward_speed", "safe_goalward_speed",
                                                   "correction_norm", "wall_active", "pair_active", "one_step_distance_change")}
                          for p in checkpoints})
    report = dict(schema="gap1_n50_heldout_postgate_probe_v1", dataset=str(dataset),
                  checkpoint_sha256={str(p):sha for p, _, _, sha in checkpoints_loaded},
                  seed=seed, per_bin=per_bin, cases=len(rows), grouped=grouped, rows=rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-bin", type=int, default=12)
    parser.add_argument("--seed", type=int, default=841)
    args = parser.parse_args()
    report = run(args.dataset, args.checkpoint, args.output, per_bin=args.per_bin, seed=args.seed)
    print(json.dumps(report["grouped"], indent=2), flush=True)


if __name__ == "__main__":
    main()
