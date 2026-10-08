"""Collect mirrored, train-only Gap1 terminal occupancy recoveries.

The perturbation uses completed one-way expert states, never an opposing
rollout. Five already-arrived agents are displaced a short, collision-free
distance from their own final goals, then the unchanged goal waypoint expert
and accepted safety projection must complete all 50 goals. Rejections and
expert failures remain visible in the report.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np

from diagnostics.double_bottleneck_eta3_basin.tools.exact_projection_retry import CertifiedHardSafetyFilter
from new_benchmark_common.dataset import DatasetWriter, RecoveryAudit, Trajectory
from shared_control.hard_projection import HardProjectionConfig
from shared_rollout_db.src.rollout_db import eta_identity, uid
from shared_rollout_db.src.planner import preflight
from .collect_scale_dagger import replay_transform, state_queue_rollout
from .environment import BottleneckEnv
from .flow_dataset import GapFlowScenario
from .scenario import Config


def _candidate(config, states, goals, *, seed, pending_count=5,
               minimum_goal_occupancy=None, max_displacement=.32):
    """Use a fixed seeded rejection sampler; no model-result feedback."""
    distance = np.linalg.norm(states - goals[None], axis=2)
    occupancy = (distance <= config.goal_tolerance).sum(axis=1)
    crossed = (states[:, :, 0] > .8).all(axis=1)
    if minimum_goal_occupancy is None:
        minimum_goal_occupancy = config.num_agents - 5
    anchors = np.flatnonzero((occupancy >= minimum_goal_occupancy) & crossed)
    if not len(anchors):
        return None, "no_train_terminal_anchor"
    # Distinct terminal snapshots across perturbation variants, but all
    # chosen solely from successful train expert data.
    anchor = int(anchors[min(seed % 3, len(anchors)-1)])
    base = states[anchor].copy()
    already = np.flatnonzero(distance[anchor] <= config.goal_tolerance)
    env = BottleneckEnv(config)
    rng = np.random.default_rng(seed)
    for attempt in range(100):
        p = base.copy()
        selected = rng.choice(already, size=pending_count, replace=False)
        valid = True
        for i in selected:
            placed = False
            for _ in range(100):
                angle = rng.uniform(-np.pi, np.pi)
                length = rng.uniform(.14, max_displacement)
                candidate_position = goals[i] + length*np.array([np.cos(angle), np.sin(angle)])
                if not env.instance.geometry.valid_points(candidate_position[None]).all():
                    continue
                separation = np.linalg.norm(p[np.arange(len(p)) != i]-candidate_position,axis=1)
                if np.min(separation) <= 2*config.agent_radius + .005:
                    continue
                p[i] = candidate_position
                placed = True
                break
            if not placed:
                valid = False
                break
        if not valid:
            continue
        return (anchor, p, selected.tolist(), attempt), None
    return None, "no_valid_displacement_after_100_attempts"


def collect(source: Path, output: Path, *, variants=8, seed=638270, pending_count=5,
            minimum_goal_occupancy=None, max_displacement=.32):
    source, output = Path(source), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True)
    manifest = json.loads((source / "manifest.json").read_text())
    config = Config(**manifest["scenario_config"])
    if (config.num_agents not in (20, 50)
            or not 1 <= pending_count <= config.num_agents // 2
            or not .14 < max_displacement <= 2.0):
        raise ValueError("N=20/50 source, valid pending count and displacement required")
    if minimum_goal_occupancy is None:
        minimum_goal_occupancy = config.num_agents - 5
    if not pending_count <= minimum_goal_occupancy <= config.num_agents:
        raise ValueError("minimum goal occupancy outside valid range")
    rows = [r for r in manifest["files"] if r["split"] == "train"
            and r["source"] == "nominal" and "_full_LR_perm0" in r["rollout_id"]]
    if not rows:
        raise ValueError("no full L->R train trajectories")
    candidates, rejected = [], []
    for row in rows:
        with np.load(source / row["file"], allow_pickle=False) as data:
            states = np.asarray(data["states"], dtype=np.float64)
            initial = json.loads(str(data["initial_state_json"].item()))
        goals = np.asarray(initial["goals"], dtype=np.float64)
        episode_config = replace(config, seed=int(initial["episode_seed"]), split="train")
        for variant in range(variants):
            perturb_seed = int(np.random.SeedSequence([seed, episode_config.seed,
                                                       variant, pending_count])
                               .generate_state(1)[0])
            candidate, reason = _candidate(episode_config, states, goals, seed=perturb_seed,
                                           pending_count=pending_count,
                                           minimum_goal_occupancy=minimum_goal_occupancy,
                                           max_displacement=max_displacement)
            if candidate is None:
                rejected.append(dict(parent=row["rollout_id"], variant=variant, reason=reason))
                continue
            anchor, positions, selected, attempt = candidate
            name = f"{row['rollout_id']}_terminal_p{pending_count}_v{variant:02d}"
            candidates.append(dict(name=name, parent=row["rollout_id"], config=episode_config,
                                   anchor=anchor, positions=positions, goals=goals,
                                   selected=selected, attempt=attempt, seed=perturb_seed))
    if not candidates:
        raise ValueError("no geometrically valid terminal candidates")
    requests = [dict(state_uid=uid("state", {
        "physical_fingerprint":config.physical_fingerprint,
        "positions":row["positions"].tolist(), "goals":row["goals"].tolist(),
        "episode_seed":row["config"].seed, "parent":row["parent"],
        "anchor":row["anchor"], "perturbation_seed":row["seed"]}),
        eta_uid=eta_identity((0., 0., 0.))[0],
        controller_uid=uid("ctl", {"controller":"same_direction_goal_waypoint_plus_accepted_safety",
                                  "horizon":config.max_steps}),
        seed_keys=[str(row["config"].seed)]) for row in candidates]
    plan = output / "planned_rollouts.json"
    plan.write_text(json.dumps({"requests":requests}, indent=2)+"\n")
    cache = preflight(plan)
    (output / "cache_preflight.json").write_text(json.dumps(cache, indent=2)+"\n")
    if cache["summary"]["ambiguous"] or cache["summary"]["genuinely_missing"] != len(candidates):
        raise RuntimeError("terminal recovery cache requires review/reuse")
    writer = DatasetWriter(output / "dataset", GapFlowScenario(config), scenario_config=asdict(config))
    projector = CertifiedHardSafetyFilter(HardProjectionConfig())
    accepted = {direction:dict(trajectories=0, transitions=0) for direction in ("LR", "RL")}
    failures = []
    for row in candidates:
        p, g, episode_config = row["positions"], row["goals"], row["config"]
        moving = np.flatnonzero(np.linalg.norm(p-g, axis=1) > config.goal_tolerance)
        result = state_queue_rollout(episode_config, p, np.zeros_like(p), g, moving, projector)
        if not result["success"]:
            failures.append(dict(name=row["name"], termination=result["termination"],
                                 reached=int(np.sum(np.linalg.norm(result["states"][-1]-g, axis=1)
                                                    <=config.goal_tolerance))))
            print(json.dumps({"case":row["name"],"success":False}), flush=True)
            continue
        for mirror in (False, True):
            direction = "RL" if mirror else "LR"
            initial, velocity, goals, states, obs, actions, clearance = replay_transform(
                episode_config, p, np.zeros_like(p), g, result["actions"], mirror=mirror)
            writer.add(Trajectory(f"{row['name']}_{direction}", "train",
                "terminal_multi_recovery",
                {"positions":initial,"velocities":velocity,"goals":goals,
                 "episode_seed":episode_config.seed,"split":"train",
                 "mode":"nonopposing_terminal_multi_recovery","direction":direction,
                 "parent":row["parent"],"source_time":row["anchor"],
                 "perturbed_agents":row["selected"]},
                states,obs,actions,
                {"success":True,"terminal_reason":"success","direction":direction,
                 "teacher":"goal_waypoint_plus_accepted_hard_safety",
                 "min_swept_clearance":clearance,"perturbation_attempt":row["attempt"]},
                RecoveryAudit(row["parent"],"train",row["anchor"],None,None,None,
                              row["seed"],True)))
            accepted[direction]["trajectories"] += 1
            accepted[direction]["transitions"] += len(actions)
        print(json.dumps({"case":row["name"],"success":True,
                          "steps":len(result["actions"])}), flush=True)
    if accepted["LR"] != accepted["RL"]:
        raise RuntimeError("directional imbalance")
    report = dict(schema=f"n{config.num_agents}_train_only_mirrored_terminal_multi_recovery_v1",
        source_dataset=str(source), source_manifest_sha256=hashlib.sha256(
            (source/"manifest.json").read_bytes()).hexdigest(),
        source_label="terminal_multi_recovery",seed=seed,variants=variants,
        perturbed_agent_count=pending_count,
        minimum_goal_occupancy=minimum_goal_occupancy,
        max_displacement=max_displacement,
        source_trajectories=len(rows),candidates=len(candidates),
        rejected_geometry=rejected,teacher_failures=failures,
        accepted=accepted,preflight=cache["summary"],
        opposing_demonstrations=0)
    writer.finalize(extra_report=report)
    (output/"collection_report.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--variants",type=int,default=8)
    parser.add_argument("--seed",type=int,default=638270)
    parser.add_argument("--pending-count",type=int,default=5)
    parser.add_argument("--minimum-goal-occupancy",type=int)
    parser.add_argument("--max-displacement",type=float,default=.32)
    args=parser.parse_args()
    print(json.dumps(collect(args.source,args.output,variants=args.variants,seed=args.seed,
                             pending_count=args.pending_count,
                             minimum_goal_occupancy=args.minimum_goal_occupancy,
                             max_displacement=args.max_displacement),
                     indent=2),flush=True)


if __name__ == "__main__":
    main()
