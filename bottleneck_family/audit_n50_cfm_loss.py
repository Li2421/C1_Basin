"""Estimate phase contributions to the *actual* frozen CFM objective.

The estimate samples joint frames from each original training trajectory with
known inverse-probability weights. It recreates the checkpoint's uniform,
early and near-goal frame mixture, and evaluates its squared CFM residual.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from new_benchmark_common.macflow import load_checkpoint
from .scenario import Config


PHASES = ("initial_passive", "approach", "entry", "crossing", "postgate_far", "near_goal", "settled")
STRATA = ("uniform", "early", "near_frame")


def _phase(observation, initially_active, config):
    position = observation[:, :2]
    goal = position + observation[:, 4:6]
    distance = np.linalg.norm(observation[:, 4:6], axis=1)
    direction = np.where(goal[:, 0] > config.barrier_x[0], 1., -1.)
    progress = (position[:, 0] - config.barrier_x[0]) * direction
    offset = config.barrier_thickness/2 + config.agent_radius + config.wall_radius + config.wall_collision_margin + .12
    result = np.full(len(position), -1, dtype=np.int8)
    result[~initially_active] = 0
    live = initially_active & (distance > config.goal_tolerance)
    result[live & (progress < -offset-.04)] = 1
    result[live & (progress >= -offset-.04) & (progress < -.3)] = 2
    result[live & (progress >= -.3) & (progress < offset-.04)] = 3
    result[live & (progress >= offset-.04) & (distance > 1.)] = 4
    result[live & (progress >= offset-.04) & (distance <= 1.)] = 5
    result[initially_active & (distance <= config.goal_tolerance)] = 6
    if (result < 0).any():
        raise AssertionError("unclassified phase")
    return result


def _sample(dataset: Path, config, *, split, samples_per_file, seed):
    manifest = json.loads((dataset / "manifest.json").read_text())
    rng = np.random.default_rng(seed)
    records = {stratum: [] for stratum in STRATA}
    population = {stratum: 0 for stratum in STRATA}
    for row in manifest["files"]:
        if row["split"] != split:
            continue
        with np.load(dataset / row["file"], allow_pickle=False) as archive:
            initial = json.loads(str(archive["initial_state_json"].item()))
            observations = np.asarray(archive["observations"][:-1])
            actions = np.asarray(archive["actions"])
        active = np.linalg.norm(np.asarray(initial["goals"])-np.asarray(initial["positions"]),axis=1) > config.goal_tolerance
        near = np.any((np.linalg.norm(observations[:, :, 4:6], axis=2) > config.goal_tolerance) &
                      (np.linalg.norm(observations[:, :, 4:6], axis=2) < 1.), axis=1)
        choices = {"uniform": np.arange(len(actions)),
                   "early": np.arange(min(50, len(actions))),
                   "near_frame": np.flatnonzero(near)}
        for stratum, available in choices.items():
            population[stratum] += len(available)
            if len(available) == 0:
                continue
            count = min(samples_per_file, len(available))
            selected = rng.choice(available, size=count, replace=False)
            importance = len(available) / count
            for index in selected:
                index = int(index)
                records[stratum].append((observations[index].copy(), actions[index].copy(),
                                         _phase(observations[index], active, config), importance))
    return records, population


def _residual(agent, entries, *, seed):
    errors = []
    for start in range(0, len(entries), 64):
        chunk = entries[start:start+64]
        observations = jnp.asarray(np.asarray([r[0] for r in chunk]))
        actions = jnp.asarray(np.asarray([r[1] for r in chunk]))
        obs_flat, x_t, time, velocity = agent.flow_bc_samples(
            {"observations":observations,"actions":actions},
            jax.random.fold_in(jax.random.PRNGKey(seed),start))
        prediction = agent.network.select("actor_bc_flow")(
            obs_flat, x_t, time, params=agent.network.params)
        error = np.asarray(jnp.mean((prediction-velocity).reshape(
            len(chunk), agent.config["num_agents"], agent.config["act_dim"])**2,axis=-1))
        errors.extend(error)
    return np.asarray(errors)


def run(dataset, checkpoints, output, *, split="train", samples_per_file=8, seed=845):
    dataset, output = Path(dataset), Path(output)
    manifest = json.loads((dataset/"manifest.json").read_text())
    config = Config(**manifest["scenario_config"])
    records, population = _sample(dataset, config, split=split,
                                   samples_per_file=samples_per_file, seed=seed)
    results = {}
    for path in map(Path, checkpoints):
        agent, _ = load_checkpoint(path,
            expected_environment_fingerprint=manifest["environment_fingerprint"])
        training = json.loads((path.parent/"config.json").read_text())
        mixture = {"early":training["early_transition_fraction"],
                   "near_frame":training["near_goal_fraction"]}
        mixture["uniform"] = 1. - sum(mixture.values())
        phase_numerator = np.zeros(len(PHASES))
        phase_weight_mass = np.zeros(len(PHASES))
        for stratum in STRATA:
            if mixture[stratum] == 0:
                continue
            entries = records[stratum]
            errors = _residual(agent, entries, seed=seed+STRATA.index(stratum))
            actions = np.asarray([r[1] for r in entries])
            phases = np.asarray([r[2] for r in entries])
            frame_importance = np.asarray([r[3] for r in entries])
            weight = 1. + (float(agent.config["motion_loss_weight"])-1.) * (np.linalg.norm(actions,axis=2)>.1)
            for index in range(len(PHASES)):
                mask = phases==index
                factor = mixture[stratum] * frame_importance[:,None] / population[stratum]
                phase_numerator[index] += float(np.sum(errors*weight*mask*factor))
                phase_weight_mass[index] += float(np.sum(weight*mask*factor))
        total = float(phase_numerator.sum())
        key = path.parent.name
        results[key] = dict(checkpoint=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            mixture=mixture, estimated_cfm_loss=float(total/phase_weight_mass.sum()),
                            phase={phase:dict(loss_fraction=float(phase_numerator[i]/total),
                                              weight_fraction=float(phase_weight_mass[i]/phase_weight_mass.sum()),
                                              conditional_loss=float(phase_numerator[i]/phase_weight_mass[i]))
                                   for i,phase in enumerate(PHASES)})
    report = dict(schema="n50_phase_cfm_loss_estimate_v1", split=split, seed=seed,
                  samples_per_file=samples_per_file, sampled_frames={k:len(v) for k,v in records.items()},
                  population_frames=population, results=results,
                  caveat="Monte Carlo estimate under each checkpoint's train sampler; fixed noise, not a full exact epoch")
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--checkpoint",type=Path,action="append",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--split",choices=("train","dev"),default="train")
    parser.add_argument("--samples-per-file",type=int,default=8)
    parser.add_argument("--seed",type=int,default=845)
    args=parser.parse_args()
    print(json.dumps(run(args.dataset,args.checkpoint,args.output,split=args.split,
                         samples_per_file=args.samples_per_file,seed=args.seed),indent=2),flush=True)


if __name__ == "__main__":
    main()
