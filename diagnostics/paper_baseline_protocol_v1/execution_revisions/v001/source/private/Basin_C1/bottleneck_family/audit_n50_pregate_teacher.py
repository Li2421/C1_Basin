"""Check the accepted one-way expert from train-only Flow gate-congestion states.

This is a feasibility diagnostic, never an opposing-flow demonstration or an
automatic source for training. The safety projection and simulator are reused.
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
from .scenario import Config


def run(dataset: Path, traces: list[Path], output: Path, *, anchor: int, horizon: int):
    dataset, output = Path(dataset), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True)
    manifest = json.loads((dataset / "manifest.json").read_text())
    config = Config(**manifest["scenario_config"])
    cases, requests = [], []
    for trace in map(Path, traces):
        with np.load(trace, allow_pickle=False) as data:
            p = np.asarray(data["positions"])
            v = np.asarray(data["velocities"])
            g = np.asarray(data["goals"])
            meta = json.loads(str(data["metadata_json"].item()))
        if (meta["split"] != "train" or meta["category"] != "full_LR"
                or not 0 < anchor < len(p) - 1):
            raise ValueError(f"invalid train-only pre-gate trace or anchor: {trace}")
        source_hash = hashlib.sha256(trace.read_bytes()).hexdigest()
        cases.append((trace, source_hash, meta, p[anchor], v[anchor], g))
        requests.append(dict(state_uid=uid("state", {
            "physical_fingerprint": config.physical_fingerprint,
            "positions": p[anchor].tolist(), "velocity": v[anchor].tolist(),
            "goals": g.tolist(), "source_trace_sha256": source_hash,
            "anchor": anchor}), eta_uid=eta_identity((0., 0., 0.))[0],
            controller_uid=uid("ctl", {
                "controller": "existing_same_direction_expert_plus_accepted_safety_v1",
                "horizon": horizon, "safety": HardProjectionConfig().to_dict()}),
            seed_keys=[str(meta["episode_seed"])]))
    plan = output / "planned_rollouts.json"
    plan.write_text(json.dumps({"requests": requests}, indent=2) + "\n")
    cache = preflight(plan)
    (output / "cache_preflight.json").write_text(json.dumps(cache, indent=2) + "\n")
    if cache["summary"]["ambiguous"] or cache["summary"]["genuinely_missing"] != len(cases):
        raise RuntimeError("preflight requires cache review/reuse")
    projector = CertifiedHardSafetyFilter(HardProjectionConfig())
    rows = []
    for trace, source_hash, meta, p, v, g in cases:
        env = BottleneckEnv(replace(config, seed=int(meta["episode_seed"]), split="train"))
        env.reset(p, g)
        env.velocities = v.copy()
        clearance = float("inf")
        for _ in range(min(horizon, config.max_steps)):
            reference = teacher_reference(env)
            safe = np.asarray(projector(env.snapshot(), reference).velocity)
            _, done, info = env.step(safe, diagnose_stalls=False)
            clearance = min(clearance, info["swept_clearance"])
            if done:
                break
        distance = np.linalg.norm(env.positions - env.goals, axis=1)
        row = dict(trace=str(trace), trace_sha256=source_hash, anchor=anchor,
                   horizon=horizon, steps=env.step_count,
                   termination=env.termination, success=env.termination == "success" and not env.collided,
                   collision=bool(env.collided), final_at_goal=int(np.sum(distance <= config.goal_tolerance)),
                   min_swept_clearance=clearance)
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = dict(schema="n50_pregate_existing_expert_feasibility_v1", rows=rows,
                  preflight=cache["summary"])
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--trace", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchor", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=12000)
    args = parser.parse_args()
    run(args.dataset, args.trace, args.output, anchor=args.anchor, horizon=args.horizon)


if __name__ == "__main__":
    main()
