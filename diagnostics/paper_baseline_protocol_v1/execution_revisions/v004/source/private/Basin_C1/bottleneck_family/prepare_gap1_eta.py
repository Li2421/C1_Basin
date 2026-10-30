"""Freeze state splits and a shared global eta pool for Gap1 scaling data.

This preparation is outcome-blind. No Flow, eta rollout, critic or generator is
executed here. The same 3-D Sobol pool and physical-state counts apply to all N.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from new_benchmark_common.safety_eta3 import (
    DOMAIN_HIGH, DOMAIN_LOW, FUTURE_ROOT, ROBUST_PROTOCOL, STANDARD_SEEDS,
)
from shared_control.basis_families import get_basis_family
from shared_control.hard_projection import HardProjectionConfig
from shared_rollout_db.src.planner import preflight
from shared_rollout_db.src.rollout_db import (
    canonical, connect, eta_identity, initialize, uid,
)

from .environment import BottleneckEnv
from .evaluate_safety_audit import SPECS
from .scenario import Config


ROOT = Path("diagnostics/gap_flow_v1")
CHECKPOINTS = {
    2: Path("diagnostics/gap_flow_competence_v3_recovery/frozen/gap1_n2_macflow_competence_v1.pkl"),
    10: Path("diagnostics/gap_flow_scale_20261007/frozen/gap1_n10_flow_navigation_v1.pkl"),
    20: Path("diagnostics/gap_flow_scale_20261007/frozen/gap1_n20_flow_navigation_v1.pkl"),
}
SPLIT_COUNTS = {"train": 32, "val": 8, "test": 8}
PHYSICAL_SPLIT = {"train": "train", "val": "dev", "test": "test"}
MASTER_SEED = 20261009
SOBOL_SEED = 20261001
ETA_COUNT = 16


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def register(output: Path) -> dict:
    """Register frozen exact identities before the shared-cache preflight."""
    design_path = output / "design_manifest.json"
    design = json.loads(design_path.read_text())
    experiment_uid = uid("exp", {"path": str(output.resolve())})
    initialize()
    with connect() as con:
        con.execute(
            "INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,metadata_json) VALUES(?,?,?,?,?,?)",
            (experiment_uid, "gap1_eta_scaling_v1", str(output.resolve()),
             sha(design_path), sha(Path(__file__)),
             canonical({"state_eta_split": "physical", "future_seeds": list(STANDARD_SEEDS)})),
        )
        for n_text, spec in design["scenario_info"].items():
            n = int(n_text)
            con.execute(
                "INSERT OR IGNORE INTO scenario(scenario_uid,name,code_config_fingerprint,metadata_json) VALUES(?,?,?,?)",
                (spec["scenario_uid"], f"Gap1_N{n}", spec["physical_fingerprint"],
                 canonical({"source_manifest": spec["source_manifest"],
                            "geometry": spec["config"]["openings"]})),
            )
            ctl = spec["controller"]
            con.execute(
                """INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,
                flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,
                success_semantics_version,conditioning_version,rng_semantics_version,
                config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (spec["controller_uid"], spec["scenario_uid"],
                 ctl["flow_checkpoint_sha256"], ctl["basis_source_sha256"],
                 ctl["safety_source_sha256"], str(ctl["horizon"]), str(ctl["dt"]),
                 "collision_free_all_agents_goal_tolerance_0p08_v1",
                 "physical_t0_eta_held_for_episode_v1",
                 canonical({"root": ctl["future_root"],
                            "derivation": ctl["future_seed_derivation"]}),
                 canonical(ctl), "EXACT_PROFILE"),
            )
        for state in design["states"]:
            scenario_uid = design["scenario_info"][str(state["N"])]["scenario_uid"]
            physical = {key: state[key] for key in ("positions", "goals", "velocities")}
            con.execute(
                """INSERT OR IGNORE INTO state(state_uid,scenario_uid,source_group,
                content_hash,physical_state_json,goals_geometry_json,provenance_json,
                identity_quality) VALUES(?,?,?,?,?,?,?,?)""",
                (state["state_uid"], scenario_uid, state["split"],
                 state["content_sha256"], canonical(physical),
                 canonical({"goals": state["goals"]}),
                 canonical({"state_id": state["state_id"],
                            "episode_seed": state["episode_seed"],
                            "physical_split": state["physical_split"]}),
                 "CONTENT_EXACT"),
            )
            con.execute("INSERT OR IGNORE INTO state_alias VALUES(?,?,?,?)",
                        (scenario_uid, state["state_id"], state["state_uid"],
                         experiment_uid))
        con.commit()
    return {"experiment_uid": experiment_uid,
            "scenarios": len(design["scenario_info"]),
            "states": len(design["states"])}


def prepare(output: Path, *, eta_count: int = ETA_COUNT,
            master_seed: int = MASTER_SEED, sobol_seed: int = SOBOL_SEED,
            exclude_design: Path | None = None) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"immutable eta design already exists: {output}")
    if eta_count < 1 or eta_count & (eta_count - 1):
        raise ValueError("eta_count must be a positive power of two")
    excluded_hashes = set()
    if exclude_design is not None:
        prior = json.loads((exclude_design / "design_manifest.json").read_text())
        excluded_hashes = {state["content_sha256"] for state in prior["states"]}
    output.mkdir(parents=True)
    basis = get_basis_family("orthoflow3")
    safety = HardProjectionConfig()
    unit = qmc.Sobol(3, scramble=True, seed=sobol_seed).random_base2(
        int(math.log2(eta_count)))
    eta_values = qmc.scale(unit, DOMAIN_LOW, DOMAIN_HIGH)
    eta_pool = {
        "schema": "gap1_shared_global_eta_pool_v1",
        "family": basis.metadata.family,
        "basis_version": basis.metadata.version,
        "basis_semantics": list(basis.metadata.names),
        "dimension": 3,
        "distribution": f"first {eta_count} points of scrambled Sobol(3), uniform box transform",
        "sobol_seed": sobol_seed,
        "bounds_low": DOMAIN_LOW.tolist(),
        "bounds_high": DOMAIN_HIGH.tolist(),
        "shared_across_N": [2, 10, 20],
        "shared_across_physical_states": True,
        "candidate_count": eta_count,
        "eta": eta_values.tolist(),
        "eta_uid": [eta_identity(value)[0] for value in eta_values],
        "K_nested_prefixes": [k for k in (1, 2, 4, 8, 16, 32, 64, 128)
                              if k <= eta_count],
    }
    dump(output / "eta_pool.json", eta_pool)
    all_requests = []
    all_states = []
    scenario_info = {}
    state_hashes = set(excluded_hashes)
    for n in (2, 10, 20):
        source_root = (Path("diagnostics/gap_flow_scale_20261007")
                       if n == 20 else ROOT)
        source_manifest = source_root / SPECS[n] / "dataset/manifest.json"
        archived = json.loads(source_manifest.read_text())
        base = Config(**archived["scenario_config"])
        checkpoint = CHECKPOINTS[n]
        scenario_uid = uid("scn", {
            "name": "Gap1", "N": n, "physical_fingerprint": base.physical_fingerprint,
            "geometry": {"barrier_x": base.barrier_x, "openings": base.openings},
        })
        controller = {
            "chain": "Flow_bound_goal_stop_hard_projection_OrthoFlow3_hard_projection",
            "flow_checkpoint_sha256": sha(checkpoint),
            "flow_observation": "competence_v2",
            "flow_sampler_source_sha256": sha(Path("new_benchmark_common/macflow.py")),
            "observation_source_sha256": sha(Path("bottleneck_family/observation.py")),
            "flow_samples_per_step": 1,
            "flow_latent": "per_step",
            "goal_stop": "zero reference for agents inside 0.08 m before first projection",
            "basis_family": basis.metadata.family,
            "basis_version": basis.metadata.version,
            "basis_source_sha256": sha(Path("shared_control/basis_families.py")),
            "basis_implementation_sha256": sha(Path("shared_control/diagnostic_corrector.py")),
            "safety_config": safety.to_dict(),
            "safety_source_sha256": sha(Path("shared_control/hard_projection.py")),
            "safety_wrapper_sha256": sha(Path(
                "diagnostics/double_bottleneck_eta3_basin/tools/exact_projection_retry.py")),
            "environment_source_sha256": sha(Path("bottleneck_family/environment.py")),
            "horizon": base.max_steps, "dt": base.dt,
            "goal_tolerance": base.goal_tolerance,
            "robust_protocol": ROBUST_PROTOCOL,
            "future_root": FUTURE_ROOT,
            "future_seed_derivation": "fold_in(fold_in(PRNGKey(root),state_uid_sha256_prefix32),future_index),then_step",
        }
        controller_uid = uid("ctl", controller)
        scenario_info[str(n)] = {
            "scenario_uid": scenario_uid,
            "controller_uid": controller_uid,
            "controller": controller,
            "checkpoint": str(checkpoint),
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": sha(source_manifest),
            "config": base.to_dict(),
            "physical_fingerprint": base.physical_fingerprint,
        }
        for split_index, (logical_split, count) in enumerate(SPLIT_COUNTS.items()):
            generator = np.random.default_rng(np.random.SeedSequence(
                [master_seed, n, split_index]
            ))
            for index in range(count):
                episode_seed = int(generator.integers(1, 2**31 - 1))
                env = BottleneckEnv(replace(base, seed=episode_seed,
                                            split=PHYSICAL_SPLIT[logical_split]))
                physical = {
                    "positions": env.positions.tolist(),
                    "goals": env.goals.tolist(),
                    "velocities": env.velocities.tolist(),
                }
                content_hash = hashlib.sha256(canonical({
                    "physical_fingerprint": base.physical_fingerprint,
                    **physical,
                }).encode()).hexdigest()
                if content_hash in state_hashes:
                    raise ValueError("duplicate physical state across splits or N")
                state_hashes.add(content_hash)
                state_uid = uid("state", {"scenario": scenario_uid,
                                          "content": content_hash})
                state = {
                    "state_id": f"gap1_n{n}_{logical_split}_{index:04d}",
                    "state_uid": state_uid,
                    "content_sha256": content_hash,
                    "N": n,
                    "split": logical_split,
                    "physical_split": PHYSICAL_SPLIT[logical_split],
                    "episode_seed": episode_seed,
                    **physical,
                }
                all_states.append(state)
                for eta_uid in eta_pool["eta_uid"]:
                    all_requests.append({
                        "scenario_uid": scenario_uid,
                        "state_uid": state_uid,
                        "eta_uid": eta_uid,
                        "controller_uid": controller_uid,
                        "seed_keys": [canonical({"future_index": int(s)})
                                      for s in STANDARD_SEEDS],
                    })
    manifest = {
        "schema": "gap1_eta_scaling_design_v1",
        "purpose": "state-conditioned Q(x,eta) and G(eta|x) supervision; no controller identity in model input",
        "N": [2, 10, 20],
        "state_count_per_N": SPLIT_COUNTS,
        "physical_state_master_seed": master_seed,
        "state_split_unit": "exact physical initial state",
        "eta_pool": str(output / "eta_pool.json"),
        "eta_pool_sha256": sha(output / "eta_pool.json"),
        "eta_count_per_state": eta_count,
        "future_indices": list(STANDARD_SEEDS),
        "rollout_budget_full_Q16": len(all_requests) * len(STANDARD_SEEDS),
        "scenario_info": scenario_info,
        "states": all_states,
    }
    if exclude_design is None:
        manifest.update({
            "N2_N10_design_frozen_before_prior_eta_outcomes": True,
            "N2_N10_prior_eta_outcomes_exist": True,
            "N20_prepared_before_any_N20_eta_outcome": True,
            "prior_N2_N10_design": "datasets/gap1_eta_scaling_v2/design_manifest.json",
            "prior_N2_N10_design_sha256": sha(Path(
                "datasets/gap1_eta_scaling_v2/design_manifest.json")),
        })
    else:
        manifest.update({
            "prior_design": str(exclude_design),
            "prior_design_sha256": sha(exclude_design / "design_manifest.json"),
            "physical_state_overlap_with_prior": 0,
            "prior_test_outcomes_seen_before_this_design": True,
            "new_test_states_unopened_at_design": True,
            "candidate_prefix_protocol": {
                "global_prefixes": [32, 64, 128],
                "first_stage": "TRAIN and VAL, first 32 eta per state; TEST sealed",
                "expansion_gate": "For both N=10 and N=20, at least 8/32 TRAIN states and 2/8 VAL states must have a robust eta; otherwise expand the same global prefix to 64 then 128, without targeting states or eta regions",
                "test_open_rule": "Only if the TRAIN/VAL coverage criterion passes at a fixed global prefix; otherwise keep TEST sealed",
            },
        })
    dump(output / "design_manifest.json", manifest)
    plan = {"schema": "gap1_eta_scaling_requests_v1", "requests": all_requests}
    dump(output / "planned_rollouts.json", plan)
    register(output)
    cache = preflight(output / "planned_rollouts.json")
    dump(output / "cache_preflight.json", cache)
    if cache["summary"]["ambiguous"]:
        raise RuntimeError("ambiguous cache identities require audit before rollouts")
    return {"states": len(all_states), "state_eta_pairs": len(all_requests),
            "full_Q16_rollout_budget": len(all_requests) * len(STANDARD_SEEDS),
            "cache_preflight": cache["summary"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("datasets/gap1_eta_scaling_v1"))
    parser.add_argument("--register-existing", action="store_true",
                        help="Register a previously frozen outcome-blind design")
    parser.add_argument("--eta-count", type=int, default=ETA_COUNT)
    parser.add_argument("--master-seed", type=int, default=MASTER_SEED)
    parser.add_argument("--sobol-seed", type=int, default=SOBOL_SEED)
    parser.add_argument("--exclude-design", type=Path)
    args = parser.parse_args()
    if args.register_existing:
        result = register(args.output)
        cache = preflight(args.output / "planned_rollouts.json")
        dump(args.output / "cache_preflight.json", cache)
        result["cache_preflight"] = cache["summary"]
    else:
        result = prepare(args.output, eta_count=args.eta_count,
                         master_seed=args.master_seed,
                         sobol_seed=args.sobol_seed,
                         exclude_design=args.exclude_design)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
