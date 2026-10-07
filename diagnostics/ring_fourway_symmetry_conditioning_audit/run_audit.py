#!/usr/bin/env python3
"""Read-only symmetry/conditioning audit for frozen Four-Way and Ring models."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import math

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

from diagnostics.orthoflow3_generator_critic_frozen_test_v1 import run_frozen_test as frozen
from diagnostics.orthoflow3_generator_critic_v1 import train_evaluate as learn
from diagnostics.orthoflow3_ring_revision_v1 import run_fresh as fresh
from new_benchmark_common.basin_dataset_v1 import TrainingRuntime, _environment_descriptor
from shared_control.hard_projection import barrier_constraints


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = ROOT / "datasets/orthoflow3_basin_dataset_v1"
TRANSFORM_PATH = OUT / "symmetry_transformations.json"
SCENARIOS = ("four_way_intersection", "ring_exchange")


def dump(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse(value):
    return json.loads(value) if isinstance(value, str) else value


def stable_int(*parts):
    return int(hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()[:16], 16)


def transform_array(value, matrix, permutation):
    return np.asarray(value, float)[np.asarray(permutation)] @ np.asarray(matrix, float).T


def inverse_array(value, matrix, permutation):
    value = np.asarray(value, float)
    out = np.empty_like(value)
    for new_index, old_index in enumerate(permutation):
        out[old_index] = value[new_index] @ np.asarray(matrix, float)
    return out


def transform_physical(physical, definition):
    matrix, permutation = definition["matrix"], definition["source_permutation"]
    result = dict(physical)
    for key in ("positions", "velocities", "goals"):
        if physical.get(key) is not None:
            result[key] = transform_array(physical[key], matrix, permutation).tolist()
    return result


def choose_states(rows, scenario, count=8):
    eligible = [x for x in rows if x["scenario"] == scenario and x["split"] in ("train", "validation")]
    return sorted(eligible, key=lambda x: hashlib.sha256(("symmetry-audit-v1|" + x["state_uid"]).encode()).hexdigest())[:count]


def physical(row):
    return parse(row["structured_state"])


def make_env(runtime, state_physical):
    env = runtime.make_env()
    p = np.asarray(state_physical["positions"], float)
    v = np.asarray(state_physical["velocities"], float)
    g = np.asarray(state_physical["goals"], float)
    if runtime.kind == "four":
        env.reset(p, v)
        env.goals = g.copy()
    else:
        env.reset(p, velocities=v, goals=g)
    env.step_count = int(state_physical.get("timestep", 0))
    return env


def observation(runtime, env):
    return np.asarray(runtime.observation(env), float)


def flow(runtime, env, key):
    return np.asarray(runtime.flow_world(env, key), float)


def context_vector(env_desc, norm, scenario):
    return fresh.context_vector(env_desc, norm, scenario)


def conditioning(runtime, env, key, norm, scenario):
    obs = observation(runtime, env)
    current_flow = flow(runtime, env, key)
    flat = np.concatenate((obs.reshape(-1), current_flow.reshape(-1)))
    n = norm["scenarios"][scenario]
    h = (flat.astype(np.float32) - np.asarray(n["h_mean"], np.float32)) / np.asarray(n["h_std"], np.float32)
    desc = _environment_descriptor(runtime)
    c = context_vector(desc, norm, scenario)
    return h, c, {"observation": obs, "reference_flow_world": current_flow, "flat": flat, "descriptor": desc}


def generator_raw(models, scenario, h, c):
    gm, gp, _, _ = models
    return np.asarray(gm.apply(gp, jnp.asarray(h[None]), jnp.asarray(c[None]),
                               method=getattr(gm, learn.METHOD[scenario])))[0]


def critic_scores(models, scenario, h, c, etas):
    _, _, cm, cp = models
    etas = np.asarray(etas, float)
    hh = np.repeat(np.asarray(h, np.float32)[None], len(etas), axis=0)
    cc = np.repeat(np.asarray(c, np.float32)[None], len(etas), axis=0)
    ee = (etas - learn.CENTER) / learn.RADIUS
    logits = np.asarray(cm.apply(cp, jnp.asarray(hh), jnp.asarray(cc), jnp.asarray(ee),
                                 method=getattr(cm, learn.METHOD[scenario])))
    return 1.0 / (1.0 + np.exp(-logits))


def proposals(raw, noise):
    mean = np.asarray(learn.eta_mean(jnp.asarray(raw)), float)
    samples = np.asarray(learn.eta_from_noise(
        jnp.repeat(jnp.asarray(raw)[None], len(noise), axis=0), jnp.asarray(noise, jnp.float32)), float)
    return mean, samples


def nearest_cloud_metrics(first, second):
    a = (np.asarray(first) - learn.CENTER) / learn.RADIUS
    b = (np.asarray(second) - learn.CENTER) / learn.RADIUS
    distances = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    ri, ci = linear_sum_assignment(distances)
    return {"hungarian_mean": float(distances[ri, ci].mean()),
            "hungarian_max": float(distances[ri, ci].max()),
            "symmetric_nearest_mean": float(0.5 * (distances.min(1).mean() + distances.min(0).mean()))}


def summarize(values):
    x = np.asarray(values, float)
    return {"count": int(x.size), "mean": float(x.mean()) if x.size else None,
            "median": float(np.median(x)) if x.size else None,
            "p95": float(np.quantile(x, .95)) if x.size else None,
            "max": float(x.max()) if x.size else None}


def exact_tests(runtimes, selected, definitions):
    tolerances = definitions["tolerances"]
    results = {}
    eta = np.asarray([0.91, 0.13, 0.28])
    for scenario in SCENARIOS:
        runtime = runtimes[scenario]
        names = (["rotate_90", "rotate_180", "rotate_270", "canonical_rotate_180_relabel", "cyclic_relabel"]
                 if scenario == "four_way_intersection" else
                 ["rotate_90", "rotate_180", "rotate_270", "rotate_90_cyclic_relabel", "cyclic_relabel", "reflect_x_axis"])
        scenario_rows = []
        for state_index, row in enumerate(selected[scenario][:6]):
            base_physical = physical(row)
            for name in names:
                definition = definitions["four_way" if runtime.kind == "four" else "ring_exchange"][name]
                matrix, permutation = np.asarray(definition["matrix"]), definition["source_permutation"]
                transformed = transform_physical(base_physical, definition)
                env0, env1 = make_env(runtime, base_physical), make_env(runtime, transformed)
                rng = np.random.default_rng(stable_int("exact", scenario, row["state_uid"], name))
                target = rng.normal(size=(4, 2))
                target *= (0.45 * runtime.config.max_speed /
                           np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-12))
                target_t = transform_array(target, matrix, permutation)

                # Dynamics/collision metamorphism under one safe, bounded step.
                dyn0, dyn1 = make_env(runtime, base_physical), make_env(runtime, transformed)
                before0, before1 = dyn0.positions.copy(), dyn1.positions.copy()
                _, _, _, info0 = dyn0.step(0.1 * target)
                _, _, _, info1 = dyn1.step(0.1 * target_t)
                expected_pos = transform_array(dyn0.positions, matrix, permutation)
                dynamics_error = float(np.max(np.abs(dyn1.positions - expected_pos)))
                wall_error = float(np.max(np.abs(np.sort(dyn0.distances()[0].ravel()) -
                                                  np.sort(dyn1.distances()[0].ravel()))))
                pair_error = float(np.max(np.abs(np.sort(dyn0.distances()[1]) - np.sort(dyn1.distances()[1]))))
                collision_match = bool(info0["termination"] == info1["termination"] and
                                       bool(info0.get("agent_collision")) == bool(info1.get("agent_collision")))

                # Hard constraints/projection.
                snap0, snap1 = runtime.safety_snapshot(env0), runtime.safety_snapshot(env1)
                a0, b0, _ = barrier_constraints(snap0, runtime.cbf)
                a1, b1, _ = barrier_constraints(snap1, runtime.cbf)
                residual0 = np.sort(a0 @ target.reshape(-1) - b0)
                residual1 = np.sort(a1 @ target_t.reshape(-1) - b1)
                constraint_error = float(np.max(np.abs(residual0 - residual1)))
                safe0, _ = runtime.project(env0, target)
                safe1, _ = runtime.project(env1, target_t)
                projection_error = float(np.max(np.abs(safe1 - transform_array(safe0, matrix, permutation))))

                # All three basis fields, full correction and mandatory second projection.
                fields0 = runtime.basis.compute(env0.positions, env0.goals, safe0, runtime.config.max_speed)
                fields1 = runtime.basis.compute(env1.positions, env1.goals, safe1, runtime.config.max_speed)
                basis_errors = [float(np.max(np.abs(v1 - transform_array(v0, matrix, permutation))))
                                for v0, v1 in zip(fields0.values, fields1.values)]
                correction0, correction1 = fields0.correction(eta), fields1.correction(eta)
                correction_error = float(np.max(np.abs(correction1 - transform_array(correction0, matrix, permutation))))
                second0, _ = runtime.project(env0, safe0 + correction0)
                second1, _ = runtime.project(env1, safe1 + correction1)
                second_error = float(np.max(np.abs(second1 - transform_array(second0, matrix, permutation))))

                # Observation is exact only under the chart's known transformation.
                obs0, obs1 = observation(runtime, env0), observation(runtime, env1)
                observation_equal_error = float(np.max(np.abs(obs1 - obs0)))
                scenario_rows.append({
                    "state_uid": row["state_uid"], "transform": name,
                    "dynamics_error": dynamics_error, "wall_distance_error": wall_error,
                    "pair_distance_error": pair_error, "collision_match": collision_match,
                    "constraint_residual_multiset_error": constraint_error,
                    "projection_error": projection_error,
                    "basis_goal_error": basis_errors[0], "basis_flow_perp_error": basis_errors[1],
                    "basis_rel_error": basis_errors[2], "full_correction_error": correction_error,
                    "second_projection_error": second_error,
                    "raw_observation_equal_error": observation_equal_error,
                })
        exact_keys = ["dynamics_error", "wall_distance_error", "pair_distance_error",
                      "constraint_residual_multiset_error", "projection_error", "basis_goal_error",
                      "basis_flow_perp_error", "basis_rel_error", "full_correction_error", "second_projection_error"]
        maxima = {key: max(x[key] for x in scenario_rows) for key in exact_keys}
        failures = []
        for x in scenario_rows:
            limits = {"dynamics_error": tolerances["dynamics_abs"],
                      "wall_distance_error": tolerances["collision_distance_abs"],
                      "pair_distance_error": tolerances["collision_distance_abs"],
                      "constraint_residual_multiset_error": tolerances["constraint_residual_multiset_abs"],
                      "projection_error": tolerances["projection_abs"],
                      "basis_goal_error": tolerances["basis_abs"],
                      "basis_flow_perp_error": tolerances["basis_abs"],
                      "basis_rel_error": tolerances["basis_abs"],
                      "full_correction_error": tolerances["basis_abs"],
                      "second_projection_error": tolerances["projection_abs"]}
            bad = [key for key, limit in limits.items() if x[key] > limit]
            if bad or not x["collision_match"]:
                failures.append({"state_uid": x["state_uid"], "transform": x["transform"], "violations": bad,
                                 "collision_match": x["collision_match"]})
        results[scenario] = {"cases": len(scenario_rows), "max_errors": maxima,
                             "exact_failures": failures, "rows": scenario_rows}
    dump("exact_component_test_results.json", results)
    return results


def macflow_tests(runtimes, selected, definitions):
    results = {}
    for scenario in SCENARIOS:
        runtime = runtimes[scenario]
        names = (["rotate_90", "rotate_180", "rotate_270", "canonical_rotate_180_relabel", "cyclic_relabel"]
                 if scenario == "four_way_intersection" else
                 ["rotate_90", "rotate_180", "rotate_270", "rotate_90_cyclic_relabel", "cyclic_relabel", "reflect_x_axis"])
        by_transform = defaultdict(lambda: defaultdict(list))
        rows = []
        for row in selected[scenario]:
            base = physical(row)
            env0 = make_env(runtime, base)
            for name in names:
                definition = definitions["four_way" if runtime.kind == "four" else "ring_exchange"][name]
                matrix, permutation = definition["matrix"], definition["source_permutation"]
                env1 = make_env(runtime, transform_physical(base, definition))
                actions0, actions1_back = [], []
                for sample in range(16):
                    key = jax.random.PRNGKey(stable_int("macflow-symmetry", scenario, row["state_uid"], sample) & 0xffffffff)
                    a0, a1 = flow(runtime, env0, key), flow(runtime, env1, key)
                    actions0.append(a0)
                    actions1_back.append(inverse_array(a1, matrix, permutation))
                actions0, actions1_back = np.asarray(actions0), np.asarray(actions1_back)
                paired_rmse = float(np.sqrt(np.mean((actions0 - actions1_back) ** 2)))
                mean_shift = float(np.sqrt(np.mean((actions0.mean(0) - actions1_back.mean(0)) ** 2)))
                scale = runtime.config.max_speed
                by_transform[name]["paired_action_rmse_over_speed"].append(paired_rmse / scale)
                by_transform[name]["distribution_mean_shift_over_speed"].append(mean_shift / scale)
                rows.append({"state_uid": row["state_uid"], "transform": name,
                             "paired_action_rmse": paired_rmse, "mean_action_shift": mean_shift})

        # Short-horizon first-projection rollouts for rotation and pure relabeling.
        horizon_names = ["rotate_90", "cyclic_relabel"]
        short = []
        for row in selected[scenario][:4]:
            for name in horizon_names:
                definition = definitions["four_way" if runtime.kind == "four" else "ring_exchange"][name]
                matrix, permutation = definition["matrix"], definition["source_permutation"]
                e0, e1 = make_env(runtime, physical(row)), make_env(runtime, transform_physical(physical(row), definition))
                path0, path1 = [e0.positions.copy()], [inverse_array(e1.positions, matrix, permutation)]
                outcome0 = outcome1 = "running"
                for step in range(25):
                    key = jax.random.PRNGKey(stable_int("macflow-short", scenario, row["state_uid"], step) & 0xffffffff)
                    u0, _ = runtime.project(e0, flow(runtime, e0, key))
                    u1, _ = runtime.project(e1, flow(runtime, e1, key))
                    _, _, d0, i0 = e0.step(u0); _, _, d1, i1 = e1.step(u1)
                    path0.append(e0.positions.copy()); path1.append(inverse_array(e1.positions, matrix, permutation))
                    outcome0, outcome1 = i0["termination"], i1["termination"]
                    if d0 or d1: break
                length = min(len(path0), len(path1))
                trajectory_rmse = float(np.sqrt(np.mean((np.asarray(path0[:length]) - np.asarray(path1[:length])) ** 2)))
                short.append({"state_uid": row["state_uid"], "transform": name,
                              "trajectory_rmse": trajectory_rmse, "outcome_original": outcome0,
                              "outcome_transformed": outcome1, "outcome_match": outcome0 == outcome1,
                              "steps_compared": length - 1})

        # Small full-horizon MACFlow+first-projection replay, without eta/generator/critic.
        full_names = (["rotate_90", "canonical_rotate_180_relabel"] if runtime.kind == "four"
                      else ["rotate_90", "rotate_90_cyclic_relabel"])
        full = []
        for row in selected[scenario][:2]:
            for name in full_names:
                definition = definitions["four_way" if runtime.kind == "four" else "ring_exchange"][name]
                matrix, permutation = definition["matrix"], definition["source_permutation"]
                e0, e1 = make_env(runtime, physical(row)), make_env(runtime, transform_physical(physical(row), definition))
                paths = [[e0.positions.copy()], [inverse_array(e1.positions, matrix, permutation)]]
                outcomes = ["timeout", "timeout"]; collisions = [False, False]
                remaining = min(runtime.config.max_steps - e0.step_count, runtime.config.max_steps - e1.step_count)
                for step in range(max(0, remaining)):
                    key = jax.random.PRNGKey(stable_int("macflow-full", scenario, row["state_uid"], name, step) & 0xffffffff)
                    for index, env in enumerate((e0, e1)):
                        if env.done: continue
                        u, _ = runtime.project(env, flow(runtime, env, key))
                        _, _, done, info = env.step(u)
                        paths[index].append(env.positions.copy() if index == 0 else inverse_array(env.positions, matrix, permutation))
                        outcomes[index] = info["termination"]
                        collisions[index] = bool(info.get("wall_collision") or info.get("obstacle_collision") or
                                                 info.get("outer_collision") or info.get("agent_collision"))
                    if e0.done and e1.done: break
                length = min(len(paths[0]), len(paths[1]))
                full.append({"state_uid": row["state_uid"], "transform": name,
                    "outcome_original": outcomes[0], "outcome_transformed": outcomes[1],
                    "outcome_match": outcomes[0] == outcomes[1],
                    "steps_original": len(paths[0])-1, "steps_transformed": len(paths[1])-1,
                    "collision_original": collisions[0], "collision_transformed": collisions[1],
                    "trajectory_rmse": float(np.sqrt(np.mean((np.asarray(paths[0][:length])-np.asarray(paths[1][:length]))**2)))})
        results[scenario] = {
            "initial_action": {name: {metric: summarize(values) for metric, values in metrics.items()}
                               for name, metrics in by_transform.items()},
            "short_horizon": {"rows": short, "trajectory_rmse": summarize([x["trajectory_rmse"] for x in short]),
                              "outcome_match_fraction": float(np.mean([x["outcome_match"] for x in short]))},
            "full_horizon_hard_safety": {"rows": full,
                "outcome_match_fraction": float(np.mean([x["outcome_match"] for x in full])),
                "trajectory_rmse": summarize([x["trajectory_rmse"] for x in full])},
            "rows": rows,
        }
    dump("macflow_symmetry_metrics.json", results)
    return results


def learned_eta_tests(runtimes, selected, definitions, norm, models):
    generator_result, critic_result, shortcut_result = {}, {}, {}
    for scenario in SCENARIOS:
        runtime = runtimes[scenario]
        names = (["rotate_90", "rotate_180", "rotate_270", "canonical_rotate_180_relabel", "cyclic_relabel"]
                 if scenario == "four_way_intersection" else
                 ["rotate_90", "rotate_180", "rotate_270", "rotate_90_cyclic_relabel", "cyclic_relabel", "reflect_x_axis"])
        gen_by = defaultdict(lambda: defaultdict(list)); crit_by = defaultdict(lambda: defaultdict(list))
        gen_rows, crit_rows, shortcut_rows = [], [], []
        for row in selected[scenario]:
            base = physical(row); e0 = make_env(runtime, base)
            key = jax.random.PRNGKey(stable_int("conditioning", scenario, row["state_uid"]) & 0xffffffff)
            h0, c0, raw_condition0 = conditioning(runtime, e0, key, norm, scenario)
            raw0 = generator_raw(models, scenario, h0, c0)
            rng = np.random.default_rng(stable_int("symmetry-proposals", scenario, row["state_uid"]))
            noise = rng.standard_normal((16, 3))
            mean0, samples0 = proposals(raw0, noise)
            all0 = np.asarray([mean0, *samples0])
            score0 = critic_scores(models, scenario, h0, c0, all0)
            mu0, sigma0 = np.asarray(learn.dist_params(jnp.asarray(raw0)))

            # Nonphysical bookkeeping-only scenario one-hot counterfactual; branch stays fixed.
            sc_index = frozen.SCENARIOS.index(scenario)
            alt_index = (sc_index + 1) % len(frozen.SCENARIOS)
            c_cf = np.asarray(c0).copy()
            offset = len(norm["environment_keys"])
            cstd = np.asarray(norm["scenarios"][scenario]["c_std"], float)
            c_cf[offset + sc_index] -= 1.0 / cstd[offset + sc_index]
            c_cf[offset + alt_index] += 1.0 / cstd[offset + alt_index]
            raw_cf = generator_raw(models, scenario, h0, c_cf)
            mean_cf = np.asarray(learn.eta_mean(jnp.asarray(raw_cf)), float)
            score_cf = critic_scores(models, scenario, h0, c_cf, all0)
            shortcut_rows.append({
                "state_uid": row["state_uid"],
                "generator_normalized_mean_l2": float(np.linalg.norm((mean_cf - mean0) / learn.RADIUS)),
                "critic_mean_abs_score_change": float(np.mean(np.abs(score_cf - score0))),
                "critic_max_abs_score_change": float(np.max(np.abs(score_cf - score0))),
                "canonical_context_max_abs": float(np.max(np.abs(c0))),
            })

            for name in names:
                definition = definitions["four_way" if runtime.kind == "four" else "ring_exchange"][name]
                e1 = make_env(runtime, transform_physical(base, definition))
                h1, c1, raw_condition1 = conditioning(runtime, e1, key, norm, scenario)
                raw1 = generator_raw(models, scenario, h1, c1)
                mean1, samples1 = proposals(raw1, noise)
                mu1, sigma1 = np.asarray(learn.dist_params(jnp.asarray(raw1)))
                cloud = nearest_cloud_metrics(samples0, samples1)
                mean_delta = float(np.linalg.norm((mean1 - mean0) / learn.RADIUS))
                mu_delta = float(np.linalg.norm(mu1 - mu0)); sigma_delta = float(np.linalg.norm(sigma1 - sigma0))
                own1 = np.asarray([mean1, *samples1])
                score1_own = critic_scores(models, scenario, h1, c1, own1)
                predicted_hits0 = int(np.sum(score0 >= .9375)); predicted_hits1 = int(np.sum(score1_own >= .9375))

                # Critic symmetry uses exactly the same eta candidate set all0.
                score1_same = critic_scores(models, scenario, h1, c1, all0)
                rho = float(spearmanr(score0, score1_same).statistic)
                if not np.isfinite(rho): rho = 1.0 if np.allclose(score0, score1_same) else 0.0
                top0, top1 = int(np.argmax(score0)), int(np.argmax(score1_same))
                regret = float(score0[top0] - score0[top1])
                gen_by[name]["normalized_mean_delta"].append(mean_delta)
                gen_by[name]["mu_l2"].append(mu_delta); gen_by[name]["sigma_l2"].append(sigma_delta)
                gen_by[name]["cloud_hungarian_mean"].append(cloud["hungarian_mean"])
                gen_by[name]["predicted_robust_hit_delta"].append(predicted_hits1 - predicted_hits0)
                crit_by[name]["mean_abs_score_delta"].append(float(np.mean(np.abs(score1_same - score0))))
                crit_by[name]["max_abs_score_delta"].append(float(np.max(np.abs(score1_same - score0))))
                crit_by[name]["spearman"].append(rho); crit_by[name]["representation_regret"].append(regret)
                crit_by[name]["top1_match"].append(float(top0 == top1))
                gen_rows.append({"state_uid": row["state_uid"], "transform": name,
                    "normalized_mean_delta": mean_delta, "mu_l2": mu_delta, "sigma_l2": sigma_delta,
                    "proposal_cloud": cloud, "predicted_robust_original": predicted_hits0,
                    "predicted_robust_transformed": predicted_hits1,
                    "observation_max_abs_change": float(np.max(np.abs(raw_condition1["observation"] - raw_condition0["observation"]))),
                    "world_flow_suffix_max_abs_change": float(np.max(np.abs(raw_condition1["reference_flow_world"] - raw_condition0["reference_flow_world"])))})
                crit_rows.append({"state_uid": row["state_uid"], "transform": name,
                    "mean_abs_score_delta": float(np.mean(np.abs(score1_same-score0))),
                    "max_abs_score_delta": float(np.max(np.abs(score1_same-score0))),
                    "spearman": rho, "top1_original": top0, "top1_transformed": top1,
                    "top1_match": top0 == top1, "representation_regret": regret})
        generator_result[scenario] = {
            "by_transform": {name: {metric: summarize(values) for metric, values in metrics.items()}
                             for name, metrics in gen_by.items()}, "rows": gen_rows,
            "robust_hit_semantics": "critic-predicted score>=0.9375 only; no new Q16 or basin rollouts were run"}
        critic_result[scenario] = {"by_transform": {name: {metric: summarize(values) for metric, values in metrics.items()}
                                                     for name, metrics in crit_by.items()}, "rows": crit_rows}
        gen_flag = sum(x["generator_normalized_mean_l2"] > .1 for x in shortcut_rows)
        critic_flag = sum(x["critic_max_abs_score_change"] > .1 for x in shortcut_rows)
        shortcut_result[scenario] = {
            "canonical_context_all_zero": all(x["canonical_context_max_abs"] < 1e-6 for x in shortcut_rows),
            "rows": shortcut_rows,
            "generator_counterfactual_flag_count": gen_flag,
            "critic_counterfactual_flag_count": critic_flag,
            "SCENARIO_ID_SHORTCUT": bool(gen_flag or critic_flag),
            "interpretation": "counterfactual is outside canonical support; canonical per-scenario normalization maps the fixed descriptor and one-hot to zero, while scenario-specific branch adapters still route by scenario",
        }
    dump("generator_proposal_distribution_metrics.json", generator_result)
    dump("critic_ranking_symmetry_metrics.json", critic_result)
    dump("scenario_id_shortcut_audit.json", shortcut_result)
    return generator_result, critic_result, shortcut_result


def normalization_and_serialization(runtimes, selected, norm):
    output = {}
    for scenario in SCENARIOS:
        runtime = runtimes[scenario]; n = norm["scenarios"][scenario]
        obs_dim = int(runtime.agent.config["obs_dim"])
        stage_mean = np.asarray(runtime.agent.config["obs_mean"], float).reshape(4, obs_dim)
        stage_scale = np.asarray(runtime.agent.config["obs_scale"], float).reshape(4, obs_dim)
        h_mean = np.asarray(n["h_mean"], float); h_std = np.asarray(n["h_std"], float)
        native_dim = 4 * obs_dim
        obs_h_mean = h_mean[:native_dim].reshape(4, obs_dim)
        obs_h_std = h_std[:native_dim].reshape(4, obs_dim)
        flow_h_mean = h_mean[native_dim:].reshape(4, 2)
        flow_h_std = h_std[native_dim:].reshape(4, 2)
        desc = _environment_descriptor(runtime)
        c = context_vector(desc, norm, scenario)
        output[scenario] = {
            "stage1_observation_shape": [4, obs_dim], "stage1_action_shape": [4, 2],
            "basin_h_shape": [native_dim + 8], "context_shape": [len(c)],
            "stage1_agent_slot_mean_dispersion": float(np.mean(np.std(stage_mean, axis=0))),
            "stage1_agent_slot_scale_dispersion": float(np.mean(np.std(stage_scale, axis=0))),
            "basin_observation_agent_slot_mean_dispersion": float(np.mean(np.std(obs_h_mean, axis=0))),
            "basin_observation_agent_slot_std_dispersion": float(np.mean(np.std(obs_h_std, axis=0))),
            "basin_world_flow_agent_slot_mean_dispersion": float(np.mean(np.std(flow_h_mean, axis=0))),
            "basin_world_flow_agent_slot_std_dispersion": float(np.mean(np.std(flow_h_std, axis=0))),
            "canonical_context_max_abs_after_normalization": float(np.max(np.abs(c))),
            "environment_descriptor": desc,
            "observation_frame": "world Cartesian" if runtime.kind == "four" else "per-agent radial/CCW-tangent local",
            "reference_flow_suffix_frame": "world Cartesian",
            "mixed_frame_conditioning": runtime.kind == "ring",
            "history": "none; instantaneous native observation plus one deterministically sampled current Flow action",
            "normalized_episode_time_in_h": False,
        }
    dump("serialization_audit.json", output)
    return output


def closed_loop(runtimes, selected, definitions, norm, models):
    results = {}
    for scenario in SCENARIOS:
        runtime = runtimes[scenario]
        transform_names = (["rotate_90", "canonical_rotate_180_relabel"] if runtime.kind == "four"
                           else ["rotate_90", "rotate_90_cyclic_relabel"])
        rows = []
        for row in selected[scenario][:2]:
          for transform_name in transform_names:
            definition = definitions["four_way" if runtime.kind == "four" else "ring_exchange"][transform_name]
            matrix, permutation = definition["matrix"], definition["source_permutation"]
            base, transformed = physical(row), transform_physical(physical(row), definition)
            pair_key = jax.random.PRNGKey(stable_int("closed-conditioning", scenario, row["state_uid"], transform_name) & 0xffffffff)

            def select(state_physical):
                env = make_env(runtime, state_physical)
                h, c, _ = conditioning(runtime, env, pair_key, norm, scenario)
                raw = generator_raw(models, scenario, h, c)
                noise = np.random.default_rng(stable_int("closed-proposals", scenario, row["state_uid"], transform_name)).standard_normal((4, 3))
                mean, samples = proposals(raw, noise); etas = np.asarray([mean, *samples])
                scores = critic_scores(models, scenario, h, c, etas)
                index = int(np.argmax(scores))
                return etas[index], index, scores.tolist(), etas.tolist()

            eta0, index0, scores0, candidates0 = select(base)
            eta1, index1, scores1, candidates1 = select(transformed)

            def rollout(state_physical, eta):
                env = make_env(runtime, state_physical); path = [env.positions.copy()]
                root = jax.random.PRNGKey(stable_int("closed-rollout", scenario, row["state_uid"], transform_name, 0) & 0xffffffff)
                termination = "timeout"; numerical = False
                try:
                    remaining = max(0, runtime.config.max_steps - env.step_count)
                    for step in range(remaining):
                        u_flow = flow(runtime, env, jax.random.fold_in(root, step))
                        u_safe, _ = runtime.project(env, u_flow)
                        fields = runtime.basis.compute(env.positions, env.goals, u_safe, runtime.config.max_speed)
                        executed, _ = runtime.project(env, u_safe + fields.correction(eta))
                        _, _, done, info = env.step(executed); path.append(env.positions.copy())
                        termination = info["termination"]
                        if done: break
                except Exception as exc:
                    termination = "numerical_failure"; numerical = True
                summary = env.summary()
                collision = bool(summary.get("wall_collision") or summary.get("obstacle_collision") or summary.get("outer_collision") or summary.get("agent_collision"))
                return {"path": np.asarray(path), "termination": termination, "success": bool(summary.get("collision_free_success")) and termination == "success",
                        "collision": collision, "numerical": numerical, "steps": len(path)-1}

            z0, z1 = rollout(base, eta0), rollout(transformed, eta1)
            back = np.asarray([inverse_array(x, matrix, permutation) for x in z1["path"]])
            length = min(len(z0["path"]), len(back))
            trajectory_rmse = float(np.sqrt(np.mean((z0["path"][:length] - back[:length])**2)))
            rows.append({"state_uid": row["state_uid"], "transform": transform_name,
                         "eta_original": eta0.tolist(), "eta_transformed": eta1.tolist(),
                         "eta_normalized_distance": float(np.linalg.norm((eta1-eta0)/learn.RADIUS)),
                         "selected_proposal_original": index0, "selected_proposal_transformed": index1,
                         "same_proposal_index": index0 == index1,
                         "critic_scores_original": scores0, "critic_scores_transformed": scores1,
                         "outcome_original": z0["termination"], "outcome_transformed": z1["termination"],
                         "success_original": z0["success"], "success_transformed": z1["success"],
                         "collision_original": z0["collision"], "collision_transformed": z1["collision"],
                         "steps_original": z0["steps"], "steps_transformed": z1["steps"],
                         "trajectory_rmse_back_transformed": trajectory_rmse, "steps_compared": length-1})
        results[scenario] = {"rows": rows,
            "outcome_match_fraction": float(np.mean([x["outcome_original"] == x["outcome_transformed"] for x in rows])),
            "proposal_index_match_fraction": float(np.mean([x["same_proposal_index"] for x in rows])),
            "trajectory_rmse": summarize([x["trajectory_rmse_back_transformed"] for x in rows]),
            "eta_normalized_distance": summarize([x["eta_normalized_distance"] for x in rows])}
    dump("closed_loop_metamorphic_replay_results.json", results)
    return results


def provenance(runtimes):
    files = [
        ROOT / "four_way_intersection/environment.py", ROOT / "four_way_intersection/scenario.py",
        ROOT / "four_way_intersection/safety.py", ROOT / "ring_exchange/environment.py",
        ROOT / "ring_exchange/local_frame.py", ROOT / "ring_exchange/safety.py",
        ROOT / "new_benchmark_common/macflow.py", ROOT / "new_benchmark_common/safety_eta3.py",
        ROOT / "new_benchmark_common/basin_dataset_v1.py", ROOT / "shared_control/basis_families.py",
        ROOT / "shared_control/hard_projection.py", frozen.GENERATOR_CKPT,
        frozen.CRITIC_CKPT, frozen.NORMALIZATION, DATA / "states.parquet",
    ]
    value = {"files": {str(x.relative_to(ROOT)): sha(x) for x in files},
             "runtime": {s: {"environment": r.env_hash, "safety": r.safety_hash,
                               "macflow_checkpoint": r.checkpoint_sha, "orthoflow3": r.orthoflow_hash,
                               "stage1_config": dict(r.agent.config)} for s, r in runtimes.items()},
             "frozen_test_states_used": 0, "training_or_retraining_performed": False,
             "success_basin_Q_experiments_repeated": False}
    dump("provenance_hashes.json", value)


def main():
    definitions = load(TRANSFORM_PATH)
    if not definitions.get("frozen_before_model_tests"):
        raise RuntimeError("transform definitions were not frozen")
    all_states = pq.read_table(DATA / "states.parquet").to_pylist()
    selected = {scenario: choose_states(all_states, scenario) for scenario in SCENARIOS}
    dump("selected_train_dev_states.json", {scenario: [{"state_uid": x["state_uid"], "state_id": x["state_id"],
          "split": x["split"], "parent_episode_id": x["parent_episode_id"]} for x in rows]
          for scenario, rows in selected.items()})
    # TrainingRuntime opens only frozen train/dev artifacts here; no test manifest is loaded.
    runtimes = {scenario: TrainingRuntime(scenario, [], parent=False) for scenario in SCENARIOS}
    norm = frozen.load(frozen.NORMALIZATION)
    models = fresh.load_models(norm)
    provenance(runtimes)
    normalization_and_serialization(runtimes, selected, norm)
    exact_tests(runtimes, selected, definitions)
    macflow_tests(runtimes, selected, definitions)
    learned_eta_tests(runtimes, selected, definitions, norm, models)
    closed_loop(runtimes, selected, definitions, norm, models)
    print(json.dumps({"status": "AUDIT_EXECUTION_COMPLETE", "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
