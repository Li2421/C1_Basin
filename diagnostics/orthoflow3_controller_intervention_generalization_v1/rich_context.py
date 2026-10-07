"""Short, paired controller-response descriptors; no success labels are read.

The same probe RNG streams are used for base and alternative controllers.
Only the base Flow is allowed to act at t0. The summary is physically
interpretable and never runs more than eight simulator steps (at most 0.4 s).
"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

from diagnostics.orthoflow3_controller_context_loso_v1.context import Runtime

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PROTOCOL = {
    "horizon_steps": 8,
    "probe_roots": [2026100417, 2026100499],
    "first_action": "base Flow for both conditions; if committed in h, use committed raw Flow",
    "future_action": "base or compatible alternate Flow checkpoint",
    "physical_groups": ["goal_progress", "clearance", "safety_response", "correction_response"],
    "no_success_or_termination_feature": True,
    "no_scene_or_checkpoint_id": True,
}
NAMES = [
    "nominal_progress_integral", "nominal_progress_late", "nominal_goal_delta",
    "nominal_min_pair_clearance", "nominal_pair_closing_max",
    "nominal_safety_mean", "nominal_safety_max", "nominal_safety_active",
    "nominal_final_speed", "nominal_flow_change",
    "eta_progress_integral", "eta_progress_late", "eta_goal_delta",
    "eta_min_pair_clearance", "eta_pair_closing_max",
    "eta_safety_mean", "eta_safety_max", "eta_safety_active",
    "eta_correction_mean", "eta_correction_max", "eta_correction_active",
    "eta_nominal_progress_difference", "eta_nominal_final_position_distance",
    "eta_nominal_min_clearance_difference",
]


def _pair_distances(positions):
    return np.asarray([np.linalg.norm(positions[i] - positions[j])
                       for i in range(len(positions)) for j in range(i)], float)


class RichRuntime:
    def __init__(self, scene, alternate_path=None):
        self.core = Runtime(scene)
        self.scene = scene
        self.alternate_path = Path(alternate_path) if alternate_path else None
        self.alt_sha = hashlib.sha256(self.alternate_path.read_bytes()).hexdigest() if self.alternate_path else None
        self.base_flow = self.core.flow
        self.alt_flow = None
        if self.alternate_path:
            if scene == "toy_giveway":
                self.alt_flow = Runtime(scene, alternate=True).flow
            elif scene == "double_bottleneck":
                import jax
                import jax.numpy as jnp
                from double_bottleneck.flowbc_4a_agent import load_checkpoint
                import pickle

                config = pickle.load(self.alternate_path.open("rb"))["config"]
                policy, _ = load_checkpoint(self.alternate_path, config["environment_fingerprint"])
                sample = jax.jit(lambda obs, key: policy.sample_actions(obs[None], key)[0])

                def alt(env, key):
                    raw = np.asarray(sample(jnp.asarray(env.observation(), jnp.float32), key), float)
                    v = self.core.cfg.max_speed
                    return raw * np.minimum(1., v / np.maximum(np.linalg.norm(raw, axis=-1, keepdims=True), 1e-30))

                self.alt_flow = alt
            else:
                from new_benchmark_common.macflow import load_checkpoint
                from new_benchmark_common.safety_eta3 import ScenarioRuntime

                rt = self.core.rt
                expected = json.loads((ROOT / "diagnostics/orthoflow3_controller_intervention_generalization_v1/protocol.json").read_text())["profiles"][scene]["environment_fingerprint"]
                alt_agent, _ = load_checkpoint(self.alternate_path, expected_environment_fingerprint=expected)
                base_agent = rt.agent

                def alt(env, key):
                    rt.agent = alt_agent
                    try:
                        return ScenarioRuntime.flow_world(rt, env, key)
                    finally:
                        rt.agent = base_agent

                self.alt_flow = alt

    def trajectory(self, physical, eta, root):
        import jax

        rt = self.core
        env = rt.reset(physical)
        v = rt.cfg.max_speed
        previous_flow = None
        records = []
        initial_goal = float(np.linalg.norm(env.goals - env.positions, axis=1).sum())
        for t in range(PROTOCOL["horizon_steps"]):
            key = jax.random.fold_in(jax.random.PRNGKey(root), t)
            flow = (np.asarray(physical["flow"], float) if t == 0 and physical["flow_committed"]
                    else self.base_flow(env, key) if t == 0 or self.alt_flow is None
                    else self.alt_flow(env, key))
            safe = rt.project(env, flow)
            corrected = safe + rt.basis.compute(env.positions, env.goals, safe, v).correction(eta)
            executed = safe if np.array_equal(eta, np.zeros(3)) else rt.project(env, corrected)
            before = float(np.linalg.norm(env.goals - env.positions, axis=1).sum())
            dist_before = _pair_distances(env.positions)
            records.append({
                "progress": None,
                "pair_clearance": float(np.min(dist_before)),
                "pair_closing": None,
                "safety": float(np.linalg.norm(safe - flow)) / max(v, 1e-12),
                "correction": float(np.linalg.norm(executed - safe)) / max(v, 1e-12),
                "flow_change": 0. if previous_flow is None else float(np.linalg.norm(flow - previous_flow)) / max(v, 1e-12),
                "speed": float(np.linalg.norm(executed, axis=1).mean()) / max(v, 1e-12),
            })
            previous_flow = flow
            env.step(executed)
            after = float(np.linalg.norm(env.goals - env.positions, axis=1).sum())
            records[-1]["progress"] = (before - after) / max(v * rt.cfg.dt * len(env.positions), 1e-12)
            records[-1]["pair_closing"] = float(np.max((dist_before - _pair_distances(env.positions)) / max(v * rt.cfg.dt, 1e-12)))
            if env.done:
                break
        final_goal = float(np.linalg.norm(env.goals - env.positions, axis=1).sum())
        return records, np.asarray(env.positions, float), (initial_goal - final_goal) / max(initial_goal, 1e-12)

    def features(self, physical, eta):
        eta = np.asarray(eta, float)
        results = []
        for root in PROTOCOL["probe_roots"]:
            nom, nom_pos, nom_goal = self.trajectory(physical, np.zeros(3), root)
            corr, corr_pos, corr_goal = self.trajectory(physical, eta, root)

            def summarize(rows, goal):
                a = {k: np.asarray([r[k] for r in rows], float) for k in rows[0]}
                late = a["progress"][max(0, len(rows) // 2):]
                return [float(a["progress"].sum()), float(late.mean()), goal,
                        float(a["pair_clearance"].min()), float(a["pair_closing"].max()),
                        float(a["safety"].mean()), float(a["safety"].max()), float((a["safety"] > 1e-6).mean()),
                        float(a["speed"][-1]), float(a["flow_change"].mean())], a

            n, na = summarize(nom, nom_goal)
            c, ca = summarize(corr, corr_goal)
            results.append(n + c[:8] + [float(ca["correction"].mean()),
                                        float(ca["correction"].max()),
                                        float((ca["correction"] > 1e-6).mean()),
                                        c[0] - n[0],
                                        float(np.linalg.norm(corr_pos - nom_pos, axis=1).mean()),
                                        c[3] - n[3]])
        x = np.asarray(results, float)
        assert x.shape == (len(PROTOCOL["probe_roots"]), len(NAMES)), x.shape
        assert np.isfinite(x).all()
        return {"mean": x.mean(0).tolist(), "probe_std": x.std(0).tolist(),
                "steps": PROTOCOL["horizon_steps"], "roots": PROTOCOL["probe_roots"]}


def key(scene, state_uid, eta, alternate_sha):
    payload = {"scene": scene, "state_uid": state_uid, "eta": np.asarray(eta, float).tolist(),
               "alternate_sha": alternate_sha, "protocol": PROTOCOL,
               "code_sha": hashlib.sha256(inspect.getsource(RichRuntime).encode()).hexdigest()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def cached(runtime, state, eta):
    cache_key = key(runtime.scene, state["state_uid"], eta, runtime.alt_sha)
    path = OUT / "rich_response_cache" / f"{cache_key}.json"
    if path.exists():
        return json.loads(path.read_text())
    row = {"state_uid": state["state_uid"], "scene": runtime.scene,
           "eta": np.asarray(eta, float).tolist(), "alternate_sha": runtime.alt_sha}
    try:
        row["features"] = runtime.features(state["physical"], eta)
        row["valid"] = True
    except Exception as exc:
        row["features"] = None
        row["valid"] = False
        row["error"] = f"{type(exc).__name__}: {exc}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2) + "\n")
    return row


def build_pilot(scene, shard=0, shards=1):
    from .intervention import OUT, OLD, read

    protocol = read(OUT / "protocol.json")
    rows = [r for r in read(OUT / "pairs.json") if r["scene"] == scene]
    states = read(OLD / "states.json")
    profile = protocol["profiles"][scene]
    base = RichRuntime(scene)
    alt = RichRuntime(scene, profile["alternate_path"])
    results = []
    for index, row in enumerate(rows):
        if index % shards != shard:
            continue
        state = states[row["state_index"]]
        a = cached(base, state, row["eta"])
        b = cached(alt, state, row["eta"])
        valid = a["valid"] and b["valid"]
        results.append({"scene": scene, "state_uid": row["state_uid"],
                        "eta_uid": row["eta_uid"], "base_cache_key": key(scene, row["state_uid"], row["eta"], None),
                        "alt_cache_key": key(scene, row["state_uid"], row["eta"], alt.alt_sha),
                        "valid": valid,
                        "feature_distance": float(np.linalg.norm(np.asarray(a["features"]["mean"]) - np.asarray(b["features"]["mean"]))) if valid else None})
        if len(results) % 8 == 0:
            print(json.dumps({"scene": scene, "shard": shard, "done": len(results)}), flush=True)
    target = OUT / "rich_features" / f"{scene}_{shard}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"scene": scene, "shard": shard, "complete": len(results)}), flush=True)


def build_toy_swap(shard=0, shards=1):
    import jax
    from diagnostics.orthoflow3_controller_context_loso_v1.context import OLD, SWAP, load

    pairs = load(SWAP / "pair_manifest.json")
    protocol = load(SWAP / "protocol.json")
    base = RichRuntime("toy_giveway")
    alt = RichRuntime("toy_giveway", protocol["flow_paths"]["1"])
    template = dict(next(s["physical"] for s in load(OLD / "states.json") if s["scenario"] == "toy_giveway"))
    results = []
    for index, row in enumerate(pairs):
        if index % shards != shard:
            continue
        env = base.core.make()
        env.reset(np.asarray(row["initial_positions"]))
        flow = base.base_flow(env, jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(42), row["rollout_id"]), 0))
        physical = dict(template)
        physical.update(positions=env.positions.tolist(), velocities=env.velocities.tolist(),
                        goals=env.goals.tolist(), flow=flow.tolist())
        state = {"state_uid": row["state_uid"], "physical": physical}
        a, b = cached(base, state, row["eta"]), cached(alt, state, row["eta"])
        valid = a["valid"] and b["valid"]
        results.append({"scene": "toy_giveway", "state_uid": row["state_uid"],
                        "eta_uid": row["eta_uid"], "source_group": row["source_group"],
                        "valid": valid,
                        "feature_distance": float(np.linalg.norm(np.asarray(a["features"]["mean"]) - np.asarray(b["features"]["mean"]))) if valid else None})
        if len(results) % 8 == 0:
            print(json.dumps({"scene": "toy_giveway", "shard": shard, "done": len(results)}), flush=True)
    target = OUT / "rich_features" / f"toy_giveway_{shard}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"scene": "toy_giveway", "shard": shard, "complete": len(results)}), flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("pilot", "toy_swap"))
    parser.add_argument("--scene", choices=("toy_giveway", "double_bottleneck", "four_way_intersection", "ring_exchange"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if args.action == "pilot":
        build_pilot(args.scene, args.shard, args.shards)
    else:
        build_toy_swap(args.shard, args.shards)
