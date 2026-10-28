"""Constant physical lifting and finite-step admissible-field projection.

Only the copied source tree is imported. No rollout registry/cache is imported.
At a fixed physical state X, U(X) is the original convex CBF+speed action set.
V(h,s) = {w: h + (1-s) w in U(X)}. Projecting the WHOLE field onto V gives
w = [P_U(h+(1-s)*(u+delta)) - h]/(1-s).
An Euler step is a convex interpolation towards a certified point of U.
The final Euler step reaches U exactly (up to the frozen solver tolerances).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path("/home/zhihan/research/Basin_C1_flow_field_poc_20261004")
SOURCE = Path("/home/zhihan/research/Basin_C1")
sys.path.insert(0, str(SOURCE))

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from new_benchmark_common.macflow import load_checkpoint, sample_bounded_actions
from shared_control.basis_families import get_basis_family
from shared_control.hard_projection import (
    HardProjectionConfig, CBFSolverError, barrier_constraints, project_velocity,
)
from diagnostics.double_bottleneck_eta3_basin.tools.exact_projection_retry import (
    tighter_identical_socp, identical_nonlinear_polish,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class FixedSafetySet:
    """Freeze physical geometry during a pseudo-time solve; reuse exact solver."""

    def __init__(self, snapshot, config):
        self.config = config
        self.A, self.lower, self.geometry = barrier_constraints(snapshot, config)
        self.max_speed = snapshot["config"]["max_speed"]
        self.calls = 0
        if max(config.gamma, config.gamma_wall) * snapshot["config"]["dt"] > 1:
            raise ValueError("The original held-action safety condition is violated")

    def certificate(self, a):
        linear = self.A @ np.asarray(a).ravel() - self.lower
        excess = np.linalg.norm(a, axis=1).max() - self.max_speed
        return {"min_linear_residual": float(linear.min()), "max_speed_excess": float(excess),
                "max_violation": float(max(0., -linear.min(), excess))}

    def assert_safe(self, a):
        d = self.certificate(a)
        if (not np.isfinite(a).all() or d["min_linear_residual"] < -self.config.feasibility_tol
                or d["max_speed_excess"] > self.config.speed_tol):
            raise CBFSolverError("unsafe_action", d)
        return d

    def project(self, target):
        self.calls += 1
        try:
            a, status, d = project_velocity(target, self.A, self.lower, self.max_speed, self.config)
        except CBFSolverError as first:
            try:
                a, status, d = tighter_identical_socp(target, self.A, self.lower, self.max_speed, self.config)
            except CBFSolverError:
                initial = first.details.get("solver_output_candidate", target)
                a, status, d = identical_nonlinear_polish(target, self.A, self.lower, self.max_speed, self.config, initial)
        self.assert_safe(a)
        return a, {**d, "status": status}

    def project_field_target(self, target):
        # Exact closed form for the product of speed balls. If it also meets
        # every CBF row it is the exact intersection projection, by set
        # inclusion and uniqueness. No tolerance or physical bound is relaxed.
        # Keep the original reference/old-policy projector untouched.
        a = np.asarray(target, dtype=np.float64).copy()
        a *= np.minimum(1., self.max_speed / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-30))
        if np.min(self.A @ a.ravel() - self.lower) >= 0.:
            self.calls += 1
            d = self.assert_safe(a)
            return a, {**d, "status": "exact_speed_ball_projection_linear_feasible"}
        return self.project(target)


def admissible_step(h, nominal_field, s, ds, safety):
    """Euclidean projection of a field, followed by one certified Euler step."""
    remaining = 1.0 - float(s)
    if remaining <= 0 or ds <= 0 or ds > remaining + 1e-12:
        raise ValueError("Invalid pseudo-time grid")
    projector = getattr(safety, "project_field_target", safety.project)
    target, diagnostic = projector(h + remaining * nominal_field)
    field = (target - h) / remaining
    rho = min(1.0, ds / remaining)
    # Algebraically h + ds*field. This form avoids cancellation on the last step.
    next_h = (1.0 - rho) * h + rho * target
    return next_h, field, target, diagnostic


class Runtime:
    def __init__(self, scenario):
        self.scenario = scenario
        if scenario == "four_way_intersection":
            ds, cp = "base_u_v13_broad_global_dataset", "base_u_v13_broad_global_source_balanced_macflow"
            from four_way_intersection.environment import Config, FourWayIntersectionEnv
            self.env_cls, self.config_cls = FourWayIntersectionEnv, Config
        elif scenario == "ring_exchange":
            ds, cp = "base_u_v10_local_dataset", "base_u_v10_local_macflow"
            from ring_exchange.environment import LocalFrameConfig, RingExchangeEnv
            self.env_cls, self.config_cls = RingExchangeEnv, LocalFrameConfig
        else:
            raise ValueError(scenario)
        base = ROOT / "assets/diagnostics" / f"{scenario}_stage1"
        self.manifest = json.loads((base / ds / "manifest.json").read_text())
        self.config = self.config_cls(**self.manifest["scenario_config"])
        self.checkpoint = base / cp / "best.pkl"
        self.agent, _ = load_checkpoint(self.checkpoint, expected_environment_fingerprint=self.manifest["environment_fingerprint"])
        self.basis = get_basis_family("orthoflow3")
        self.cbf = HardProjectionConfig()
        self.steps = int(self.agent.config["flow_steps"])
        c = self.agent.config
        if not c["normalize"]:
            raise ValueError("POC explicitly expects the frozen normalized checkpoints")
        self.scale = np.asarray(c["act_scale"], dtype=np.float64).reshape(-1, 2)
        self.mean = np.asarray(c["act_mean"], dtype=np.float64).reshape(-1, 2)
        if np.any(self.scale <= 0):
            raise ValueError("Noninvertible action normalization")

        @jax.jit
        def trace(obs, key):
            flat = obs.reshape(1, c["joint_obs_dim"])
            flat = (flat - jnp.asarray(c["obs_mean"])) / jnp.asarray(c["obs_scale"])
            z = jax.random.normal(key, (1, c["joint_act_dim"]))
            path = [z[0]]
            for k in range(self.steps):
                t = jnp.full((1, 1), k / self.steps)
                z = z + self.agent.network.select("actor_bc_flow")(flat, z, t) / self.steps
                path.append(z[0])
            return jnp.stack(path)

        @jax.jit
        def velocity(obs, z, s):
            flat = obs.reshape(1, c["joint_obs_dim"])
            flat = (flat - jnp.asarray(c["obs_mean"])) / jnp.asarray(c["obs_scale"])
            return self.agent.network.select("actor_bc_flow")(flat, z.reshape(1, -1), s.reshape(1, 1))[0]

        self.trace_fn, self.velocity_fn = trace, velocity
        self.native_fn = jax.jit(lambda obs, key: sample_bounded_actions(self.agent, obs[None], key)[0])

    def make_env(self, state):
        env = self.env_cls(self.config)
        p = state["physical"]
        if self.scenario == "four_way_intersection":
            env.reset(np.asarray(p["positions"]), np.asarray(p["velocities"]))
            np.testing.assert_allclose(env.goals, p["goals"], rtol=0, atol=1e-12)
        else:
            env.reset(np.asarray(p["positions"]), velocities=np.asarray(p["velocities"]), goals=np.asarray(p["goals"]))
        if p.get("timestep", 0) != 0:
            raise ValueError("This POC only restores true-t0 states, not hidden histories")
        return env

    def observation(self, env):
        if self.scenario == "four_way_intersection":
            return env.observation()
        from ring_exchange.local_frame import local_observation
        return local_observation(env.positions, env.velocities, env.goals, env.config)

    def to_world(self, a, env):
        if self.scenario == "ring_exchange":
            from ring_exchange.local_frame import local_actions_to_world
            return local_actions_to_world(a, env.positions)
        return np.asarray(a)

    def to_local(self, a, env):
        if self.scenario == "ring_exchange":
            from ring_exchange.local_frame import world_actions_to_local
            return world_actions_to_local(a, env.positions)
        return np.asarray(a)

    def safety(self, env):
        if self.scenario == "ring_exchange":
            from ring_exchange.safety import polygonal_obstacle_snapshot
            snapshot = polygonal_obstacle_snapshot(env, sides=48)
        else:
            snapshot = env.snapshot()
        return FixedSafetySet(snapshot, self.cbf)

    def nominal(self, env, key, traced=False):
        obs = jnp.asarray(self.observation(env))
        # Preserve all original clipping, JAX speed bound and float64 re-bound.
        raw = np.asarray(self.native_fn(obs, key), dtype=np.float64)
        flow = self.to_world(raw, env).copy()
        flow *= np.minimum(1., self.config.max_speed / np.maximum(np.linalg.norm(flow, axis=1, keepdims=True), 1e-30))
        path = None
        if traced:
            zpath = np.asarray(self.trace_fn(obs, key)).reshape(self.steps + 1, -1, 2)
            path = np.stack([self.to_world(self.mean + self.scale * z, env) for z in zpath])
            # Independent equivalence check of the reconstructed native sampler.
            decoded = np.clip(zpath[-1] * self.scale + self.mean, -1., 1.)
            decoded *= np.minimum(1., self.config.max_speed / np.maximum(np.linalg.norm(decoded, axis=1, keepdims=True), 1e-30))
            decoded = self.to_world(decoded, env)
            np.testing.assert_allclose(decoded, flow, rtol=1e-10, atol=1e-11)
        return obs, flow, path

    def prepare(self, env, key, eta, traced=False):
        obs, flow, path = self.nominal(env, key, traced)
        safety = self.safety(env)
        safe, _ = safety.project(flow)
        delta = self.basis.compute(env.positions, env.goals, safe, self.config.max_speed).correction(eta)
        return obs, safety, flow, safe, delta, path

    def integrate(self, env, obs, key, delta, safety, *, project=True, traced=False):
        z0 = np.asarray(jax.random.normal(key, (1, self.agent.config["joint_act_dim"]))).reshape(-1, 2)
        h = self.to_world(self.mean + self.scale * z0, env)
        path, fields, targets, records = [h.copy()], [], [], []
        for k in range(self.steps):
            s, ds = k / self.steps, 1. / self.steps
            z = (self.to_local(h, env) - self.mean) / self.scale
            fz = np.asarray(self.velocity_fn(obs, jnp.asarray(z), jnp.asarray(s))).reshape(-1, 2)
            u = self.to_world(self.scale * fz, env)
            proposed = u + delta  # constant physical g_eta; translation has zero derivative
            if project:
                new_h, w, target, d = admissible_step(h, proposed, s, ds, safety)
            else:
                new_h, w, target, d = h + ds * proposed, proposed, h + ds * proposed, {}
            if traced:
                fields.append(w.copy()); targets.append(target.copy())
                records.append({"s": s, "field_projection_norm": float(np.linalg.norm(w - proposed)),
                                "target_certificate": safety.certificate(target) if project else None,
                                "solver_status": d.get("status")})
            h = new_h
            path.append(h.copy())
        if project:
            safety.assert_safe(h)  # assertion only; no terminal correction/clipping
        return h, {"path": np.asarray(path), "fields": np.asarray(fields),
                   "targets": np.asarray(targets), "steps": records}

    def action(self, env, key, eta, chain):
        obs, safety, flow, safe, delta, _ = self.prepare(env, key, eta)
        if chain == "old":
            executed, _ = safety.project(safe + delta)
        elif chain == "field":
            executed, _ = self.integrate(env, obs, key, delta, safety)
        else:
            raise ValueError(chain)
        return executed, safety.assert_safe(executed), safety.calls

    def paired_probe(self, env, key, eta):
        obs, safety, flow, safe, delta, old_path = self.prepare(env, key, eta, traced=True)
        old, _ = safety.project(safe + delta)
        new, trace = self.integrate(env, obs, key, delta, safety, traced=True)
        lifted, no_safety_trace = self.integrate(env, obs, key, delta, safety, project=False, traced=True)
        np.testing.assert_array_equal(trace["path"][0], no_safety_trace["path"][0])
        np.testing.assert_allclose(trace["path"][0], old_path[0], atol=1e-12, rtol=0)
        distances = []
        for h in trace["path"]:
            projected, _ = safety.project_field_target(h)
            distances.append(float(np.linalg.norm(h - projected)))
        dist = np.asarray(distances)
        # A certified witness avoids mistaking optimization error in repeated
        # distance measurements for a failure of the distance contraction.
        witness, _ = safety.project_field_target(trace["path"][0])
        witnesses = [witness.copy()]
        for k, target in enumerate(trace["targets"]):
            rho = 1. / (self.steps-k)
            witness = (1.-rho)*witness + rho*target
            safety.assert_safe(witness)
            witnesses.append(witness.copy())
        witnesses = np.asarray(witnesses)
        bound = np.arange(self.steps, -1, -1) / self.steps * np.linalg.norm(trace["path"][0]-witnesses[0])
        witness_distance = np.linalg.norm((trace["path"]-witnesses).reshape(self.steps+1,-1),axis=1)
        if np.max(witness_distance - bound) > 1e-11:
            raise AssertionError("Certified finite-time witness envelope violated")
        return {"flow": flow, "safe_reference": safe, "delta": delta, "old_action": old,
                "new_action": new, "terminal_l2": float(np.linalg.norm(old-new)),
                "old_flow_path": old_path, "new_flow_path": trace["path"],
                "lift_only_path": no_safety_trace["path"], "lift_only_terminal": lifted,
                "projected_fields": trace["fields"], "safe_targets": trace["targets"],
                "approx_distance_to_safe_set": dist, "distance_envelope": bound,
                "safe_witness_path": witnesses, "certified_distance_upper_bound": witness_distance,
                "projection_steps": trace["steps"], "new_certificate": safety.assert_safe(new)}
