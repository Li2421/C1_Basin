"""Physical-only 3-D MACFlow -> hard projection -> fixed-eta correction.

No route, teacher, phase, or deadlock-triggered fallback is used by the actor.
TT and FF reproduce the repository's distinct terminal/field semantics.
"""
from dataclasses import asdict
import hashlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from new_benchmark_common.macflow import load_checkpoint, sample_bounded_actions
from shared_control.basis_families import ORTHOFLOW3_SCALE
from shared_rollout_db.src.rollout_db import uid
from . import pilot


def environment_fingerprint():
    return uid("drone3d", dict(config=asdict(pilot.Config()), geometry=[a.tolist() for a in pilot.geometry()],
                              simulator_sha256=hashlib.sha256(Path(pilot.__file__).read_bytes()).hexdigest()))


def observation(positions, velocities, goals):
    p, v, g = (np.asarray(x, dtype=float) for x in (positions, velocities, goals))
    if p.shape != (2, 3) or v.shape != p.shape or g.shape != p.shape:
        raise ValueError("expected two agents with three physical coordinates")
    return np.concatenate([p, v, g-p, p[::-1]-p], axis=1).astype(np.float32)


def bounded(u, speed):
    u = np.asarray(u, dtype=float)
    return u * np.minimum(1., speed / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-30))


def basis(positions, goals, safe, speed):
    """Version: orthoflow3_xyz_v1; algebraic dimension lift, old scale explicit.

    Three eta coefficients are independent of physical space dimension.
    The perpendicular column inherits actual lateral/vertical Flow components.
    """
    goal = bounded(np.asarray(goals)-positions, speed)
    denominator = np.sum(goal*goal, axis=1, keepdims=True) + (1e-6*speed)**2
    perp = ORTHOFLOW3_SCALE * (safe - np.sum(safe*goal, axis=1, keepdims=True)/denominator*goal)
    relation = bounded(np.asarray(positions)-np.asarray(positions)[::-1], speed)
    return np.stack([goal, perp, relation], axis=-1)  # [N,3,eta_dim=3]


class FlowRuntime:
    def __init__(self, checkpoint):
        self.checkpoint = str(Path(checkpoint).resolve())
        self.agent, self.metadata = load_checkpoint(checkpoint, expected_environment_fingerprint=environment_fingerprint())
        c = self.agent.config
        if (c["num_agents"], c["obs_dim"], c["act_dim"], c["flow_steps"]) != (2,12,3,10):
            raise ValueError("incompatible 3-D checkpoint")
        self.mean = np.asarray(c["act_mean"], float).reshape(2,3)
        self.scale = np.asarray(c["act_scale"], float).reshape(2,3)

        @jax.jit
        def sampled(obs, key):
            return sample_bounded_actions(self.agent, obs[None], key)[0]

        @jax.jit
        def velocity(obs, z, t):
            x = (obs.reshape(1,24)-jnp.asarray(c["obs_mean"]))/jnp.asarray(c["obs_scale"])
            return self.agent.network.select("actor_bc_flow")(x, z.reshape(1,6), t.reshape(1,1))[0].reshape(2,3)

        self.sampled, self.velocity = sampled, velocity

    def action(self, positions, velocities, goals, key, eta, placement, config):
        obs = observation(positions, velocities, goals)
        raw = bounded(np.asarray(self.sampled(jnp.asarray(obs),key)), config.max_speed)
        safe = pilot.project(raw, positions, config)
        fields = basis(positions, goals, safe, config.max_speed)
        delta = fields @ np.asarray(eta, float)
        if placement == "TT":
            executed = pilot.project(safe+delta, positions, config)
        elif placement == "FF":
            # Same Flow key/initial noise, frozen physical snapshot and delta.
            h = self.mean+self.scale*np.asarray(jax.random.normal(key,(1,6))).reshape(2,3)
            for k in range(10):
                s=k/10.; remaining=1.-s
                z=(h-self.mean)/self.scale
                field=self.scale*np.asarray(self.velocity(jnp.asarray(obs),jnp.asarray(z),jnp.asarray(s)))
                target=pilot.project(h+remaining*(field+delta),positions,config)
                rho=min(1.,.1/remaining)
                h=(1.-rho)*h+rho*target
            # Final rho=1 guarantees membership; no terminal projection/clip.
            executed=h
            matrix,lower=pilot.constraints(positions,config)
            if ((matrix@executed.ravel()-lower).min() < -1e-8
                    or np.linalg.norm(executed,axis=1).max()>config.max_speed+1e-8):
                raise RuntimeError("FF final safety certificate failed")
        else:
            raise ValueError("placement must be TT or FF")
        return executed, dict(raw=raw,safe=safe,delta=delta,fields=fields)


def rollout(state, runtime, eta, placement="TT", seed=0, config=None):
    config = config or pilot.Config()
    p=np.asarray(state["positions"],float); v=np.asarray(state["velocities"],float); g=np.asarray(state["goals"],float)
    if np.min(pilot.swept_clearances(p,p,config)) <= 0:
        raise ValueError("unsafe initial state")
    path=[p.copy()]; actions=[]; margins=[]; raw=[]; safe=[]; corrections=[]
    error=None; event=None; termination="timeout"
    root=jax.random.PRNGKey(seed)
    for step in range(config.max_steps):
        try:
            u,d=runtime.action(p,v,g,jax.random.fold_in(root,step),eta,placement,config)
        except (ValueError,RuntimeError,FloatingPointError) as exc:
            error=f"{type(exc).__name__}: {exc}"; termination="numerical_failure"; break
        q=p+config.dt*u
        clearance=pilot.swept_clearances(p,q,config)
        path.append(q.copy()); actions.append(u.copy()); margins.append(clearance)
        raw.append(d["raw"]); safe.append(d["safe"]); corrections.append(d["delta"])
        p=q; v=u
        if clearance.min() < -1e-9:
            termination="collision"; break
        if (np.linalg.norm(p-g,axis=1)<=config.goal_tol).all():
            termination="success"; break
        if pilot.blocked(path,actions,g,config):
            termination="deadlock"; event=step+1; break
    trace=dict(positions=np.asarray(path),actions=np.asarray(actions),goals=g,
               raw_flow=np.asarray(raw),safe_flow=np.asarray(safe),corrections=np.asarray(corrections),
               swept_clearance=np.asarray(margins),steps=np.arange(len(path)),
               bounds=pilot.geometry()[0],obstacles=pilot.geometry()[1])
    mins=np.min(margins,axis=0).tolist() if margins else [None]*3
    result=dict(outcome=termination,success=termination=="success",deadlock=termination=="deadlock",
                timeout=termination=="timeout",collision=termination=="collision",
                numerical_failure=termination=="numerical_failure",numerical_error=error,
                episode_length=len(actions),deadlock_event_step=event,config=asdict(config),
                minimum_wall_clearance=mins[0],minimum_agent_clearance=mins[1],minimum_pair_ellipsoid_gap=mins[2],
                final_goal_distances=np.linalg.norm(p-g,axis=1).tolist(),
                checkpoint=runtime.checkpoint,placement=placement,online_teacher=False,
                deadlock_criterion="pilot mutual_block_v1 unchanged",basis="orthoflow3_xyz_v1")
    return trace,result
