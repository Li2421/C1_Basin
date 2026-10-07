"""Closed-loop controller assembly for the four-agent Double-Bottleneck.

The frozen Flow-BC checkpoint remains a two-agent joint policy.  A fixed,
predeclared opposing-dyad adapter calls that same checkpoint twice per physical
step.  Both hard projections and the eta relation basis remain global.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from shared_control.diagnostic_corrector import DiagnosticCorrector, DiagnosticEta, bounded_rows
from shared_control.hard_projection import HardSafetyFilter


DEFAULT_FLOW_DYADS = ((0, 2), (1, 3))


@dataclass(frozen=True)
class FlowStep:
    raw_flow: np.ndarray
    u_flow: np.ndarray
    key_data: np.ndarray
    dyads: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ControllerStep:
    raw_flow: np.ndarray
    u_flow: np.ndarray
    u_safe: np.ndarray
    b_goal: np.ndarray
    b_rel: np.ndarray
    g_raw: np.ndarray
    w: np.ndarray
    u_exec: np.ndarray
    flow_key_data: np.ndarray
    first_projection: dict
    second_projection: dict


class FixedDyadFlowAdapter:
    """Adapt the unchanged two-agent joint Flow policy to four agents.

    This is an explicitly frozen tensor-interface adapter, not a learned
    four-agent Flow model.  Each agent occurs in exactly one opposing dyad.
    """

    def __init__(self, policy, dyads: tuple[tuple[int, int], ...] = DEFAULT_FLOW_DYADS):
        self.policy = policy
        self.dyads = tuple(tuple(map(int, pair)) for pair in dyads)
        flattened = [agent for pair in self.dyads for agent in pair]
        if sorted(flattened) != [0, 1, 2, 3]:
            raise ValueError("dyads must partition the four global agent indices")

    def sample_flow(self, env, episode_key, step: int) -> FlowStep:
        import jax
        import jax.numpy as jnp

        step_key = jax.random.fold_in(episode_key, int(step))
        raw_flow = np.empty((4, 2), dtype=np.float64)
        key_data = []
        for dyad_id, (first, second) in enumerate(self.dyads):
            pair_key = jax.random.fold_in(step_key, dyad_id)
            observation = env.pair_observation(first, second)
            output = np.asarray(
                self.policy.sample_actions(jnp.asarray(observation[None]), seed=pair_key)[0],
                dtype=np.float64,
            )
            if output.shape != (2, 2) or not np.isfinite(output).all():
                raise ValueError("frozen pair policy must return finite [2,2] actions")
            raw_flow[[first, second]] = output
            key_data.append(np.asarray(jax.random.key_data(pair_key)).copy())
        u_flow = bounded_rows(raw_flow, env.config.max_speed)
        return FlowStep(
            raw_flow=raw_flow,
            u_flow=u_flow,
            key_data=np.asarray(key_data),
            dyads=self.dyads,
        )


def load_frozen_pair_policy(checkpoint: str | Path):
    """Load the existing checkpoint without changing its architecture/config."""
    from single_integrator.evaluate import load_policy

    policy, provenance = load_policy(Path(checkpoint), allow_legacy=False)
    return policy, provenance


class DoubleBottleneckController:
    """Flow -> global projection -> eta correction -> global projection."""

    def __init__(self, policy, projection: HardSafetyFilter | None = None):
        self.flow = policy if hasattr(policy, "sample_flow") else FixedDyadFlowAdapter(policy)
        self.projection = projection or HardSafetyFilter()

    def action(
        self,
        env,
        eta: DiagnosticEta | tuple[float, float, float] | np.ndarray,
        episode_key,
        step: int,
    ) -> ControllerStep:
        eta = DiagnosticEta.from_value(eta)
        flow = self.flow.sample_flow(env, episode_key, step)
        snapshot = env.snapshot()
        first = self.projection(snapshot, flow.u_flow)
        u_safe = np.asarray(first.velocity, dtype=np.float64)
        corrector = DiagnosticCorrector(eta)
        g_raw, b_goal, b_rel = corrector(
            env.positions,
            env.goals,
            u_safe,
            env.config.max_speed,
        )
        w = u_safe + g_raw
        second = self.projection(snapshot, w)
        u_exec = np.asarray(second.velocity, dtype=np.float64)
        if any(value.shape != (4, 2) for value in (flow.u_flow, u_safe, g_raw, w, u_exec)):
            raise AssertionError("four-agent controller produced an invalid action shape")
        return ControllerStep(
            raw_flow=flow.raw_flow.copy(),
            u_flow=flow.u_flow.copy(),
            u_safe=u_safe.copy(),
            b_goal=b_goal.copy(),
            b_rel=b_rel.copy(),
            g_raw=g_raw.copy(),
            w=w.copy(),
            u_exec=u_exec.copy(),
            flow_key_data=flow.key_data.copy(),
            first_projection={"status": first.status, **first.diagnostics},
            second_projection={"status": second.status, **second.diagnostics},
        )
