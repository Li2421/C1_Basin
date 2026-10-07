"""Small full-episode runner for integration and smoke diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .controller import ControllerStep, DoubleBottleneckController
from .environment import Config, DoubleBottleneckEnv


@dataclass(frozen=True)
class RolloutRecord:
    step: int
    positions_before: np.ndarray
    control: ControllerStep
    positions_after: np.ndarray
    event: str
    info: dict


@dataclass(frozen=True)
class RolloutResult:
    eta: tuple[float, float, float]
    regime: str
    seed: int
    rollout_id: int
    summary: dict
    records: tuple[RolloutRecord, ...]
    positions: np.ndarray


def run_rollout(
    policy,
    eta,
    regime: str = "weakly_asymmetric",
    seed: int = 0,
    max_steps: int | None = None,
    projection=None,
    rollout_id: int = 0,
    config: Config | None = None,
) -> RolloutResult:
    """Run one persistent-eta closed-loop episode.

    ``max_steps`` is only a smoke-test horizon override.  With its default
    ``None``, the scenario's unchanged 850-step (42.5 s) horizon is used.
    """
    import jax

    config = config or Config(initial_regime=regime)
    if config.initial_regime != regime:
        config = replace(config, initial_regime=regime)
    if max_steps is not None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        config = replace(config, max_steps=int(max_steps))
    eta_value = tuple(float(value) for value in np.asarray(eta, dtype=np.float64))
    if len(eta_value) != 3:
        raise ValueError("eta must contain exactly three values")
    env = DoubleBottleneckEnv(config)
    controller = DoubleBottleneckController(policy, projection=projection)
    episode_key = jax.random.fold_in(jax.random.PRNGKey(seed), int(rollout_id))
    positions = [env.positions.copy()]
    records = []
    info = None
    for _ in range(config.max_steps):
        absolute_step = env.step_count
        before = env.positions.copy()
        control = controller.action(env, eta_value, episode_key, absolute_step)
        _, _, done, info = env.step(control.u_exec)
        positions.append(env.positions.copy())
        records.append(
            RolloutRecord(
                step=absolute_step,
                positions_before=before,
                control=control,
                positions_after=env.positions.copy(),
                event=str(info["termination"]),
                info=info,
            )
        )
        if done:
            break
    if info is None or not env.done:
        raise AssertionError("full-horizon rollout did not reach a terminal event")
    summary = env.summary()
    summary.update(
        termination=str(info["termination"]),
        eta=list(eta_value),
        regime=regime,
        seed=int(seed),
        rollout_id=int(rollout_id),
        rng_protocol="fold_in(fold_in(fold_in(PRNGKey(seed), rollout_id), step), dyad_id)",
        flow_dyads=[[0, 2], [1, 3]],
    )
    return RolloutResult(
        eta=eta_value,
        regime=regime,
        seed=int(seed),
        rollout_id=int(rollout_id),
        summary=summary,
        records=tuple(records),
        positions=np.asarray(positions),
    )
