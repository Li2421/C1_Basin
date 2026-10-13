"""Explicit TT hybrid: physical near-goal servo, no route or teacher phase.

This is a separate controller variant, never a pure-Flow result. The original
TT/FF implementation and its cached outcomes remain unchanged.
"""
import jax.numpy as jnp
import numpy as np

from . import pilot, pipeline

RADIUS = .6


def terminal_mask(positions, goals, config):
    bounds, obstacles = pilot.geometry()
    active = []
    for p, g in zip(positions, goals):
        clear = all(pilot.segment_box_distance(p, g, box) > config.radius + config.wall_buffer
                    for box in obstacles)
        inside = np.all(g > bounds[0]+config.radius+config.wall_buffer) and np.all(g < bounds[1]-config.radius-config.wall_buffer)
        active.append(np.linalg.norm(p-g) < RADIUS and clear and inside)
    return np.array(active, bool)


class TerminalServoRuntime(pipeline.FlowRuntime):
    def action(self, positions, velocities, goals, key, eta, placement, config):
        if placement != "TT":
            raise ValueError("terminal_servo_v1 is declared only for TT")
        obs = pipeline.observation(positions, velocities, goals)
        raw_flow = pipeline.bounded(np.asarray(self.sampled(jnp.asarray(obs), key)), config.max_speed)
        active = terminal_mask(positions, goals, config)
        nominal = np.where(active[:,None], pilot.nominal(positions, goals, config), raw_flow)
        safe = pilot.project(nominal, positions, config)
        fields = pipeline.basis(positions, goals, safe, config.max_speed)
        delta = fields @ np.asarray(eta, float)
        executed = pilot.project(safe+delta, positions, config)
        return executed, dict(raw=raw_flow, safe=safe, delta=delta, fields=fields)
