"""Differentiable C1 closed loop: frozen Flow-BC, residual field, and SOCP."""
import jax
import jax.numpy as jnp


def bounded_nominal(raw, max_speed):
    raw = jnp.asarray(raw, dtype=jnp.float64).reshape(*raw.shape[:-1], 2, 2)
    norm = jnp.sqrt(jnp.maximum(jnp.sum(raw * raw, axis=-1, keepdims=True), 1e-60))
    return (raw * jnp.minimum(1., max_speed / norm)).reshape(*raw.shape[:-2], 4)


class ResidualFlowField:
    """Frozen Flow-BC sampler and post-Safety residual (C1 V0 only)."""
    def __init__(self, baseline, residual):
        self.baseline, self.residual = baseline, residual

    def correction(self, params, observations, safe):
        # Reuse the residual MLP tensor convention: h is now the physical
        # safe control and s=0, never an internal Flow integration time.
        obs, _ = self.baseline._flatten(observations, jnp.zeros_like(safe))
        return self.residual.apply(params, safe, jnp.zeros((len(safe), 1)), obs)

    def prepare(self, params, observations, noise, A, b, speed, projection):
        nominal = bounded_nominal(self.baseline_sample(observations, noise), speed)
        safe = projection(nominal, A, b, speed)
        correction = self.correction(params, observations, safe)
        candidate = safe + correction
        return dict(nominal=nominal, safe=safe, correction=correction,
                    candidate=candidate)

    def control(self, params, observations, noise, A, b, speed, projection):
        result = self.prepare(params, observations, noise, A, b, speed, projection)
        result['applied'] = projection(result['candidate'], A, b, speed)
        return result

    def baseline_sample(self, observations, noise):
        """Frozen counterpart with identical observation and flow noise."""
        with jax.experimental.disable_x64():
            observations = jnp.asarray(observations, dtype=jnp.float32)
            obs, _ = self.baseline._flatten(observations, jnp.zeros((len(observations), 2, 2), dtype=jnp.float32))
            h = jnp.asarray(noise, dtype=jnp.float32)
            for step in range(self.baseline.config['flow_steps']):
                s = jnp.full((len(h), 1), jnp.asarray(step / self.baseline.config['flow_steps'], dtype=jnp.float32), dtype=jnp.float32)
                h = h + self.baseline.network.select('actor_bc_flow')(obs, h, s) / self.baseline.config['flow_steps']
            if self.baseline.config.get('normalize', False):
                h = h * jnp.asarray(self.baseline.config['act_scale']) + jnp.asarray(self.baseline.config['act_mean'])
            return jnp.clip(h, jnp.asarray(-1., jnp.float32), jnp.asarray(1., jnp.float32))


def barrier_constraints(positions, walls, plant, cbf):
    """JAX equivalent of the authoritative 17-row CBF geometry."""
    p, walls = jnp.asarray(positions, jnp.float64), jnp.asarray(walls, jnp.float64)
    a, z = walls[:, 0], walls[:, 1]
    segment = z - a
    length2 = jnp.sum(segment * segment, axis=-1)
    q = p[:, None] - a
    fraction = jnp.clip(jnp.sum(q * segment, axis=-1) / length2, 0., 1.)
    delta = q - fraction[..., None] * segment
    distance = jnp.sqrt(jnp.maximum(jnp.sum(delta * delta, axis=-1), 1e-30))
    gradient = delta / distance[..., None]
    wall_h = distance - plant['agent_radius'] - plant['wall_radius'] - plant['wall_collision_margin'] - cbf.separation_buffer
    relative = p[0] - p[1]
    d_safe = 2 * plant['agent_radius'] + plant['agent_collision_margin'] + cbf.separation_buffer
    pair_h = jnp.dot(relative, relative) - d_safe * d_safe
    pair = jnp.concatenate((2 * relative, -2 * relative))[None]
    # Each finite-wall gradient occupies only its owning agent's control block.
    wall_rows = jnp.concatenate((jnp.concatenate((gradient[0], jnp.zeros_like(gradient[0])), -1),
                                 jnp.concatenate((jnp.zeros_like(gradient[1]), gradient[1]), -1)), axis=0)
    rows = jnp.concatenate((pair, wall_rows), axis=0)
    lower = jnp.concatenate((jnp.array([-cbf.gamma * pair_h]), (-cbf.gamma_wall * wall_h).reshape(-1)))
    h = jnp.concatenate((jnp.array([pair_h]), wall_h.reshape(-1)))
    return rows, lower, h
