"""Check the certificate, not merely parity with its implementation."""
import jax
import jax.numpy as jnp
import numpy as np
import unittest
from single_integrator.c1.risk.progress_debt import certificate, goal_potential


def test_timeout_bound_on_random_nonnegative_paths():
    rng = np.random.default_rng(19)
    for _ in range(30):
        v = np.maximum(0., 5.+np.cumsum(rng.normal(0., .04, 851)))
        result = certificate(jnp.asarray(v), jnp.ones(850, bool))
        assert float(result['timeout_bound']) >= 1.-2e-5


def test_low_speed_interval_implies_deadlock_bound_even_after_recovery():
    # The fastest decrease permitted by the supplied Lipschitz bound during
    # 101 slow actions. Afterward the task completes and is absorbed.
    v = np.maximum(0., 6.-.04*np.arange(851))
    v[50:152] = v[50]-.1*.05*np.arange(102)
    v[152:] = np.maximum(0., v[151]-.04*np.arange(1, 700))
    alive = np.arange(850) < np.flatnonzero(v == 0)[0]
    result = certificate(jnp.asarray(v), jnp.asarray(alive))
    assert float(result['deadlock_bound']) >= 1.-2e-5
    assert v[-1] == 0


def test_fast_completion_can_have_zero_risk():
    v = jnp.maximum(0., 4.-.025*jnp.arange(851))
    result = certificate(v, jnp.arange(850) < 160)
    assert float(result['J_live']) < 1e-5


def test_potential_lipschitz_and_finite_goal_gradient():
    rng = np.random.default_rng(23)
    goals = jnp.array([[1.09, 0.], [-1.09, 0.]])
    x = rng.normal(size=(100, 2, 2)); delta = rng.normal(size=x.shape)*.02
    change = np.abs(np.asarray(goal_potential(x+delta, goals)-goal_potential(x, goals)))
    assert np.all(change <= 4*np.max(np.linalg.norm(delta, axis=-1), axis=-1)+1e-5)
    assert np.isfinite(jax.grad(lambda z: goal_potential(z, goals))(goals)).all()


def test_invalid_certificate_is_not_silently_rescaled():
    with unittest.TestCase().assertRaises(ValueError):
        certificate(jnp.zeros(851), jnp.ones(850, bool), target_rate=.1)
    result = certificate(jnp.ones(851)*20, jnp.ones(850, bool))
    assert np.isnan(float(result['J_live']))


if __name__ == '__main__':
    suite = unittest.TestSuite(unittest.FunctionTestCase(fn) for name, fn in list(globals().items())
                               if name.startswith('test_') and callable(fn))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)

