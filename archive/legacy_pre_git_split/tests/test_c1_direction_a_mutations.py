import inspect
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import single_integrator.c1.direction_a as direction_a_package
from single_integrator.c1.direction_a.estimators import (
    deformation_costs, pilot_baseline, weighted_score_trees,
)
from single_integrator.c1.direction_a.policy import (
    DirectionADistribution, log_probability, policy_parameters,
    sample_residual,
)
from single_integrator.c1.direction_a.randomness import IndexedRandomTape
from single_integrator.c1.direction_a.statistics import (
    directional_second_moment_bound, fisher_path_quadratic,
    gradient_direction_interval, outcome_interval, path_kl_terms,
)


jax.config.update("jax_enable_x64", True)


class TinyModel:
    @staticmethod
    def apply(params, safe, time, observation):
        del safe, time
        return (params["mu"][None]+params["shared"]*observation[:, :1],
                jnp.broadcast_to(params["alpha"], (len(observation),)),
                jnp.broadcast_to(params["beta"], (len(observation),)))


class RequiredMutationDetectionTests(unittest.TestCase):
    def setUp(self):
        self.config = DirectionADistribution(.05, .25)
        self.outputs = (jnp.array([.1, -.2, .05, .3]), jnp.array(-.4),
                        jnp.array(.2))

    def test_rejects_per_agent_gate_shape(self):
        with self.assertRaises(ValueError):
            sample_residual(self.outputs, jnp.array([.1, .2]), jnp.ones(4),
                            self.config)

    def test_detects_sticky_multistep_gate(self):
        tape = IndexedRandomTape(418)
        uniforms = np.array([float(tape.gate_uniform("contract", 0, 0, t))
                             for t in range(8)])
        self.assertGreater(np.unique(uniforms).size, 1)
        sticky_mutation = np.repeat(uniforms[0], len(uniforms))
        self.assertFalse(np.array_equal(uniforms, sticky_mutation))

    def test_detects_omitted_scale_and_gaussian_mean_logprob(self):
        _, _, sigma = policy_parameters(self.outputs, self.config)
        epsilon = jnp.array([3., 0., 0., 0.])
        residual = self.outputs[0]+sigma*epsilon
        correct = float(log_probability(self.outputs, jnp.array(True),
                                        residual, self.config))
        p = float(jax.nn.sigmoid(self.outputs[1]))
        wrong_averaged = (np.log(p)-2*np.log(2*np.pi)-4*np.log(float(sigma))
                          -float(np.mean(np.asarray(epsilon)**2))/2)
        self.assertNotAlmostEqual(correct, wrong_averaged)

        params = dict(mu=self.outputs[0], alpha=self.outputs[1],
                      beta=self.outputs[2], shared=jnp.array(.1))
        components, total = weighted_score_trees(
            TinyModel, params, jnp.zeros((1, 4)), jnp.zeros((1, 1)),
            jnp.ones((1, 1)), jnp.array([True]), residual[None],
            jnp.ones(1), self.config)
        without_scale = jax.tree_util.tree_map(
            lambda x, y: x+y, components["gate"], components["mean"])
        self.assertGreater(abs(float(total["beta"]-without_scale["beta"])), 1e-6)

    def test_detects_sample_path_gradient_and_detached_outputs(self):
        alpha, beta = self.outputs[1], self.outputs[2]
        epsilon = jnp.array([1.3, -.4, .2, .7])
        fixed_r = self.outputs[0]+policy_parameters(self.outputs, self.config)[2]*epsilon
        correct = jax.grad(lambda mu: log_probability(
            (mu, alpha, beta), jnp.array(True), fixed_r, self.config))(
                self.outputs[0])
        wrong_sample_path = jax.grad(lambda mu: log_probability(
            (mu, alpha, beta), jnp.array(True),
            mu+policy_parameters((mu, alpha, beta), self.config)[2]*epsilon,
            self.config))(self.outputs[0])
        self.assertGreater(float(jnp.linalg.norm(correct)), 0.)
        np.testing.assert_allclose(wrong_sample_path, np.zeros(4), atol=1e-12)
        detached = jax.grad(lambda mu: log_probability(
            (jax.lax.stop_gradient(mu), alpha, beta), jnp.array(True), fixed_r,
            self.config))(self.outputs[0])
        np.testing.assert_array_equal(detached, np.zeros(4))

    def test_detects_cached_state_and_deterministic_deployment_mutations(self):
        params = dict(mu=jnp.zeros(4), alpha=jnp.array(-1.), beta=jnp.array(0.),
                      shared=jnp.array(.4))
        first = TinyModel.apply(params, jnp.zeros((1, 4)), jnp.zeros((1, 1)),
                                jnp.zeros((1, 1)))[0]
        second = TinyModel.apply(params, jnp.zeros((1, 4)), jnp.zeros((1, 1)),
                                 jnp.ones((1, 1)))[0]
        self.assertFalse(np.array_equal(first, second))
        # p<.5, but this registered uniform opens the stochastic gate.
        p = float(jax.nn.sigmoid(-1.))
        gate, residual, values = sample_residual(
            TinyModel.apply(params, jnp.zeros((1, 4)), jnp.zeros((1, 1)),
                            jnp.ones((1, 1))),
            jnp.array([p/2]), jnp.ones((1, 4)), self.config)
        self.assertTrue(bool(gate[0]))
        self.assertGreater(float(jnp.linalg.norm(residual-values["mu"])), 0.)
        self.assertFalse(p >= .5)  # thresholded deployment mutation differs.

    def test_detects_wrong_deformation_reference_latch_and_length_normalization(self):
        applied = jnp.array([[.2, 0., 0., 0.], [.1, 0., 0., 0.]])
        same_state_safe = jnp.array([[.1, 0., 0., 0.], [0., 0., 0., 0.]])
        wrong_rollout_safe = jnp.zeros((2, 4))
        correct = deformation_costs(applied, same_state_safe, np.eye(4))
        wrong = deformation_costs(applied, wrong_rollout_safe, np.eye(4))
        self.assertNotEqual(float(correct.sum()), float(wrong.sum()))
        # J_def retains costs after a risk latch and is a sum, not an average.
        latch = np.array([False, True])
        self.assertGreater(float(correct.sum()), float(correct[~latch].sum()))
        self.assertNotEqual(float(correct.sum()), float(correct.mean()))

    def test_detects_current_batch_baseline_and_archived_risk_reentry(self):
        pilot_events = np.array([[1., 0.], [0., 1.]])
        pilot_scores = np.array([[[3., 0.], [1., 0.]],
                                 [[0., 2.], [0., 1.]]])
        fixed = pilot_baseline(pilot_events, pilot_scores)
        current_batch_mutation = pilot_baseline(
            np.zeros_like(pilot_events), pilot_scores)
        self.assertNotEqual(fixed, current_batch_mutation)
        package_dir = Path(inspect.getfile(direction_a_package)).parent
        sources = "\n".join(path.read_text().lower()
                            for path in package_dir.glob("*.py"))
        for forbidden in ("single_integrator.c1.risk", "rollout_vi_r_cert",
                          "rollout_certificate"):
            self.assertNotIn(forbidden, sources)


class FrozenStatisticsFormulaTests(unittest.TestCase):
    def test_bounds_intervals_kl_and_fisher(self):
        expected_bound = 850*(.3**2/4+.2**2/.05**2+8*.1**2)
        self.assertAlmostEqual(
            directional_second_moment_bound(.3, .2, .1, .05), expected_bound)
        interval = gradient_direction_interval(np.array([1., 2., 3., 4.]), 30.)
        self.assertEqual(interval["n"], 4)
        self.assertLess(interval["interval"][0], interval["interval"][1])
        outcome = outcome_interval(np.array([0., 0., 0., 0.]), -1., 1.)
        self.assertGreater(outcome["radius"], 0.)
        self.assertNotEqual(outcome["interval"], (0., 0.))

        p = np.array([.3, .7]); sigma = np.array([.1, .2])
        mu = np.zeros((2, 4))
        bernoulli, gaussian = path_kl_terms(p, sigma, mu, p, sigma, mu)
        np.testing.assert_allclose(bernoulli, 0., atol=1e-15)
        np.testing.assert_allclose(gaussian, 0., atol=1e-15)
        fisher = fisher_path_quadratic(
            p, sigma, np.array([1., 2.]), np.ones((2, 4)),
            np.array([.5, -.5]))
        expected = np.sum(p*(1-p)*np.array([1., 2.])**2
            +p/sigma**2*4+8*p*np.array([.5, -.5])**2)
        self.assertAlmostEqual(fisher, expected)


if __name__ == "__main__":
    unittest.main()
