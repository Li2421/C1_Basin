import itertools
import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import norm

from single_integrator.c1.direction_a.estimators import (
    deformation_costs, log_probability_components, pilot_baseline,
    risk_gradient_contributions, weighted_score_trees,
)
from single_integrator.c1.direction_a.policy import (
    DirectionADistribution, initialization_logits, log_probability, policy_parameters,
    sample_residual,
)
from single_integrator.c1.direction_a.randomness import IndexedRandomTape


jax.config.update("jax_enable_x64", True)


class LinearModel:
    """Small independent test model with a parameter shared by all heads."""
    mean_direction = jnp.array([1.0, -0.5, 0.25, 2.0])

    @staticmethod
    def apply(params, safe, s, observations):
        del safe, s
        x = observations
        shared = params["shared"]
        mu = params["mean"][None, :] + shared*x[:, 1, None]*LinearModel.mean_direction
        alpha = params["alpha"] + shared*x[:, 0]
        beta = params["beta"] + shared*x[:, 2]
        return mu, alpha, beta


class SamplingAndPointScoreTests(unittest.TestCase):
    def setUp(self):
        self.config = DirectionADistribution(0.05, 0.25)

    def test_one_joint_gate_and_exact_active_gaussian(self):
        outputs = (jnp.array([0.1, -0.2, 0.3, -0.4]), jnp.array(0.0),
                   jnp.array(0.0))
        epsilon = jnp.array([1.0, 2.0, -1.0, 0.5])
        off, r_off, _ = sample_residual(outputs, jnp.array(0.75), epsilon,
                                         self.config)
        on, r_on, values = sample_residual(outputs, jnp.array(0.25), epsilon,
                                            self.config)
        self.assertEqual(np.asarray(off).shape, ())
        self.assertFalse(bool(off))
        np.testing.assert_array_equal(r_off, np.zeros(4))
        self.assertTrue(bool(on))
        np.testing.assert_allclose(r_on, outputs[0]+values["sigma"]*epsilon,
                                   rtol=0, atol=0)
        with self.assertRaises(ValueError):
            sample_residual(outputs, jnp.zeros(2), epsilon, self.config)

    def test_approved_physical_initial_values_map_back_exactly(self):
        alpha, beta = initialization_logits(.2, .1, self.config)
        _, p, sigma = policy_parameters(
            (jnp.zeros(4), jnp.asarray(alpha), jnp.asarray(beta)), self.config)
        self.assertAlmostEqual(float(p), .2, places=15)
        self.assertAlmostEqual(float(sigma), .1, places=15)

    def test_indexed_sources_replay_without_consumption_shift(self):
        tape = IndexedRandomTape(9271)
        first = np.asarray(tape.flow_noise("P", 3, 2, 8))
        repeat = np.asarray(tape.flow_noise("P", 3, 2, 8))
        np.testing.assert_array_equal(first, repeat)
        self.assertFalse(np.array_equal(first,
            np.asarray(tape.gaussian_residual("P", 3, 2, 8))))
        # Whether time 8's gate is used cannot affect the address at time 9.
        later = np.asarray(tape.flow_noise("P", 3, 2, 9))
        _ = tape.gate_uniform("P", 3, 2, 8)
        np.testing.assert_array_equal(later,
            np.asarray(tape.flow_noise("P", 3, 2, 9)))
        self.assertFalse(np.array_equal(first, later))

    def test_exact_log_probability_and_all_parameter_scores(self):
        params = dict(alpha=jnp.array(-0.3), beta=jnp.array(0.2),
                      mean=jnp.array([0.1, -0.2, 0.05, 0.3]),
                      shared=jnp.array(0.4))
        observation = jnp.array([[0.7, -0.5, 0.2]])
        safe, time = jnp.zeros((1, 4)), jnp.zeros((1, 1))
        outputs = LinearModel.apply(params, safe, time, observation)
        mu, p, sigma = policy_parameters(outputs, self.config)
        epsilon = jnp.array([[0.5, -1.0, 0.25, 2.0]])
        residual = mu+sigma[:, None]*epsilon
        gate = jnp.array([True])
        observed = float(log_probability(outputs, gate, residual, self.config)[0])
        expected = (math.log(float(p[0]))-2*math.log(2*math.pi)
                    -4*math.log(float(sigma[0]))
                    -float(np.sum(np.asarray(epsilon)**2))/2)
        self.assertAlmostEqual(observed, expected, places=12)

        components, total = weighted_score_trees(
            LinearModel, params, safe, time, observation, gate, residual,
            jnp.ones(1), self.config)
        dlog_sigma_dbeta = ((self.config.sigma_max-self.config.sigma_min)
            *jax.nn.sigmoid(outputs[2])[0]*(1-jax.nn.sigmoid(outputs[2])[0])
            /sigma[0])
        expected_gate = 1-p[0]
        expected_mean = np.asarray(epsilon[0]/sigma[0])
        expected_scale = (float(np.sum(np.asarray(epsilon)**2))-4)*dlog_sigma_dbeta
        self.assertAlmostEqual(float(components["gate"]["alpha"]),
                               float(expected_gate), places=12)
        np.testing.assert_allclose(components["mean"]["mean"], expected_mean,
                                   rtol=0, atol=2e-12)
        self.assertAlmostEqual(float(components["scale"]["beta"]),
                               float(expected_scale), places=12)
        expected_shared = (expected_gate*observation[0, 0]
            + jnp.vdot(expected_mean,
                       observation[0, 1]*LinearModel.mean_direction)
            + expected_scale*observation[0, 2])
        self.assertAlmostEqual(float(total["shared"]), float(expected_shared),
                               places=11)

    def test_inactive_sample_has_gate_score_only(self):
        params = dict(alpha=jnp.array(0.4), beta=jnp.array(-0.7),
                      mean=jnp.ones(4), shared=jnp.array(0.2))
        safe, time, obs = jnp.zeros((1, 4)), jnp.zeros((1, 1)), jnp.ones((1, 3))
        components, _ = weighted_score_trees(
            LinearModel, params, safe, time, obs, jnp.array([False]),
            jnp.zeros((1, 4)), jnp.ones(1), self.config)
        for key in ("mean", "scale"):
            for leaf in jax.tree_util.tree_leaves(components[key]):
                np.testing.assert_array_equal(leaf, np.zeros_like(leaf))
        self.assertNotEqual(float(components["gate"]["alpha"]), 0.0)

    def test_branch_is_taken_from_gate_even_if_active_residual_is_zero(self):
        outputs = (jnp.zeros(4), jnp.array(0.), jnp.array(0.))
        active = float(log_probability(outputs, jnp.array(True), jnp.zeros(4),
                                       self.config))
        inactive = float(log_probability(outputs, jnp.array(False), jnp.zeros(4),
                                         self.config))
        self.assertNotEqual(active, inactive)

    def test_scale_score_below_equal_above_four(self):
        beta = jnp.array(0.1)
        for norm2, expected_sign in ((1.0, -1), (4.0, 0), (9.0, 1)):
            epsilon = jnp.array([[math.sqrt(norm2), 0., 0., 0.]])
            outputs = (jnp.zeros((1, 4)), jnp.zeros(1), beta[None])
            _, _, sigma = policy_parameters(outputs, self.config)
            r = sigma[:, None]*epsilon
            f = lambda value: jnp.sum(log_probability_components(
                (outputs[0], outputs[1], value[None]), jnp.array([True]), r,
                self.config)[2])
            score = float(jax.grad(f)(beta))
            if expected_sign == 0:
                self.assertLess(abs(score), 2e-15)
            else:
                self.assertEqual(int(np.sign(score)), expected_sign)


class MomentAndOracleTests(unittest.TestCase):
    def setUp(self):
        self.config = DirectionADistribution(0.04, 0.30)
        self.nodes = np.array([-math.sqrt(3), 0., math.sqrt(3)])
        self.weights = np.array([1/6, 2/3, 1/6])

    def test_conditional_zero_mean_fisher_and_zero_cross_moments(self):
        theta = jnp.array([0.2, -0.1, 0.3])
        A = jnp.array([0.7, -0.4, 0.2])
        J = jnp.array([[.3, -.2, .1], [0., .5, -.1],
                       [-.4, .2, .6], [.1, .3, -.2]])
        q = jnp.array([-.2, .5, .4])

        def outputs(value):
            return J@value, A@value, q@value
        base = outputs(theta)
        _, p, sigma = policy_parameters(base, self.config)
        dlog_dbeta = ((self.config.sigma_max-self.config.sigma_min)
            *jax.nn.sigmoid(base[2])*(1-jax.nn.sigmoid(base[2]))/sigma)
        C = dlog_dbeta*q

        records = []
        for b in (0, 1):
            pb = float(p if b else 1-p)
            for indices in itertools.product(range(3), repeat=4):
                eps = jnp.asarray(self.nodes[list(indices)])
                weight = pb*float(np.prod(self.weights[list(indices)]))
                fixed_r = jax.lax.stop_gradient(base[0]+sigma*eps) if b else jnp.zeros(4)
                def terms(value):
                    return jnp.stack(log_probability_components(
                        outputs(value), jnp.asarray(bool(b)), fixed_r,
                        self.config))
                components = np.asarray(jax.jacrev(terms)(theta))
                records.append((weight, components))
        mean = sum(w*c.sum(axis=0) for w, c in records)
        second = sum(w*np.outer(c.sum(axis=0), c.sum(axis=0)) for w, c in records)
        expected = (float(p*(1-p))*np.outer(A, A)
                    +float(p/(sigma*sigma))*(np.asarray(J).T@np.asarray(J))
                    +8*float(p)*np.outer(C, C))
        np.testing.assert_allclose(mean, np.zeros(3), rtol=0, atol=2e-12)
        np.testing.assert_allclose(second, expected, rtol=0, atol=3e-12)
        for left, right in ((0, 1), (0, 2), (1, 2)):
            cross = sum(w*np.outer(c[left], c[right]) for w, c in records)
            np.testing.assert_allclose(cross, np.zeros((3, 3)), rtol=0,
                                       atol=2e-12)

    def test_nonzero_indicator_oracle_for_negative_zero_positive_z(self):
        alpha = -0.25
        beta = 0.35
        _, p, sigma = policy_parameters(
            (jnp.zeros(4), jnp.asarray(alpha), jnp.asarray(beta)), self.config)
        p, sigma = float(p), float(sigma)
        dsigma_dbeta = ((self.config.sigma_max-self.config.sigma_min)
            *float(jax.nn.sigmoid(beta)*(1-jax.nn.sigmoid(beta))))
        dlog_sigma_dbeta = dsigma_dbeta/sigma
        threshold = 0.2
        for z in (-1.2, 0., 1.1):
            mu = threshold-z*sigma
            tail, density = norm.sf(z), norm.pdf(z)
            expected = np.array([
                p*(1-p)*tail,
                p*density/sigma,
                p*z*density*dlog_sigma_dbeta,
            ])
            def objective(values):
                a, m, be = values
                _, pp, ss = policy_parameters(
                    (jnp.zeros(4), a, be), self.config)
                zz = (threshold-m)/ss
                return pp*0.5*jax.scipy.special.erfc(zz/jnp.sqrt(2.0))
            actual = jax.grad(objective)(jnp.array([alpha, mu, beta]))
            np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
            shared_direction = np.array([.4, -.7, .2])
            self.assertAlmostEqual(float(actual@shared_direction),
                                   float(expected@shared_direction), places=12)

    def test_two_step_first_gate_event_and_termination_action_score(self):
        p = 0.37
        # Complete paths are 1, 01 and 00.  No score is appended after path 1.
        paths = [
            (p, 1, 1-p),
            ((1-p)*p, 1, -p+(1-p)),
            ((1-p)**2, 0, -2*p),
        ]
        probability = sum(weight*event for weight, event, _ in paths)
        derivative = sum(weight*event*score for weight, event, score in paths)
        self.assertAlmostEqual(probability, 1-(1-p)**2, places=15)
        self.assertAlmostEqual(derivative, 2*p*(1-p)**2, places=15)

    def test_deformation_value_and_score_oracle(self):
        config = self.config
        alpha, beta = -0.2, 0.3
        mu = jnp.array([.1, -.2, .05, .15])
        outputs = (mu, jnp.asarray(alpha), jnp.asarray(beta))
        _, p, sigma = policy_parameters(outputs, config)
        dlog_sigma_dbeta = ((config.sigma_max-config.sigma_min)
            *jax.nn.sigmoid(beta)*(1-jax.nn.sigmoid(beta))/sigma)
        expected_value = .05*p*(jnp.vdot(mu, mu)+4*sigma*sigma)
        expected_gradient = np.r_[
            .05*p*(1-p)*(jnp.vdot(mu, mu)+4*sigma*sigma),
            np.asarray(.1*p*mu),
            .05*p*8*sigma*sigma*dlog_sigma_dbeta,
        ]
        estimate = np.zeros(6)
        value = 0.0
        for b in (0, 1):
            pb = float(p if b else 1-p)
            for indices in itertools.product(range(3), repeat=4):
                eps = jnp.asarray(self.nodes[list(indices)])
                weight = pb*float(np.prod(self.weights[list(indices)]))
                r = mu+sigma*eps if b else jnp.zeros(4)
                cost = .05*float(jnp.vdot(r, r))
                def packed(v):
                    terms = log_probability_components(
                        (v[1:5], v[0], v[5]), jnp.asarray(bool(b)), r, config)
                    return jnp.stack(terms)
                scores = np.asarray(jax.jacrev(packed)(
                    jnp.r_[jnp.asarray(alpha), mu, jnp.asarray(beta)])).sum(axis=0)
                value += weight*cost
                estimate += weight*cost*scores
        self.assertAlmostEqual(value, float(expected_value), places=13)
        np.testing.assert_allclose(estimate, expected_gradient,
                                   rtol=0, atol=3e-12)

    def test_deformation_edge_cases_and_pilot_baseline(self):
        metric = np.eye(4)
        safe = jnp.zeros((1, 4))
        self.assertEqual(float(deformation_costs(safe, safe, metric)[0]), 0.)
        # A nonzero raw residual may be projected back to the same baseline.
        self.assertEqual(float(deformation_costs(safe, safe, metric)[0]), 0.)
        applied = jnp.array([[.1, -.2, .3, -.4]])
        self.assertGreater(float(deformation_costs(applied, safe, metric)[0]), 0.)
        events = np.array([[1., 0.], [0., 0.]])
        scores = np.array([[[2., 0.], [1., 0.]], [[0., 1.], [0., 1.]]])
        # numerator=(4+0)/2 + 0; denominator=(4+1)/2 +(1+1)/2.
        self.assertAlmostEqual(pilot_baseline(events, scores), 2/3.5)
        self.assertIsNone(pilot_baseline(np.zeros((1, 2)), np.zeros((1, 2, 3))))
        result = risk_gradient_contributions(events, scores, .25)
        expected = ((events-.25)[..., None]*scores).mean(axis=1).mean(axis=0)
        np.testing.assert_array_equal(result["gradient"], expected)


if __name__ == "__main__":
    unittest.main()
