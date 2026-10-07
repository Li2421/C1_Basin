from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from diagnostics.eta_complexity_stress_test.benchmark import freeze_benchmark, generate_benchmark, load_benchmark
from diagnostics.eta_complexity_stress_test.controller import rollout
from diagnostics.eta_complexity_stress_test.environment import CoupledDualIntersectionEnv, StressConfig
from diagnostics.eta_complexity_stress_test.oracle import rollout_oracle
from diagnostics.eta_complexity_stress_test.search import EtaBounds, basin_components


class StressPlantTests(unittest.TestCase):
    def test_eight_agent_observation_and_projection_snapshot_contract(self):
        env = CoupledDualIntersectionEnv()
        self.assertEqual(env.observation().shape, (8, 34))
        self.assertEqual(env.snapshot()["pair_indices"].shape, (28, 2))
        self.assertGreater(len(env.walls), 0)
        self.assertFalse(env.outside(env.positions).any())
        wall, pair = env.distances()
        self.assertGreater(wall.min(), env.config.wall_collision_margin)
        self.assertGreater(pair.min(), env.config.agent_collision_margin)

    def test_frozen_benchmark_round_trip(self):
        config = StressConfig()
        items = generate_benchmark(count=8, seed=17, config=config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ics.json"
            freeze_benchmark(path, items, config)
            restored = load_benchmark(path, config)
        self.assertEqual([item.identifier for item in restored], [item.identifier for item in items])
        for left, right in zip(items, restored):
            np.testing.assert_array_equal(left.positions, right.positions)

    def test_oracle_has_explicit_distinct_orders(self):
        start = CoupledDualIntersectionEnv().positions
        first = rollout_oracle(start, "ltr_then_vertical_then_rtl")
        second = rollout_oracle(start, "vertical_then_ltr_then_rtl")
        self.assertTrue(first.success, first.termination)
        self.assertTrue(second.success, second.termination)
        self.assertNotEqual(first.active_agents.tolist(), second.active_agents.tolist())

    def test_eta_zero_has_exactly_zero_raw_correction(self):
        config = replace(StressConfig(), max_steps=3, terminate_on_deadlock=False)
        trace = rollout(CoupledDualIntersectionEnv(config).positions, eta=(0.0, 0.0, 0.0), variant="eta", config=config)
        np.testing.assert_array_equal(trace.correction, np.zeros_like(trace.correction))
        # The second exact SOCP receives exactly u_safe; independent solver
        # calls may differ at certified numerical tolerance.
        np.testing.assert_allclose(trace.u_safe, trace.u_exec, rtol=0, atol=2e-6)

    def test_basin_components_are_sample_resolution_diagnostic(self):
        bounds = EtaBounds()
        points = np.asarray(((0.1, -0.8, -0.8), (0.11, -0.8, -0.8), (1.4, 0.8, 0.8)))
        result = basin_components(points, np.asarray((True, True, True)), bounds, radius=0.05)
        self.assertEqual(result["components"], 2)
        self.assertEqual(result["success_points"], 3)


if __name__ == "__main__":
    unittest.main()
