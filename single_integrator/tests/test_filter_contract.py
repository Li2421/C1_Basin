import tempfile
from pathlib import Path
import pickle
import unittest
import numpy as np
import jax
from single_integrator.environment import Config
from single_integrator.filters import FilterResult, IdentityFilter
from single_integrator.evaluate import rollout, load_policy


class NoisePolicy:
    def sample_actions(self,observations,seed):
        return .1*jax.random.normal(seed,(1,2,2))


class FilterTests(unittest.TestCase):
    def test_identity_matches_unfiltered_exactly(self):
        initial=np.array([[-2.,0.],[2.,0.]])
        config=Config(max_steps=12)
        a,aa=rollout(NoisePolicy(),initial,config,42,3)
        b,bb=rollout(NoisePolicy(),initial,config,42,3,IdentityFilter())
        self.assertEqual(a,b)
        for key in ['raw_policy_velocity','nominal_velocity','executed_velocity','positions','velocities','wall_collision','agent_collision','deadlock']:
            np.testing.assert_array_equal(aa[key],bb[key])

    def test_filtered_action_is_executed_exactly(self):
        def stop(snapshot,nominal):return FilterResult(np.zeros((2,2)),status='test_stop')
        initial=np.array([[-2.,0.],[2.,0.]])
        summary,arrays=rollout(NoisePolicy(),initial,Config(max_steps=3),42,0,stop)
        np.testing.assert_array_equal(arrays['positions'],np.repeat(initial[None],3,axis=0))
        np.testing.assert_array_equal(arrays['executed_velocity'],np.zeros((3,2,2)))
        self.assertEqual(summary['max_tracking_error'],0.)

    def test_overspeed_filter_output_is_rejected_not_clipped(self):
        def invalid(snapshot,nominal):return FilterResult(np.ones((2,2)))
        with self.assertRaises(ValueError):rollout(NoisePolicy(),[[-2.,0.],[2.,0.]],Config(max_steps=1),0,0,invalid)

    def test_legacy_checkpoint_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'legacy.pkl'
            with path.open('wb') as f:pickle.dump({},f)
            with self.assertRaisesRegex(ValueError,'no SI training provenance'):load_policy(path)


if __name__=='__main__':unittest.main()
