"""Exercise the real trainer lifecycle with a cheap differentiable test rollout.

Physics/Flow/SOCP are tested separately; this isolates checkpoint/RNG/selection
orchestration, including interruption before the first optimizer update.
"""
import contextlib
import io
import json
import pickle
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from single_integrator.c1 import train
from single_integrator.c1.training.dataset_starts import load_dataset_starts
from single_integrator.environment import Config


def synthetic_rollout(params, field, projection, initial, noise, plant, cbf, risk,
                 return_details=False, return_mask=False):
    value = jnp.sum(params['params']['zero_initialized_output']['bias'])
    B, T = noise.shape[:2]
    current = jnp.ones((B, T, 4)) * value
    safe = jnp.zeros_like(current)
    risks = jnp.full((B,), .5) - .1 * value
    mask = jnp.ones((B, T), bool)
    result = (current, safe, risks)
    if return_details:
        return (*result, dict(episode_mask=mask, cone_risk_t=jnp.full((B,T),.1),
            stall_risk_t=jnp.full((B,T),.2), terminal_codes=jnp.full((B,),5),
            episode_lengths=jnp.full((B,),T), terminal_risk=risks))
    return (*result, mask) if return_mask else result


class TrainerRecoveryTests(unittest.TestCase):
    def test_initial_interruption_resume_and_stale_history_repair(self):
        previous = jax.config.x64_enabled
        self.addCleanup(jax.config.update, 'jax_enable_x64', previous)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = Path('datasets/give_way_si_short_v1')
            plant = Config(**json.loads((data/'environment.json').read_text())['evaluation_environment'])
            for kind, split in (('calibration','train'), ('validation','val')):
                starts, _ = load_dataset_starts(data, split, plant)
                np.savez(root/f'{kind}.npz', initial_positions=starts,
                         noise=np.zeros((len(starts),850,4), np.float32))

            def invoke(name, resume=False, interrupt=False):
                argv = ['train', '--checkpoint', 'baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl',
                        '--out-dir', str(root/name), '--seed', '0', '--updates', '2',
                        '--batch-size', '2', '--start-distribution', 'dataset',
                        '--calibration-set', str(root/'calibration.npz'),
                        '--validation-set', str(root/'validation.npz'), '--validation-every', '2']
                if resume:
                    argv.append('--resume')
                def rollout(*args, **kwargs):
                    if interrupt and kwargs.get('return_details'):
                        raise RuntimeError('simulated validation interruption')
                    return synthetic_rollout(*args, **kwargs)
                with patch('sys.argv', argv), patch.object(train, 'rollout_terms', rollout), contextlib.redirect_stdout(io.StringIO()):
                    train.main()

            with self.assertRaisesRegex(RuntimeError, 'simulated'):
                invoke('interrupted', interrupt=True)
            checkpoint = root/'interrupted/residual.pkl'
            self.assertEqual(pickle.loads(checkpoint.read_bytes())['completed_updates'], 0)
            invoke('interrupted', resume=True)
            invoke('reference')
            actual = pickle.loads(checkpoint.read_bytes())
            expected = pickle.loads((root/'reference/residual.pkl').read_bytes())
            self.assertEqual(actual['completed_updates'], 2)
            self.assertEqual(actual['rng_state'], expected['rng_state'])
            np.testing.assert_array_equal(actual['jax_key'], expected['jax_key'])
            for a, b in zip(jax.tree_util.tree_leaves(actual['params']), jax.tree_util.tree_leaves(expected['params'])):
                np.testing.assert_array_equal(a, b)
            self.assertGreater(actual['history'][1]['gradient_norm'], 0.)
            self.assertEqual(actual['history'][0]['step_control']['status'], 'no_op')
            for row in actual['history']:
                self.assertLessEqual(row['post_step']['loss'], row['loss'])
                self.assertAlmostEqual(row['post_step']['loss'], row['post_step']['J_def'] +
                    row['lambda_used'] * row['post_step']['constraint'])
            self.assertEqual(int(actual['optimizer_state'][0].count),
                sum(row['step_control']['status'] == 'accepted' for row in actual['history']))
            # Completed resume must repair a broken secondary history and must
            # not perform another optimizer update or alter checkpoint bytes.
            before = checkpoint.read_bytes()
            (root/'interrupted/history.json').write_text('broken json')
            invoke('interrupted', resume=True)
            self.assertEqual(before, checkpoint.read_bytes())
            self.assertEqual(len(json.loads((root/'interrupted/history.json').read_text())), 2)
            self.assertEqual(len(json.loads((root/'interrupted/validation.json').read_text())), 2)
            with np.load(root/'validation.npz') as data:
                starts = data['initial_positions'].copy()
                noise = data['noise'].copy()
            np.savez(root/'validation.npz', initial_positions=starts, noise=noise+1)
            with self.assertRaisesRegex(ValueError, 'configuration differs'):
                invoke('interrupted', resume=True)
            self.assertEqual(before, checkpoint.read_bytes())


if __name__ == '__main__':
    unittest.main()
