import json
from pathlib import Path
import tempfile
import unittest
import jax
import numpy as np

from single_integrator.c1.training.dataset_starts import load_dataset_starts, load_fixed_inputs, require_population
from single_integrator.c1.training.persistence import atomic_save, restored_history
from single_integrator.c1.training.primal_dual import PrimalDualState, dual_update
from single_integrator.environment import Config


class AuditRegressions(unittest.TestCase):
    def test_nonfinite_risk_cannot_reset_dual(self):
        for risk in (np.nan, np.inf, -np.inf):
            with self.assertRaisesRegex(ValueError, 'finite'):
                dual_update(PrimalDualState(), risk, .2, .01)

    def test_dataset_sampling_uses_only_train_pair_starts(self):
        from single_integrator.c1.train import sample_batch
        root = Path('datasets/give_way_si_short_v1')
        plant = Config(**json.loads((root/'environment.json').read_text())['evaluation_environment'])
        train, meta = load_dataset_starts(root, 'train', plant)
        val, _ = load_dataset_starts(root, 'val', plant)
        self.assertEqual((len(train), len(val)), (200, 25))
        self.assertFalse(set(map(tuple, train.reshape(-1,4))) & set(map(tuple, val.reshape(-1,4))))
        with jax.experimental.enable_x64():
            sampled, noise = sample_batch(np.random.default_rng(0), None, 32, 3,
                jax.random.PRNGKey(0), None, 0., train)
        self.assertEqual(noise.shape, (32, 3, 4))
        self.assertTrue(set(map(tuple, np.asarray(sampled).reshape(-1,4))) <= set(map(tuple, train.reshape(-1,4))))
        require_population(train[::-1], train, 'calibration')
        for bad in (train[:-1], np.concatenate((train, train[:1])), val):
            with self.assertRaises(ValueError):
                require_population(bad, train, 'calibration')

    def test_invalid_validation_and_content_change(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'set.npz'
            for starts in (np.zeros((0,2,2)), np.full((1,2,2), np.nan)):
                np.savez(path, initial_positions=starts, noise=np.zeros((len(starts),2,4)))
                with self.assertRaises(ValueError):
                    load_fixed_inputs(path, 2)
            np.savez(path, initial_positions=np.zeros((1,2,2)), noise=np.zeros((1,2,4)))
            _, _, first = load_fixed_inputs(path, 2)
            np.savez(path, initial_positions=np.zeros((1,2,2)), noise=np.ones((1,2,4)))
            _, _, second = load_fixed_inputs(path, 2)
            self.assertNotEqual(first, second)

    def test_checkpoint_recovers_after_history_publication_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'history.json'
            atomic_save(path, [{'update':0}])
            saved = dict(completed_updates=2, history=[{'update':0}, {'update':1}])
            self.assertEqual(len(restored_history(saved, path)), 2)
            saved['history'][1]['update'] = 3
            with self.assertRaises(ValueError):
                restored_history(saved, path)
            # Failed serialization must leave the last published artifact intact.
            with self.assertRaises(ValueError):
                atomic_save(path, [float('nan')])
            self.assertEqual(json.loads(path.read_text()), [{'update':0}])

    def test_initial_checkpoint_recovers_without_history_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'history.json'
            saved = dict(completed_updates=0, history=[])
            history = restored_history(saved, path)
            atomic_save(path, history)
            self.assertEqual(json.loads(path.read_text()), [])


if __name__ == '__main__':
    unittest.main()
