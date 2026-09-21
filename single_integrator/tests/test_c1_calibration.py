import tempfile
import json
import unittest
from pathlib import Path

import numpy as np

from single_integrator.c1.training.calibration import calibrate_fixed_set, reuse_fixed_calibration


class CalibrationTests(unittest.TestCase):
    def test_reference_checks_provenance_input_hash_and_actual_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            path, reference = Path(directory)/'set.npz', Path(directory)/'config.json'
            starts = np.zeros((3,2,2)); starts[:,0,0] = [.1,.3,.8]
            noise = np.zeros((3,2,4))
            np.savez(path, initial_positions=starts, noise=noise)
            evaluate = lambda s,n: s[:,0,0]
            mean, report = calibrate_fixed_set(path,2,2,evaluate)
            metadata = dict(environment={'max_steps':2},training=dict(horizon=2,
                            baseline_J_live=mean,calibration=report))
            reference.write_text(json.dumps(metadata))
            expected = dict(environment={'max_steps':2})
            self.assertEqual(reuse_fixed_calibration(reference,path,2,2,expected,evaluate),(mean,report))
            with self.assertRaises(ValueError):
                reuse_fixed_calibration(reference,path,2,2,dict(environment={}),evaluate)
            with self.assertRaises(AssertionError):
                reuse_fixed_calibration(reference,path,2,2,expected,lambda s,n:evaluate(s,n)+.1)
            np.savez(path,initial_positions=starts,noise=noise+1)
            with self.assertRaises(ValueError):
                reuse_fixed_calibration(reference,path,2,2,expected,evaluate)

    def test_episode_weighting_includes_zero_and_partial_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'set.npz'
            starts = np.zeros((3, 2, 2))
            starts[:, 0, 0] = [0, .3, .9]
            np.savez(path, initial_positions=starts, noise=np.zeros((3, 2, 4)))
            mean, report = calibrate_fixed_set(path, 2, 2, lambda s, n: s[:, 0, 0])
            self.assertAlmostEqual(mean, .4)
            self.assertEqual(report['risks'], [0, .3, .9])
            self.assertEqual(report['size'], 3)
            with self.assertRaises(ValueError):
                calibrate_fixed_set(path, 3, 2, lambda s, n: s[:, 0, 0])
            for value in (0., -1., float('nan'), float('inf')):
                with self.assertRaises(ValueError):
                    calibrate_fixed_set(path, 2, 2, lambda s, n: np.full(len(s), value))

    def test_empty_or_nonfinite_inputs_rejected_before_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'set.npz'
            for starts in (np.zeros((0, 2, 2)), np.full((1, 2, 2), np.nan)):
                np.savez(path, initial_positions=starts, noise=np.zeros((len(starts), 2, 4)))
                with self.assertRaises(ValueError):
                    calibrate_fixed_set(path, 2, 1, lambda s, n: self.fail('evaluated invalid input'))
