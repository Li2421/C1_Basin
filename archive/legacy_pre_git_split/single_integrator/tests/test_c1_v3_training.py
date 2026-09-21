"""V3 orchestration tests; physical gradients are audited separately."""
import contextlib
import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import jax.numpy as jnp
from single_integrator.c1 import train_v3


def synthetic_rollout(params,field,initial,noise,plant,cbf):
    v=params['params']['zero_initialized_output']['bias'].sum()
    # Nonzero noise influence makes accidental reuse visible in saved values.
    live=.5-.1*v+1e-4*noise[0,0]
    return dict(J_live=live,J_def=v*v,P=live,g=jnp.array(0.),
                success=jnp.array(True),steps=jnp.array(850)), {}


class V3TrainingTests(unittest.TestCase):
    def test_calibration_dual_fresh_noise_and_infeasible_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);sets=root/'sets.json';out=root/'run'
            with patch('sys.argv',['v3','--prepare','--sets',str(sets),
                    '--n-train','2','--n-val','2','--n-test','2']), contextlib.redirect_stdout(io.StringIO()):
                train_v3.main()
            with patch.object(train_v3,'rollout',synthetic_rollout), patch('sys.argv',
                    ['v3','--sets',str(sets),'--out',str(out),'--updates','2','--batch-size','1']), contextlib.redirect_stdout(io.StringIO()):
                train_v3.main()
            calibration=json.loads((out/'calibration.json').read_text())
            self.assertEqual(len(calibration['episodes']),4)
            self.assertAlmostEqual(calibration['epsilon'],.8*calibration['J_live'])
            history=json.loads((out/'history.json').read_text())
            self.assertEqual(history[0]['step_control']['status'],'no_op')
            self.assertGreater(history[0]['lambda_after'],0)
            self.assertGreater(history[1]['gradient_norm'],0)
            self.assertNotEqual(history[0]['noise_seed'],history[1]['noise_seed'])
            self.assertFalse((out/'best_feasible.pkl').exists())
            for row in history:
                self.assertLessEqual(row['post']['loss'],row['pre']['loss'])
                self.assertAlmostEqual(row['post']['loss'],row['post']['J_def']+
                                       row['lambda_used']*row['post']['constraint'])
            state=pickle.loads((out/'checkpoint.pkl').read_bytes())
            self.assertEqual(state['completed_updates'],2)
            self.assertTrue((out/'final_train.json').exists())
            self.assertEqual(len(list(out.glob('*test*'))),0)


if __name__=='__main__':
    unittest.main()
