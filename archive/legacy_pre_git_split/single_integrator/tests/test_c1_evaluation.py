import hashlib
import pickle
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from single_integrator.c1.risk.risk_v1 import RiskV1Config
from single_integrator.c1.risk.evaluation import risk_from_metadata, audit_trace
from single_integrator.cbf import CBFConfig
from single_integrator.environment import Config


class EvaluationTests(unittest.TestCase):
    def test_risk_version_replay(self):
        old=asdict(RiskV1Config());old.pop('unfinished_mode')
        self.assertEqual(risk_from_metadata(dict(risk_version='R_risk_v1',risk=old)).unfinished_mode,'joint_sum')
        self.assertEqual(risk_from_metadata(dict(risk_version='R_risk_v1_1',risk=asdict(RiskV1Config()))).unfinished_mode,'per_agent')
        with self.assertRaises(ValueError):
            risk_from_metadata(dict(risk_version='R_risk_v1',risk=asdict(RiskV1Config())))

    def test_executed_trace_matches_training_risk_and_correction_metrics(self):
        from single_integrator.c1.train import rollout_terms
        class Field:
            def control(self,params,obs,noise,A,b,speed,projection):
                safe=jnp.broadcast_to(jnp.array([.01,0.,-.01,0.]),(len(obs),4))
                correction=jnp.broadcast_to(jnp.array([.005,0.,-.005,0.]),safe.shape)
                return dict(nominal=safe,safe=safe,correction=correction,
                            candidate=safe+correction,applied=safe+.4*correction)
        with jax.experimental.enable_x64():
            plant=Config(corridor_half_length=1.3);cbf=CBFConfig()
            risk=RiskV1Config(window_seconds=.1)
            u,safe,r,details=rollout_terms(None,Field(),None,jnp.array([[[-.18,.008],[.18,-.005]]]),
                                         jnp.zeros((1,5,4)),plant,cbf,risk,return_details=True)
            before=np.asarray(details['positions'])[0]
            trace=dict(positions_before=before,positions=before+plant.dt*np.asarray(u)[0].reshape(-1,2,2),
                       executed_velocity=np.asarray(u)[0].reshape(-1,2,2),
                       c1_safe=np.asarray(safe)[0],c1_correction=np.asarray(details['correction'])[0],
                       c1_candidate=np.asarray(details['candidate'])[0])
            summary,fields=audit_trace(trace,plant,cbf,risk)
            self.assertAlmostEqual(summary['trajectory_risk'],float(r[0]),places=12)
            self.assertEqual(summary['risk_valid_steps'],3)
            self.assertAlmostEqual(summary['J_def'],2*.002**2,places=12)
            self.assertAlmostEqual(summary['mean_raw_correction_norm'],np.sqrt(2)*.005,places=12)
            self.assertAlmostEqual(summary['mean_applied_correction_norm'],np.sqrt(2)*.002,places=12)
            self.assertAlmostEqual(summary['mean_final_projection_norm'],np.sqrt(2)*.003,places=12)
            short={k:v[:2] for k,v in trace.items()}
            summary,fields=audit_trace(short,plant,cbf,risk)
            self.assertIsNone(summary['trajectory_risk'])
            self.assertEqual(summary['risk_valid_steps'],0)

    def test_formal_pair_rejects_duplicate_and_mixed_seeds(self):
        from scripts.plan_c1_400 import ROOT, validate_residual_pair
        metadata=[]
        for seed in (0,1):
            checkpoint=ROOT/f'baseline_309_314/checkpoints/seed{seed}/ckpt_0025000.pkl'
            metadata.append(dict(method='c1_v0',baseline_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                                 risk_version='R_risk_v1_1',risk=asdict(RiskV1Config())))
        with tempfile.TemporaryDirectory() as tmp:
            paths=[Path(tmp)/f'seed{s}.pkl' for s in (0,1)]
            for path,meta in zip(paths,metadata):
                path.write_bytes(pickle.dumps(dict(metadata=meta)))
            validate_residual_pair(paths)
            with self.assertRaises(ValueError):validate_residual_pair([paths[0],paths[0]])
            with self.assertRaises(ValueError):validate_residual_pair(paths[::-1])
            metadata[1]['training'] = dict(updates=3)
            paths[1].write_bytes(pickle.dumps(dict(metadata=metadata[1])))
            with self.assertRaisesRegex(ValueError, 'training updates'):
                validate_residual_pair(paths)
            metadata[1].pop('training')
            metadata[1]['risk_version']='R_risk_v1'
            paths[1].write_bytes(pickle.dumps(dict(metadata=metadata[1])))
            with self.assertRaises(ValueError):validate_residual_pair(paths)


if __name__=='__main__':unittest.main()
