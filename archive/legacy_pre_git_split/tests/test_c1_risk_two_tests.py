import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scripts.analyze_c1_risk_two_tests import analyze
from single_integrator.c1.action_probe import calibrate_offsets
from single_integrator.cbf import CBFConfig


class TwoRiskTests(unittest.TestCase):
    def test_projection_amplitude_calibration_and_unreachable_direction(self):
        d=np.array([[1.,2.,-1.,.5],[.5,-1.,2.,1.]])
        A=np.repeat(np.array([[[1.,0.,0.,0.]]]),2,axis=0)
        samples=[(np.zeros((2,4)),A,np.full((2,1),-1.))]
        offsets,report=calibrate_offsets(d,samples,CBFConfig())
        self.assertTrue(report['matched'])
        self.assertAlmostEqual(np.sqrt(np.mean(offsets**2)),.0025,delta=.000025)
        blocked=np.zeros((2,4));blocked[:,0]=-1
        _,report=calibrate_offsets(blocked,[(np.zeros((2,4)),A,np.zeros((2,1)))],CBFConfig())
        self.assertFalse(report['matched'])

    def test_successful_resolution_passes_but_timeout_substitution_fails(self):
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name)
            write=lambda p,data:(folder/p).write_text(json.dumps(data))
            arms=['reference','negative','positive','random']
            write('protocol.json',dict(starts=[None]*25,evaluation_seeds=[1,2,3,4],arms=arms))
            write('complete.json',dict(replays=400,independent_test_opened=False))
            write('diagnostics.json',[dict(rid=i,prediction_risk=float(25-i),gradient_norm=1.,
                calibration={a:dict(matched=True) for a in arms[1:]}) for i in range(25)])
            rows=[]
            for i in range(25):
                for s in [1,2,3,4]:
                    for a in arms:
                        dead=i<10 and a!='negative'
                        rows.append(dict(rid=i,seed=s,arm=a,either_deadlock=dead,success=not dead,
                            outcome='safe_deadlock' if dead else 'success',
                            direct_action_sum_squares=400*.0025**2,direct_action_components=400,
                            safety={k:0 for k in ['agent_collision_steps','wall_collision_steps',
                                'outside_endpoints','cbf_violations','speed_violations']}))
            write('records.json',rows)
            result=analyze(folder)
            self.assertTrue(result['both_tests_and_controls_passed'])
            for row in rows:
                if row['arm']=='negative' and row['rid']<10:
                    row.update(success=False,outcome='other_timeout')
            write('records.json',rows)
            result=analyze(folder)
            self.assertFalse(result['gates']['gradient_usefulness'])
            self.assertFalse(result['both_tests_and_controls_passed'])
            self.assertEqual(result['arms']['negative']['original_deadlock_to_timeout'],40)


if __name__=='__main__':unittest.main()
