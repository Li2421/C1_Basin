"""Synthetic paired outcomes check that timeout conversion is not success."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class EarlyAnalysisTests(unittest.TestCase):
    def fixture(self, folder):
        arms=['reference','negative_early','positive_early','random_early','negative_full']
        seeds=[1,2,3,4]
        (folder/'protocol.json').write_text(json.dumps(dict(evaluation_seeds=seeds,arms=arms)))
        (folder/'complete.json').write_text(json.dumps(dict(replays=400)))
        (folder/'gradients.json').write_text(json.dumps([dict(gradient_norm=1.,mean_prediction_risk=2.,
            negative_prediction_risk=1.) for _ in range(20)]))
        rows=[]
        for rid in range(20):
            for seed in seeds:
                for arm in arms:
                    outcome='safe_deadlock' if rid<5 else 'success'
                    if rid<5 and arm in ('negative_early','negative_full'):outcome='success'
                    if rid<5 and arm=='random_early':outcome='other_timeout'
                    if rid==5 and arm=='negative_full':outcome='safe_deadlock'
                    rows.append(dict(rid=rid,seed=seed,arm=arm,outcome=outcome,
                        either_deadlock=outcome=='safe_deadlock',success=outcome=='success',
                        actual_early_action_rms=0.,safety={k:0 for k in ('agent_collision_steps',
                        'wall_collision_steps','outside_endpoints','cbf_violations','speed_violations')}))
        (folder/'records.json').write_text(json.dumps(rows))
        return rows

    def run_analysis(self,folder):
        script=Path(__file__).resolve().parents[2]/'scripts/analyze_c1_early_gradient.py'
        return subprocess.run([sys.executable,str(script),str(folder)],capture_output=True,text=True)

    def test_conversion_and_paired_difference(self):
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name);self.fixture(folder)
            result=self.run_analysis(folder)
            self.assertEqual(result.returncode,0,result.stderr)
            analysis=json.loads((folder/'analysis.json').read_text())
            self.assertTrue(analysis['difficulty_passed'])
            arms=analysis['arms']
            self.assertEqual(arms['negative_early']['baseline_deadlock_to_success'],20)
            self.assertEqual(arms['random_early']['baseline_deadlock_to_success'],0)
            self.assertEqual(arms['random_early']['baseline_deadlock_to_timeout'],20)
            self.assertEqual(arms['random_early']['success_increase_vs_reference']['mean'],0.)
            self.assertEqual(arms['negative_early']['deadlock_reduction_vs_reference']['mean'],.25)
            self.assertEqual(arms['negative_full']['new_deadlocks'],4)

    def test_duplicate_pair_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name);rows=self.fixture(folder);rows[-1]=rows[0]
            (folder/'records.json').write_text(json.dumps(rows))
            self.assertNotEqual(self.run_analysis(folder).returncode,0)
            self.assertFalse((folder/'analysis.json').exists())
