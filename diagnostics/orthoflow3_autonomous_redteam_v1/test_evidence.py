import ast
import unittest
import numpy as np
from audit import ROOT,OUT,read
from scientific_evidence import summarize_canonical16,exact_label_or_proxy,finite_logit_argmax

def seed_rows(s,f,u=0):
    return [dict(future_index=i,success=i<s,numerical_failure=i>=s+f) for i in range(s+f+u)]

class EvidenceRepairTests(unittest.TestCase):
    def test_numerical_is_interval_not_exact_Q(self):
        out=summarize_canonical16(seed_rows(15,0,1))
        self.assertIsNone(out['Q16']);self.assertTrue(out['robust'])
        self.assertEqual((out['Q_lower'],out['Q_upper']),(.9375,1.))
    def test_underresolved_not_task_failure(self):
        out=summarize_canonical16(seed_rows(14,1,1))
        self.assertIsNone(out['robust']);self.assertIsNone(out['Q16'])
    def test_exact_rejection_without_fabricating_Q(self):
        out=summarize_canonical16(seed_rows(0,2))
        self.assertFalse(out['robust']);self.assertIsNone(out['Q16'])
        self.assertEqual(out['Q_upper'],14/16)
    def test_full_Q_control(self):
        out=summarize_canonical16(seed_rows(14,2))
        self.assertFalse(out['robust']);self.assertEqual(out['Q16'],.875)
    def test_duplicate_seed_rejected(self):
        with self.assertRaises(ValueError):summarize_canonical16(seed_rows(1,0)*2)
    def test_conflicting_numerical_success_rejected(self):
        with self.assertRaises(ValueError):summarize_canonical16([dict(future_index=0,success=True,numerical_failure=True)])
    def test_nearest_cannot_be_exact(self):
        req=dict(state_uid='s',eta_uid='e1',controller_uid='c')
        other={**req,'eta_uid':'e2','robust_15of16':True}
        out=exact_label_or_proxy(req,other)
        self.assertIsNone(out['robust']);self.assertFalse(out['eligible_for_readiness_decision'])
        self.assertTrue(exact_label_or_proxy(req,{**other,'eta_uid':'e1'})['robust'])
    def test_sigmoid_saturation_counterexample(self):
        z=np.array([18.,22.,20.],np.float32);p=1/(1+np.exp(-z))
        self.assertEqual(len(set(p.tolist())),1)
        self.assertEqual(int(np.argmax(p)),0)
        self.assertEqual(finite_logit_argmax(z.tolist()),1)
        permutation=[2,0,1]
        self.assertEqual(permutation[finite_logit_argmax(z[permutation].tolist())],1)
    def test_real_top1_and_candidate_reconstruction(self):
        out=read(OUT/'H3_inference_results.json')['summary']
        self.assertEqual(out['exact_eta_reproductions'],76)
        self.assertEqual(out['top1_match'],76)
        self.assertEqual(out['duplicate_seed_count'],0)
    def test_legacy_reporting_witness_is_real(self):
        tree=ast.parse((ROOT/'diagnostics/orthoflow3_ring_k16_diagnostic_v1/run_k16.py').read_text())
        fun=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='evidence')
        ns={};exec(compile(ast.Module(body=[fun],type_ignores=[]),'legacy_evidence','exec'),ns)
        rows=[dict(r,collision=False,timeout=False) for r in seed_rows(15,0,1)]
        before=ns['evidence'](rows);after=summarize_canonical16(rows)
        self.assertEqual(before['Q16'],.9375)
        self.assertIsNone(after['Q16'])
        self.assertIs(before['robust'],after['robust'])
    def test_real_H2_headlines_invariant(self):
        out=read(OUT/'H2_k16_corrected_summary.json')
        self.assertEqual((out['oracle_B15'],out['critic_B15'],out['misses'],out['exploitation_confirmed']),(60,53,7,4))
        self.assertEqual(out['collision_seeds'],0)
    def test_optional_stopping_likelihood_control(self):
        rows=read(OUT/'H4_stopped_evidence.json')['exact_tree_control']
        self.assertTrue(all(abs(r['expected_binomial_score_at_truth'])<1e-10 for r in rows))
        self.assertTrue(any(abs(r['bias'])>.1 for r in rows))
    def test_outcome_balancing_is_not_stopping_alone(self):
        rows=read(OUT/'H4_stopped_evidence.json')['exact_tree_control']
        for r in rows:
            self.assertAlmostEqual(r['outcome_conditioned_sampling_control'][0]['population_likelihood_optimum'],r['p'])
        r=next(r for r in rows if r['p']==.9)
        self.assertGreater(r['outcome_conditioned_sampling_control'][2]['population_likelihood_optimum'],15/16)

if __name__=='__main__':unittest.main(verbosity=2)
