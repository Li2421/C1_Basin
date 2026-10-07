import unittest
import numpy as np
from analyze import auc, ap, membership, state_cases, score_cases, evaluate

class ScientificContractTests(unittest.TestCase):
    def test_unknown_is_not_failure(self):
        np.testing.assert_array_equal(membership(np.array([15,14,13]),np.array([0,1,2])),[1,-1,0])
    def test_joint_reversal_requires_both_directions(self):
        s=np.array([[16,0],[0,16]]);f=16-s;cases=state_cases(s,f)['B15_swap']
        self.assertEqual(len(cases),1)
        self.assertFalse(score_cases(cases,np.array([[2,1],[2,1]]))[0])
        self.assertTrue(score_cases(cases,np.array([[2,1],[1,2]]))[0])
    def test_global_difficulty_is_not_within_state_ranking(self):
        y=np.array([[1,1],[0,0]]);z=np.array([[3,3],[0,0]])
        self.assertEqual(auc(y.ravel(),z.ravel()),1)
        self.assertTrue(np.isnan(auc(y[0],z[0])))
    def test_tie_credit_and_ap(self):
        self.assertEqual(auc(np.array([1,0]),np.array([1,1])),.5)
        self.assertEqual(ap(np.array([1,0]),np.array([1,1])),.5)
    def test_proposal_vs_ranking_failure(self):
        s=np.array([[16,0],[8,0]]);f=16-s;z=np.array([[0,1],[0,1]])
        r,_=evaluate(s,f,z,['a','b'],'test','test')
        self.assertEqual((r['oracle_eligible'],r['ranking_failures'],r['proposal_failures']),(1,1,1))
    def test_gap_uses_bounds_not_observed_denominator(self):
        s=np.array([[12,4],[4,12]]);f=np.zeros_like(s)
        self.assertEqual(len(state_cases(s,f)['Q16_gap_025']),0)

if __name__=='__main__':unittest.main()
