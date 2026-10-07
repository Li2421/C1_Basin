"""Offline synthetic correctness tests; no learned predictions or TEST labels."""
import unittest
import numpy as np
from .evaluate import point_metrics,reversals,interval


class EvidenceTests(unittest.TestCase):
    def test_unknown_not_failed(self):
        s=np.array([[15.,1.],[14.,16.],[0.,0.]])
        f=np.array([[0.,15.],[1.,0.],[16.,16.]])
        u=16-s-f;z=np.array([[10.,0.],[10.,0.],[0.,1.]])
        m,a=point_metrics(z,s,f,u,s,f)
        self.assertEqual(m['B15'],1)
        self.assertEqual(m['unresolved'],1)
        self.assertEqual(m['oracle_B15'],2)
        self.assertEqual(m['proposal_failure'],1)
        self.assertEqual(m['selection_failure'],0)
        self.assertEqual(m['selection_unresolved'],1)

    def test_eta_only_reversal_half(self):
        s=np.zeros((4,16));s[:2,0]=16;s[2:,1]=16;f=16-s
        eta=np.broadcast_to(np.linspace(2,-2,16),(4,16)).copy()
        rr=reversals(s,f,eta)
        self.assertEqual(rr['reversal_eta_pairs'],1)
        self.assertEqual(rr['stable_Wilson_reversal_eta_pairs'],1)
        self.assertEqual(rr['reversal_balanced_accuracy'],.5)
        rr2=reversals(s,f,np.log((s+.5)/(f+.5)))
        self.assertEqual(rr2['reversal_balanced_accuracy'],1.)

    def test_tied_eta_is_fixed_not_statewise_oracle(self):
        s=np.zeros((4,16));s[:2,0]=16;s[2:,1]=16;f=16-s
        self.assertEqual(reversals(s,f,np.zeros_like(s))['reversal_balanced_accuracy'],.5)

    def test_paired_unit(self):
        self.assertEqual(interval(np.ones(16)),[1.,1.])
        self.assertEqual(interval(np.zeros(16)),[0.,0.])


if __name__=='__main__':unittest.main()
