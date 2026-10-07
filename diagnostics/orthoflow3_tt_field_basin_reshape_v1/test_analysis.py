"""Small independent checks for inferential units and unresolved outcomes."""
import itertools
import unittest
import numpy as np
from analyze import membership, signflip_exact, summarize_complete, holm


class AnalysisChecks(unittest.TestCase):
    def test_numerical_unknown_is_not_failure(self):
        a=np.zeros((4,16,3),dtype=np.int8)
        a[0,:15,0]=1;a[0,15,1]=1   # robust despite one unresolved
        a[1,:14,0]=1;a[1,14:,1]=1  # could be robust or not
        a[2,:13,0]=1;a[2,13:15,1]=1  # could be 15 successes
        a[3,:12,0]=1;a[3,12:14,1]=1  # cannot reach 15
        np.testing.assert_array_equal(membership(a)[0],[1,-1,-1,0])

    def test_whole_state_sign_permutations(self):
        d=np.array([2,-1,4,0])
        brute=np.mean([abs(sum(d*np.array(s)))>=abs(d.sum())
                       for s in itertools.product((-1,1),repeat=len(d))])
        self.assertEqual(signflip_exact(d),brute)

    def test_contingency_and_volume_identity(self):
        a=np.array([[0,0,1,1],[0,1,1,1]])
        b=np.array([[0,1,0,1],[1,1,1,1]])
        states=[{'scenario':'one'},{'scenario':'one'}]
        r=summarize_complete(a,b,states,np.array([[0,1],[1,0]]))
        self.assertEqual([r[k] for k in ('both_failure','rescue','break_count','both_success')],[1,2,1,4])
        self.assertEqual(r['change_rate'],3/8)
        self.assertEqual(r['volume_delta'],1/8)
        self.assertEqual(r['volume_increase_states'],1)
        self.assertEqual(r['volume_tie_states'],1)
        self.assertEqual(r['state_sign_p'],1.)  # one state, not eight cells

    def test_holm_preserves_input_order(self):
        np.testing.assert_allclose(holm([.04,.001,.02]),[.04,.003,.04])


if __name__=='__main__':unittest.main()
