"""Synthetic counterexamples for selection and paired inference; zero rollouts."""
import guard
import itertools,unittest
from fractions import Fraction as F
from statistic_rules import *

class Contracts(unittest.TestCase):
    def test_scene_weighting_changes_the_winner(self):
        def rows(nsuccess,last):
            return[dict(scene='A',state=i,success=int(i<nsuccess),collision=0,truncated_time=2.)for i in range(10)]+[dict(scene='B',state=0,success=last,collision=0,truncated_time=2.)]
        first,second=score(1,rows(10,0)),score(2,rows(8,1))
        self.assertEqual(first.success,F(1,2));self.assertEqual(second.success,F(9,10));self.assertLess(second.key(),first.key())
    def test_missing_evidence_is_not_a_zero(self):
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            score(1,[dict(scene='A',state=0,success=0,collision=0,truncated_time=2)],{('A',0,'paper'),('A',1,'paper')})
    def test_second_round_cannot_pick_a_cached_outsider(self):
        make=lambda i,s:Score(i,F(s,10),0,0,1.,1,1)
        shortlist=first_round([make(i,10-i)for i in range(1,6)])
        result=final_choice(shortlist,{i:make(i,i)for i in range(1,6)})
        self.assertEqual(result.index,4)
    def test_expected_max_equals_exhaustive_subsets_with_ties(self):
        values=[F(0),F(1,2),F(1,2),F(1)]
        for row in expected_max_curve(values):
            brute=sum((max(x)for x in itertools.combinations(values,row['k'])),F())/math.comb(4,row['k'])
            self.assertEqual(F(*row['fraction']),brute)
    def test_exact_three_percent_expansion_boundary(self):
        # One exceptional setting is present in a random8-subset with probability1/2.
        self.assertTrue(saturation([F(0)]*15+[F(6,100)])['expand'])
        self.assertFalse(saturation([F(0)]*15+[F(59,1000)])['expand'])
    def test_holm_keeps_missing_preregistered_comparisons(self):
        self.assertEqual(holm({'a':.01,'b':.04,'c':.03}),{'a':.03,'c':.06,'b':.06})
        self.assertEqual(holm({'a':.01,'b':None})['a'],.02)
        self.assertIsNone(holm({'a':.01,'b':None})['b'])
    def test_exact_scene_signflip(self):
        self.assertEqual(paired_signflip([F(1)]*4)['p'],2/16)
        self.assertEqual(paired_signflip([F(0)]*4)['p'],1.)
    def test_bootstrap_keeps_physical_state_averages(self):
        result=hierarchical_interval({'one':{'A':[.5]*8},'two':{'B':[.5]*8}},resamples=30)
        self.assertEqual(result['physical_states'],16);self.assertEqual(result['ci95'],[.5,.5])
    def test_nominals_collapse_to_two_physical_states_not_four(self):
        rows=[dict(scene='A',state=si,nominal=nom,value=value)for si,values in enumerate([(1,0),(1,1)])for nom,value in zip(['one','two'],values)]
        values=collapse_nominals(rows,['one','two'])
        self.assertEqual(values,{('A',0):F(1,2),('A',1):F(1)})
        result=hierarchical_interval({'type':{'A':list(values.values())}},resamples=30)
        self.assertEqual(result['physical_states'],2)
    def test_missing_nominal_cannot_be_silently_averaged_away(self):
        with self.assertRaisesRegex(ValueError,'Incomplete nominal'):
            collapse_nominals([dict(scene='A',state=0,nominal='one',value=1)],['one','two'])
    def test_protocol_strict_boundaries(self):
        self.assertEqual(verdict(.01,[.001,.1]),'优于')
        self.assertEqual(verdict(.05,[.001,.1]),'不劣于')
        self.assertEqual(verdict(.5,[-.05,.1]),'无法区分')

if __name__=='__main__':unittest.main()
