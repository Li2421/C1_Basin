from experiment import *
from migration_statistics import paired
import unittest

class Pairing(unittest.TestCase):
    def test_identical_methods_cancel_and_unknown_is_bounded(self):
        rows=[];q=[]
        for name,typ in [('a','type1'),('b','type2')]:
            q.append(dict(controller=name,scene='synthetic',qualified=True))
            for state in range(3):
                for seed in (0,1):
                    for method in ('safety','fixed_eta10'):
                        success=int((state+seed)%2==0)
                        rows.append(dict(success=success,collision=0,timeout=1-success,numerical_failure=0,case=dict(state=dict(scene='synthetic',N=2,index=state),seed=seed,method=method,controller_payload=dict(nominal=dict(controller=name,type_code=typ)))))
        result=paired(rows,q,'synthetic',['safety','fixed_eta10'])
        self.assertTrue(all(r['delta_ci95']==[0.,0.] for r in result['statistics']))
        changed=next(r for r in rows if r['success'] and r['case']['method']=='fixed_eta10');changed.update(success=0,timeout=0,numerical_failure=1)
        result=paired(rows,q,'synthetic',['safety','fixed_eta10']);self.assertEqual(result['harms_vs_safety']['fixed_eta10'],dict(known=0,upper_including_unknown=1))
        r=next(r for r in result['statistics'] if r['type']=='type_equal' and r['method']=='fixed_eta10')
        self.assertLess(r['success_rate'],r['success_upper']);self.assertLessEqual(r['identified_delta_bounds'][0],0.);self.assertEqual(r['identified_delta_bounds'][1],0.)

if __name__=='__main__':unittest.main()
