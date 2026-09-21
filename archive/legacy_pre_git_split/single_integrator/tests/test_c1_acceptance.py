import copy
import unittest
from single_integrator.c1.acceptance import assess


def rows():
    result=[]
    for rid in range(40):
        for method in ('Safety','C1'):
            dead=rid<12 and method=='Safety'
            result.append(dict(rid=rid,noise_seed=123,method=method,success=not dead,any_deadlock=dead,
                timeout=False,six_class_outcome='safe_deadlock' if dead else 'success',
                safety=dict(agent_collision_steps=0,wall_collision_steps=0,outside_endpoints=0,cbf_violations=0,speed_violations=0)))
    return result


class AcceptanceTests(unittest.TestCase):
    def test_full_paired_resolution_passes_empirical_gates(self):
        self.assertTrue(assess(rows(),range(40),[123],family_size=6)['all_endpoint_gates_passed'])

    def test_relabelling_deadlock_as_timeout_is_not_resolution(self):
        records=rows()
        for r in records:
            if r['method']=='C1' and r['rid']<12:
                r.update(success=False,timeout=True,six_class_outcome='other_timeout')
        report=assess(records,range(40),[123],family_size=6)
        self.assertTrue(report['gates']['relative_deadlock_reduction_at_least_80_percent'])
        self.assertFalse(report['all_endpoint_gates_passed'])

    def test_even_favorable_complete_subset_is_rejected(self):
        with self.assertRaises(ValueError):assess(rows()[:-2],range(40),[123],family_size=6)

    def test_one_safety_violation_prevents_acceptance(self):
        records=copy.deepcopy(rows());records[1]['safety']['cbf_violations']=1
        self.assertFalse(assess(records,range(40),[123],family_size=6)['all_endpoint_gates_passed'])


if __name__=='__main__':unittest.main()
