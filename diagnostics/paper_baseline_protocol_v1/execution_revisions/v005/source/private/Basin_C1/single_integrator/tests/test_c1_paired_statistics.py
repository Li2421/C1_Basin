import unittest
from single_integrator.c1.paired_statistics import summarize


def rows(n=4, replicas=3, baseline_dead=True, corrected_success=True):
    return [dict(rid=i, noise_seed=j, method=name,
                 any_deadlock=baseline_dead if name=='Safety' else False,
                 success=(not baseline_dead) if name=='Safety' else corrected_success)
            for i in range(n) for j in range(replicas) for name in ('Safety','C1')]


class PairedStatisticsTests(unittest.TestCase):
    def test_deadlock_to_timeout_is_not_resolution(self):
        r = summarize(rows(corrected_success=False), replicates=1000)
        self.assertEqual(r['relative_deadlock_reduction'], 1.)
        self.assertEqual(r['safety_deadlock_to_success_fraction'], 0.)
        self.assertEqual(r['failure_rate_difference'], 0.)

    def test_no_baseline_events_is_undefined_not_perfect_improvement(self):
        r = summarize(rows(baseline_dead=False), replicates=1000)
        self.assertIsNone(r['relative_deadlock_reduction'])
        self.assertIsNone(r['relative_deadlock_reduction_ci'])
        self.assertFalse(r['rate_reduction_interval_excludes_zero'])

    def test_replicating_noise_does_not_inflate_cluster_evidence(self):
        def varying(replicas):
            data = rows(n=12, replicas=replicas)
            for r in data:
                if r['rid'] < 6:
                    r['any_deadlock'] = False
                    r['success'] = True
            return summarize(data, replicates=1000)
        self.assertEqual(varying(1)['deadlock_rate_difference_ci'], varying(8)['deadlock_rate_difference_ci'])

    def test_duplicate_or_missing_policy_rejected(self):
        original = rows()
        for data in (original[:-1], original+[original[0]]):
            with self.assertRaises(ValueError):
                summarize(data, replicates=1000)

    def test_stalled_endpoint_does_not_relabel_every_timeout(self):
        data = rows(baseline_dead=False)
        for r in data:
            r['six_class_outcome'] = 'success'
            if r['method']=='Safety':
                r['success'] = False
                r['six_class_outcome'] = 'stalled_deadlock' if r['rid'] < 2 else 'other_timeout'
        strict = summarize(data, replicates=1000)
        both = summarize(data, replicates=1000, endpoint='strict_or_stalled')
        self.assertEqual(strict['safety_deadlocks'], 0)
        self.assertEqual(both['safety_deadlocks'], 6)
        self.assertEqual(both['safety_failures'], 12)


if __name__ == '__main__':
    unittest.main()
