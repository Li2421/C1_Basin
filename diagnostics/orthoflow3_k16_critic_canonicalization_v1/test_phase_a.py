"""Lightweight protocol tests: no learning and no rollout execution."""
import unittest
import numpy as np
from diagnostics.orthoflow3_k16_critic_canonicalization_v1 import phase_a as a


class PhaseAProtocolTest(unittest.TestCase):
    def test_v2_and_canonical_artifacts_unchanged(self):
        for name,value in a.load(a.OUT/'canonical_hashes_before.json').items():
            self.assertEqual(a.sha(a.ROOT/name),value,name)
        addendum=a.OUT/'runtime_artifacts.json'
        if addendum.exists():
            for name,value in a.load(addendum)['files'].items():
                self.assertEqual(a.sha(a.ROOT/name),value,name)

    def test_all_states_original_parent_splits_and_valid_conditioning(self):
        rows=a.states();self.assertEqual(len(rows),245)
        norm=a.load(a.frozen.NORMALIZATION);parents={}
        for row in rows:
            key=(row['scenario'],row['parent_episode_id'])
            if key in parents:self.assertEqual(parents[key],row['split'])
            parents[key]=row['split']
            h,c=a.inputs(row,norm)
            self.assertEqual(h.shape,(100 if row['scenario']=='ring_exchange' else 80,))
            self.assertEqual(c.shape,(33,));self.assertTrue(np.isfinite(h).all())

    def test_exact_nested_proposals_and_no_extra_candidates(self):
        rows=a.load(a.OUT/'phase_a/proposals.json')['states'];by={r['state_uid']:r for r in rows}
        old=a.load(a.ROOT/'diagnostics/orthoflow3_generator_critic_v1/generator/frozen_validation_proposals.json')['states']
        for r in old:
            self.assertEqual(by[r['state_uid']]['etas'][1:5],r['samples'])
            self.assertEqual(by[r['state_uid']]['etas'][0],r['generator_mean'])
        for r in rows:
            self.assertEqual(len(r['etas']),17)
            e=np.asarray(r['etas']);self.assertTrue((e>=a.learn.LOW).all());self.assertTrue((e<=a.learn.HIGH).all())
            self.assertFalse(np.any(np.all(e==0,axis=1)))
            self.assertEqual(r['proposal_hash'],a.digest(r['etas']))

    def test_initial_expansion_inherits_parent_splits(self):
        rows=a.load(a.OUT/'initial_expansion/states.json')
        parents={r['source_initial_state_id']:r for r in a.states() if r['scenario']!='double_bottleneck'}
        self.assertEqual(len(rows),80)
        for r in rows:
            source=parents[r['state_uid']]
            self.assertEqual(source['split'],r['split'])
            self.assertEqual(source['parent_episode_id'],r['parent_episode_id'])
            self.assertEqual(r['timestep'],0)
            self.assertEqual(r['provenance']['opened_test_archives'],0)


if __name__=='__main__':unittest.main()
