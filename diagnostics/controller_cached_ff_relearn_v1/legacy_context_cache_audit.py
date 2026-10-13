"""Audit a controller-omitting key in the legacy H20 feature cache.

Single existing TRAIN state, existing eta, exact historical probe streams.
No new Q labels or training examples; all artifacts retain diagnostic status.
"""
from collections import defaultdict
import sys
import numpy as np
from .study import HERE, OLD, FIELD, read, write, freeze, sha


def run():
    source=read(OLD/'decision_pair_critic_v3/source_pairs.json')['records']
    groups=defaultdict(dict)
    for row in source:groups[(row['state_uid'],tuple(row['eta']))][row['controller_condition']]=row
    equal=[];unequal=[]
    for key,rs in groups.items():
        difference=np.max(np.abs(np.asarray(rs['base']['context'][:10])-rs['held']['context'][:10]))
        (equal if difference<1e-12 else unequal).append(key)
    selected=sorted(k for k in equal if groups[k]['base']['split']=='train')[0]
    state=next(s for s in read(HERE/'protocol.json')['states'] if s['uid']==selected[0])
    rows=groups[selected]
    freeze(HERE/'legacy_context_probe_preflight.json',dict(
        scope='Input-feature correctness diagnostic, not success-label generation',
        state_uid=state['uid'],split='train',eta=list(selected[1]),
        Q_rollouts_requested=0,new_training_examples=0,
        cached_base_context='Available and checked against direct function reconstruction',
        cached_held_context='Nominal prefix exactly identical to base; ambiguous under advertised controller-specific semantics. Recompute one case to test cache-key collision.',
        context_seeds=[2026100417,2026100499],steps_per_context_trajectory=20,
        outcome_based_selection=False,selection='Lexicographically first TRAIN state/eta with the cache-alias signature',
        cache_key_code=str(OLD/'learn.py'),code_sha256=sha(OLD/'learn.py')))
    import jax
    sys.path.insert(0,str(FIELD))
    from field_pipeline_v1 import learn
    from field_pipeline_v1.ring_t0_source import runtime
    from field_pipeline_v1.held import HeldRuntime
    base=runtime();held=HeldRuntime('ring_exchange');jax.config.update('jax_enable_x64',False)
    count=[0];old_step=base.env_cls.step
    def counted_step(self,*args,**kwargs):
        count[0]+=1
        return old_step(self,*args,**kwargs)
    base.env_cls.step=counted_step
    try:
        learn.NOMINAL_CACHE.clear()
        base_rebuilt=learn.h20_context(base,state,np.asarray(selected[1]))
        held_polluted=learn.h20_context(held,state,np.asarray(selected[1]))
        learn.NOMINAL_CACHE.clear()
        held_correct=learn.h20_context(held,state,np.asarray(selected[1]))
    finally:
        base.env_cls.step=old_step;learn.NOMINAL_CACHE.clear()
    cached_base=np.asarray(rows['base']['context']);cached_held=np.asarray(rows['held']['context'])
    np.testing.assert_allclose(base_rebuilt,cached_base,rtol=1e-5,atol=1e-5)
    np.testing.assert_allclose(held_polluted,cached_held,rtol=1e-5,atol=1e-5)
    np.testing.assert_array_equal(base_rebuilt[:10],held_polluted[:10])
    assert np.max(np.abs(held_correct[:10]-held_polluted[:10]))>1e-5
    affected=np.flatnonzero(np.abs(held_correct-held_polluted)>1e-6).tolist()
    result=dict(
        finding='CONFIRMED_CONTROLLER_OMITTED_FROM_H20_NOMINAL_CACHE_KEY',
        source_state_eta_pairs=len(groups),source_pairs_with_identical_base_held_nominal_prefix=len(equal),
        source_states_with_identical_prefix=len({k[0] for k in equal}),
        source_states_with_different_prefix=len({k[0] for k in unequal}),
        case_state_uid=state['uid'],case_eta=list(selected[1]),
        cached_base_reconstruction_max_error=float(np.max(np.abs(base_rebuilt-cached_base))),
        cached_held_polluted_reconstruction_max_error=float(np.max(np.abs(held_polluted-cached_held))),
        held_correct_vs_polluted_max_error=float(np.max(np.abs(held_correct-held_polluted))),
        affected_context_indices=affected,physical_steps_for_input_audit=count[0],new_Q_rollouts=0,
        root_cause='NOMINAL_CACHE key is (scenario,state_uid,probe_root), omitting the controller. In a paired collector, held-controller nominal trajectory is replaced by the previously cached base-controller trajectory.',
        consequence='A mixed feature contract: controller-specific baseline in separately collected cohorts, base-controller baseline in paired cohorts. Q16 rollout labels are unaffected.',
        evidence_boundary='This proves a feature-generation defect. It does not quantify how much old critic generalization failure it caused; corrected-branch static features avoid this cache entirely.',
        code_sha256=sha(OLD/'learn.py'))
    write(HERE/'legacy_context_cache_audit.json',result)
    np.savez_compressed(HERE/'legacy_context_probe_vectors.npz',base=base_rebuilt,held_polluted=held_polluted,held_correct=held_correct)
    print(result,flush=True)


if __name__=='__main__':run()
