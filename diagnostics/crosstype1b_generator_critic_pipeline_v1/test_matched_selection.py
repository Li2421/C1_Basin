"""Matched no-state shrinkage has the same source-only selection boundary."""
from core import *
import select_matched_v2b as module


def main():
    rng=np.random.default_rng(23);types=np.asarray(['a','b','c','outer'])
    data=dict(names=np.asarray(['A','B','C','D']),types=types,labels=rng.integers(0,2,(4,2,13,2)))
    pf=rng.uniform(.1,.9,(3,4,2,13)).astype(np.float32);pn=rng.uniform(.1,.9,pf.shape).astype(np.float32)
    saved=[module.inner_selection,module.inner_action,module.predictions,module.action_predictions]
    def inner(*args):return dict(selected=dict(initialization='scratch',step=100))
    def pred(source,mode,selection):return pf,data['names'],dict(source_types=sorted(source),mode='no_state')
    def act(source,selection):return pn,data['names'],dict(source_types=sorted(source),mode='action_only')
    try:
        module.inner_selection=inner;module.inner_action=inner;module.predictions=pred;module.action_predictions=act
        p,m,r=module.select_no_state(['a','b','c'],{},data)
        changed={**data,'labels':data['labels'].copy()};changed['labels'][-1]=1-changed['labels'][-1]
        p2,m2,r2=module.select_no_state(['a','b','c'],{},changed)
        assert np.array_equal(p,p2) and m==m2 and r==r2
        assert m['mode']=='no_state' and m['no_controller']['mode']=='action_only'
    finally:module.inner_selection,module.inner_action,module.predictions,module.action_predictions=saved
    write(HERE/'MATCHED_SELECTION_TESTS.json',dict(utc=utc(),passed=True,outer_label_perturbation_exactly_invariant=True))
    seal('MATCHED_SELECTION_TESTS.receipt.json',[HERE/'MATCHED_SELECTION_TESTS.json',HERE/'test_matched_selection.py',HERE/'select_matched_v2b.py'])
    print('Matched source-only selection checks passed',flush=True)


if __name__=='__main__':main()
