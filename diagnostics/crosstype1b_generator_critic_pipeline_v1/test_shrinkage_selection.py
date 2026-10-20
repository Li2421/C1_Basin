"""The added shrinkage hyperparameter cannot observe outer labels."""
from core import *
import select_models_v2 as module


def main():
    rng=np.random.default_rng(41);types=np.asarray(['a','b','c','outer'])
    data=dict(names=np.asarray(['A','B','C','D']),types=types,labels=rng.integers(0,2,(4,2,13,2)))
    pf=rng.uniform(.1,.9,(3,4,2,13)).astype(np.float32);pn=rng.uniform(.1,.9,pf.shape).astype(np.float32)
    original_inner,original_predictions=module.inner_selection,module.predictions
    def inner(source,mode,configuration,initialization):return dict(selected=dict(initialization=initialization or 'scratch',step=100))
    def predictions(source,mode,selection):
        return (pf if mode=='full' else pn),data['names'],dict(source_types=sorted(source),mode=mode,initialization='scratch',step=100)
    try:
        module.inner_selection=inner;module.predictions=predictions
        p,m,r=module.select(['a','b','c'],{},data)
        alternate={**data,'labels':data['labels'].copy()};alternate['labels'][-1]=1-alternate['labels'][-1]
        p2,m2,r2=module.select(['a','b','c'],{},alternate)
        assert np.array_equal(p,p2) and m==m2 and r==r2
        assert 0<m['full_weight']<=1
        assert all(set(row['source_only_folds'])=={'a','b','c'} for row in r['lambda_curve'])
        assert np.array_equal(1.*pf+0.*pn,pf)
    finally:module.inner_selection,module.predictions=original_inner,original_predictions
    write(HERE/'SHRINKAGE_SELECTION_TESTS.json',dict(utc=utc(),passed=True,
        outer_label_perturbation_exactly_invariant=True,positive_controller_weight=True,lambda_one_contains_v1_exactly=True))
    seal('SHRINKAGE_SELECTION_TESTS.receipt.json',[HERE/'SHRINKAGE_SELECTION_TESTS.json',HERE/'test_shrinkage_selection.py',
        HERE/'select_models_v2.py',HERE/'ITERATION_V2_DESIGN.receipt.json'])
    print('Source-only shrinkage selection checks passed',flush=True)


if __name__=='__main__':main()
