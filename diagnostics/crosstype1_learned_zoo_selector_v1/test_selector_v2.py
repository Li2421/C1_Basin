"""Source-only fitting and serialization for the second development iteration."""
from pathlib import Path
import datetime,json
import numpy as np
from selector_candidates_v2 import Predictor,grid,leave_one_type_candidate,restore
from selector import BASELINE


def main():
    rng=np.random.default_rng(202610180811);types=np.repeat(['a','b','c','d'],6)
    x=rng.normal(size=(24,4));rates=rng.uniform(size=(24,8));x[-1]=np.nan
    for strategy in ('armwise_tree','type_kernel'):
        choices,folds=leave_one_type_candidate(x,rates,types,strategy)
        changed=rates.copy();changed[types=='d']=1-rates[types=='d']
        other,ofolds=leave_one_type_candidate(x,changed,types,strategy)
        a=next(f for f in folds if f['held_type']=='d');b=next(f for f in ofolds if f['held_type']=='d')
        assert json.dumps(a,sort_keys=True)==json.dumps(b,sort_keys=True)
        assert np.array_equal(choices[types=='d'],other[types=='d'])
        model=restore(a['model']);assert np.array_equal(model.choose(x[types=='d']),choices[types=='d'])
        assert model.choose(np.full((1,4),np.nan))[0]==BASELINE
        pca=Predictor(grid(strategy)[0],2).fit(x,rates,types)
        assert np.allclose(pca.predict(x),restore(pca.to_dict()).predict(x))
        if strategy=='armwise_tree':
            def visit(node,depth=0):
                assert depth<=model.config['depth']
                if 'left' in node:visit(node['left'],depth+1);visit(node['right'],depth+1)
                else:assert node['controllers']>=3
            for arm,tree in model.models.items():visit(tree.tree)
    cfg=dict(kind='type_kernel',k=3,bandwidth=.5,shrinkage=0.,margin=0.)
    m=Predictor(cfg).fit(x[:18],rates[:18],types[:18]);idx=np.r_[np.arange(18),np.arange(6)]
    n=Predictor(cfg).fit(x[idx],rates[idx],types[idx]);assert np.allclose(m.predict(x),n.predict(x))
    result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,synthetic_only=True,
        held_labels_do_not_change_held_model=True,source_type_duplication_invariant=True,
        pca_and_model_serialization=True,all_missing_fallback=True,independent_arm_tree_bounds=True)
    (Path(__file__).parent/'SELECTOR_V2_TESTS.json').write_text(json.dumps(result,indent=2)+'\n');print(result)


if __name__=='__main__':main()
