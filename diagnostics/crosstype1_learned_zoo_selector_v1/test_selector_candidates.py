"""No held-type labels, outcome-free serialization, type weighting and leaf size."""
from pathlib import Path
import json,datetime
import numpy as np
from selector_candidates import Predictor,grid,leave_one_type_candidate,restore
from selector import BASELINE


def main():
    rng=np.random.default_rng(202610180811)
    types=np.repeat(['a','b','c','d'],6);x=rng.normal(size=(24,4));rates=rng.uniform(size=(24,8))
    x[-1]=np.nan
    for strategy in ('ridge_margin','nearest_type','shallow_tree'):
        choices,folds=leave_one_type_candidate(x,rates,types,strategy)
        changed=rates.copy();changed[types=='d']=1-rates[types=='d']
        other,ofolds=leave_one_type_candidate(x,changed,types,strategy)
        assert np.array_equal(choices[types=='d'],other[types=='d'])
        a=next(f for f in folds if f['held_type']=='d');b=next(f for f in ofolds if f['held_type']=='d')
        assert json.dumps(a,sort_keys=True)==json.dumps(b,sort_keys=True)
        model=restore(a['model']);assert np.array_equal(model.choose(x[types=='d']),choices[types=='d'])
        assert model.choose(np.full((1,4),np.nan))[0]==BASELINE
        if strategy=='shallow_tree':
            def visit(node,depth=0):
                assert depth<=model.config['depth']
                if 'left' in node:visit(node['left'],depth+1);visit(node['right'],depth+1)
                else:assert node['controllers']>=3
            visit(model.tree)
        pca=Predictor(grid(strategy)[0],2).fit(x,rates,types)
        assert np.allclose(pca.predict(x),restore(pca.to_dict()).predict(x))
    # Duplicating source instances inside a type cannot alter type-centroid KNN.
    config=dict(kind='nearest_type',k=2,weight='equal',margin=0.)
    model=Predictor(config).fit(x[:18],rates[:18],types[:18])
    duplicated=np.r_[np.arange(18),np.arange(6)]
    other=Predictor(config).fit(x[duplicated],rates[duplicated],types[duplicated])
    assert np.allclose(model.predict(x),other.predict(x))
    out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,synthetic_only=True,
        held_type_labels_cannot_change_held_predictions=True,serialization_exact_choices=True,
        all_missing_fallback=True,tree_depth_and_minimum_leaf=True,type_centroid_duplication_invariant=True)
    (Path(__file__).parent/'SELECTOR_CANDIDATE_TESTS.json').write_text(json.dumps(out,indent=2)+'\n');print(out)


if __name__=='__main__':main()
