"""Historical strong fixed B3 and safety-only reference, unchanged frozen settings."""
import json,sys
from basis_select import R,HERE,case,stage
def plan():
    p=json.load(open(R.OUT/'HOLDOUT_FROZEN.json'))
    prior=json.load(open(HERE.parent/'b5_oracle_20261027/INDEPENDENT_FROZEN.json'))
    dev=['xsc_hs_plus_w025','xsc_hs_box_n09_r014_f25','xsc_hs_swap_n07','hs_obst_n08_R140_r045']
    test=json.load(open(R.OUT/'SELECTION_FROZEN.json'))['test_scenes'];mapping=dict(zip(test,dev));cases=[];labels={}
    for c in p['cases']:
        for name,sp in dict(B3_historical_frozen=dict(eta=prior['b3'][mapping[c['scene']]+'__'+c['controller']]['eta'][:3],coefficients={}),safety_only=dict(eta=[0.,0.,0.],coefficients={})).items():
            cc=case(c,sp,p['seed']);cases.append(cc);labels.setdefault(name,[]).append(cc['rollout_uid'])
    R.dump(R.OUT/'REFERENCE_LABELS.json',dict(labels=labels,evidence='Original INDEPENDENT_FROZEN.json copied exactly. Supplement because central B3 setting is weaker than previously tuned per-scene/controller B3; no new fit or test-label choice.'))
    return cases
if __name__=='__main__':
    cs=plan()
    if '--plan'in sys.argv:R.register(cs,'reference')
    else:stage(cs,'reference')
