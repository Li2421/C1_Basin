"""Frozen fixed policies and matched finite-pool oracle; no test-based selection."""
import sys,json
from basis_select import R,OLD,HERE,case,stage,support
from scipy.stats import qmc
from shared_control.basis_dictionary import SIGNED

def design():
    path=R.OUT/'HOLDOUT_FROZEN.json'
    if path.exists():return json.load(path.open())
    selected=json.load(open(R.OUT/'FINAL_FROZEN.json'))
    initial=json.load(open(R.OUT/'SELECTION_FROZEN.json'))
    # Original fixed B3 was frozen before any of the new test states.
    prior=json.load(open(OLD/'CONFIRMATORY_FROZEN.json'))
    print('prior confirmatory keys',list(prior),flush=True)
    # A central, predetermined legal B3 setting; fixed and Oracle named separately.
    b3=dict(eta=[.875,0.,.375],coefficients={})
    families=[dict(name='B3',names=['G','P','R'],spec=b3)]+[
        dict(name='B'+str(x['dimension']),names=x['names'],spec=x['spec'])for x in selected['best_by_dimension']]
    for k,fam in enumerate(families):
        bounds=[(.5,1.25)if n=='G'else(-.5,.5)if n=='P'else(0.,.75)if n=='R'else(-1.,1.)if n in SIGNED else(0.,1.)for n in fam['names']]
        pts=qmc.Sobol(len(bounds),scramble=True,seed=202610280+k).random_base2(5)
        pool=[fam['spec']]
        for row in pts[:31]:
            vals={n:float(a+(b-a)*v)for n,v,(a,b)in zip(fam['names'],row,bounds)}
            sp=dict(eta=[vals.get(n,0.)for n in ('G','P','R')],coefficients={n:v for n,v in vals.items()if n not in ('G','P','R')})
            if 'T5'in vals:sp['t5_seed']=fam['spec'].get('t5_seed','tangent')
            if 'Y3'in vals:sp['clearance_side']=fam['spec'].get('clearance_side',1)
            pool.append(R.behavior(sp))
        fam['pool']=pool
    _,controllers,_,_=R.inputs()
    cohorts=[dict(scene=s,controller=c['type_code'],state_index=i)for s in initial['test_scenes']for c in controllers for i in initial['test_states']]
    d=dict(families=families,cases=cohorts,seed=initial['test_seed'],budgets=[1,4,8,16,32],selection='All family identities/coefficients frozen on development data; report every family. Pure family Oracle@K and B3 union@32 separately. Union = first16 B3 + first16 family. Never choose the family on test labels.',evidence_gap='States22..25 seed5 not present in preceding discovery. Native production nominal, safety, dynamics, thresholds unchanged.')
    R.dump(path,d);return d

def cases(d):return[case(c,sp,d['seed'])for c in d['cases']for f in d['families']for sp in f['pool']]

if __name__=='__main__':
    d=design();cs=cases(d)
    if '--plan'in sys.argv:R.register(cs,'holdout')
    else:stage(cs,'holdout')
