"""Automatic family selection, common-configuration transfer, untouched holdout."""
import json
from collections import defaultdict
import numpy as np
from scipy.stats import qmc
from runner import OUT,OLD,inputs,dump,digest,make_case,old_oracle,behavior
from discovery_design import freeze,case,key
from screen import rows,pool_plan,minimal_cover
from shared_control.basis_dictionary import ALL,NEW,SIGNED


def support(sp):return {n for n,v in zip(ALL[:3],sp['eta'])if v!=0}|set(sp['coefficients'])
def eligible(r):
    sp=r['spec']
    return r['seed']==0 and sp.get('t5_seed','tangent')=='tangent'and sp.get('clearance_side',1)==1 and not sp.get('schedule')and all(abs(v)<=1 for v in sp['coefficients'].values())

def rank_spec(sp,rr):
    hits=[r for r in rr if digest(r['spec'])==digest(sp)]
    return(-sum(r['success']for r in hits),sum(r['collision']for r in hits),np.mean([r['truncated_time']for r in hits]),digest(sp))

def candidate_pool(family,rr,n=8):
    names=set(family);candidates={digest(r['spec']):r['spec']for r in rr if r['success']and support(r['spec'])<=names}
    ordered=sorted(candidates.values(),key=lambda sp:rank_spec(sp,rr))
    return ordered[:n]

def transfer():
    if (OUT/'HOLDOUT_FROZEN.json').exists() and (OUT/'transfer_execution.json').exists() and json.load(open(OUT/'transfer_execution.json')).get('complete'):
        return json.load(open(OUT/'HOLDOUT_FROZEN.json'))['families']
    path=OUT/'TRANSFER_FROZEN.json';p=freeze()
    if path.exists():design=json.load(path.open())
    else:
        minimum=minimal_cover();rr=[r for r in rows()if eligible(r)];families=minimum['minimal_families'][:2]
        # Include the strongest single class as a low-complexity reference, chosen
        # by uniform canonical single-screen rescue, not by a hand-picked name.
        hard={(c['scene'],c['controller'],c['state_index'])for c in p['cases']if c['hard']}
        scores={n:len({key(r)for r in rr if r['success']and set(r['spec']['coefficients'])=={n}}&hard)for n in NEW}
        single=min(scores,key=lambda n:(-scores[n],n));single_names=['G','P','R',single]
        if not any(set(f['names'])==set(single_names)for f in families):families.append(dict(names=single_names,reference='best_single_class'))
        design=dict(families=[dict(**f,id='family'+str(i),pool=candidate_pool(f['names'],rr))for i,f in enumerate(families)],
            single_scores=scores,rule='First two canonical minimum witness-cover families; tie lexical, excluding T5 cost when possible; plus strongest single-class reference.',
            split='22 discovery cases only; holdout states18/19 seed3 not read')
        dump(path,design)
    todo=[];labels={}
    for family in design['families']:
        for sp in family['pool']:
            for c in p['cases']:
                cc=case(c,sp);todo.append(cc);labels.setdefault(family['id'],[]).append(cc['rollout_uid'])
    dump(OUT/'TRANSFER_LABELS.json',labels);result=pool_plan('transfer',todo,None)
    frozen=[]
    for family in design['families']:
        pool=family['pool'];ranked=sorted(pool,key=lambda sp:rank_spec(sp,result));best=ranked[0]
        names=family['names'];sobol=qmc.Sobol(len(names),scramble=True,seed=2026102807).random_base2(3)
        generic=[]
        for point in sobol:
            e=[0.,0.,0.];co={}
            for name,u in zip(names,point):
                if name=='G':e[0]=.5+.75*u
                elif name=='P':e[1]=-.5+u
                elif name=='R':e[2]=.75*u
                else:co[name]=2*u-1 if name in SIGNED else u
            generic.append(behavior(dict(eta=e,coefficients=co)))
        frozen.append(dict(id=family['id'],names=names,best_fixed=best,oracle_pool=ranked[:8]+generic,training_candidates=len(ranked)))
    dump(OUT/'HOLDOUT_FROZEN.json',dict(families=frozen,test_scenes=p['test_scenes'],states=p['test_states'],seeds=p['test_seeds'],
        budgets='B3 @16 -> @48; if failure compare additional16 B3 versus additional<=16 per frozen family, all retaining original48.',
        no_test_selection=True))
    return frozen


def holdout():
    frozen=json.load(open(OUT/'HOLDOUT_FROZEN.json'));p=freeze();scenes,ctls,acts,_=inputs()
    test=[dict(scene=s,controller=ctl['type_code'],state_index=si)for s in p['test_scenes']for ctl in ctls for si in p['test_states']]
    seed=p['test_seeds'][0];original=old_oracle.b3_pool()[:16]
    points=qmc.scale(qmc.Sobol(3,scramble=True,seed=2026102808).random_base2(6),[.5,-.5,0.],[1.25,.5,.75])
    def specs(lo,hi):return [dict(eta=e.tolist(),coefficients={})for e in points[lo:hi]]
    initial=[case(c,dict(eta=sp['eta'][:3],coefficients={}),seed)for c in test for sp in original]
    a16=pool_plan('test_b3_16',initial,None)
    def failed(cohort,rr):return [c for c in cohort if not any(r['success']for r in rr if key(r)==(c['scene'],c['controller'],c['state_index']))]
    f16=failed(test,a16);a48=pool_plan('test_b3_48',[case(c,sp,seed)for c in f16 for sp in specs(0,32)],None)
    f48=failed(f16,a48);dump(OUT/'TEST_B3_FAILURES.json',dict(at16=f16,at48=f48))
    pool_plan('test_b3_64',[case(c,sp,seed)for c in f48 for sp in specs(32,48)],None)
    labels={};todo=[]
    for family in frozen['families']:
        label=family['id']
        for c in f48:
            for sp in family['oracle_pool']:
                cc=case(c,sp,seed);todo.append(cc);labels.setdefault(label,[]).append(cc['rollout_uid'])
    dump(OUT/'TEST_ORACLE_LABELS.json',labels);pool_plan('test_family_oracle',todo,None)
    # Fixed transfer policies evaluated on the entire cohort, regardless of B3 success.
    prior=json.load(open(OLD/'INDEPENDENT_FROZEN.json'))
    dev=['xsc_hs_plus_w025','xsc_hs_box_n09_r014_f25','xsc_hs_swap_n07','hs_obst_n08_R140_r045'];mapping=dict(zip(p['test_scenes'],dev))
    todo=[];labels={}
    for c in test:
        b3=prior['b3'][mapping[c['scene']]+'__'+c['controller']]['eta'][:3]
        methods={'B3_frozen':dict(eta=b3,coefficients={}), 'safety':dict(eta=[0.,0.,0.],coefficients={}),
                 'RHR_B':dict(eta=[0.,0.,0.],coefficients={},rhr='B')}
        for f in frozen['families']:
            methods[f['id']]=f['best_fixed']
        # Same basis and coefficients, a separately identified temporal ablation.
        methods['family0_release']=dict(frozen['families'][0]['best_fixed'],schedule='release_third')
        for label,sp in methods.items():
            cc=case(c,sp,seed);todo.append(cc);labels.setdefault(label,[]).append(cc['rollout_uid'])
    dump(OUT/'TEST_FIXED_LABELS.json',labels);pool_plan('test_fixed',todo,None)


def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['transfer','test','all']);arg=ap.parse_args()
    if arg.stage in ('transfer','all'):transfer()
    if arg.stage in ('test','all'):holdout()

if __name__=='__main__':main()
