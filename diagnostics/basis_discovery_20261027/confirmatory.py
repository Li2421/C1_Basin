"""Confirm automatically compressed family on a fresh physical-state/future split."""
import json
from scipy.stats import qmc
from runner import OUT,OLD,inputs,dump,digest,behavior,old_oracle
from discovery_design import freeze,case,key
from screen import rows,pool_plan,minimal_cover
from holdout import support,eligible,candidate_pool,rank_spec
from shared_control.basis_dictionary import ALL,SIGNED


def main():
    p=freeze();path=OUT/'CONFIRMATORY_FROZEN.json'
    if path.exists():design=json.load(path.open())
    else:
        minimum=minimal_cover();family=minimum['minimal_families'][0]['names'];rr=[r for r in rows()if eligible(r)and key(r)in {(c['scene'],c['controller'],c['state_index'])for c in p['cases']}]
        pool=candidate_pool(family,rr);design=dict(names=family,transfer_pool=pool,states=[20,21],seeds=[4],scenes=p['test_scenes'],
            selection='Minimum canonical family from discovery seed0 only; fresh states20/21 seed4. No earlier heldout success labels used.')
        dump(path,design)
    todo=[case(c,sp)for c in p['cases']for sp in design['transfer_pool']];rr=pool_plan('compressed_transfer',todo,None)
    ranked=sorted(design['transfer_pool'],key=lambda sp:rank_spec(sp,rr));best=ranked[0];family=design['names']
    points=qmc.Sobol(len(family),scramble=True,seed=2026102814).random_base2(3);generic=[]
    for point in points:
        e=[0.,0.,0.];co={}
        for n,u in zip(family,point):
            if n=='G':e[0]=.5+.75*u
            elif n=='P':e[1]=-.5+u
            elif n=='R':e[2]=.75*u
            else:co[n]=2*u-1 if n in SIGNED else u
        generic.append(behavior(dict(eta=e,coefficients=co)))
    final_path=OUT/'CONFIRMATORY_FINAL.json'
    if final_path.exists():final=json.load(final_path.open());best=final['best_fixed']
    else:final=dict(design,best_fixed=best,oracle_pool=ranked+generic);dump(final_path,final)
    ctls=inputs()[1];test=[dict(scene=s,controller=ctl['type_code'],state_index=si)for s in design['scenes']for ctl in ctls for si in design['states']];seed=4
    original=old_oracle.b3_pool()[:16];points=qmc.scale(qmc.Sobol(3,scramble=True,seed=2026102808).random_base2(6),[.5,-.5,0.],[1.25,.5,.75])
    def specs(lo,hi):return [dict(eta=e.tolist(),coefficients={})for e in points[lo:hi]]
    def failed(cohort,results):return [c for c in cohort if not any(r['success']for r in results if key(r)==(c['scene'],c['controller'],c['state_index']))]
    rr=pool_plan('confirm_b3_16',[case(c,dict(eta=sp['eta'][:3],coefficients={}),seed)for c in test for sp in original],None);f16=failed(test,rr)
    rr=pool_plan('confirm_b3_48',[case(c,sp,seed)for c in f16 for sp in specs(0,32)],None);f48=failed(f16,rr)
    dump(OUT/'CONFIRMATORY_B3_FAILURES.json',dict(at16=f16,at48=f48))
    pool_plan('confirm_b3_64',[case(c,sp,seed)for c in f48 for sp in specs(32,48)],None)
    pool_plan('confirm_family_oracle',[case(c,sp,seed)for c in f48 for sp in final['oracle_pool']],None)
    prior=json.load(open(OLD/'INDEPENDENT_FROZEN.json'));dev=['xsc_hs_plus_w025','xsc_hs_box_n09_r014_f25','xsc_hs_swap_n07','hs_obst_n08_R140_r045'];mapping=dict(zip(design['scenes'],dev))
    t5=next(f['best_fixed']for f in json.load(open(OUT/'HOLDOUT_FROZEN.json'))['families']if 'T5'in f['names'])
    labels={};todo=[]
    for c in test:
        methods={'B3_frozen':dict(eta=prior['b3'][mapping[c['scene']]+'__'+c['controller']]['eta'][:3],coefficients={}),
            'safety':dict(eta=[0.,0.,0.],coefficients={}), 'RHR_B':dict(eta=[0.,0.,0.],coefficients={},rhr='B'),
            'compressed_fixed':best,'compressed_release':dict(best,schedule='release_third'),'T5_fixed':t5}
        for label,sp in methods.items():
            cc=case(c,sp,seed);todo.append(cc);labels.setdefault(label,[]).append(cc['rollout_uid'])
    dump(OUT/'CONFIRMATORY_FIXED_LABELS.json',labels);pool_plan('confirm_fixed',todo,None)
    # Test the same compressed frozen pool on genuine B3-resistant failed states.
    restart=json.load(open(OUT/'RESTART_FROZEN.json'));failed_names=set(json.load(open(OUT/'RESTART_LABELS.json'))['failed_after_B3'])
    from runner import make_case
    todo=[]
    for c in restart['cases']:
        if c['scene']['name']not in failed_names:continue
        ctl=next(x for x in ctls if x['type_code']==c['controller'])
        for sp in final['oracle_pool']:todo.append(make_case(c['scene'],ctl,0,sp,restart['seed']))
    pool_plan('restart_compressed',todo,None)

if __name__=='__main__':main()
