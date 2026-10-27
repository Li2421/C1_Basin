"""Freeze paper parameter choices on development only, then identical physical holdout."""
import json,sys
import paper_run as P
R=P.R;HERE=P.HERE

def freeze():
    path=R.OUT/'DEVELOPMENT_FROZEN.json'
    if path.exists():return json.load(path.open())
    old=json.load(open(HERE.parent/'basis_discovery_20261027/PROTOCOL.json'));seen=set();cases=[]
    for c in old['cases']:
        key=(c['scene'],c['state_index'])
        if key in seen:continue
        seen.add(key);cases.append(c)
    wang=[dict(alpha=a,gamma=g)for a in (1.,5.)for g in (.1,1.,10.)]
    variants={'dac_author_si':[{},dict(threshold_profile='author_narrow')],
        'mgr_author_unicycle_tracking':[dict(tau=.5),dict(tau=1.)],
        'chen_impc_ob_si_tracking':[dict(K=k,scale_lengths=s)for k in (8,24)for s in (False,True)],
        'wang2017_full_di_tracking':wang,'wang2017_consistent_perturbation_di_tracking':wang,
        'zhang2025_risk_si':[dict(gamma=g,q=q)for g in (1.,2.,5.)for q in (.01,1.)]}
    d=dict(cases=cases,variants=variants,seed=0,selection='Maximum success on entire unique physical development cohort; then fewer collisions, fewer numerical unknown, then lower all-cohort capped time. No heldout paper result observed. Default and published parameter profiles plus predeclared modest physical scaling/gain sensitivity; no success-label recipe choice.',holdout_states=[22,23],holdout_seed=5,holdout_scenes=old['test_scenes'],scope='Full mechanisms with explicitly recorded common-plant adaptations; no old undocumented heuristics.',evidence_gap='Earlier pilot demonstrated import/interface execution and revealed Wang QP infeasibility; it did not establish robust parameter transfer. Existing compatible identical requests reused, new parameter/controller variants separately fingerprinted.')
    R.dump(path,d);return d

def development():
    d=freeze();cs=[P.make(c,m,d['seed'],o)for m,options in d['variants'].items()for o in options for c in d['cases']];R.register(cs,'development');print('planned',len(cs),'physical dev states',len(d['cases']))

def holdout():
    d=freeze();done={r['rollout_uid']:r for r in map(json.loads,open(R.OUT/'candidates.jsonl'))};planned=json.load(open(R.OUT/'development_planned.json'))['exact_configs'];assert all(c['rollout_uid']in done for c in planned)
    frozen=R.OUT/'FINAL_FROZEN.json'
    if frozen.exists():final=json.load(frozen.open())
    else:
        rankings={};selected={}
        for method,options in d['variants'].items():
            rank=[]
            for o in options:
                rr=[done[c['rollout_uid']]for c in planned if c['controller_payload']['paper_method']==method and c['controller_payload']['paper_options']==o]
                rank.append(dict(options=o,n=len(rr),success=sum(r['success']for r in rr),collision=sum(r['collision']for r in rr),unknown=sum(r['numerical_failure']for r in rr),truncated_mean=sum(r['truncated_time']for r in rr)/len(rr)))
            rank.sort(key=lambda r:(-r['success'],r['collision'],r['unknown'],r['truncated_mean'],R.digest(r['options'])));rankings[method]=rank;selected[method]=rank[0]['options']
        final=dict(selected=selected,rankings=rankings,selection=d['selection']);R.dump(frozen,final)
    cohort=[dict(scene=s,state_index=i)for s in d['holdout_scenes']for i in d['holdout_states']]
    cases=[P.make(c,m,d['holdout_seed'],o)for c in cohort for m,o in final['selected'].items()];R.register(cases,'holdout');print('planned holdout',len(cases))
if __name__=='__main__':
    if '--holdout'in sys.argv:holdout()
    else:development()
