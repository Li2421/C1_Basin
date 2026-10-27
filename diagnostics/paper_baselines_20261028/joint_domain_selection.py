"""Minimum observed support across external development and legacy TT/FF evidence.

Uses all existing eligible witnesses, not hand-picked B4/B5 proposals. Results
are finite-pool coverage, not impossibility certificates for unobserved supports.
"""
import json,itertools,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parents[1]))
from shared_control.basis_dictionary import ALL
def main():
    protocol=json.load(open(HERE.parent/'basis_discovery_20261027/PROTOCOL.json'));dev={(c['scene'],c['controller'],c['state_index'])for c in protocol['cases']}
    witnesses=[]
    for r in map(json.loads,open(HERE.parent/'basis_discovery_20261027/candidates.jsonl')):
        if not r['success']or r['seed']!=0 or (r['scene'],r['controller'],r['state_index'])not in dev:continue
        sp=r['spec']
        if sp.get('schedule')or sp.get('rhr')or sp.get('t5_seed','tangent')!='tangent'or sp.get('clearance_side',1)!=1:continue
        if any(abs(v)>1 for v in sp['coefficients'].values()):continue
        names=[n for n,v in zip(('G','P','R'),sp['eta'])if v]+[n for n,v in sp['coefficients'].items()if v]
        key=('external_dev',r['scene'],r['controller'],r['state_index']);witnesses.append((key,names,sp,r['rollout_uid']))
    files=[HERE.parent/'b5_oracle_20261027/legacy_candidates.jsonl',HERE.parent/'b5_oracle_20261027/legacy_prior_candidates.jsonl',HERE/'legacy/candidates.jsonl']
    for file in files:
        for r in map(json.loads,file.open()):
            if not r['success']:continue
            sp=r['spec'];mode=sp['mode'];e=sp['eta'];co={}
            if mode in ('b3','ff_b3'):pass
            elif mode=='add'and not e[4]:co={'T1':e[3]}if e[3]else{}
            elif mode in ('dictionary_tt','ff_dictionary'):co={'C2':sp['C2']}
            else:continue
            spec=dict(eta=e[:3],coefficients=co);names=[n for n,v in zip(('G','P','R'),e)if v]+[n for n,v in co.items()if v]
            key=('legacy_FF'if mode.startswith('ff')else'legacy_TT',r['scenario'],r['legacy_state_uid'],r['seed']);witnesses.append((key,names,spec,r['rollout_uid']))
    keys=sorted({w[0]for w in witnesses});index={k:i for i,k in enumerate(keys)};bit={n:1<<i for i,n in enumerate(ALL)};coverage={}
    for k,names,sp,uid in witnesses:
        mask=sum(bit[n]for n in names);coverage[mask]=coverage.get(mask,0)|(1<<index[k])
    # Subset zeta-transform: exact pool coverage for every dictionary support.
    cover=[0]*(1<<len(ALL))
    for mask,v in coverage.items():cover[mask]=v
    for i in range(len(ALL)):
        b=1<<i
        for mask in range(len(cover)):
            if mask&b:cover[mask]|=cover[mask^b]
    target=(1<<len(keys))-1;minimal=[]
    for dim in range(0,6):
        for inds in itertools.combinations(range(len(ALL)),dim):
            mask=sum(1<<i for i in inds)
            if cover[mask]==target:minimal.append(dict(names=[ALL[i]for i in inds],dimension=dim,mask=mask))
        if minimal:break
    if not minimal:raise RuntimeError('No <=5-column observed support covers all known successes; report instead of silently dropping cases')
    selected=min(minimal,key=lambda x:(x['dimension'],x['mask']));family=set(selected['names'])
    pool={json.dumps(sp,sort_keys=True):sp for k,names,sp,uid in witnesses if set(names)<=family}
    out=dict(selected=selected,minimal_families=minimal,known_solved_cases=len(keys),cohort_success_keys=keys,eligible_witnesses=len(witnesses),candidate_pool=list(pool.values()),criterion='Smallest support covering every already-solved development+legacy case; tie by canonical dictionary mask. Uses legacy16/17 for discovery only; fresh-seed confirmation required. Finite observed witness cover, not continuous minimality.',warning='Original four checkpoint transfer falsified fixed GC2 generalization. Family identity now derived automatically from combined-domain evidence; no external holdout22/23 labels used here.')
    (HERE/'CROSS_DOMAIN_FROZEN.json').write_text(json.dumps(out,indent=2)+'\n');print(selected,'known solved',len(keys),'pool',len(pool))
if __name__=='__main__':main()
