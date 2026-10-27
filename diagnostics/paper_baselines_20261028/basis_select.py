"""Reuse discovery evidence, evaluate common fixed settings, freeze <=5 columns."""
import sys,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;OLD=HERE.parent/'basis_discovery_20261027';sys.path.insert(0,str(OLD))
import runner as R
from discovery_design import case,key
from scipy.stats import qmc
import numpy as np
R.OUT=HERE/'basis';R.OUT.mkdir(exist_ok=True);R.EXP='exp_paper_baselines_basis_20261028'

def support(sp):return [n for n,v in zip(('G','P','R'),sp['eta'])if v!=0]+[n for n,v in sp['coefficients'].items()if v!=0]
def stage(cases,name):
    mark=R.OUT/f'{name}_execution.json'
    if mark.exists()and json.load(mark.open())['complete']:
        ids={c['rollout_uid']for c in json.load(open(R.OUT/f'{name}_planned.json'))['exact_configs']}
        return[r for r in map(json.loads,open(R.OUT/'candidates.jsonl'))if r['rollout_uid']in ids]
    prior={r['rollout_uid']:r for r in map(json.loads,open(OLD/'candidates.jsonl'))}
    output=R.OUT/'candidates.jsonl';done={r['rollout_uid']for r in map(json.loads,output.open())}if output.exists()else set()
    with output.open('a')as f:
        for c in cases:
            uid=c['rollout_uid']
            if uid in prior and uid not in done:
                r=dict(prior[uid],source='exact_prior_dictionary_reuse',wall_seconds=0.);assert r['spec']==c['spec'];f.write(json.dumps(r)+'\n');done.add(uid)
    return R.execute(cases,name)

def main():
    p=json.load(open(OLD/'PROTOCOL.json'));keys={tuple(c[k]for k in ('scene','controller','state_index'))for c in p['cases']}
    rows=[r for r in map(json.loads,open(OLD/'candidates.jsonl'))if r['seed']==0 and key(r)in keys and not r['spec'].get('schedule') and not r['spec'].get('rhr') and r['spec'].get('t5_seed','tangent')=='tangent' and r['spec'].get('clearance_side',1)==1 and all(abs(v)<=1 for v in r['spec']['coefficients'].values())]
    group={}
    for r in rows:group.setdefault(R.digest(r['spec']),[]).append(r)
    path=R.OUT/'SELECTION_FROZEN.json'
    if path.exists():design=json.load(path.open())
    else:
        pool=[]
        for dim in range(1,6):
            eligible=[v for v in group.values()if len(support(v[0]['spec']))==dim]
            eligible.sort(key=lambda v:(-len({key(r)for r in v if r['success']}),-len(v),sum(r['collision']for r in v),np.mean([r['truncated_time']for r in v]),R.digest(v[0]['spec'])))
            chosen=[];families={}
            for v in eligible:
                fam=tuple(sorted(support(v[0]['spec'])));families[fam]=families.get(fam,0)+1
                if families[fam]>2:continue
                chosen.append(v[0]['spec'])
                if len(chosen)>=12:break
            pool+=chosen
        design=dict(pool=pool,dev_cases=p['cases'],selection='Top <=12 existing configurations per exact dimension1..5, <=2 per support; shared full22 development evaluation. No test labels.',test_scenes=p['test_scenes'],test_states=[22,23,24,25],test_seed=5)
        R.dump(path,design)
    rr=stage([case(c,sp,0)for sp in design['pool']for c in p['cases']],'selection')
    frozen=R.OUT/'FINAL_FROZEN.json'
    if not frozen.exists():
        ranked=[]
        for sp in design['pool']:
            hits=[r for r in rr if r['spec']==sp]
            ranked.append(dict(spec=sp,names=support(sp),dimension=len(support(sp)),success=sum(r['success']for r in hits),collision=sum(r['collision']for r in hits),success_mean_time=float(np.mean([r['completion_time']for r in hits if r['success']]))if any(r['success']for r in hits)else None,truncated_mean=float(np.mean([r['truncated_time']for r in hits]))))
        ranked.sort(key=lambda x:(-x['success'],x['dimension'],x['collision'],x['truncated_mean'],R.digest(x['spec'])))
        bydim=[next(x for x in ranked if x['dimension']==d)for d in range(1,6)if any(x['dimension']==d for x in ranked)]
        R.dump(frozen,dict(best=ranked[0],best_by_dimension=bydim,ranking=ranked,criterion='Maximum full-dev success, then minimum dimension, then collisions and capped time; frozen before states22..25.'))
    print(json.load(frozen.open())['best'],flush=True)
if __name__=='__main__':main()
