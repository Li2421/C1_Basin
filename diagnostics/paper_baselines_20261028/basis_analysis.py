"""All frozen families, including failures; no test-selected winner."""
import json,csv
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from basis_select import R,HERE

def metrics(rows):
    n=len(rows);hits=[r for r in rows if r['success']];mean=lambda v:float(np.mean(v))if v else None
    ci=binomtest(len(hits),n).proportion_ci(method='wilson')
    return dict(n=n,success=len(hits),success_fraction=len(hits)/n,wilson95_descriptive=[ci.low,ci.high],success_mean_time=mean([r['completion_time']for r in hits]),truncated_mean_time=mean([r['truncated_time']for r in rows]),collisions=sum(r['collision']for r in rows),timeouts=sum(r['timeout']for r in rows),numerical_unknown=sum(r['numerical_failure']for r in rows),mean_wall_seconds=mean([r['wall_seconds']for r in rows]))

def main():
    d=json.load(open(R.OUT/'HOLDOUT_FROZEN.json'));rows={r['rollout_uid']:r for r in map(json.loads,open(R.OUT/'candidates.jsonl'))}
    from basis_holdout import case
    def get(f,k):
        return [[rows[case(c,sp,d['seed'])['rollout_uid']]for sp in f['pool'][:k]]for c in d['cases']]
    def choose(pool):return next((r for r in pool if r['success']),pool[0])
    baseline=get(d['families'][0],32);fixed_base=[p[0]for p in baseline];oracle_base=[choose(p)for p in baseline]
    labels=json.load(open(R.OUT/'REFERENCE_LABELS.json'))['labels']
    references={k:metrics([rows[uid]for uid in ids])for k,ids in labels.items()}
    strong={ (rows[uid]['scene'],rows[uid]['controller'],rows[uid]['state_index']):rows[uid]for uid in labels['B3_historical_frozen']}
    rng=np.random.default_rng(2026102809);summaries=[]
    for f in d['families']:
        full=get(f,32);fixed=[p[0]for p in full]
        entry=dict(name=f['name'],columns=f['names'],dimension=len(f['names']),fixed=metrics(fixed),oracle=[],by_scene={},by_nominal={})
        for k in d['budgets']:
            pools=[p[:k]for p in full];chosen=[choose(p)for p in pools];m=metrics(chosen)
            m.update(K=k,definition='First successful candidate in frozen ordering; all candidates actually executed; no online selector.',candidate_n=sum(map(len,pools)),candidate_collisions=sum(r['collision']for p in pools for r in p),candidate_numerical_unknown=sum(r['numerical_failure']for p in pools for r in p),candidate_wall_seconds=sum(r['wall_seconds']for p in pools for r in p))
            entry['oracle'].append(m)
        union=[choose(a[:16]+b[:16])for a,b in zip(baseline,full)]
        entry['union_with_B3_at32']=metrics(union)
        entry['rescued_vs_B3_at32']=sum(a['success']and not b['success']for a,b in zip(union,oracle_base))
        entry['lost_vs_B3_at32']=sum(not a['success']and b['success']for a,b in zip(union,oracle_base))
        for label,key in [('by_scene','scene'),('by_nominal','controller')]:
            for value in sorted({r[key]for r in fixed}):entry[label][value]=metrics([r for r in fixed if r[key]==value])
        pairs=[(a,b)for a,b in zip(fixed,fixed_base)if a['success']and b['success']]
        entry['paired_success_vs_fixed_B3']=dict(n=len(pairs),method_mean=float(np.mean([a['completion_time']for a,b in pairs]))if pairs else None,B3_mean=float(np.mean([b['completion_time']for a,b in pairs]))if pairs else None)
        aligned=[strong[(r['scene'],r['controller'],r['state_index'])]for r in fixed]
        pairs=[(a,b)for a,b in zip(fixed,aligned)if a['success']and b['success']]
        physical=sorted({(r['scene'],r['state_index'])for r in fixed});blocks=[]
        for key in physical:
            blocks.append(np.mean([a['success']-b['success']for a,b in zip(fixed,aligned)if (a['scene'],a['state_index'])==key]))
        bootstrap=np.mean(np.asarray(blocks)[rng.integers(0,len(blocks),size=(20000,len(blocks)))],axis=1)
        entry['vs_historical_strong_B3']=dict(paired_success_n=len(pairs),paired_method_time=float(np.mean([a['completion_time']for a,b in pairs]))if pairs else None,paired_B3_time=float(np.mean([b['completion_time']for a,b in pairs]))if pairs else None,rescued=sum(a['success']and not b['success']for a,b in zip(fixed,aligned)),lost=sum(not a['success']and b['success']for a,b in zip(fixed,aligned)),physical_block_rate_difference_percentile95=np.quantile(bootstrap,[.025,.975]).tolist(),warning='Only8 physical blocks; exploratory bootstrap, no superiority claim.')
        summaries.append(entry)
    R.dump(R.OUT/'summary.json',dict(families=summaries,references=references,independence='16 scene/controller/state cases but only8 physical initial states; Wilson intervals descriptive, not independent-trial significance. One future seed per case, no B15 robustness certification.',selection='Chosen family remains development winner G,C2. No test-driven reselection.',state_amendment=d.get('pre_run_amendment')))
    with (R.OUT/'budget_curves.csv').open('w')as f:
        writer=csv.writer(f);writer.writerow(['family','columns','K','success','n','success_mean_time','truncated_mean_time','candidate_wall_seconds','candidate_numerical_unknown'])
        for e in summaries:
            for m in e['oracle']:writer.writerow([e['name'],','.join(e['columns']),m['K'],m['success'],m['n'],m['success_mean_time'],m['truncated_mean_time'],m['candidate_wall_seconds'],m['candidate_numerical_unknown']])
    for e in summaries:print(e['name'],e['columns'],'fixed',e['fixed']['success'],'time',e['fixed']['success_mean_time'],'oracle32',e['oracle'][-1]['success'],'union32',e['union_with_B3_at32']['success'],flush=True)
if __name__=='__main__':main()
