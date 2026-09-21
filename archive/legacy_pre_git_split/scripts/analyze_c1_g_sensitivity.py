"""Paired branch-level evidence for prediction sensitivity, with no new scorer."""
import json,csv
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/c1_jax_training_audit';BASE=ROOT/'results/c1_pretraining_audits'
a=json.loads((BASE/'ablation.json').read_text());p=json.loads((BASE/'mismatch/protocol.json').read_text());all_results={}
for before,after in [('P','P+g'),('P+S','P+g+S')]:
    rows=[]
    for state in a['states']:
        left,right=state['selected'][before],state['selected'][after]
        if left['cid']==right['cid']:continue
        entry=dict(rid=state['rid'])
        for label,branch in [('before',left),('after',right)]:
            runs=[json.loads((BASE/'mismatch/rows'/f"{state['rid']}_{seed}_{branch['cid']}.json").read_text()) for seed in p['seeds']]
            entry[label]=dict(cid=branch['cid'],nominal=branch['costs'],nominal_stagnation=branch['outcome']['stagnation_seconds'],
                empirical_success=float(np.mean([r['outcome']['success'] for r in runs])),
                empirical_stagnation=float(np.mean([r['outcome']['stagnation_seconds'] for r in runs])),
                stagnation_std=float(np.std([r['outcome']['stagnation_seconds'] for r in runs])),
                seeds=[dict(seed=r['seed'],**r['outcome']) for r in runs])
        rows.append(entry)
    ds=np.array([r['after']['empirical_success']-r['before']['empirical_success'] for r in rows]);dt=np.array([r['after']['empirical_stagnation']-r['before']['empirical_stagnation'] for r in rows])
    rng=np.random.default_rng(20261005);idx=rng.integers(0,len(rows),(10000,len(rows)))
    all_results[before+' -> '+after]=dict(changed_states=len(rows),mean_success_change=float(ds.mean()),success_change95=np.quantile(ds[idx].mean(1),[.025,.975]).tolist(),
        mean_stagnation_change=float(dt.mean()),stagnation_change95=np.quantile(dt[idx].mean(1),[.025,.975]).tolist(),
        states_success_worse=int((ds<0).sum()),states_success_better=int((ds>0).sum()),
        mean_before_stagnation_std=float(np.mean([r['before']['stagnation_std'] for r in rows])),
        mean_after_stagnation_std=float(np.mean([r['after']['stagnation_std'] for r in rows])),rows=rows)
    with (OUT/('sensitivity_'+before.replace('+','_')+'.csv')).open('w',newline='') as f:
        fields=['rid','branch','cid','P','g','Stwo','score','nominal_stagnation','empirical_success','empirical_stagnation','stagnation_std'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:
            for side in ['before','after']:
                b=r[side];w.writerow(dict(rid=r['rid'],branch=side,cid=b['cid'],**{k:b['nominal'][k] for k in ['P','g','Stwo','score']},**{k:b[k] for k in fields[7:]}))
(OUT/'g_sensitivity.json').write_text(json.dumps(all_results,indent=2)+'\n')
print(json.dumps({k:{a:b for a,b in v.items() if a!='rows'} for k,v in all_results.items()},indent=2))
