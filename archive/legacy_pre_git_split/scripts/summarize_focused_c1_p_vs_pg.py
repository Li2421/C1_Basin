"""Strictly paired P versus P+g report for the frozen direct-G_phi study."""
import csv, json, hashlib
from pathlib import Path
import numpy as np
from focused_c1_p_vs_pg import ROOT, OUT, hashes, ARMS, SEEDS

def describe(rows):
    return dict(n=len(rows),success=sum(r['success'] for r in rows),deadlock=sum(r['deadlock'] for r in rows),
      timeout=sum(r['timeout'] for r in rows),collision=sum(r['collision'] for r in rows),controller_errors=sum(r['controller_error'] is not None for r in rows),
      any_deadlock_events=sum(r['any_deadlock'] for r in rows),success_rate=float(np.mean([r['success'] for r in rows])),
      mean_stagnation=float(np.mean([r['stagnation'] for r in rows])),median_stagnation=float(np.median([r['stagnation'] for r in rows])),
      total_stagnation=float(sum(r['stagnation'] for r in rows)))

def main():
    p=json.loads((OUT/'protocol.json').read_text());assert hashes()==p['source_hashes']
    rows=[];run_stats=[]
    for arm in ARMS:
      for seed in SEEDS:
        complete=json.loads((OUT/'runs'/f'{arm}_seed{seed}'/'complete.json').read_text())
        rr=[json.loads(f.read_text()) for f in (OUT/'evaluation'/f'{arm}_seed{seed}').glob('5*.json')]
        assert len(rr)==384 and complete['source_hashes_verified']
        rows.extend(rr)
        for condition in ['matched','independent']:run_stats.append(dict(arm=arm,training_seed=seed,condition=condition,**describe([r for r in rr if r['condition']==condition])))
    lookup={(r['arm'],r['training_seed'],r['rid'],r['execution_seed']):r for r in rows};rng=np.random.default_rng(20261117);contrasts={}
    for condition in ['matched','independent']:
      exseeds=[p['matched_execution_seed']] if condition=='matched' else p['independent_execution_seeds']
      ds=np.zeros((len(SEEDS),128));dt=np.zeros((len(SEEDS),128))
      for a,seed in enumerate(SEEDS):
       for i,rid in enumerate(p['evaluation_ids']):
        left=[lookup[('P',seed,rid,k)] for k in exseeds];right=[lookup[('Pg',seed,rid,k)] for k in exseeds]
        ds[a,i]=np.mean([float(q['success'])-float(r['success']) for r,q in zip(left,right)])
        dt[a,i]=np.mean([q['stagnation']-r['stagnation'] for r,q in zip(left,right)])
      si=rng.integers(0,len(SEEDS),(20000,len(SEEDS)));ii=rng.integers(0,128,(20000,128))
      bs=ds[si[:,:,None],ii[:,None,:]].mean((1,2));bt=dt[si[:,:,None],ii[:,None,:]].mean((1,2))
      contrasts[condition]=dict(success_difference=float(ds.mean()),success_difference_by_seed=ds.mean(1).tolist(),success_difference95=np.quantile(bs,[.025,.975]).tolist(),
        stagnation_difference=float(dt.mean()),stagnation_difference_by_seed=dt.mean(1).tolist(),stagnation_difference95=np.quantile(bt,[.025,.975]).tolist(),
        P_wins=sum((ds<0).sum() for _ in [0]),Pg_wins=sum((ds>0).sum() for _ in [0]),ties=sum((ds==0).sum() for _ in [0]))
    pooled={arm:{condition:describe([r for r in rows if r['arm']==arm and r['condition']==condition]) for condition in ['matched','independent']} for arm in ARMS}
    safety=dict(counts={k:sum(r['safety'][k] for r in rows) for k in ['steps','agent_collision_steps','wall_collision_steps','outside_endpoints','cbf_violations','speed_violations']},
       minima={k:min(r['safety'][k] for r in rows) for k in ['min_center_separation','min_agent_surface_clearance','min_wall_surface_clearance','min_pair_cbf_residual','min_wall_cbf_residual']})
    curves=[]
    for arm in ARMS:
      for seed in [3,4,5]:
       h=json.loads((OUT/'runs'/f'{arm}_seed{seed}'/'history.json').read_text())
       for r in h:curves.append(dict(arm=arm,training_seed=seed,step=r['step']+1,accepted=r['accepted'],P=r['terms'][0],g=r['terms'][1],S=r['terms'][2],full=r['terms'][3],gradient_norm=r['gradient_norm'],pre=r['pre'],post=r['post']))
    with (OUT/'training_curves_new_seeds.csv').open('w',newline='') as f:
      writer=csv.DictWriter(f,fieldnames=list(curves[0]));writer.writeheader();writer.writerows(curves)
    result=dict(protocol=p,pooled=pooled,per_training_seed=run_stats,contrasts=contrasts,safety=safety,total_executions=len(rows),
      verification=dict(frozen_source_hashes=True,direct_G_phi_only=all(r['direct_G_phi'] and not r['candidate_search'] for r in rows),
      existing_seed_checkpoints_reused=True,final_checkpoints_only=True),
      uncertainty='Paired two-level bootstrap resamples 6 training seeds and 128 held-out states; independent execution replicas are averaged within state.')
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(pooled=pooled,contrasts=contrasts,safety=safety,verification=result['verification']),indent=2))

if __name__=='__main__':main()
