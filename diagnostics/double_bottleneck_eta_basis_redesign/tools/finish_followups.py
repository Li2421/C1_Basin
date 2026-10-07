"""Validate completed seed matrices and schedule only missing fixed follow-ups."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import qmc

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'diagnostics/double_bottleneck_eta_basis_redesign'
SOURCE=ROOT/'diagnostics/double_bottleneck_eta3_full_sobol'

def read(pattern):
    return [json.loads(line) for p in sorted((OUT/'raw').glob(pattern)) for line in p.read_text().splitlines() if line.strip()]

def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')

def main():
    full=json.loads((OUT/'full_global_summary.json').read_text())['P1-OrthoFlow3']['episodes']
    rows=read('P1_full_seed16_cached_pilot.jsonl')+read('P1_full_seed_pending_shard*.jsonl')
    expected={(e['episode_id'],c['eta_index'],s) for e in full for c in e['candidates'] for s in range(2001,2017)}
    keys=[(r['episode_id'],r['eta_index'],r['seed']) for r in rows]
    assert len(keys)==len(set(keys)) and set(keys)==expected, 'seed matrix incomplete/duplicate'
    groups=defaultdict(list)
    for r in rows: groups[r['episode_id'],r['eta_index']].append(r)
    centers=[]
    for e in full:
        cs=[dict(c,Q_seed=sum(r['success'] for r in groups[e['episode_id'],c['eta_index']])/16) for c in e['candidates']]
        best=min(cs,key=lambda c:(-c['Q_seed'],-c['successful_neighbors_among_8'],c['normalized_distance_to_eta_zero'],c['eta_index']))
        centers.append(dict(e,eta_robust=best,Q_max=best['Q_seed'],candidate_seed_results=cs))
    summary={'episodes':centers,'median_Q_max':float(np.median([e['Q_max'] for e in centers])),
        'counts':{str(t):sum(e['Q_max']>=t for e in centers) for t in (.25,.5,.75)},'rollouts':len(rows)}
    save('full_seed_robustness.json',summary)
    offsets=(2*qmc.Sobol(3,scramble=True,seed=2026092703).random_base2(4)-1)*.05
    low=np.array([.5,-.5,0]); width=np.array([.75,1,.75])
    scale=json.loads((OUT/'P1_SCALE.json').read_text())['scale']
    cached_local={(r['episode_id'],r['local_index'],r['seed']) for r in read('pilot_local_shard*.jsonl')}
    local=[]
    for e in centers:
        if e['Q_max']<.5: continue
        c=e['eta_robust']; unit=(np.array(c['theta'])-low)/width
        for k,off in enumerate(offsets):
            for seed in range(3001,3009):
                if (e['episode_id'],k,seed) in cached_local: continue
                local.append({**{key:e[key] for key in ('episode_id','population','set','family_id','regime','rollout_id','baseline_outcome')},
                    'stage':'full_local','representation':'P1-OrthoFlow3','eta_index':c['eta_index'],'center_eta_index':c['eta_index'],
                    'local_index':k,'parameter_id':f'L{k:02d}','theta':(low+np.clip(unit+off,0,1)*width).tolist(),
                    'ortho_scale':scale,'seed':seed,'job_id':f"full_local|{e['episode_id']}|{k}|{seed}"})
    controls=[e for e in json.loads((SOURCE/'episode_catalog.json').read_text())['episodes'] if e['population']=='baseline_success_control']
    cached_controls={(r['eta_index'],r['episode_id'],r['seed']) for r in read('P1_pilot_control_shard*.jsonl')}
    unique={e['eta_robust']['eta_index']:e['eta_robust'] for e in centers}
    control_jobs=[]
    for idx,c in sorted(unique.items()):
        for e in controls:
            for seed in range(4001,4005):
                if (idx,e['episode_id'],seed) in cached_controls: continue
                control_jobs.append({**e,'stage':'full_controls','representation':'P1-OrthoFlow3','eta_index':idx,
                    'parameter_id':c['parameter_id'],'theta':c['theta'],'ortho_scale':scale,'seed':seed,
                    'job_id':f"full_control|{idx}|{e['episode_id']}|{seed}"})
    save('jobs/full_local_pending.json',{'jobs':local})
    save('jobs/full_control_pending.json',{'jobs':control_jobs})
    save('followup_counts.json',{'local_pending':len(local),'local_expected':sum(e['Q_max']>=.5 for e in centers)*128,
        'controls_pending':len(control_jobs),'controls_expected':len(unique)*96,'unique_centers':sorted(unique)})
    print(json.dumps({k:v for k,v in summary.items() if k!='episodes'}))
    print('local pending',len(local),'controls pending',len(control_jobs))

if __name__=='__main__': main()
