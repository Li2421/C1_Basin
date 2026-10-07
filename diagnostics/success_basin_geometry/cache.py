"""Resolve exact compatible existing samples before allocating new main-map work."""
import json
from collections import defaultdict
import numpy as np
from diagnostics.success_basin_geometry.setup import HERE,ROOT,write,sha

def main():
    plan=json.loads((HERE/'protocol.json').read_text());jobs=json.loads((HERE/'main_jobs.json').read_text())
    by=defaultdict(list)
    for r in jobs:by[(r['state_id'],tuple(r['phi']))].append(r)
    candidates=defaultdict(list)
    old=ROOT/'diagnostics/true_q_geometry/raw'
    for stage in ['q_map','direction_validation']:
        base=old/stage
        for r in json.loads((base/'manifest.json').read_text())['records']:
            key=(r['state_id'],tuple(r['phi']))
            if key not in by or r['flow_seed']==19073:continue
            path=base/r['relative_path'];assert sha(path)==r['sha256']
            candidates[key].append({'state_id':r['state_id'],'phi':r['phi'],'seed':r['flow_seed'],
                'cell_id':by[key][0]['cell_id'],'outcome':r['outcome'],'steps':r['continuation_steps'],
                'file':str(path),'sha256':r['sha256'],'origin':'compatible_fresh_cached'})
    reused=[];new=[]
    for key,group in by.items():
        cached=candidates[key][:16];assert len({r['seed'] for r in cached})==len(cached)
        reused.extend(cached);new.extend(group[:16-len(cached)])
    write('cache_index.json',{'records':reused,'count':len(reused),'selection_conditioned_seed_excluded':19073})
    for sid in plan['primary_states']:write(f'main_{sid}.json',[r for r in new if r['state_id']==sid])
    # Numerical/backend benchmark uses the same random keys, so compare full executed traces.
    cpu=json.loads((HERE/'raw/benchmark_cpu/manifest.json').read_text());gpu=json.loads((HERE/'raw/benchmark_gpu/manifest.json').read_text())
    maxpos=0.;same=0
    for a,b in zip(cpu['records'],gpu['records']):
        same+=int(a['outcome']==b['outcome'] and a['steps']==b['steps'])
        with np.load(HERE/a['file']) as da,np.load(HERE/b['file']) as db:
            if len(da['event'])==len(db['event']):maxpos=max(maxpos,float(np.max(abs(da['positions_after']-db['positions_after']))))
    write('execution_plan.json',{'cached_samples':len(reused),'new_main_rollouts':len(new),
        'per_state_new':{sid:sum(r['state_id']==sid for r in new) for sid in plan['primary_states']},
        'CPU_seconds':cpu['elapsed_s'],'GPU_seconds':gpu['elapsed_s'],'same_outcome_and_length':same,
        'max_position_backend_difference':maxpos,'choice':'CPU, 3 parallel Slurm jobs with 2 cores each, batch16',
        'main_max_physical_steps':sum(850-next(s['start_step'] for s in plan['state_catalog'] if s['state_id']==r['state_id']) for r in new)})
    print(json.loads((HERE/'execution_plan.json').read_text()))

if __name__=='__main__':main()
