#!/usr/bin/env python3
"""Matched-seed Q64 analysis of frozen DB mean and K16 critic."""
import csv,json,statistics
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path('/home/zhihan/research/Basin_C1')
HERE=Path(__file__).resolve().parent
PRIOR=ROOT/'diagnostics/orthoflow3_db_mode_free_mustdo_v1'
BASE=ROOT/'diagnostics/orthoflow3_db_generator_necessity_v1'
manifest=json.loads((HERE/'frozen_q64_manifest.json').read_text())
choice={s['state_id']:s['critic_choice_K16'] for s in json.loads((PRIOR/'db_frozen_proposals.json').read_text())['states']}
rows=defaultdict(dict)
for p in list((PRIOR/'raw').glob('mean_shard*.jsonl'))+list((PRIOR/'raw').glob('proposals_shard*.jsonl'))+list((HERE/'raw').glob('q64_shard*.jsonl')):
    for line in p.open():
        r=json.loads(line);sid=r['state_id'];fi=int(r['future_index'])
        if r.get('controller')=='generator_mean':method='generator_mean'
        elif r.get('controller')=='generator_proposal':
            if int(r['sample_index'])!=choice[sid]:continue
            method='critic_K16'
        else:method=r['method']
        key=(sid,method,fi)
        if key in rows[sid]:
            assert rows[sid][key]['success']==r['success'],key
        rows[sid][key]=r
baseline={r['state_id']:r for r in csv.DictReader((BASE/'discrete_baselines.csv').open())}
assert len(baseline)==len(rows)==48
out=[];methods=['safety','fixed_common','generator_mean','critic_K16']
for sid in sorted(rows):
    b=baseline[sid]
    rec={'state_id':sid,'source_group':b['source_group']}
    for method in ('generator_mean','critic_K16'):
        trials=[rows[sid][(sid,method,i)] for i in range(64)]
        assert sorted(int(r['future_index']) for r in trials)==list(range(64))
        r0=next(r for r in manifest['states'] if r['state_id']==sid and r['method']==method)
        assert all(r['eta']==r0['eta'] for r in trials)
        rec[method+'_success']=sum(int(r['success']) for r in trials)
        rec[method+'_fresh48_success']=sum(int(r['success']) for r in trials[16:])
        rec[method+'_Q64']=rec[method+'_success']/64
        rec[method+'_B63']=int(rec[method+'_success']>=63)
        rec[method+'_J_def']=statistics.mean(float(r['J_def']) for r in trials)
        rec[method+'_episode_length']=statistics.mean(float(r['episode_steps']) for r in trials)
        rec[method+'_collision']=sum(int(r.get('agent_collision',False) or r.get('wall_collision',False)) for r in trials)
        rec[method+'_numerical_failure']=sum(int(not r['scientific_outcome_valid']) for r in trials)
        rec[method+'_eta']=json.dumps(r0['eta'])
    for method,prefix in [('safety','safety'),('fixed_common','fixed')]:
        rec[method+'_success']=int(b[prefix+'_success'])
        rec[method+'_Q64']=float(b[prefix+'_Q64'])
        rec[method+'_B63']=int(b[prefix+'_B63']=='True')
        rec[method+'_J_def']=float(b[prefix+'_J_def'])
        rec[method+'_collision']=int(b[prefix+'_collision'])
    out.append(rec)
with (HERE/'q64_per_state.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
summary={'states':len(out),'fresh_seed_indices':list(range(16,64)),'matched_controller_uid':manifest['controller_uid'],'methods':{},'paired':{},'integrity':{'source_groups_unique':len({r['source_group'] for r in out})==48,
    'mean_and_critic_exact_64_seeds_each':True,'new_rollout_count':48*2*48}}
for method in methods:
    summary['methods'][method]={'B63_states':sum(r[method+'_B63'] for r in out),
        'mean_Q64':statistics.mean(r[method+'_Q64'] for r in out),
        'successes':sum(r[method+'_success'] for r in out),
        'collision':sum(r[method+'_collision'] for r in out),
        'J_def':statistics.mean(r[method+'_J_def'] for r in out)}
    if method in ('generator_mean','critic_K16'):
        summary['methods'][method]['fresh48_mean_success']=sum(r[method+'_fresh48_success'] for r in out)/(48*48)
for a,b in [('generator_mean','safety'),('generator_mean','fixed_common'),('critic_K16','generator_mean'),('critic_K16','fixed_common')]:
    delta=np.asarray([r[a+'_Q64']-r[b+'_Q64'] for r in out])
    rng=np.random.default_rng(20261002)
    bs=delta[rng.integers(0,len(delta),(20000,len(delta)))].mean(axis=1)
    summary['paired'][a+'_vs_'+b]={'B63_rescue':sum(r[a+'_B63'] and not r[b+'_B63'] for r in out),
        'B63_break':sum(r[b+'_B63'] and not r[a+'_B63'] for r in out),
        'mean_Q64_delta':float(np.mean(delta)),'Q64_delta_bootstrap_95CI':[float(x) for x in np.quantile(bs,[.025,.975])]}
(HERE/'q64_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
