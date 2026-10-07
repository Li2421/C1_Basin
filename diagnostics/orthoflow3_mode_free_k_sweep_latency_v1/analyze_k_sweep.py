#!/usr/bin/env python3
"""Frozen K-prefix coverage, critic selection, and K4 failure geometry."""
from __future__ import annotations

import csv
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path('/home/zhihan/research/Basin_C1')
OLD = ROOT / 'diagnostics/orthoflow3_mode_free_generator_critic_hard_cohort_v1'
OUT = Path(__file__).resolve().parent
CENTER = np.array([0.625, 0.0, 0.375])
RADIUS = np.array([0.625, 0.5, 0.375])


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def robust_references(con, state_uid, controller_uid):
    seeds = [json.dumps({'future_index':i}, separators=(',',':'),sort_keys=True) for i in range(16)]
    query = '''SELECT e.eta_uid,e.eta1,e.eta2,e.eta3,COUNT(DISTINCT r.seed_key),SUM(r.success)
               FROM rollout r JOIN eta e ON e.eta_uid=r.eta_uid
               WHERE r.state_uid=? AND r.controller_uid=? AND r.seed_key IN (''' + ','.join('?' for _ in seeds) + ''')
                 AND r.conflict_quarantined=0 AND r.numerical_failure=0
               GROUP BY e.eta_uid HAVING COUNT(DISTINCT r.seed_key)=16 AND SUM(r.success)>=15'''
    return [{'eta_uid':r[0], 'eta':[r[1],r[2],r[3]], 'successes':r[5]} for r in con.execute(query,(state_uid,controller_uid,*seeds))]


def paired(a,b):
    a,b=np.asarray(a,bool),np.asarray(b,bool)
    return {'rescue':int(np.sum(a & ~b)), 'break':int(np.sum(~a & b)),
            'net':int(np.sum(a)-np.sum(b))}


def main():
    post = json.loads((OUT / 'cache_postflight.json').read_text())['summary']
    # One frozen continuation has a reproducible safety-solver numerical failure.
    # Retain it as censored evidence, never silently certify it as a success.
    if post['ambiguous'] or post['incompatible'] or post['genuinely_missing']>1:
        raise RuntimeError(f'Unexpected cache status: {post}')
    old = list(csv.DictReader((OLD / 'per_state_results.csv').open()))
    old_frozen = json.loads((OLD / 'frozen_proposals.json').read_text())['states']
    frozen = json.loads((OUT / 'frozen_proposals.json').read_text())['states']
    if len(old)!=200 or len(frozen)!=200: raise RuntimeError('Frozen cohort changed')
    new_rows=[]
    for path in sorted((OUT / 'raw').glob('shard*.jsonl')):
        new_rows += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    invalid=[r for r in new_rows if not r['scientific_outcome_valid']]
    if len(invalid)!=1 or (invalid[0]['episode_index'],invalid[0]['kind'],invalid[0]['future_index'])!=(187,'sample_12',15):
        raise RuntimeError(f'Unexpected invalid rollouts: {invalid[:3]}')
    keyed = {(r['episode_index'],r['kind'],r['future_index']):r for r in new_rows}
    if len(new_rows)!=38400 or len(keyed)!=38400: raise RuntimeError(f'Incomplete local matrix {len(keyed)}')
    q=np.zeros((200,16),np.float64)
    q_upper=np.zeros((200,16),np.float64)
    for i in range(200):
        for j in range(4):
            q[i,j]=float(old[i][f'Q16_sample_{j}'])
            q_upper[i,j]=q[i,j]
        for j in range(4,16):
            rec=[keyed[(i,f'sample_{j}',seed)] for seed in range(16)]
            q[i,j]=sum(r['success'] for r in rec if r['scientific_outcome_valid'])/16
            q_upper[i,j]=q[i,j]+sum(not r['scientific_outcome_valid'] for r in rec)/16
    if np.any((q>=15/16)!=(q_upper>=15/16)):
        raise RuntimeError('Numerical censoring changes B15 classification')
    score=np.asarray([s['all_critic_scores'] for s in frozen],np.float64)
    if score.shape!=(200,16): raise RuntimeError('Critic score array changed')
    mean=np.asarray([float(r['Q16_generator_mean']) for r in old])
    fixed=np.asarray([float(r['Q16_fixed_common']) for r in old])
    selector=np.asarray([float(r['Q16_old_selector_anchor']) for r in old])
    Kvals=(1,2,4,8,16)
    rows=[]; state_rows=[]
    for K in Kvals:
        oracle=np.max(q[:,:K],axis=1)
        chosen=np.argmax(score[:,:K],axis=1)
        critic=q[np.arange(200),chosen]
        sample_robust=q[:,:K]>=15/16
        covered=np.any(sample_robust,axis=1)
        rb=critic>=15/16
        row={'K':K,'random_first_sample_B15_states':int(np.sum(q[:,0]>=15/16)),
             'random_draw_B15_rate':float(sample_robust.mean()),
             'random_draw_mean_Q16':float(q[:,:K].mean()),
             'oracle_B15_states':int(np.sum(covered)), 'oracle_mean_Q16':float(oracle.mean()),
             'critic_B15_states':int(np.sum(rb)), 'critic_mean_Q16':float(critic.mean()),
             'coverage':float(covered.mean()), 'oracle_critic_B15_gap_states':int(np.sum(covered)-np.sum(rb)),
             'oracle_critic_mean_Q16_gap':float(np.mean(oracle-critic)),
             'critic_vs_mean_rescue':paired(rb,mean>=15/16)['rescue'],
             'critic_vs_mean_break':paired(rb,mean>=15/16)['break'],
             'critic_vs_fixed_rescue':paired(rb,fixed>=15/16)['rescue'],
             'critic_vs_fixed_break':paired(rb,fixed>=15/16)['break'],
             'oracle_vs_mean_rescue':paired(covered,mean>=15/16)['rescue'],
             'oracle_vs_mean_break':paired(covered,mean>=15/16)['break'],
             'oracle_vs_fixed_rescue':paired(covered,fixed>=15/16)['rescue'],
             'oracle_vs_fixed_break':paired(covered,fixed>=15/16)['break'],
             'critic_gap_to_old_selector_states':int(np.sum(selector>=15/16)-np.sum(rb)),
             'oracle_gap_to_old_selector_states':int(np.sum(selector>=15/16)-np.sum(covered))}
        rows.append(row)
        for i in range(200):
            state_rows.append({'episode_index':i,'K':K,'oracle_Q16':float(oracle[i]),
                               'critic_choice':int(chosen[i]),'critic_Q16':float(critic[i]),
                               'covered_B15':int(covered[i]),'critic_B15':int(rb[i]),
                               'generator_mean_Q16':float(mean[i]),'fixed_common_Q16':float(fixed[i])})
    write_csv('k_sweep.csv',rows)
    (OUT/'numerical_censoring.json').write_text(json.dumps({
        'affected_episode':187,'affected_kind':'sample_12','affected_future_index':15,
        'Q16_interval':[float(q[187,12]),float(q_upper[187,12])],
        'B15_classification_certain':True,
        'max_mean_Q16_uncertainty':1/(200*16),
        'cache_postflight':post},indent=2,sort_keys=True)+'\n')
    write_csv('per_state_k.csv',state_rows)
    mapping={}
    for r in csv.DictReader((OLD / 'candidate_cache_keys.csv').open()):
        if r['kind']=='old_selector_anchor': mapping[int(r['episode_index'])]=(r['state_uid'],r['controller_uid'])
    con=sqlite3.connect(ROOT/'shared_rollout_db/rollout.sqlite')
    failures=[]
    for i in range(200):
        if np.any(q[i,:4]>=15/16): continue
        state=frozen[i]
        samples=np.asarray([old_frozen[i]['eta'][f'sample_{j}']
                            for j in range(4)]+[state['eta'][f'sample_{j}'] for j in range(4,16)])
        old_state=old_frozen[i]
        m=np.asarray(old_state['eta']['generator_mean'],np.float64)
        f=np.asarray(old_state['eta']['fixed_common'],np.float64)
        a=np.asarray(old_state['eta']['old_selector_anchor'],np.float64)
        sigma=np.asarray(state['sigma'],np.float64)
        mu=np.arctanh(np.clip((m-CENTER)/RADIUS,-0.999999,0.999999))
        uid,ctl=mapping[i]
        refs=robust_references(con,uid,ctl)
        def normdist(x,y): return float(np.linalg.norm((np.asarray(x)-np.asarray(y))/RADIUS))
        for ref in refs:
            eta=np.asarray(ref['eta'])
            z=np.arctanh(np.clip((eta-CENTER)/RADIUS,-0.999999,0.999999))
            ref['norm_dist_mean']=normdist(eta,m)
            ref['latent_sigma_dist']=float(np.linalg.norm((z-mu)/sigma))
        refs.sort(key=lambda r:r['latent_sigma_dist'])
        extended_hit=bool(np.any(q[i,4:16]>=15/16))
        mean_hit=bool(mean[i]>=15/16)
        if mean_hit or extended_hit: cls='LOW_PROBABILITY_BASIN'
        elif refs and min(r['latent_sigma_dist'] for r in refs)>3: cls='MISSING_BASIN_COVERAGE'
        else: cls='UNDERRESOLVED'
        failures.append({'episode_index':i,'category':cls,'mean_Q16':float(mean[i]),
                         'K4_best_Q16':float(q[i,:4].max()),'K8_best_Q16':float(q[i,:8].max()),
                         'K16_best_Q16':float(q[i,:16].max()),'first_extra_B15_index':next((j for j in range(4,16) if q[i,j]>=15/16),''),
                         'predicted_sigma_json':json.dumps(sigma.tolist()),
                         'sample_dispersion_norm_mean':float(np.mean([normdist(s,m) for s in samples[:4]])),
                         'sample_dispersion_norm_max':float(max(normdist(s,m) for s in samples[:4])),
                         'fixed_common_Q16':float(fixed[i]),'fixed_common_norm_dist_mean':normdist(f,m),
                         'old_selector_Q16':float(selector[i]),'old_selector_norm_dist_mean':normdist(a,m),
                         'old_selector_nearest_K4_sample_norm_dist':float(min(normdist(a,s) for s in samples[:4])),
                         'compatible_B15_reference_count':len(refs),
                         'nearest_B15_ref_norm_dist_mean':refs[0]['norm_dist_mean'] if refs else '',
                         'nearest_B15_ref_latent_sigma_dist':refs[0]['latent_sigma_dist'] if refs else '',
                         'compatible_B15_refs_json':json.dumps(refs[:8],sort_keys=True)})
    if len(failures)!=27: raise RuntimeError(f'Expected 27 frozen K4 failure states, got {len(failures)}')
    write_csv('k4_failure_state_audit.csv',failures)
    summary={'K4_failure_states':len(failures),'category_counts':dict(Counter(r['category'] for r in failures)),
             'new_K16_hits_among_K4_failures':sum(bool(r['first_extra_B15_index']!='') for r in failures),
             'mean_B15_among_K4_failures':sum(r['mean_Q16']>=15/16 for r in failures),
             'median_nearest_reference_latent_sigma_dist':float(np.median([r['nearest_B15_ref_latent_sigma_dist']
                                                              for r in failures if r['nearest_B15_ref_latent_sigma_dist']!=''])),
             'K32_new_rollout_estimate':200*16*16}
    (OUT/'failure_audit_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'K':[{k:r[k] for k in ('K','oracle_B15_states','critic_B15_states','coverage')} for r in rows],
                      'failure_categories':summary['category_counts']}))


if __name__=='__main__': main()
