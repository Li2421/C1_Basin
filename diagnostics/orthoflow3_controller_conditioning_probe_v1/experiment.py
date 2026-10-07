"""Freeze, merge and analyze a matched-controller information-sufficiency probe."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest, spearmanr
from shared_rollout_db.src.rollout_db import canonical, uid, eta_identity, connect, transaction
from diagnostics.orthoflow3_mode_free_generator_critic_hard_cohort_v1.prepare_cache_plan import CORRECTION_CONFIG, SCENARIO

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
OLD=ROOT/'diagnostics/orthoflow3_mode_free_generator_critic_hard_cohort_v1'
FLOW_ROOT=Path('/home/zhihan/research/02_C1_Toy_GiveWay/baseline_309_314/checkpoints')
EXP='exp_controller_conditioning_probe_v1'

def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def json_default(x):
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)
def dump(name,x):(OUT/name).write_text(json.dumps(x,indent=2,sort_keys=True,default=json_default)+'\n')
def csvout(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def prepare():
    if (OUT/'protocol.json').exists():print('already frozen');return
    states=load(OLD/'frozen_proposals.json')['states']
    states=sorted(states,key=lambda r:hashlib.sha256(('controller-probe-v1|'+r['source_group']).encode()).hexdigest())[:16]
    assert len({r['source_group'] for r in states})==16
    flow_paths={str(k):str(FLOW_ROOT/f'seed{k}/ckpt_0025000.pkl') for k in (0,1)}
    hashes={k:sha(p) for k,p in flow_paths.items()}
    assert hashes['0']==CORRECTION_CONFIG['flow_sha256']
    wide=load(ROOT/'diagnostics/gphi_wide_ic_cadence_v1/frozen_benchmark_manifest.json')
    alt={**CORRECTION_CONFIG,'flow_sha256':hashes['1'],'committed_t0_flow_sha256':hashes['0'],
         'conditioning':'committed_flow0_t0_then_frozen_flow1_future_v1','runtime_sha256':sha(OUT/'runtime.py'),
         'environment_config':wide['environment'],'cbf_config':wide['cbf'],
         'base_controller_uid':uid('ctl',CORRECTION_CONFIG)}
    protocol={'task':'ORTHOFLOW3_CONTROLLER_CONDITIONING_PROBE_V1','experiment_uid':EXP,
      'design':'Same physical state, eta, current committed Flow0 reference, t0 executed action and matched seeds; change future frozen Flow only at t>=1.',
      'primary_question':'Can exactly the same current h and eta have different Q under a missing future-controller condition?',
      'states':16,'state_selection':'first16 SHA256(controller-probe-v1|source_group), frozen prior to reading pair outcomes',
      'candidate_kinds':['safety','fixed_common','generator_mean','sample_0'],'seeds':list(range(16)),
      'base_controller_uid':uid('ctl',CORRECTION_CONFIG),'alternate_controller_uid':uid('ctl',alt),
      'alternate_config':alt,'flow_paths':flow_paths,'flow_sha256':hashes,'runtime_sha256':sha(OUT/'runtime.py'),
      'environment':wide['environment'],'cbf':wide['cbf'],'features_path':str(OLD/'cohort_features.npz'),
      'analysis':'all64 pairs, B15 switches, paired Q difference, exact paired seed tests, Holm correction; cluster uncertainty by source family',
      'limits':'Mechanism intervention on archived Toy states; not fresh cross-scene confirmation. Flow1 with Flow0 committed t0 is an explicitly defined counterfactual controller, not native Flow1 deployment.',
      'planned_new_cap':1024,'generator_changed':False,'safety_changed':False,'models_trained':False}
    pairs=[];requests=[]
    for s in states:
        content=hashlib.sha256(canonical({'initial_positions':s['initial_positions']}).encode()).hexdigest()
        sid=uid('state',{'scenario':SCENARIO,'content':content})
        for kind in protocol['candidate_kinds']:
            eta=[0.,0.,0.] if kind=='safety' else s['eta'][kind]
            eid=eta_identity(eta)[0]
            pairs.append({k:s[k] for k in ('episode_index','source_group','initial_positions','rollout_id','h_sha256')}|
                         {'kind':kind,'state_uid':sid,'eta_uid':eid,'eta':eta})
            for cid in (protocol['base_controller_uid'],protocol['alternate_controller_uid']):
                requests.append({'state_uid':sid,'eta_uid':eid,'controller_uid':cid,'seed_keys':[canonical({'future_index':j}) for j in range(16)]})
    dump('protocol.json',protocol);dump('pair_manifest.json',pairs);dump('planned_rollouts.json',{'requests':requests})
    with connect() as c,transaction(c):
        for r in pairs:
            assert c.execute('SELECT 1 FROM state WHERE state_uid=?',(r['state_uid'],)).fetchone()
            eid,v,hx=eta_identity(r['eta'])
            c.execute('INSERT OR IGNORE INTO eta(eta_uid,eta1,eta2,eta3,canonical_hex) VALUES(?,?,?,?,?)',(eid,*v,hx))
        c.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',
          (EXP,protocol['task'],str(OUT),sha(OUT/'protocol.json'),sha(OUT/'runtime.py'),canonical({'mechanism_probe':True})))
        c.execute('INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
          (protocol['alternate_controller_uid'],SCENARIO,hashes['1'],alt['basis_sha256'],alt['safety'],str(alt['horizon']),str(alt['dt']),alt['success_semantics'],alt['conditioning'],canonical(alt['rng']),canonical(alt),'EXACT_PROFILE'))
    print(json.dumps({'requested':len(requests)*16,'base_slots':1024,'alternate_slots':1024,'outcome_blind_states':16}))

def merge():
    p=load(OUT/'protocol.json');d=ROOT/'shared_rollout_db/journals'/EXP
    stats=defaultdict(int)
    with connect() as c:
        cfg=c.execute('SELECT config_json FROM controller_config WHERE controller_uid=?',(p['alternate_controller_uid'],)).fetchone()[0]
        assert uid('ctl',json.loads(cfg))==p['alternate_controller_uid']
        allowed={(r['state_uid'],r['eta_uid']) for r in load(OUT/'pair_manifest.json')}
        for path in sorted(d.glob('*.jsonl')):
            lines=path.read_text().splitlines();src=uid('src',{'path':str(path.resolve())})
            with transaction(c):
                c.execute('INSERT OR IGNORE INTO source_file(source_uid,experiment_uid,path,sha256,file_type,classification,rows_seen) VALUES(?,?,?,?,?,?,?)',
                          (src,EXP,str(path.resolve()),sha(path),'.jsonl','SEED_EXACT',len(lines)))
                for line_index,line in enumerate(lines,1):
                    env=json.loads(line);assert env['schema']=='explicit_controller_rollout_v1';r=env['record']
                    assert (r['state_uid'],r['eta_uid']) in allowed and r['controller_uid']==p['alternate_controller_uid']
                    assert eta_identity(r['eta'])[0]==r['eta_uid'] and 0<=r['future_index']<16
                    sk=canonical({'future_index':r['future_index']})
                    rid=uid('roll',{'state':r['state_uid'],'eta':r['eta_uid'],'controller':r['controller_uid'],'seed':sk})
                    core=tuple(int(r[k]) for k in ('success','deadlock','timeout','collision','numerical_failure'))
                    existing=c.execute('SELECT * FROM rollout WHERE rollout_uid=?',(rid,)).fetchone()
                    if existing:
                        prior=tuple(existing[k] for k in ('success','deadlock','timeout','collision','numerical_failure'))
                        if prior!=core:
                            c.execute('UPDATE rollout SET conflict_quarantined=1 WHERE rollout_uid=?',(rid,))
                            conflict=uid('conflict',{'rid':rid,'core':core})
                            c.execute('INSERT OR IGNORE INTO conflict VALUES(?,?,?,?,?,?,?,CURRENT_TIMESTAMP)',(conflict,'ROLLOUT',rid,canonical(dict(existing)),canonical(r),str(path),'CONFLICT_QUARANTINED'))
                            stats['conflicts']+=1;continue
                        stats['duplicates']+=1
                    else:
                        c.execute('INSERT INTO rollout(rollout_uid,state_uid,eta_uid,controller_uid,seed_key,continuation_seed_json,success,deadlock,timeout,collision,numerical_failure,episode_length,j_def,min_wall_distance,min_agent_distance,outcome,experiment_uid,original_source_file,timestamp,compatibility_quality,raw_record_hash) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                                  (rid,r['state_uid'],r['eta_uid'],r['controller_uid'],sk,sk,*core,r['episode_length'],r['J_def'],r['minimum_wall_clearance'],r['minimum_agent_clearance'],r['outcome'],EXP,str(path),r['timestamp'],'EXACT_REUSE',hashlib.sha256(canonical(r).encode()).hexdigest()))
                        stats['inserted']+=1
                    c.execute('INSERT OR IGNORE INTO rollout_source VALUES(?,?,?)',(rid,src,line_index))
            stats['journals']+=1
        c.execute('UPDATE experiment SET new_rollout_count=(SELECT COUNT(*) FROM rollout WHERE experiment_uid=?),reused_rollout_count=1024,end_time=CURRENT_TIMESTAMP WHERE experiment_uid=?',(EXP,EXP));c.commit()
    dump('merge_audit.json',dict(stats));print(json.dumps(dict(stats)))

def refresh_unexecuted_runtime():
    """Record an import-path correction before any continuation exists."""
    p=load(OUT/'protocol.json');oldcid=p['alternate_controller_uid']
    d=ROOT/'shared_rollout_db/journals'/EXP
    assert not d.exists() or not list(d.glob('*.jsonl'))
    with connect() as c,transaction(c):
        assert c.execute('SELECT COUNT(*) FROM rollout WHERE experiment_uid=?',(EXP,)).fetchone()[0]==0
        alt={**p['alternate_config'],'runtime_sha256':sha(OUT/'runtime.py')}
        p['runtime_sha256']=alt['runtime_sha256'];p['alternate_config']=alt;p['alternate_controller_uid']=uid('ctl',alt)
        p['preexecution_correction']={'reason':'Authority guard caught import path ROOT before TOY; corrected to frozen Toy runner order TOY before ROOT before any rollout.','old_controller_uid':oldcid,'new_outcomes_before_change':0}
        req=load(OUT/'planned_rollouts.json')
        for r in req['requests']:
            if r['controller_uid']==oldcid:r['controller_uid']=p['alternate_controller_uid']
        c.execute('INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
          (p['alternate_controller_uid'],SCENARIO,alt['flow_sha256'],alt['basis_sha256'],alt['safety'],str(alt['horizon']),str(alt['dt']),alt['success_semantics'],alt['conditioning'],canonical(alt['rng']),canonical(alt),'EXACT_PROFILE'))
        dump('protocol.json',p);dump('planned_rollouts.json',req)
        c.execute('UPDATE experiment SET protocol_hash=?,code_hash=? WHERE experiment_uid=?',(sha(OUT/'protocol.json'),sha(OUT/'runtime.py'),EXP))
    print(json.dumps(p['preexecution_correction']))

def analyze():
    p=load(OUT/'protocol.json');rows=[]
    with connect(True) as c:
        for pair in load(OUT/'pair_manifest.json'):
            r={**pair};by=[]
            for key in ('base','alternate'):
                cid=p[key+'_controller_uid']
                x=[dict(z) for z in c.execute('SELECT * FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=? AND conflict_quarantined=0 AND compatibility_quality="EXACT_REUSE"',(pair['state_uid'],pair['eta_uid'],cid))]
                x={json.loads(z['seed_key'])['future_index']:z for z in x if not z['numerical_failure'] and json.loads(z['seed_key']).get('future_index') in range(16)}
                assert len(x)==16,(key,pair['state_uid'],pair['kind'],len(x))
                n=sum(z['success'] for z in x.values());r[key+'_success']=n;r[key+'_Q16']=n/16;r[key+'_B15']=n>=15;by.append(x)
            loss=sum(by[0][i]['success'] and not by[1][i]['success'] for i in range(16))
            gain=sum(by[1][i]['success'] and not by[0][i]['success'] for i in range(16))
            r['alternate_rescue_seeds']=gain;r['alternate_break_seeds']=loss
            r['paired_p']=binomtest(gain,gain+loss).pvalue if gain+loss else 1.
            r['delta_Q16']=r['alternate_Q16']-r['base_Q16'];rows.append(r)
    order=np.argsort([r['paired_p'] for r in rows]);last=0.
    for rank,i in enumerate(order):
        last=max(last,min(1.,rows[i]['paired_p']*(len(rows)-rank)));rows[i]['holm_p']=last
    csvout('controller_pair_results.csv',rows)
    raw=[]
    for path in (ROOT/'shared_rollout_db/journals'/EXP).glob('*.jsonl'):
        raw.extend(json.loads(x)['record'] for x in path.read_text().splitlines())
    assert len({(r['state_uid'],r['eta_uid'],r['future_index']) for r in raw})==len(raw)
    response_by=defaultdict(list)
    for r in raw:response_by[(r['state_uid'],r['eta_uid'])].append(r['response'])
    response_rows=[]
    for r in rows:
        responses=response_by[(r['state_uid'],r['eta_uid'])]
        response_rows.append({k:r[k] for k in ('state_uid','source_group','episode_index','kind','base_Q16','alternate_Q16','delta_Q16','holm_p')}|
            {key:float(np.mean([z[key] for z in responses if key in z])) for key in ('raw_response_distance','safe_response_distance','executed_response_distance')})
    csvout('local_response_audit.csv',response_rows)
    a=np.asarray([r['base_Q16'] for r in rows]);b=np.asarray([r['alternate_Q16'] for r in rows])
    entropy=lambda q:-q*np.log(np.clip(q,1e-15,1))-(1-q)*np.log(np.clip(1-q,1e-15,1))
    summary={'pairs':len(rows),'states':len({r['state_uid'] for r in rows}),'new_continuations':len(raw),
      'same_h_max_abs_error':max(r['feature_max_abs_error'] or 0 for r in raw),
      'collisions':sum(r['collision'] for r in raw),'numerical_failure':sum(r['numerical_failure'] for r in raw),
      'pairs_abs_delta_Q_ge_0_25':sum(abs(r['delta_Q16'])>=.25 for r in rows),
      'pairs_abs_delta_Q_ge_0_5':sum(abs(r['delta_Q16'])>=.5 for r in rows),
      'pairs_Holm_p_lt_0_05':sum(r['holm_p']<.05 for r in rows),
      'source_families_with_Holm_significant_Q_change':len({r['source_group'] for r in rows if r['holm_p']<.05}),
      'pairs_full_success_to_zero_success':sum(r['base_success']==16 and r['alternate_success']==0 for r in rows),
      'B15_lost':sum(r['base_B15'] and not r['alternate_B15'] for r in rows),
      'B15_gained':sum(r['alternate_B15'] and not r['base_B15'] for r in rows),
      'response_at_same_post_t0_state_mean_executed_distance':float(np.mean([r['response']['executed_response_distance'] for r in raw if 'executed_response_distance' in r['response']])),
      'absolute_delta_Q_vs_raw_response_distance_spearman_descriptive':float(spearmanr(np.abs(a-b),[r['raw_response_distance'] for r in response_rows]).statistic),
      'absolute_delta_Q_vs_executed_response_distance_spearman_descriptive':float(spearmanr(np.abs(a-b),[r['executed_response_distance'] for r in response_rows]).statistic),
      'empirical_hidden_controller_pair_equal_MAE_lower_bound':float(np.abs(a-b).mean()/2),
      'empirical_hidden_controller_NLL_grouping_penalty_nats':float(np.mean(entropy((a+b)/2)-(entropy(a)+entropy(b))/2)),
      'by_kind':{k:{'base_B15':sum(r['base_B15'] for r in rows if r['kind']==k),'alternate_B15':sum(r['alternate_B15'] for r in rows if r['kind']==k),
                   'base_Q':float(np.mean([r['base_Q16'] for r in rows if r['kind']==k])),
                   'alternate_Q':float(np.mean([r['alternate_Q16'] for r in rows if r['kind']==k]))} for k in p['candidate_kinds']},
      'scope':'Identifiability under an explicitly changed future controller, with shared t0 reference. Does not establish the cause of original four-scene LOSO failure.'}
    dump('summary.json',summary);print(json.dumps(summary,indent=2,default=json_default))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','merge','analyze','refresh_unexecuted_runtime']);a=ap.parse_args()
    globals()[a.command]()
