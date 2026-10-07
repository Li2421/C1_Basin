"""Fresh-seed confirmation of one exact shared-eta ranking reversal.

Discovery uses the archived matrix; seeds16..31 are held out until freeze.
This is a replication of seed robustness, not an unseen-state model test.
"""
import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from shared_rollout_db.src.rollout_db import canonical, connect, eta_identity, transaction, uid
from shared_rollout_db.src.cache_writer import append_journal

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent/'ring_confirmation'
PANEL=ROOT/'diagnostics/orthoflow3_state_eta_interaction_panel_v1'
EXP='exp_core_followup_ring_reversal_confirmation_v1'

def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(name,x):
    OUT.mkdir(exist_ok=True);(OUT/name).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')

def runtime():
    from new_benchmark_common import basin_dataset_v1 as bd
    p=load(OUT/'protocol.json');bd.OUT=OUT/'registry';bd.WORK=OUT/'runtime'
    rows=[]
    for i,r in enumerate(load(OUT/'states.json')):
        rows.append({'uid':r['state_uid'],'alias':r['state_id'],'index':i,'physical':r['physical'],
                     'content_hash':r['physical_snapshot_hash'],'conditioning':r['conditioning'],
                     'environment_descriptor':r['environment_descriptor'],'source_group':r['parent_episode_id'],
                     'provenance':{'task':EXP,'source_split':r['split'],'parent_episode_id':r['parent_episode_id'],
                                   'frozen_test_used':False,'fresh_confirmation_seeds':list(range(16,32))}})
    rt=bd.TrainingRuntime('ring_exchange',rows,parent=False)
    assert rt.controllers['orthoflow3']['uid']==p['controller_uid']
    rt.experiment_uid=EXP
    return rt

def prepare():
    if (OUT/'protocol.json').exists():print('already frozen');return
    mat=load(PANEL/'Q16_matrices.json')['ring_exchange'];q=np.asarray(mat['Q16'],float)
    candidates=[]
    for i,j in itertools.combinations(range(len(q)),2):
        for a,b in itertools.combinations(range(1,q.shape[1]),2):
            values=q[[i,i,j,j],[a,b,a,b]]
            if not np.isfinite(values).all():continue
            da,db=float(values[0]-values[1]),float(values[2]-values[3])
            if da*db<0 and min(abs(da),abs(db))>=.25:
                candidates.append((min(abs(da),abs(db)),abs(da)+abs(db),mat['state_uids'][i],mat['state_uids'][j],a,b,values.tolist()))
    candidates.sort(key=lambda z:(-z[0],-z[1],z[2],z[3],z[4],z[5]))
    best=candidates[0];states={r['state_uid']:r for r in load(PANEL/'states_manifest.json')['states']}
    chosen=[states[best[2]],states[best[3]]]
    assert chosen[0]['parent_episode_id']!=chosen[1]['parent_episode_id']
    cid=chosen[0]['controller_uid_source'];assert chosen[1]['controller_uid_source']==cid
    etas=load(PANEL/'common_eta_panel.json')['eta_values']
    jobs=[{'state_uid':sid,'eta':etas[e],'eta_uid':eta_identity(etas[e])[0],'eta_index':e,'controller_uid':cid,
           'seed_keys':[canonical({'future_index':k}) for k in range(16,32)]} for sid in best[2:4] for e in best[4:6]]
    p={'task':'RING_FRESH_SEED_RANKING_REVERSAL_CONFIRMATION','experiment_uid':EXP,'controller_uid':cid,
       'selection_rule':'Among four-complete-cell nonzero-eta reversals, maximize minimum absolute opposite gap, then total gap; tie state UID and eta index. Discovery outcomes are allowed; confirmation outcomes cannot select witness.',
       'discovery_Q16':best[6],'eta_indices':list(best[4:6]),'discovery_standard_seeds':list(range(16)),
       'confirmation_future_indices':list(range(16,32)),'state_uids':list(best[2:4]),
       'hypothesis':'Opposite eta ordering replicates with gap>=.25 in both states and paired two-sided p<=.05 in each state.',
       'interpretation':'Two previously studied development states, held-out seeds. Not new source-family generalization evidence.',
       'safety_semantics':'current Ring safety only; exact source controller UID required','no_model_selection':True,
       'source_matrix_sha256':sha(PANEL/'Q16_matrices.json'),'runtime_script_sha256':sha(__file__),
       'numerical_policy':'One execution per missing seed; retain numerical unresolved, no imputation.'}
    dump('protocol.json',p);dump('states.json',chosen);dump('planned_rollouts.json',{'requests':jobs})
    rt=runtime()
    with connect() as c,transaction(c):
        c.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',
                  (EXP,p['task'],str(OUT),sha(OUT/'protocol.json'),sha(__file__),canonical({'fresh_seeds':True,'model_training':False})))
    print(json.dumps({'discovery_Q16':best[6],'eta_indices':list(best[4:6]),'requested':64}))

def run():
    p=load(OUT/'protocol.json');assert sha(__file__)==p['runtime_script_sha256'];rt=runtime()
    states={r['uid']:r for r in rt.states};pf=load(OUT/'cache_preflight.json')
    missing={(r['state_uid'],r['eta_uid'],s) for r in pf['details'] for s in r['missing_seeds']}
    existing=set();folder=ROOT/'shared_rollout_db/journals'/EXP
    if folder.exists():
        for f in folder.glob('*.jsonl'):
            for line in f.read_text().splitlines():
                r=json.loads(line)['record'];existing.add((r['state_uid'],eta_identity(r['eta'])[0],canonical({'future_index':r['future_index']})))
    n=0
    for job in load(OUT/'planned_rollouts.json')['requests']:
        pending=[]
        for sk in job['seed_keys']:
            key=(job['state_uid'],job['eta_uid'],sk)
            if key not in missing or key in existing:continue
            seed=json.loads(sk)['future_index']
            row=rt.rollout(states[job['state_uid']],np.asarray(job['eta'],float),seed,'orthoflow3')
            row['eta_index']=job['eta_index'];row['confirmation_only']=True
            pending.append({'schema':'explicit_ring_reversal_confirmation_v1','record':row});n+=1
        if pending:append_journal(pending,EXP,'confirmation')
        print(json.dumps({'new':n,'state':job['state_uid'],'eta_index':job['eta_index']}),flush=True)

def merge():
    p=load(OUT/'protocol.json');allowed={(r['state_uid'],r['eta_uid']) for r in load(OUT/'planned_rollouts.json')['requests']}
    stats={'inserted':0,'duplicates':0,'journals':0}
    with connect() as c:
        for path in sorted((ROOT/'shared_rollout_db/journals'/EXP).glob('*.jsonl')):
            lines=path.read_text().splitlines();src=uid('src',{'path':str(path.resolve())})
            with transaction(c):
                c.execute('INSERT OR IGNORE INTO source_file(source_uid,experiment_uid,path,sha256,file_type,classification,rows_seen) VALUES(?,?,?,?,?,?,?)',
                          (src,EXP,str(path.resolve()),sha(path),'.jsonl','SEED_EXACT',len(lines)))
                for line_index,line in enumerate(lines,1):
                    e=json.loads(line);assert e['schema']=='explicit_ring_reversal_confirmation_v1';r=e['record']
                    eid=eta_identity(r['eta'])[0];assert (r['state_uid'],eid) in allowed
                    assert r['controller_uid']==p['controller_uid'] and r['future_index'] in range(16,32)
                    sk=canonical({'future_index':r['future_index']});rid=uid('roll',{'state':r['state_uid'],'eta':eid,'controller':r['controller_uid'],'seed':sk})
                    core=tuple(int(r[k]) for k in ('success','deadlock','timeout','collision','numerical_failure'))
                    old=c.execute('SELECT * FROM rollout WHERE rollout_uid=?',(rid,)).fetchone()
                    if old:
                        assert core==tuple(old[k] for k in ('success','deadlock','timeout','collision','numerical_failure')),rid
                        stats['duplicates']+=1
                    else:
                        c.execute('INSERT INTO rollout(rollout_uid,state_uid,eta_uid,controller_uid,seed_key,continuation_seed_json,success,deadlock,timeout,collision,numerical_failure,episode_length,j_def,min_wall_distance,min_agent_distance,outcome,experiment_uid,original_source_file,timestamp,compatibility_quality,raw_record_hash) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                                  (rid,r['state_uid'],eid,r['controller_uid'],sk,sk,*core,r['episode_length'],r.get('J_def'),r.get('minimum_wall_clearance'),r.get('minimum_agent_clearance'),r['outcome'],EXP,str(path),r['timestamp'],'EXACT_REUSE',hashlib.sha256(canonical(r).encode()).hexdigest()))
                        stats['inserted']+=1
                    c.execute('INSERT OR IGNORE INTO rollout_source VALUES(?,?,?)',(rid,src,line_index))
            stats['journals']+=1
        c.execute('UPDATE experiment SET new_rollout_count=(SELECT COUNT(*) FROM rollout WHERE experiment_uid=?),end_time=CURRENT_TIMESTAMP WHERE experiment_uid=?',(EXP,EXP));c.commit()
    dump('merge_audit.json',stats);print(json.dumps(stats))

def analyze():
    p=load(OUT/'protocol.json');rows=[]
    with connect(True) as c:
        for req in load(OUT/'planned_rollouts.json')['requests']:
            found=c.execute('SELECT * FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=? AND conflict_quarantined=0',
                            (req['state_uid'],req['eta_uid'],req['controller_uid'])).fetchall()
            by={json.loads(x['seed_key'])['future_index']:dict(x) for x in found if json.loads(x['seed_key']).get('future_index') in range(16,32)}
            good={i:r for i,r in by.items() if not r['numerical_failure']}
            s=sum(r['success'] for r in good.values());f=len(good)-s
            rows.append({'state_uid':req['state_uid'],'eta_index':req['eta_index'],'success':s,'n_valid':len(good),
                         'Q_lower':s/16,'Q_upper':1-f/16,'numerical':sum(r['numerical_failure'] for r in by.values()),
                         'seed_success':{i:r['success'] for i,r in good.items()}})
    state_results=[]
    for sid in p['state_uids']:
        a,b=[r for r in rows if r['state_uid']==sid];ids=set(a['seed_success'])&set(b['seed_success'])
        plus=sum(a['seed_success'][k]>b['seed_success'][k] for k in ids)
        minus=sum(a['seed_success'][k]<b['seed_success'][k] for k in ids)
        state_results.append({'state_uid':sid,'eta_indices':[a['eta_index'],b['eta_index']],
                              'delta_Q_lower':a['Q_lower']-b['Q_upper'],'delta_Q_upper':a['Q_upper']-b['Q_lower'],
                              'paired_exact_p':binomtest(plus,plus+minus).pvalue if plus+minus else 1.,
                              'a_better_seeds':plus,'b_better_seeds':minus})
    a,b=state_results
    reversed_=(a['delta_Q_lower']>=.25 and b['delta_Q_upper']<=-.25) or (a['delta_Q_upper']<=-.25 and b['delta_Q_lower']>=.25)
    result={'rows':rows,'states':state_results,'replicated_opposite_margin':reversed_,
            'both_p_le_0_05':all(r['paired_exact_p']<=.05 for r in state_results),
            'discovery_Q16':p['discovery_Q16'],'scope':p['interpretation']}
    dump('result.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','run','merge','analyze']);a=ap.parse_args();globals()[a.command]()
