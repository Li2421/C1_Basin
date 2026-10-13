"""Label-free 200-step behavior probes, indexed separately from success labels.

The shared schema has NOT NULL success and cannot encode unlabeled trajectories.
The single merger registers source_file evidence, never counterfeit success=0.
"""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from common import *
import copy,time
from dataclasses import replace
from collections import defaultdict
import numpy as np
from shared_rollout_db.src.rollout_db import connect,transaction,uid,canonical,ROOT as DBROOT
from shared_rollout_db.src.cache_writer import append_journal
from diagnostics.controller_q_diversity_v1.experiment import case
EXP='exp_controller_flow_blend_v1_behavior_probes'

def create():
    d=HERE/'stage_2/probes';src=read(HERE/'stage_2/protocol.json');p=copy.deepcopy(src);p.update(experiment_uid=EXP,states=src['probe_states'])
    p['controller_order']=src['anchor_order']+src['controller_order'];p['execution_controller_uids']={}
    for name in p['controller_order']:
        c=copy.deepcopy(p['controllers']['ring_exchange'][name]);v=c['payload'];p['execution_controller_uids'][name]=c['controller_uid']
        v.update(probe_protocol='200 physical steps, unlabeled statistics only, no early success/collision termination',horizon=200,success_semantics='UNLABELED_BEHAVIOR_NO_TASK_OUTCOMES',probe_code_sha256=sha(__file__))
        c['controller_uid']=uid('ctl',v);p['controllers']['ring_exchange'][name]=c
    freeze(d/'protocol.json',p)
    rows=[case(p,s,n,j,0) for n in p['controller_order'] for s in p['states'] for j in (10,13)];assert len(rows)==832
    freeze(d/'CASES.json',rows);freeze(d/'PLANNED_ROLLOUTS.json',dict(requests=[dict(state_uid=r['state_uid'],eta_uid=r['eta_uid'],controller_uid=r['controller_uid'],seed_keys=[r['seed_key']]) for r in rows]))
    with connect() as db,transaction(db):
        db.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',
            (EXP,'Round5 unlabeled behavior probes',str(d),sha(d/'protocol.json'),sha(__file__),canonical(dict(no_terminal_labels=True,source_file_only_merge=True))))
        for name in p['controller_order']:
            c=p['controllers']['ring_exchange'][name];v=c['payload']
            db.execute('INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                (c['controller_uid'],v['scenario_uid'],v['flow_checkpoint_sha256'],v['orthoflow3_sha256'],uid('safe',dict(code=v['safety_projection_sha256'],cbf=v['cbf'])),str(v['horizon']),str(v['dt']),v['success_semantics'],v['conditioning'],canonical(v['rng']),canonical(v),'EXACT_PROFILE'))
    # Fresh probe states make previous traces impossible at these exact identities.
    # Source-file index is checked too, because probes have no DB outcome row.
    with connect(True) as db:
        existing=db.execute('SELECT count(*) FROM source_file WHERE experiment_uid=?',(EXP,)).fetchone()[0]
    freeze(d/'HISTORICAL_TRACE_REUSE_AUDIT.json',dict(fresh_state_disjoint_receipt_sha256=sha(HERE/'stage_2/STATE_DISJOINT_RECEIPT.json'),historical_matching_probe_sources=existing,
        shared_success_schema_cannot_encode_unlabeled=True,decision='No equivalent evidence at newly generated content hashes; fresh unlabeled probe only; never generate duplicate completed raw'))

class SafetyTracker:
    def __init__(self,base):self.base=base;self.active=False
    def __getattr__(self,k):return getattr(self.base,k)
    def project(self,target):
        a,d=self.base.project(target);self.active|=np.linalg.norm(np.asarray(a)-target)>1e-6;return a,d
    def project_field_target(self,target):
        a,d=self.base.project_field_target(target);self.active|=np.linalg.norm(np.asarray(a)-target)>1e-6;return a,d
class ProbeController:
    def __init__(self,rt):self.rt=rt;self.last=None
    def __getattr__(self,k):return getattr(self.rt,k)
    def safety(self,env):self.last=SafetyTracker(self.rt.safety(env));return self.last
    def prepare(self,*args,**kwargs):
        from poc.field_controller import Runtime
        return Runtime.prepare(self,*args,**kwargs)
    def action(self,env,key,eta):
        from poc.field_controller import Runtime
        return Runtime.action(self,env,key,eta,'field')

def worker(index,workers):
    from blend_runtime import make
    from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for
    d=HERE/'stage_2/probes';p=read(HERE/'stage_2/protocol.json');q=read(d/'protocol.json');rows=read(d/'CASES.json')[index::workers];cache={};states={s['state_uid']:s for s in q['states']}
    assert read(d/'CACHE_PREFLIGHT.json')['summary']['ambiguous']==0
    for r in rows:
        dest=d/'raw'/(r['rollout_uid']+'.json')
        if dest.exists():continue
        s=states[r['state_uid']];rt=make(r['controller_name'],p,cache);probe=ProbeController(rt);env=rt.make_env(s)
        env.config=replace(env.config,terminate_on_success=False,terminate_on_collision=False)
        start=time.monotonic();p0=env.positions.copy();error=None;path=[p0];actions=[];active=[]
        for t in range(200):
            try:
                a,_,_=probe.action(env,key_for(s,0,t),np.asarray(r['eta']))
                # Return tuple contains terminal/reward fields; none are read.
                env.step(a);path.append(env.positions.copy());actions.append(a.copy());active.append(probe.last.active)
            except Exception as exc:error=dict(step=t,type=type(exc).__name__,message=str(exc)[:500]);break
        path=np.asarray(path);actions=np.asarray(actions)
        complete=len(actions)==200
        statistics=None
        if complete:
            initial=np.linalg.norm(env.goals-p0,axis=-1).sum();final=np.linalg.norm(env.goals-path[-1],axis=-1).sum()
            distances=np.linalg.norm(path[:,:,None,:]-path[:,None,:,:],axis=-1)[:,np.triu_indices(4,1)[0],np.triu_indices(4,1)[1]]
            speeds=np.linalg.norm(actions,axis=-1)
            statistics=[float((initial-final)/initial),float(np.mean(active)),float(distances.min()),float(speeds.mean()),float((speeds.mean(-1)<.01).sum())]
        trace=dest.with_suffix('.npz');trace.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(trace,positions=path,actions=actions,projection_active=np.array(active),goals=env.goals)
        record=dict(schema='unlabeled_flow_behavior_probe_v1',experiment_uid=EXP,protocol_sha256=sha(d/'protocol.json'),case=r,physical_steps=len(actions),complete=complete,
            statistics=statistics,error=error,elapsed_seconds=time.monotonic()-start,jax_enable_x64=False,terminal_labels_read=False,trace_path=str(trace),trace_sha256=sha(trace),probe_code_sha256=sha(__file__))
        write(dest,record);append_journal([record],EXP,f'probe_{index}')
        print(json.dumps(dict(controller=r['controller_name'],state=r['state_index'],eta=r['eta_index'],steps=len(actions),complete=complete)),flush=True)

def merge():
    d=HERE/'stage_2/probes';p=read(d/'protocol.json');rows=read(d/'CASES.json');features=[];index=[];unknown=[]
    for name in p['controller_order']:
        feats=[]
        for r in [z for z in rows if z['controller_name']==name]:
            rec=read(d/'raw'/(r['rollout_uid']+'.json'));assert rec['case']==r and rec['protocol_sha256']==sha(d/'protocol.json')
            assert rec['terminal_labels_read'] is False and not any(k in rec for k in ('success','timeout','collision','outcome'))
            feats.extend(rec['statistics'] if rec['complete'] else [np.nan]*5)
            if not rec['complete']:unknown.append(r['rollout_uid'])
            index.append(dict(probe_uid=r['rollout_uid'],raw_path=str(d/'raw'/(r['rollout_uid']+'.json')),raw_sha256=sha(d/'raw'/(r['rollout_uid']+'.json')),trace_path=rec['trace_path'],trace_sha256=rec['trace_sha256']))
        features.append(feats);assert len(feats)==160
    with connect() as db,transaction(db):
        assert db.execute('SELECT protocol_hash FROM experiment WHERE experiment_uid=?',(EXP,)).fetchone()[0]==sha(d/'protocol.json')
        for path in sorted((DBROOT/'journals'/EXP).glob('*.jsonl')):
            suid=uid('source',dict(path=str(path),sha256=sha(path)))
            rr=[json.loads(s) for s in path.read_text().splitlines()]
            assert all(r['schema']=='unlabeled_flow_behavior_probe_v1' for r in rr)
            db.execute('INSERT OR IGNORE INTO source_file(source_uid,experiment_uid,path,sha256,file_type,classification,rows_seen,rows_imported) VALUES(?,?,?,?,?,?,?,?)',
                (suid,EXP,str(path),sha(path),'.jsonl','UNLABELED_SHORT_PROBE',len(rr),0))
        db.execute('UPDATE experiment SET end_time=CURRENT_TIMESTAMP,new_rollout_count=?,metadata_json=? WHERE experiment_uid=?',(len(rows),canonical(dict(no_terminal_labels=True,unknown=len(unknown),source_file_only_merge=True)),EXP))
    np.savez_compressed(d/'R5_FEATURES.npz',features=np.asarray(features,np.float32),controller_names=np.array(p['controller_order']))
    write(d/'RAW_TRACE_INDEX.json',index);write(d/'POSTFLIGHT.json',dict(status='COMPLETE' if not unknown else 'COMPLETE WITH UNKNOWN PROBES; R5 ARM INPUTS UNAVAILABLE',requests=832,known_statistics=832-len(unknown),unknown_probe_uids=unknown,missing=0,terminal_label_rows_created=0,
        source_file_only_merge=True,why='Shared rollout.success is NOT NULL; unlabeled progress is evidence, not success=0 or numerical_failure=1. No schema change/no fabricated task labels.'))

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['create','worker','merge']);ap.add_argument('--index',type=int,default=0);ap.add_argument('--workers',type=int,default=24);a=ap.parse_args()
    if a.command=='worker':worker(a.index,a.workers)
    else:globals()[a.command]()
