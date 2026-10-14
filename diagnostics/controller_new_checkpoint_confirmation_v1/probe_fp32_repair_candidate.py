from core import *
import time,copy
from dataclasses import replace
from collections import Counter
from shared_rollout_db.src.rollout_db import connect,transaction,uid,canonical,eta_identity
from shared_rollout_db.src.cache_writer import append_journal
PROBE_EXP=EXP+'_unlabeled'

def create():
    prereg();source=read(R5/'stage_2/probes/protocol.json');base=read(HERE/'FINAL_BASE_PROFILES.json');p=dict(schema='round12_unlabeled_200step_probe_v1',experiment_uid=PROBE_EXP,states=source['states'],controller_order=base['eligible_controller_order'],controllers={'ring_exchange':base['profiles']},eta=source['eta'],probe_code_sha256=sha(__file__),core_sha256=sha(HERE/'core.py'),adapter_sha256=sha(P1/'flow_adapter.py'),historical_probe_code_sha256=sha(R5/'probe.py'),compatibility_audit_sha256=sha(HERE/'COMPATIBILITY_AUDIT.json'))
    freeze(HERE/'probe_protocol.json',p);rows=[]
    for n in p['controller_order']:
        v=copy.deepcopy(base['profiles'][n]['payload']);v.update(success_semantics='UNLABELED_BEHAVIOR_NO_TASK_OUTCOMES',probe_steps=200,no_early_termination=True,probe_code_sha256=sha(__file__));ctl=uid('ctl',v)
        for si,s in enumerate(p['states']):
            for eta in [10,13]:
                sk=canonical(dict(future_index=0,future_root=2026100403,rng_namespace=s['rng_namespace']));e=eta_identity(p['eta'][eta])[0];pid=uid('probe',dict(state=s['state_uid'],eta=e,controller=ctl,seed=sk));rows.append(dict(probe_uid=pid,state_uid=s['state_uid'],state_index=si,controller_name=n,controller_uid=ctl,eta_uid=e,eta_index=eta,eta=p['eta'][eta],seed=0,seed_key=sk))
    freeze(HERE/'probe_CASES.json',rows);freeze(HERE/'probe_PLANNED_ROLLOUTS.json',dict(requests=[dict(state_uid=r['state_uid'],eta_uid=r['eta_uid'],controller_uid=r['controller_uid'],seed_keys=[r['seed_key']],probe_uid=r['probe_uid']) for r in rows]))
    with connect(True) as db:sources=[dict(r) for r in db.execute("SELECT source_uid,experiment_uid,path,sha256 FROM source_file WHERE classification='UNLABELED_SHORT_PROBE'")]
    assert not any(r['experiment_uid']==PROBE_EXP for r in sources),'Unexpected existing experiment: audit before rerun'
    missing=[r for r in rows if not (HERE/'probe/raw'/f"{r['probe_uid']}.json").exists()];write(HERE/'probe_MISSING_CASES.json',missing);write(HERE/'probe_CACHE_PREFLIGHT.json',dict(status='PASS',planned=len(rows),genuinely_missing=len(missing),ambiguous=0,metadata_sources_scanned=len(sources),known_matching_final_sources=0,source_file_only=True,success_labels_generated=False,unknown_never_rerun=True))

def worker(index,workers):
    prereg();p=read(HERE/'probe_protocol.json');assert sha(__file__)==p['probe_code_sha256'];assert sha(HERE/'core.py')==p['core_sha256'];assert sha(P1/'flow_adapter.py')==p['adapter_sha256'];assert read(HERE/'probe_CACHE_PREFLIGHT.json')['status']=='PASS'
    import jax
    jax.config.update('jax_enable_x64',False)
    from diagnostics.controller_q_transfer_round3_v1.flow_adapter import RingFlowAdapter
    sys.path.insert(0,str(R5))
    from diagnostics.controller_flow_blend_v1.probe import ProbeController
    from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for
    jax.config.update('jax_enable_x64',False)
    assert not jax.config.x64_enabled
    rows=read(HERE/'probe_MISSING_CASES.json');perm=np.random.default_rng(202610131220).permutation(len(rows));rows=[rows[int(j)] for j in perm[index::workers]];states={s['state_uid']:s for s in p['states']};cache={};elapsed=[]
    for c in rows:
        dest=HERE/'probe/raw'/f"{c['probe_uid']}.json"
        if dest.exists():continue
        t0=time.monotonic();name=c['controller_name'];v=p['controllers']['ring_exchange'][name]['payload']
        if name not in cache:cache[name]=RingFlowAdapter(v['flow_checkpoint_path'],read(v['adapter_manifest_path']),v['observation_frame'])
        rt=cache[name];probe=ProbeController(rt);s=states[c['state_uid']];env=rt.make_env(s);env.config=replace(env.config,terminate_on_success=False,terminate_on_collision=False);path=[env.positions.copy()];actions=[];active=[];error=None
        for t in range(200):
            try:a,_,_=probe.action(env,key_for(s,0,t),np.asarray(c['eta']));env.step(a);path.append(env.positions.copy());actions.append(a.copy());active.append(probe.last.active)
            except Exception as exc:error=dict(step=t,type=type(exc).__name__,message=str(exc)[:500]);break
        path=np.asarray(path);actions=np.asarray(actions);complete=len(actions)==200;statistics=None
        if complete:
            initial=np.linalg.norm(env.goals-path[0],axis=-1).sum();final=np.linalg.norm(env.goals-path[-1],axis=-1).sum();distances=np.linalg.norm(path[:,:,None]-path[:,None,:],axis=-1)[:,np.triu_indices(4,1)[0],np.triu_indices(4,1)[1]];speeds=np.linalg.norm(actions,axis=-1)
            if initial>0:statistics=[float((initial-final)/initial),float(np.mean(active)),float(distances.min()),float(speeds.mean()),float((speeds.mean(-1)<.01).sum())]
            else:complete=False;error=dict(step=200,type='ZeroProgressDenominator',message='Statistic unavailable; no zero imputation')
        trace=dest.with_suffix('.npz');trace.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(trace,positions=path,actions=actions,projection_active=np.asarray(active),goals=env.goals)
        record=dict(jax_enable_x64=bool(jax.config.x64_enabled),schema='round12_unlabeled_200step_probe_v1',experiment_uid=PROBE_EXP,case=c,physical_steps=len(actions),complete=complete,statistics=statistics,error=error,terminal_labels_read=False,trace_path=str(trace),trace_sha256=sha(trace),protocol_sha256=sha(HERE/'probe_protocol.json'),probe_code_sha256=sha(__file__),elapsed_seconds=time.monotonic()-t0);assert not any(k in record for k in ['success','outcome','timeout','collision']);write(dest,record);append_journal([record],PROBE_EXP,f'probe_{index}');elapsed.append(record['elapsed_seconds'])
        if len(elapsed)%8==0:print(json.dumps(dict(worker=index,completed=len(elapsed),assigned=len(rows),mean_seconds=float(np.mean(elapsed)))),flush=True)
    write(HERE/f'probe/worker_{index}.json',dict(complete=True,assigned=len(rows),new=len(elapsed)))

def merge():
    prereg();p=read(HERE/'probe_protocol.json');names=p['controller_order'];features=np.full((len(names),16,2,5),np.nan);unknown=[];index=[];error_max=0.
    for c in read(HERE/'probe_CASES.json'):
        path=HERE/'probe/raw'/f"{c['probe_uid']}.json";r=read(path);assert r['case']==c and not r['terminal_labels_read'];assert r['protocol_sha256']==sha(HERE/'probe_protocol.json');assert sha(Path(r['trace_path']))==r['trace_sha256'];z=np.load(r['trace_path']);assert len(z['positions'])==r['physical_steps']+1;i=names.index(c['controller_name']);si=c['state_index'];ei=[10,13].index(c['eta_index'])
        if r['complete']:
            pos=z['positions'];a=z['actions'];goals=z['goals'];d0=np.linalg.norm(goals-pos[0],axis=-1).sum();df=np.linalg.norm(goals-pos[-1],axis=-1).sum();pairs=np.linalg.norm(pos[:,:,None]-pos[:,None,:],axis=-1)[:,np.triu_indices(4,1)[0],np.triu_indices(4,1)[1]];speeds=np.linalg.norm(a,axis=-1);values=np.array([(d0-df)/d0,z['projection_active'].mean(),pairs.min(),speeds.mean(),(speeds.mean(-1)<.01).sum()]);err=float(np.max(abs(values-r['statistics'])));assert err<1e-12;error_max=max(error_max,err);features[i,si,ei]=r['statistics']
        else:assert r['statistics'] is None;unknown.append(dict(case=c,error=r['error']))
        index.append(dict(probe_uid=c['probe_uid'],raw_path=str(path),raw_sha256=sha(path),trace_sha256=r['trace_sha256']))
    np.savez_compressed(HERE/'R5_FINAL_FEATURES.npz',features=features.reshape(len(names),160).astype(np.float32),controller_names=np.array(names));write(HERE/'R5_RAW_INDEX.json',index);write(HERE/'R5_PROBE_AUDIT.json',dict(status='COMPLETE_WITH_UNKNOWN' if unknown else 'COMPLETE',requests=len(index),unknown_count=len(unknown),unknowns=unknown,statistics_reproduction_max_error=error_max,features_shape=[len(names),160],dropped_columns=list(range(105,110)),success_rows_created=0,terminal_labels_read=False))
    with connect() as db,transaction(db):
        db.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',(PROBE_EXP,'Round12 unlabeled final checkpoint probes',str(HERE/'probe'),sha(HERE/'probe_protocol.json'),sha(__file__),canonical(dict(no_terminal_labels=True))))
        for path in sorted((HERE/'journals'/PROBE_EXP).glob('*.jsonl')):
            rows=[json.loads(x) for x in path.read_text().splitlines()];assert all(x['terminal_labels_read'] is False for x in rows);suid=uid('source',dict(path=str(path),sha256=sha(path)));db.execute('INSERT OR IGNORE INTO source_file(source_uid,experiment_uid,path,sha256,file_type,classification,rows_seen,rows_imported) VALUES(?,?,?,?,?,?,?,?)',(suid,PROBE_EXP,str(path),sha(path),'.jsonl','UNLABELED_SHORT_PROBE',len(rows),0))
        db.execute('UPDATE experiment SET end_time=CURRENT_TIMESTAMP,new_rollout_count=?,metadata_json=? WHERE experiment_uid=?',(len(index),canonical(dict(unlabeled=True,unknown=len(unknown),no_success_rows=True)),PROBE_EXP))
    freeze(HERE/'R5_FROZEN.receipt.json',dict(utc=utc(),features_sha256=sha(HERE/'R5_FINAL_FEATURES.npz'),probe_audit_sha256=sha(HERE/'R5_PROBE_AUDIT.json'),raw_index_sha256=sha(HERE/'R5_RAW_INDEX.json'),preregistration_sha256=sha(HERE/'PREREGISTRATION.md'),final_success_labels_read_or_generated=False))
    print(json.dumps(dict(probes=len(index),unknown=len(unknown),frozen=True)))
if __name__=='__main__':
    if sys.argv[1]=='worker':worker(int(sys.argv[2]),int(sys.argv[3]))
    else:globals()[sys.argv[1]]()
