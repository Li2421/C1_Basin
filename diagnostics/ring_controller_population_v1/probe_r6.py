from core import *
import copy,time
from dataclasses import replace
from collections import Counter
from shared_rollout_db.src.rollout_db import connect,transaction,uid,canonical,eta_identity
from shared_rollout_db.src.cache_writer import append_journal
PROBE_EXP=EXP+'_r6_unlabeled';COLS=['relative_progress','projection_fraction','min_pair_distance','mean_action_norm','stalled_steps','arrival_count_change','final_stalled_robot_count']

def create():
    prereg();q=read(R5/'stage_2/probes/protocol.json');base=read(HERE/'FINAL_BASE_PROFILES.json');names=base['eligible_controller_order'];assert len(names)==56 and len(q['states'])==16;p=dict(eta=q['eta'],controllers={'ring_exchange':base['profiles']})
    protocol=dict(schema='round15_unlabeled_late_probe_v1',experiment_uid=PROBE_EXP,states=q['states'],controller_order=names,controllers=p['controllers'],eta=p['eta'],prefix_steps=400,branch_steps=200,seed=0,statistics=COLS,adapter_sha256=sha(P1/'flow_adapter.py'),probe_code_sha256=sha(__file__),preregistration_sha256=sha(HERE/'PREREGISTRATION.md'))
    freeze(HERE/'r6_probe_protocol.json',protocol);rows=[]
    for n in names:
        base=p['controllers']['ring_exchange'][n];payload=copy.deepcopy(base['payload']);payload.update(probe_protocol='400 eta13 + paired200 eta13/eta10 branches; unlabeled; no early termination',success_semantics='UNLABELED_BEHAVIOR_NO_TASK_OUTCOMES',probe_code_sha256=sha(__file__),prefix_steps=400,branch_steps=200);ctl=uid('ctl',payload)
        for si,s in enumerate(q['states']):
            sk=canonical(dict(future_index=0,future_root=2026100403,rng_namespace=s['rng_namespace']));eta=eta_identity(p['eta'][13])[0];pid=uid('probe',dict(state=s['state_uid'],controller=ctl,seed=sk,eta=eta,paired_branch_protocol=sha(HERE/'r6_probe_protocol.json')))
            rows.append(dict(probe_uid=pid,state_uid=s['state_uid'],state_index=si,controller_name=n,controller_uid=ctl,eta_uid=eta,seed_key=sk,branches=[p['eta'][13],p['eta'][10]],prefix_eta=p['eta'][13],seed=0))
    freeze(HERE/'r6_probe_CASES.json',rows);freeze(HERE/'r6_probe_PLANNED_ROLLOUTS.json',dict(requests=[dict(state_uid=r['state_uid'],eta_uid=r['eta_uid'],controller_uid=r['controller_uid'],seed_keys=[r['seed_key']],probe_uid=r['probe_uid']) for r in rows]))
    # Query unlabeled evidence metadata only; do not use rollout success lookups.
    with connect(True) as db:
        sources=[dict(x) for x in db.execute("SELECT source_uid,experiment_uid,path,sha256,classification FROM source_file WHERE classification='UNLABELED_SHORT_PROBE'")]
    matches=[]
    for x in sources:
        # Exact code/branch protocol is new; identical source metadata must identify our dedicated experiment.
        if x['experiment_uid']==PROBE_EXP:matches.append(x)
    assert not matches or (HERE/'R6_FROZEN.receipt.json').exists(),'Existing dedicated sources require explicit resume audit'
    missing=[r for r in rows if not (HERE/'r6_probe/raw'/f"{r['probe_uid']}.json").exists()]
    write(HERE/'r6_probe_MISSING_CASES.json',missing);write(HERE/'r6_probe_CACHE_PREFLIGHT.json',dict(status='PASS',requests=len(rows),genuinely_missing=len(missing),existing_raw=len(rows)-len(missing),ambiguous=0,metadata_source_rows_scanned=len(sources),dedicated_sources=matches,success_labels_read=False,compatible_historical_400_plus_two200_sources=0,old200_step_probes_incompatible=True,source_file_only=True,unknown_never_rerun=True))

def stats(env,start,path,actions,active):
    d0=np.linalg.norm(env.goals-start,axis=-1).sum();df=np.linalg.norm(env.goals-path[-1],axis=-1).sum();pairs=np.linalg.norm(path[:,:,None]-path[:,None,:],axis=-1)[:,np.triu_indices(4,1)[0],np.triu_indices(4,1)[1]];speed=np.linalg.norm(actions,axis=-1)
    arrivals=lambda pos:int((np.linalg.norm(env.goals-pos,axis=-1)<=env.config.goal_tolerance).sum())
    return [float((d0-df)/d0) if d0>0 else 0.,float(np.mean(active)),float(pairs.min()),float(speed.mean()),int((speed.mean(-1)<.01).sum()),arrivals(path[-1])-arrivals(start),int((speed[-1]<.01).sum())],bool(d0==0)

def worker(index,workers):
    prereg();p=read(HERE/'r6_probe_protocol.json');assert sha(__file__)==p['probe_code_sha256'];assert sha(P1/'flow_adapter.py')==p['adapter_sha256'];assert read(HERE/'r6_probe_CACHE_PREFLIGHT.json')['status']=='PASS'
    import jax
    jax.config.update('jax_enable_x64',False);assert not jax.config.x64_enabled
    from diagnostics.controller_q_transfer_round3_v1.flow_adapter import RingFlowAdapter
    from diagnostics.controller_flow_blend_v1.probe import ProbeController
    from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for
    cases=read(HERE/'r6_probe_MISSING_CASES.json');perm=np.random.default_rng(202610131110).permutation(len(cases));cases=[cases[int(j)] for j in perm[index::workers]];states={s['state_uid']:s for s in p['states']};cache={};elapsed=[]
    for r in cases:
        if datetime.now().astimezone().hour*60+datetime.now().astimezone().minute>=600:raise SystemExit('10:00 UTC+8 probe deadline')
        dest=HERE/'r6_probe/raw'/f"{r['probe_uid']}.json"
        if dest.exists():continue
        t0=time.monotonic();s=states[r['state_uid']];
        if r['controller_name'] not in cache:
            v=p['controllers']['ring_exchange'][r['controller_name']]['payload'];cache[r['controller_name']]=RingFlowAdapter(v['flow_checkpoint_path'],read(v['adapter_manifest_path']),v['observation_frame'])
            assert all(np.asarray(x).dtype==np.float32 for x in jax.tree_util.tree_leaves(cache[r['controller_name']].agent.network.params))
        rt=cache[r['controller_name']];probe=ProbeController(rt);env=rt.make_env(s);env.config=replace(env.config,terminate_on_success=False,terminate_on_collision=False)
        prefix=[env.positions.copy()];prefix_vel=[env.velocities.copy()];error=None
        for t in range(400):
            if datetime.now().astimezone().hour*60+datetime.now().astimezone().minute>=600:raise SystemExit('10:00 UTC+8 probe deadline')
            try:a,_,_=probe.action(env,key_for(s,0,t),np.asarray(p['eta'][13]));env.step(a);prefix.append(env.positions.copy());prefix_vel.append(env.velocities.copy())
            except Exception as exc:error=dict(step=t,type=type(exc).__name__,message=str(exc)[:500]);break
        boundary=copy.deepcopy(env);branches=[];arrays=dict(prefix_positions=np.asarray(prefix),prefix_velocities=np.asarray(prefix_vel),goals=env.goals)
        for bi,eta_index in enumerate([13,10]):
            branch=copy.deepcopy(boundary);path=[branch.positions.copy()];actions=[];active=[];err=error.copy() if error else None
            if error is None:
                for t in range(400,600):
                    if datetime.now().astimezone().hour*60+datetime.now().astimezone().minute>=600:raise SystemExit('10:00 UTC+8 probe deadline')
                    try:a,_,_=probe.action(branch,key_for(s,0,t),np.asarray(p['eta'][eta_index]));branch.step(a);path.append(branch.positions.copy());actions.append(a.copy());active.append(probe.last.active)
                    except Exception as exc:err=dict(step=t,type=type(exc).__name__,message=str(exc)[:500]);break
            path=np.asarray(path);actions=np.asarray(actions);complete=len(actions)==200;values,zero=stats(branch,path[0],path,actions,active) if complete else (None,False)
            branches.append(dict(eta_index=eta_index,steps=len(actions),complete=complete,statistics=values,error=err,zero_progress_denominator=zero))
            arrays.update({f'branch{bi}_positions':path,f'branch{bi}_actions':actions,f'branch{bi}_projection_active':np.asarray(active,bool)})
        trace=dest.with_suffix('.npz');trace.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(trace,**arrays)
        record=dict(schema='round15_unlabeled_late_probe_v1',experiment_uid=PROBE_EXP,case=r,prefix_steps=len(prefix)-1,prefix_error=error,branches=branches,trace_path=str(trace),trace_sha256=sha(trace),terminal_labels_read=False,protocol_sha256=sha(HERE/'r6_probe_protocol.json'),probe_code_sha256=sha(__file__),elapsed_seconds=time.monotonic()-t0)
        assert not any(k in record for k in ['success','outcome','timeout','collision']);write(dest,record);append_journal([record],PROBE_EXP,f'probe_{index}');elapsed.append(record['elapsed_seconds'])
        if len(elapsed)%4==0:print(json.dumps(dict(worker=index,completed=len(elapsed),assigned=len(cases),mean_seconds=float(np.mean(elapsed)),last_seconds=elapsed[-1])),flush=True)
    write(HERE/f'r6_probe/worker_{index}.json',dict(complete=True,assigned=len(cases),new=len(elapsed),mean_seconds=float(np.mean(elapsed)) if elapsed else None))

def merge():
    prereg();p=read(HERE/'r6_probe_protocol.json');names=p['controller_order'];features=np.full((56,16,2,7),np.nan);unknown=[];index=[];errors=Counter();zero=0;steps=0
    for r in read(HERE/'r6_probe_CASES.json'):
        path=HERE/'r6_probe/raw'/f"{r['probe_uid']}.json";rec=read(path);assert rec['case']==r and rec['terminal_labels_read'] is False;assert rec['protocol_sha256']==sha(HERE/'r6_probe_protocol.json');assert sha(Path(rec['trace_path']))==rec['trace_sha256'];i=names.index(r['controller_name']);si=r['state_index'];steps+=rec['prefix_steps']
        for j,b in enumerate(rec['branches']):
            steps+=b['steps'];zero+=b['zero_progress_denominator']
            if b['complete']:features[i,si,j]=b['statistics']
            else:unknown.append(dict(probe_uid=r['probe_uid'],controller=r['controller_name'],state=si,branch_eta=b['eta_index'],error=b['error']));errors[b['error']['type']]+=1
        index.append(dict(probe_uid=r['probe_uid'],raw_path=str(path),raw_sha256=sha(path),trace_sha256=rec['trace_sha256']))
    valid=np.isfinite(features).all(axis=(2,3));delta=np.full((56,7),np.nan)
    for i in range(56):
        if valid[i].any():delta[i]=(features[i,valid[i],1]-features[i,valid[i],0]).mean(0)
    np.savez_compressed(HERE/'R6_FEATURES.npz',raw=features.reshape(56,224),branch_statistics=features,delta=delta,paired_state_count=valid.sum(1),controller_names=np.array(names),statistic_names=np.array(COLS));write(HERE/'R6_RAW_INDEX.json',index)
    audit=dict(status='COMPLETE_WITH_UNKNOWN' if unknown else 'COMPLETE',requests=896,branch_requests=1792,unknown_branches=len(unknown),unknowns=unknown,error_types=errors,paired_state_counts=dict(zip(names,valid.sum(1))),actual_physical_steps=steps,maximum_planned_steps=716800,zero_progress_denominators=zero,terminal_labels_read=False,success_rows_created=0,raw_feature_shape=[56,224],delta_shape=[56,7],finite_delta=np.isfinite(delta).all(axis=1).tolist(),per_statistic_distribution={COLS[k]:dict(min=float(np.nanmin(features[:,:,:,k])),median=float(np.nanmedian(features[:,:,:,k])),max=float(np.nanmax(features[:,:,:,k]))) for k in range(7)})
    write(HERE/'R6_AUDIT.json',audit)
    # Only source_file/experiment metadata are written for probes, never success rows.
    with connect() as db,transaction(db):
        db.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',(PROBE_EXP,'Round11 unlabeled paired late probes',str(HERE),sha(HERE/'r6_probe_protocol.json'),sha(__file__),canonical(dict(no_terminal_labels=True))))
        for path in sorted((HERE/'journals'/PROBE_EXP).glob('*.jsonl')):
            rows=[json.loads(x) for x in path.read_text().splitlines()];assert all(x['terminal_labels_read'] is False for x in rows);suid=uid('source',dict(path=str(path),sha256=sha(path)))
            db.execute('INSERT OR IGNORE INTO source_file(source_uid,experiment_uid,path,sha256,file_type,classification,rows_seen,rows_imported) VALUES(?,?,?,?,?,?,?,?)',(suid,PROBE_EXP,str(path),sha(path),'.jsonl','UNLABELED_SHORT_PROBE',len(rows),0))
        db.execute('UPDATE experiment SET end_time=CURRENT_TIMESTAMP,new_rollout_count=?,metadata_json=? WHERE experiment_uid=?',(896,canonical(dict(unlabeled=True,unknown_branches=len(unknown),no_success_rows=True)),PROBE_EXP))
    write(HERE/'R6_FROZEN.receipt.json',dict(utc=utc(),features_sha256=sha(HERE/'R6_FEATURES.npz'),audit_sha256=sha(HERE/'R6_AUDIT.json'),raw_index_sha256=sha(HERE/'R6_RAW_INDEX.json'),protocol_sha256=sha(HERE/'r6_probe_protocol.json'),preregistration_sha256=sha(HERE/'PREREGISTRATION.md'),terminal_labels_read_before_freeze=False,prior_round_outcome_exposure_disclosed=True))
    print(json.dumps(clean({k:v for k,v in audit.items() if k not in ['unknowns','per_statistic_distribution']})))
if __name__=='__main__':
    command=sys.argv[1]
    if command=='worker':worker(int(sys.argv[2]),int(sys.argv[3]))
    else:globals()[command]()
