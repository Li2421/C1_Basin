"""Only action-index eta changes; exact E8 native FF runtime and event logic."""
from core import *
import time
from shared_rollout_db.src.cache_writer import append_journal

def augmented_initial(env):
    if hasattr(env,'augmented_state'):return env.augmented_state()
    # Read every strict-monitor attribute from the old Toy simulator, without
    # mutating reset, progress history, events or the controller.
    keys=['distance_history','candidate_since','stuck_timer','max_stuck_timer','ever_candidate_deadlock','first_success_step','first_deadlock_step','first_wall_collision_step','first_agent_collision_step','done']
    return dict(env.snapshot(),**{k:getattr(env,k) for k in keys})

def execute(phase,index,workers):
    prereg()
    if phase=='main':gate()
    ppath=HERE/('gate_protocol.json' if phase=='gate' else 'protocol.json');p=read(ppath);allowed={p['run_sha256']}
    if (HERE/'IO_REPAIR.receipt.json').exists():
        repair=read(HERE/'IO_REPAIR.receipt.json');assert sha(HERE/'run_v1_snapshot_interface_failure.py')==repair['original_run_sha256'];allowed.add(repair['corrected_run_sha256'])
    if (HERE/'IO_REPAIR2.receipt.json').exists():
        repair2=read(HERE/'IO_REPAIR2.receipt.json');assert sha(HERE/'run_v2_initial_file_race.py')==repair2['before_sha256'];allowed.add(repair2['after_sha256'])
    assert sha(__file__) in allowed and sha(HERE/'core.py')==p['core_sha256'];assert read(HERE/f'{phase}_CACHE_PREFLIGHT.json')['summary']['ambiguous']==0
    for f,h in p['source_hashes'].items():assert sha(f)==h,f
    import jax
    from diagnostics.controller_q_diversity_v1.runtime import native_runtime
    from diagnostics.controller_flow_endpoint_probe_v1.runtime import FlowEndpointMatchedField
    from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for
    rows=read(HERE/f'{phase}_MISSING_CASES.json');perm=np.random.default_rng(202610131320).permutation(len(rows));rows=[rows[int(k)] for k in perm[index::workers]];states={s['state_uid']:s for s in p['states']};bases={};runtimes={};seconds=[];buffer=[];completed=0
    write(HERE/phase/f'START_AUTHORIZATION_{index}.json',dict(utc=utc(),preregistration_receipt_sha256=sha(HERE/'PREREGISTRATION.receipt.json'),phase=phase,worker=index,gate_sha256=sha(HERE/'CONSISTENCY_GATE.json') if phase=='main' else None,protocol_sha256=sha(ppath)))
    for c in rows:
        dest=HERE/phase/'raw'/f"{c['rollout_uid']}.json"
        if dest.exists():buffer.append(read(dest));continue
        sc=c['scene'];name=c['base_controller_name'];v=p['base_profiles'][sc][name]['payload'];x64=v['jax_enable_x64']
        from adapter import FourFlowAdapter
        jax.config.update('jax_enable_x64',False)
        if (sc,name) not in runtimes:
            base=FourFlowAdapter(v['flow_checkpoint_path'])
            assert sha(base.checkpoint)==v['flow_checkpoint_sha256']
            assert clean(base.config.to_dict())==p['environments'][sc]['config']['config']
            jax.config.update('jax_enable_x64',False)
            runtimes[sc,name]=FlowEndpointMatchedField(base,'isotropic')
        rt=runtimes[sc,name];s=states[c['state_uid']];env=rt.make_env(s);initial=clean(augmented_initial(env));initialsha=hashlib.sha256(json.dumps(initial,sort_keys=True,separators=(',',':')).encode()).hexdigest();initialpath=HERE/'actual_initial_states'/f"{initialsha}_{os.getpid()}.json"
        if not initialpath.exists():write(initialpath,initial)
        pos=[env.positions.copy()];vel=[env.velocities.copy()];actions=[];clear=[];events=[];first=None;err=None;term='timeout';calls=0;steps=0;t0=time.monotonic()
        for t in range(rt.config.max_steps):
            try:
                assert bool(jax.config.x64_enabled)==x64;eta=c['eta_a'] if t<c['switch_step'] else c['eta_b'];a,_,n=rt.action(env,key_for(s,c['seed'],t),np.asarray(eta),'field');calls+=n;steps+=1
                if first is None:first=np.asarray(a).tolist()
                _,_,done,info=env.step(a);term=info['termination'];pos.append(env.positions.copy());vel.append(env.velocities.copy());actions.append(a.copy());cfg=rt.config
                wall=float(info.get('min_swept_wall_distance',np.inf))-getattr(cfg,'wall_collision_margin',getattr(cfg,'collision_margin',0.));pair=float(info.get('min_swept_agent_distance',np.min(info.get('swept_agent_distances',[np.inf]))))-cfg.agent_collision_margin
                if sc=='ring_exchange':wall=min(float(np.min(info['swept_obstacle_distance'])),float(np.min(info['swept_outer_distance'])))-cfg.collision_margin
                clear.append(min(wall,pair));events.append(dict(step=t+1,termination=term,candidate_deadlock=bool(info.get('candidate_deadlock',False)),deadlock=bool(info.get('deadlock',False)),stuck_timer=info.get('stuck_timer')))
                if done:break
            except Exception as ex:err=dict(step=t,type=type(ex).__name__,message=str(ex)[:500]);term='numerical_failure';break
        summary=env.summary();collision=any(summary.get(k,False) for k in ['wall_collision','obstacle_collision','outer_collision','agent_collision']);success=int(summary['collision_free_success'] and term=='success' and err is None);physical=len(pos)-1
        meta=dict(batch_id=EXP,scenario='toy_giveway' if sc=='toy_give_way' else sc,rollout_id=c['rollout_uid'],controller=c['controller_uid'],checkpoint=dict(path=v['flow_checkpoint_path'],sha256=v['flow_checkpoint_sha256']),seed=c['seed'],source_record=str(dest),initial_state_sha256=initialsha,augmented_initial_state_path=str(initialpath),augmented_initial_state_file_sha256=sha(initialpath),config=rt.config.to_dict(),start_step=0,episode_steps=physical,complete=bool(err is None and env.done and physical==env.step_count),termination=term,collision=bool(collision),safety_enabled=True,deadlock_detected=bool(term=='deadlock'),deadlock_criterion=clean({k:val for k,val in rt.config.to_dict().items() if k in ['detection_protocol','progress_window_seconds','deadlock_hold_seconds','progress_epsilon','speed_epsilon_fraction','terminate_on_deadlock']}) if sc in ['toy_give_way','double_bottleneck'] else None,arm=c['arm'],switch_step=c['switch_step'],eta_a=c['eta_a'],eta_b=c['eta_b'])
        trace=HERE/phase/'traces'/f"{c['rollout_uid']}.npz";trace.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(trace,positions=np.asarray(pos),velocities=np.asarray(vel),actions=np.asarray(actions),steps=np.arange(physical+1),swept_clearance=np.asarray(clear),goals=env.goals,walls=np.asarray(getattr(env,'walls',np.empty((0,2,2)))),metadata_json=np.array(json.dumps(clean(meta))))
        r=dict(schema='round13_schedule_seed_v1',experiment_uid=EXP+'__gate' if phase=='gate' else EXP,protocol_sha256=sha(ppath),case=c,**{k:c[k] for k in ['state_uid','eta_uid','controller_uid','seed_key']},success=success,deadlock=int(term=='deadlock'),timeout=int(term=='timeout'),collision=int(collision),numerical_failure=int(err is not None),outcome=term,episode_length=steps,physical_steps=physical,elapsed_seconds=time.monotonic()-t0,first_action=first,projection_calls=calls,error=err,summary=summary,jax_enable_x64=bool(jax.config.x64_enabled),execution_code_sha256=sha(__file__),trace_path=str(trace),trace_sha256=sha(trace),checkpoint_sha256=v['flow_checkpoint_sha256'],events=events,minimum_swept_clearance=min(clear) if clear else None)
        assert not (success and (collision or err));write(dest,r);buffer.append(r);seconds.append(r['elapsed_seconds']);completed+=1
        if len(buffer)>=8:append_journal(clean(buffer),r['experiment_uid'],f'{phase}_{index}');buffer=[]
        if completed%8==0:print(json.dumps(dict(phase=phase,worker=index,completed=completed,assigned=len(rows),mean_seconds=float(np.mean(seconds)),last_seconds=seconds[-1])),flush=True)
    if buffer:append_journal(clean(buffer),EXP+'__gate' if phase=='gate' else EXP,f'{phase}_{index}')
    write(HERE/phase/f'worker_{index}.json',dict(utc=utc(),complete=True,assigned=len(rows),new=completed,seconds=sum(seconds)))
if __name__=='__main__':execute(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]))
