"""Frozen FF runtime: the only intervention is eta at action index t."""
from core import *
import time,types,subprocess
from shared_rollout_db.src.cache_writer import append_journal as original_append_journal
def append_journal(records,*args,**kwargs):return original_append_journal(clean(records),*args,**kwargs)
def queued_cases(rows,phase):
    """Durable wave queue; released case locks let surviving workers reclaim interruptions."""
    import fcntl
    qdir=HERE/phase/'queue';qdir.mkdir(parents=True,exist_ok=True);cursor=qdir/'next_index';locks=qdir/'locks';locks.mkdir(exist_ok=True)
    def cutoff():return False
    def done(c):return (HERE/phase/'raw'/f"{c['rollout_uid']}.json").exists() and (HERE/phase/'journaled'/f"{c['rollout_uid']}.json").exists()
    def claim_cursor():
        with cursor.open('a+') as f:
            fcntl.flock(f,fcntl.LOCK_EX);f.seek(0);v=int(f.read().strip() or '0');f.seek(0);f.truncate();f.write(str(v+1));f.flush();fcntl.flock(f,fcntl.LOCK_UN)
        return v
    def try_case(c,blocking):
        lock=(locks/c['rollout_uid']).open('a+')
        try:fcntl.flock(lock,fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:lock.close();return None
        return lock
    while not cutoff():
        i=claim_cursor()
        if i>=len(rows):break
        c=rows[i];lock=try_case(c,True)
        try:
            if not done(c):yield c
        finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    while not cutoff():
        missing=[c for c in rows if not done(c)]
        if not missing:return
        claimed=False
        for c in missing:
            if cutoff():return
            lock=try_case(c,False)
            if lock is None:continue
            try:
                if not done(c):claimed=True;yield c
            finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
        if not claimed:time.sleep(2)

class SafetyTracker:
    def __init__(self,base):self.base=base;self.active=False
    def __getattr__(self,k):return getattr(self.base,k)
    def project(self,target):
        a,d=self.base.project(target);self.active|=np.linalg.norm(np.asarray(a)-target)>1e-6;return a,d
    def project_field_target(self,target):
        a,d=self.base.project_field_target(target);self.active|=np.linalg.norm(np.asarray(a)-target)>1e-6;return a,d

def make_runtime(name,p,cache):
    import jax
    from diagnostics.controller_q_transfer_round3_v1.flow_adapter import RingFlowAdapter
    from diagnostics.controller_q_diversity_v1.runtime import native_runtime
    jax.config.update('jax_enable_x64',False)
    if name in cache:return cache[name]
    v=p['base_profiles'][name]['payload']
    rt=native_runtime('ring_exchange') if name=='base_u_v10_local_macflow' else RingFlowAdapter(v['flow_checkpoint_path'],read(v['adapter_manifest_path']),v['observation_frame'])
    assert not jax.config.x64_enabled and sha(rt.checkpoint)==v['flow_checkpoint_sha256']
    original=rt.safety
    def observed_safety(self,env):
        self.last_safety_tracker=SafetyTracker(original(env));return self.last_safety_tracker
    rt.safety=types.MethodType(observed_safety,rt);rt.last_safety_tracker=None;cache[name]=rt;return rt

def execute(rows,phase,index):
    prereg();p=read(HERE/'protocol.json');assert sha(__file__)==p['runner_sha256'] and sha(HERE/'core.py')==p['core_sha256']
    if phase=='gate':features_frozen();assert (HERE/'STAGE0_SUMMARY.md').exists()
    elif phase=='train':train_labels_allowed()
    elif phase=='test':test_labels_allowed()
    else:raise ValueError(phase)
    assert read(HERE/f'{phase}_CACHE_PREFLIGHT.json')['summary']['ambiguous']==0
    if phase in ('train','test'):
        sq=subprocess.run(['squeue','-h','-u','zhihan','-o','%i|%t|%C|%b|%j'],capture_output=True,text=True,check=True).stdout.splitlines()
        running=[x for x in sq if '|R|' in x];cpu=sum(int(x.split('|')[2]) for x in running);gpu=sum('shard:' in x for x in running)
        write(HERE/phase/f'RESOURCE_worker_{index}.json',dict(utc=utc(),cpu_shards=cpu,gpu_shards=gpu,squeue=running))
        assert cpu<=18
        assert gpu<=2
    import jax
    from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for
    states={s['state_uid']:s for s in p['states']};cache={};buffer=[];new=0;seconds=[];start=time.time();tag=f'{phase}_{index}'
    for c in queued_cases(rows,phase) if phase in ('train','test') else rows:
        out=HERE/phase/'raw'/f"{c['rollout_uid']}.json";out.parent.mkdir(parents=True,exist_ok=True)
        if out.exists():
            r=read(out);assert r['case']==c and r['protocol_sha256']==sha(HERE/'protocol.json');buffer.append(r)
            if len(buffer)>=1:append_journal(buffer,EXP,tag);buffer=[]
            if phase in ('train','test'):write(HERE/phase/'journaled'/f"{c['rollout_uid']}.json",dict(utc=utc(),raw_sha256=sha(out),recovered_existing_raw=True))
            continue
        name=c['base_controller_name'];rt=make_runtime(name,p,cache);s=states[c['state_uid']];env=rt.make_env(s);initial_hash=hashlib.sha256(json.dumps(clean(env.augmented_state()),sort_keys=True,separators=(',',':')).encode()).hexdigest();before=time.monotonic();eta_a=np.asarray(c['eta_a'],dtype=np.float64);eta_b=np.asarray(c['eta_b'],dtype=np.float64);t=c['switch_step']
        pos=[env.positions.copy()];vel=[env.velocities.copy()];active=[];clearance=[];term='timeout';error=None;steps=0;calls=0;first=None;switch_snapshot=None
        for step in range(rt.config.max_steps):
            if step==t:
                switch_snapshot=dict(step=step,positions=env.positions.copy(),velocities=env.velocities.copy(),goals=env.goals.copy(),alive=not env.done)
            try:
                eta=eta_a if step<t else eta_b
                action,_,n=rt.action(env,key_for(s,c['seed'],step),eta,'field');calls+=n;steps+=1
                if first is None:first=np.asarray(action).tolist()
                _,_,done,info=env.step(action);term=info['termination'];pos.append(env.positions.copy());vel.append(env.velocities.copy());active.append(bool(rt.last_safety_tracker.active))
                cfg=rt.config;clearance.append(min(float(np.min(info['swept_obstacle_distance']))-cfg.collision_margin,float(np.min(info['swept_outer_distance']))-cfg.collision_margin,float(np.min(info['swept_agent_distances']))-cfg.agent_collision_margin))
                if done:break
            except Exception as exc:
                error=dict(step=step,type=type(exc).__name__,message=str(exc)[:500]);term='numerical_failure';break
        summary=env.summary();collision=any(summary.get(k,False) for k in ('wall_collision','obstacle_collision','outer_collision','agent_collision'));pos=np.asarray(pos);vel=np.asarray(vel);physical_steps=len(pos)-1;config=rt.config.to_dict();sample=np.unique(np.r_[np.arange(0,len(pos),10),physical_steps,min(t,physical_steps)]).astype(int)
        trace=HERE/phase/'traces'/f"{c['rollout_uid']}.npz";trace.parent.mkdir(parents=True,exist_ok=True)
        meta=dict(batch_id=EXP,scenario='ring_exchange',rollout_id=c['rollout_uid'],controller=c['controller_uid'],checkpoint=dict(path=str(rt.checkpoint),sha256=p['base_profiles'][name]['payload']['flow_checkpoint_sha256']),seed=c['seed'],source_record=str(out),initial_state_sha256=initial_hash,config=config,start_step=0,episode_steps=physical_steps,complete=bool(error is None and env.done and physical_steps==env.step_count),termination=term,collision=bool(collision),safety_enabled=True,deadlock_detected=False,eta_a=c['eta_a'],eta_b=c['eta_b'],switch_step=t,parent_controller=name)
        arrays=dict(positions=pos,velocities=vel,steps=np.arange(len(pos)),projection_active=np.asarray(active,bool),swept_clearance=np.asarray(clearance,float),goals=env.goals,walls=np.empty((0,2,2)),sample_steps=sample,sample_positions=pos[sample],sample_velocities=vel[sample],sample_projection_active=np.array([False if i==0 else active[i-1] for i in sample]),metadata_json=np.array(json.dumps(clean(meta))))
        if switch_snapshot is not None:arrays.update(switch_positions=switch_snapshot['positions'],switch_velocities=switch_snapshot['velocities'],switch_goals=switch_snapshot['goals'])
        np.savez_compressed(trace,**arrays)
        record=dict(schema='ring_controller_population_seed_v1',experiment_uid=EXP,protocol_sha256=sha(HERE/'protocol.json'),case=c,**{k:c[k] for k in ('state_uid','eta_uid','controller_uid','seed_key')},success=int(summary['collision_free_success'] and term=='success' and error is None),deadlock=int(term=='deadlock'),timeout=int(term=='timeout'),collision=int(collision),numerical_failure=int(error is not None),episode_length=steps,physical_steps=physical_steps,outcome=term,elapsed_seconds=time.monotonic()-before,first_action=first,projection_calls=calls,error=error,summary=summary,execution_semantics_version='scene_precision_guard_eta_schedule_v1',jax_enable_x64=bool(jax.config.x64_enabled),execution_code_sha256=sha(__file__),flow_adapter_sha256=p['flow_adapter_sha256'],checkpoint_sha256=p['base_profiles'][name]['payload']['flow_checkpoint_sha256'],trace_path=str(trace),trace_sha256=sha(trace),switch_reached=switch_snapshot is not None,switch_snapshot=switch_snapshot,minimum_swept_clearance=float(np.min(clearance)) if len(clearance) else None)
        assert not (record['success'] and (record['collision'] or record['numerical_failure']));assert not record['jax_enable_x64'];write(out,record);buffer.append(record);new+=1;seconds.append(record['elapsed_seconds'])
        if len(buffer)>=1:append_journal(buffer,EXP,tag);buffer=[]
        if phase in ('train','test'):write(HERE/phase/'journaled'/f"{c['rollout_uid']}.json",dict(utc=utc(),raw_sha256=sha(out),recovered_existing_raw=False))
        if new%8==0:print(json.dumps(dict(phase=phase,worker=index,new=new,assigned=len(rows),last_controller=name,last_seconds=seconds[-1],wall_seconds=time.time()-start)),flush=True)
    if buffer:append_journal(buffer,EXP,tag)
    complete=all((HERE/phase/'raw'/f"{c['rollout_uid']}.json").exists() and (phase=='gate' or (HERE/phase/'journaled'/f"{c['rollout_uid']}.json").exists()) for c in rows)
    write(HERE/phase/f'worker_{index}.json',dict(utc=utc(),complete=complete,assigned=len(rows),new=new,episode_cpu_seconds=sum(seconds),mean_seconds=float(np.mean(seconds)) if seconds else None,wall_seconds=time.time()-start))
    if not complete:raise RuntimeError('Wave incomplete; inspect worker/queue and resume missing cases')
if __name__=='__main__':
    phase=sys.argv[1];index=int(sys.argv[2]);workers=int(sys.argv[3]);rows=read(HERE/f'{phase}_MISSING_CASES.json');execute(rows if phase in ('train','test') else rows[index::workers],phase,index)
