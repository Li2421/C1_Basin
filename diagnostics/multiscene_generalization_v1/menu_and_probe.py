"""Post-capability menus and outcome-free200-step behavior measurements."""
from common import *
from design_generated import identities,ARMS
from registry import register

def eligible_families():
    return {f:v['eligible'] for f,v in read(HERE/'ALL_CAPABILITY_FINAL.json')['families'].items() if v['family_passed']}

def prepare(phase):
    assert phase in ['pilot_full','probe','evaluation_AF','evaluation_G']
    cat=read(HERE/'CATALOG_FINAL.json')['catalog'];rows=[];regs=[]
    for family,names in eligible_families().items():
        if phase=='evaluation_AF' and family=='G':continue
        if phase=='evaluation_G' and family!='G':continue
        role='pilot' if phase=='pilot_full' else 'probe' if phase=='probe' else 'evaluation'
        for name in names:
            cp=HERE/'policies'/name/'best.msgpack'
            for s in cat[family]:
                if s['role']!=role:continue
                arms=['F10','F13'] if phase=='probe' else ARMS[2:] if phase=='pilot_full' else ARMS
                seeds=[0] if phase in ['probe','pilot_full'] else [0,1]
                for arm in arms:
                    for seed in seeds:
                        c,r=identities(s,'learned',arm,seed,cp,s['H']);c['controller_name']=name
                        if phase=='probe':
                            # Probe identity declares no task outcome, so cannot be reused as an evaluation label.
                            import json
                            from shared_rollout_db.src.rollout_db import uid,canonical
                            for table,v in r:
                                if table=='controller_config':
                                    payload=json.loads(v['config_json']);payload['probe_steps']=200;payload['termination_flags_off']=True
                                    new=uid('ctl',payload);v.update(controller_uid=new,config_json=canonical(payload),success_semantics_version='UNLABELED_BEHAVIOR_NO_TASK_OUTCOMES')
                                    c['controller_uid']=new;c['rollout_uid']=uid('probe',dict(state=c['state_uid'],eta=c['eta_uid'],controller=new,seed=c['seed_key']))
                            c.update(probe_steps=200,contains_task_labels=False)
                        rows.append(c);regs+=r
    if phase=='evaluation_G':assert (HERE/'AF_ANALYSIS_FROZEN.receipt.json').exists() and (HERE/'FINAL_MODEL_FROZEN.receipt.json').exists()
    if rows:register(phase,rows,regs)
    return rows

def behavior_statistics(positions,actions,active,goals,radius,speed):
    n=len(goals);initial=np.linalg.norm(goals-positions[0],axis=1)
    progress=float(np.mean(initial-np.linalg.norm(goals-positions[-1],axis=1)>0))
    vel=np.linalg.norm(actions,axis=-1);streak=np.zeros(n,int);ever=np.zeros(n,bool)
    for row in vel:streak=np.where(row<.01,streak+1,0);ever|=streak>=40
    ii,jj=np.triu_indices(n,1);dist=np.linalg.norm(positions[1:,:,None]-positions[1:,None,:],axis=-1)[:,ii,jj]
    return [progress,float(ever.mean()),float(dist.min()/radius),float(vel.mean()/speed),float(np.mean(active))]

def probe_worker(index):
    from generated_worker import make
    from generated_runtime import action
    from policies import load_policy
    import fcntl
    folder=HERE/'probe';folder.mkdir(exist_ok=True);rows=read(HERE/'probe_MISSING_CASES.json')
    while True:
        with (folder/'cursor').open('a+') as f:
            fcntl.flock(f,fcntl.LOCK_EX);f.seek(0);i=int(f.read().strip() or 0);f.seek(0);f.truncate();f.write(str(i+1));f.flush()
        if i>=len(rows):break
        c=rows[i];out=folder/'raw'/f"{c['rollout_uid']}.json"
        if out.exists():continue
        env=make(c);policy=load_policy(c['checkpoint']);positions=[env.positions.copy()];actions=[];active=[];error=None
        for t in range(200):
            try:
                env.done=False # Explicitly disabled probe terminal flags; inherited physical step unchanged.
                u,a=action(env,policy,c,t,diagnostics=True);env.step(u,diagnose_stalls=False)
                positions.append(env.positions.copy());actions.append(u);active.append(a)
            except Exception as e:error=dict(type=type(e).__name__,message=str(e),step=t);break
        complete=len(actions)==200
        values=behavior_statistics(np.asarray(positions),np.asarray(actions),active,env.goals,.16,.5) if complete else None
        trace=folder/'traces'/f"{c['rollout_uid']}.npz";trace.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(trace,positions=np.asarray(positions),actions=np.asarray(actions),projection_active=active,goals=env.goals)
        r=dict(schema='round39_unlabeled_probe_v1',experiment_uid=EXP,case=c,physical_steps=len(actions),complete=complete,statistics=values,error=error,trace_path=str(trace),trace_sha256=sha(trace),terminal_labels_read=False)
        write(out,r);journal=cache_writer.append_journal([r],EXP,f'probe_{index}')
        assert journal.resolve().is_relative_to(ROOT/'shared_rollout_db/journals'/EXP)

def freeze_probe():
    rows=read(HERE/'probe_CASES.json');values={};index=[]
    for c in rows:
        path=HERE/'probe/raw'/f"{c['rollout_uid']}.json";r=read(path)
        assert not r['terminal_labels_read'] and not any(k in r for k in ['success','outcome','termination'])
        if r['complete']:values.setdefault(c['controller_name'],[]).append(r['statistics'])
        index.append(dict(raw_path=str(path),raw_sha256=sha(path),complete=r['complete'],trace_sha256=r['trace_sha256']))
    summary={n:np.mean(v,axis=0) for n,v in values.items()}
    write(HERE/'PROBE_FEATURES.json',dict(features=summary,index=index,G_features_only_for_prediction=True))
    seal('PROBE_FEATURES_FROZEN.receipt.json',[HERE/'PROBE_FEATURES.json',HERE/'menu_and_probe.py'],before_target_labels=True)

def postflight_probe():
    from shared_rollout_db.src.planner import preflight
    rows=read(HERE/'probe_CASES.json');missing=[];incomplete=[]
    for c in rows:
        path=HERE/'probe/raw'/f"{c['rollout_uid']}.json"
        if not path.exists():missing.append(c['rollout_uid']);continue
        r=read(path);assert sha(r['trace_path'])==r['trace_sha256']
        if not r['complete']:incomplete.append(c['rollout_uid'])
    write(HERE/'probe_POSTFLIGHT.json',dict(planner=preflight(HERE/'probe_PLANNED_ROLLOUTS.json'),unlabeled=True,
        evidence_class='SUMMARY_ONLY; no task outcome rows permitted',missing_raw=missing,incomplete_preserved=incomplete))
    assert not missing

if __name__=='__main__':probe_worker(int(sys.argv[2])) if sys.argv[1]=='worker' else prepare(sys.argv[1])
