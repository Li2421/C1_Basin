"""Outcome-free fixed200-step probes; five statistics match round34 exactly."""
from common import *
import fcntl

def behavior_statistics(positions,actions,active,goals,radius,speed):
    n=len(goals);initial=np.linalg.norm(goals-positions[0],axis=1)
    progress=float(np.mean(initial-np.linalg.norm(goals-positions[-1],axis=1)>0))
    vel=np.linalg.norm(actions,axis=-1);streak=np.zeros(n,int);ever=np.zeros(n,bool)
    for row in vel:streak=np.where(row<.01,streak+1,0);ever|=streak>=40
    ii,jj=np.triu_indices(n,1);dist=np.linalg.norm(positions[1:,:,None]-positions[1:,None,:],axis=-1)[:,ii,jj]
    return [progress,float(ever.mean()),float(dist.min()/radius),float(vel.mean()/speed),float(np.mean(active))]

def worker(stage,index):
    from multiagent_worker40 import make
    from nominal39_callback import load_policy
    from generated_runtime import action
    assert read(HERE/'REPRODUCTION_GATE.json')['passed']
    assert read(HERE/'ROUND39_GEOMETRY_EQUIVALENCE.json')['passed']
    folder=HERE/stage;folder.mkdir(exist_ok=True);rows=read(HERE/f'{stage}_MISSING_CASES.json');cache={}
    while True:
        with (folder/'cursor').open('a+') as f:
            fcntl.flock(f,fcntl.LOCK_EX);f.seek(0);i=int(f.read().strip() or 0);f.seek(0);f.truncate();f.write(str(i+1));f.flush()
        if i>=len(rows):break
        c=rows[i];out=folder/'raw'/f"{c['rollout_uid']}.json"
        if out.exists():continue # Incomplete observations remain unknown, never rerun.
        env=make(c);cp=c['checkpoint']
        if cp not in cache:cache[cp]=load_policy(cp)
        policy=cache[cp];positions=[env.positions.copy()];actions=[];active=[];error=None
        for t in range(200):
            try:
                env.done=False # Disable probe termination only, native step remains inherited.
                u,a=action(env,policy,c,t,diagnostics=True);env.step(u,diagnose_stalls=False)
                positions.append(env.positions.copy());actions.append(u);active.append(a)
            except Exception as e:error=dict(type=type(e).__name__,message=str(e),step=t);break
        complete=len(actions)==200
        values=behavior_statistics(np.asarray(positions),np.asarray(actions),active,env.goals,env.config.agent_radius,env.config.max_speed) if complete else None
        trace=folder/'traces'/f"{c['rollout_uid']}.npz";trace.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(trace,positions=np.asarray(positions),actions=np.asarray(actions),projection_active=active,goals=env.goals)
        r=dict(schema='round40_unlabeled_probe_v1',experiment_uid=EXP,case=c,physical_steps=len(actions),complete=complete,statistics=values,error=error,trace_path=str(trace),trace_sha256=sha(trace),terminal_labels_read=False,jax_enable_x64=False)
        write(out,r);journal=cache_writer.append_journal([r],EXP,f'{stage}_{index}')
        assert journal.resolve().is_relative_to(ROOT/'shared_rollout_db/journals'/EXP)
        print(json.dumps(dict(stage=stage,index=index,case=i,physical_steps=len(actions),complete=complete)),flush=True)
    write(folder/f'worker_{index}.json',dict(complete=True))

if __name__=='__main__':worker(sys.argv[1],int(sys.argv[2]))
