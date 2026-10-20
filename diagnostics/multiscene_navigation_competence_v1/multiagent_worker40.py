"""Workers append journals only; design and merger stay in the parent process."""
from common import *
from dataclasses import replace
import fcntl,time
from geometry import GeneratedGeometry,GeneratedEnv
from single_agent import SingleAgentGeneratedEnv
from bottleneck_family.scenario import Config
from sequential_expert import SequentialExpert
from observation import observe,navigation_action

def make(case):
    s=case['state']; cfg=dict(case['config']);n=len(s['positions'])
    if n==1:cfg['num_agents']=2
    geometry=GeneratedGeometry(Config(**cfg),s['rectangles'])
    cls=SingleAgentGeneratedEnv if n==1 else GeneratedEnv
    return cls(geometry,s['positions'],s['goals'])

def execute(case,stage):
    env=make(case);before=time.monotonic();pos=[env.positions.copy()];actions=[];observations=[];clear=[];error=None;term='timeout'
    if case['kind']=='sequential_expert':
        try:
            plan=SequentialExpert(.25,.015).rollout(env)
            pos=list(plan['states']);actions=list(plan['actions']);term=plan['reason']
            if term not in ['success','collision','timeout']:error=dict(type='ReferenceUnresolved',message=term)
        except Exception as e:error=dict(type=type(e).__name__,message=str(e));term='numerical_failure'
    else:
        paths=[env.instance.geometry.path(p,g) for p,g in zip(env.positions,env.goals)]
        if any(p is None for p in paths):raise ValueError('registered navigation state has no path')
        policy=None
        if case['kind']=='learned':
            from nominal39_callback import load_policy
            policy=load_policy(case['checkpoint'])
        for step in range(env.config.max_steps):
            try:
                observations.append(observe(env))
                if policy is None:u=navigation_action(env,paths)
                else:
                    from generated_runtime import action
                    u=action(env,policy,case,step)
                _,done,info=env.step(u,diagnose_stalls=False)
                actions.append(np.asarray(u).copy());pos.append(env.positions.copy());clear.append(info['min_swept_wall_clearance']);term=info['termination']
                if done:break
            except Exception as e:error=dict(type=type(e).__name__,message=str(e),step=step);term='numerical_failure';break
    trace=HERE/stage/'traces'/f"{case['rollout_uid']}.npz";trace.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(trace,positions=np.asarray(pos),actions=np.asarray(actions).reshape((-1,env.num_agents,2)),
        observations=np.asarray(observations),goals=env.goals,walls=env.walls,wall_clearance=np.asarray(clear))
    s=case['state'];result=dict(cache_identity_schema='registered_exact_uid_v1',experiment_uid=EXP,scenario='Generated '+s['family'],
        **{k:case[k] for k in ['scenario_uid','state_uid','controller_uid','eta_uid','rollout_uid','seed_key']},
        eta=case['eta'],future_index=case['seed'],future_root=2026101739,
        initial_positions=s['positions'],initial_velocities=s['velocities'],goals=s['goals'],
        flow_checkpoint_sha256=case['flow_checkpoint_sha256'],success=int(term=='success' and not env.collided and error is None),
        collision=int(env.collided),timeout=int(term=='timeout'),deadlock=0,numerical_failure=int(error is not None),
        outcome=term,error=error,episode_length=len(actions),first_action=actions[0] if actions else None,
        trace_path=str(trace),trace_sha256=sha(trace),case=case,elapsed_seconds=time.monotonic()-before)
    return clean(result)

def worker(stage,index):
    prereg();assert read(HERE/'REPRODUCTION_GATE.json')['passed'];assert read(HERE/'ROUND39_GEOMETRY_EQUIVALENCE.json')['passed']
    rows=read(HERE/f'{stage}_MISSING_CASES.json');folder=HERE/stage;folder.mkdir(exist_ok=True)
    while True:
        with (folder/'cursor').open('a+') as f:
            fcntl.flock(f,fcntl.LOCK_EX);f.seek(0);i=int(f.read().strip() or 0);f.seek(0);f.truncate();f.write(str(i+1));f.flush()
        if i>=len(rows):break
        c=rows[i];path=folder/'raw'/f"{c['rollout_uid']}.json"
        if path.exists():continue
        result=execute(c,stage);write(path,result)
        journal=cache_writer.append_journal([result],EXP,f'{stage}_{index}')
        assert journal.resolve().is_relative_to(ROOT/'shared_rollout_db/journals'/EXP)
        print(json.dumps(dict(stage=stage,index=index,case=i,outcome=result['outcome'])),flush=True)
    write(folder/f'worker_{index}.json',dict(complete=True))

if __name__=='__main__':worker(sys.argv[1],int(sys.argv[2]))
