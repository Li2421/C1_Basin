"""Independent native solo evaluation with bitwise-equivalent shared field queries."""
from common import *
from geometry import GeneratedGeometry
from single_agent import SingleAgentGeneratedEnv
from bottleneck_family.scenario import Config
from navigation_field import NavigationGrid
from batch_fields39 import IndependentFields
from policies39 import load_policy
import fcntl,jax

def evaluate(envs,actors,keys):
    grid=NavigationGrid(envs[0].instance.geometry);fields=[grid.field(e.goals[0]) for e in envs];batch=IndependentFields(fields)
    buffers=[dict(positions=[e.positions.copy()],actions=[],wall_clearance=[]) for e in envs];errors=[None]*len(envs);active=list(range(len(envs)))
    while active:
        try:observed=batch.observed([envs[i] for i in active],active)
        except Exception as error:
            for i in active:errors[i]=dict(type=type(error).__name__,message=str(error))
            break
        continuing=[]
        for row,i in enumerate(active):
            env=envs[i];b=buffers[i]
            try:
                action=actors[i].from_observation(env,observed[row:row+1],jax.random.fold_in(keys[i],env.step_count));_,done,info=env.step(action,diagnose_stalls=False)
                b['positions'].append(env.positions.copy());b['actions'].append(action);b['wall_clearance'].append(info['min_swept_wall_clearance'])
                if not done:continuing.append(i)
            except Exception as error:errors[i]=dict(type=type(error).__name__,message=str(error))
        active=continuing
    return [dict(positions=np.asarray(b['positions']),actions=np.asarray(b['actions']).reshape(-1,1,2),wall_clearance=np.asarray(b['wall_clearance']),outcome=env.termination if error is None else 'numerical_failure',error=error) for env,b,error in zip(envs,buffers,errors)]

def worker(stage,index):
    assert read(HERE/'DESCRIPTIVE_BATCH40_TEST.json')['passed']
    cases=read(HERE/f'{stage}_MISSING_CASES.json');groups=read(HERE/f'{stage}_MISSING_GROUPS.json');folder=HERE/stage;folder.mkdir(exist_ok=True)
    while True:
        with (folder/'cursor').open('a+') as file:
            fcntl.flock(file,fcntl.LOCK_EX);file.seek(0);i=int(file.read().strip() or 0);file.seek(0);file.truncate();file.write(str(i+1));file.flush()
        if i>=len(groups):break
        selected=[cases[j] for j in groups[i] if not (folder/'raw'/f"{cases[j]['rollout_uid']}.json").exists()]
        if not selected:continue
        s=selected[0]['state'];geo=GeneratedGeometry(Config(**dict(selected[0]['config'],num_agents=4)),s['rectangles'])
        envs=[SingleAgentGeneratedEnv(geo,c['state']['positions'],c['state']['goals'],Config(**dict(c['config'],num_agents=4))) for c in selected]
        actors=[load_policy(c['checkpoint']) for c in selected];keys=[jax.random.fold_in(jax.random.PRNGKey(2026101740),int(hashlib.sha256(c['state_uid'].encode()).hexdigest()[:8],16)) for c in selected]
        try:records=evaluate(envs,actors,keys)
        except Exception as error:records=[dict(positions=np.asarray(c['state']['positions'])[None],actions=np.zeros((0,1,2)),wall_clearance=np.zeros(0),outcome='numerical_failure',error=dict(type=type(error).__name__,message=str(error))) for c in selected]
        for c,r in zip(selected,records):
            s=c['state'];trace=folder/'traces'/f"{c['rollout_uid']}.npz";trace.parent.mkdir(exist_ok=True);np.savez_compressed(trace,**{k:r[k] for k in ['positions','actions','wall_clearance']})
            term=r['outcome'];actions=r['actions'];result=dict(cache_identity_schema='registered_exact_uid_v1',experiment_uid=EXP,scenario='Generated '+s['family'],**{k:c[k] for k in ['scenario_uid','state_uid','controller_uid','eta_uid','rollout_uid','seed_key']},eta=c['eta'],future_index=c['seed'],future_root=2026101740,initial_positions=s['positions'],initial_velocities=s['velocities'],goals=s['goals'],flow_checkpoint_sha256=c['flow_checkpoint_sha256'],success=int(term=='success' and r['error'] is None),collision=int(term=='collision'),timeout=int(term=='timeout'),deadlock=0,numerical_failure=int(r['error'] is not None),outcome=term,error=r['error'],episode_length=len(actions),first_action=actions[0] if len(actions) else None,trace_path=str(trace),trace_sha256=sha(trace),case=c)
            write(folder/'raw'/f"{c['rollout_uid']}.json",result);journal=cache_writer.append_journal([clean(result)],EXP,f'descriptive40_{index}');assert journal.is_relative_to(ROOT/'shared_rollout_db/journals'/EXP)
        print(i,len(records),[r['outcome'] for r in records],flush=True)

if __name__=='__main__':worker(sys.argv[1],int(sys.argv[2]))
