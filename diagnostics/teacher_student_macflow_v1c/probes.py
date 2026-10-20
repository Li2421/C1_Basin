"""Unlabeled R5 probes for private one-step students through frozen D-local."""
from core import *
from audit_utils import source_inventory
from registration import *
from student import Student
from pipeline import decentralized_action,LocalSafety,local_basis
from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for
from shared_rollout_db.src.planner import preflight
from dataclasses import replace
import jax
jax.config.update('jax_enable_x64',False)
def design():
    names=read(HERE/'CAPABILITY_GATE.json')['eligible'];base=read(D/'decentralized_pipeline_v1/main_protocol.json');states=[normalized_state(s) for s in read(D/'controller_flow_blend_v1/stage_2/probes/protocol.json')['states']];rows=[];registered=[state_row(s) for s in states];ts=read(HERE/'TEACHERS.json')
    for n in names:
        i=base['controller_order'].index(n);ck=Path(read(HERE/'STUDENTS.json')[str(i)]['checkpoint']);payload=dict(student=n,checkpoint_sha256=sha(ck),pipeline_sha256=sha(HERE/'pipeline.py'),probe_sha256=sha(__file__),success_semantics='UNLABELED_SHORT_PROBE',probe_steps=200,terminate_on_success=False,terminate_on_collision=False,own_observation=True,private_robot_noise=True,jax_enable_x64=False);ctl=uid('ctl',payload);scid=states[0]['scenario_uid'];registered.append(('controller_config',dict(controller_uid=ctl,scenario_uid=scid,flow_checkpoint_sha256=sha(ck),config_json=canonical(payload),compatibility_quality='EXACT_PROFILE')))
        for si,s in enumerate(states):
            for ei in [10,13]:
                eid,ev,hx=eta_identity(base['eta'][ei]);registered.append(('eta',dict(eta_uid=eid,eta1=ev[0],eta2=ev[1],eta3=ev[2],canonical_hex=hx)));sk=canonical(dict(future_index=0));rows.append(dict(probe_uid=uid('probe',dict(state=s['state_uid'],eta=eid,controller=ctl,seed=sk)),controller_name=n,student_index=i,state_index=si,state_uid=s['state_uid'],eta_index=ei,eta_uid=eid,eta=ev,controller_uid=ctl,seed_key=sk))
    register('probe_controllers',registered);write(HERE/'probe_protocol.json',dict(states=states,controller_order=names));write(HERE/'probe_CASES.json',rows);manifest=HERE/'probe_PLANNED_ROLLOUTS.json';write(manifest,dict(requests=[{k:c[k] for k in ['state_uid','eta_uid','controller_uid','seed_key']} for c in rows]));pf=preflight(manifest);assert pf['summary']['genuinely_missing']==len(rows) and pf['summary']['ambiguous']==0;write(HERE/'probe_CACHE_PREFLIGHT.json',pf);seal('probe_PREFLIGHT.receipt.json',[manifest,HERE/'probe_CACHE_PREFLIGHT.json',HERE/'probe_CASES.json',HERE/'probes.py'],before_submission=True,task_labels=False)
def activity(rt,env,key,eta):
    # Observe projection intervention by repeated deterministic queries only.
    nominal=rt.nominal(env,key)[1];fs=[LocalSafety(env.positions,i,env.config,1.5) for i in range(4)];safe=np.stack([f.project(nominal[i:i+1])[0][0] for i,f in enumerate(fs)]);target=safe.copy()
    for i in range(4):target[i]+=np.asarray(eta)@local_basis(env.positions,env.goals,safe,i,1.5,env.config.max_speed)
    action=np.stack([f.project(target[i:i+1])[0][0] for i,f in enumerate(fs)])
    return np.linalg.norm(safe-nominal)>1e-6 or np.linalg.norm(action-target)>1e-6

def worker(index,workers):
    assert (HERE/'probe_PREFLIGHT.receipt.json').exists();p=read(HERE/'probe_protocol.json');cache={}
    for c in read(HERE/'probe_CASES.json')[index::workers]:
        out=HERE/'probe/raw'/f"{c['probe_uid']}.json"
        if out.exists():continue
        n=c['controller_name']
        if n not in cache:cache[n]=Student(c['student_index'])
        rt=cache[n];s=p['states'][c['state_index']];env=rt.make_env(s);env.config=replace(env.config,terminate_on_success=False,terminate_on_collision=False);positions=[env.positions.copy()];actions=[];active=[];error=None
        for step in range(200):
            try:
                key=key_for(s,0,step);a,_=decentralized_action(rt,env,key,c['eta'],'local',np.zeros(4,int));flag=activity(rt,env,key,c['eta']);env.step(a);positions.append(env.positions.copy());actions.append(a.copy());active.append(flag)
            except Exception as e:error=dict(step=step,type=type(e).__name__,message=str(e)[:500]);break
        trace=out.with_suffix('.npz');trace.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(trace,positions=positions,actions=actions,projection_active=active,goals=env.goals)
        write(out,dict(case=c,complete=len(actions)==200,error=error,trace_path=str(trace),trace_sha256=sha(trace),terminal_labels_read=False,jax_enable_x64=bool(jax.config.x64_enabled)));print('probe',index,n,c['state_index'],c['eta_index'],flush=True)
    source_inventory(f'imports_probe_{index}.json')
def freeze():
    p=read(HERE/'probe_protocol.json');names=p['controller_order'];f=np.full((len(names),16,2,5),np.nan,np.float32);index=[];unknown=[]
    for c in read(HERE/'probe_CASES.json'):
        path=HERE/'probe/raw'/f"{c['probe_uid']}.json";r=read(path);assert not r['terminal_labels_read'] and not r['jax_enable_x64'];assert sha(r['trace_path'])==r['trace_sha256']
        if r['complete']:
            z=np.load(r['trace_path']);pos=z['positions'];acts=z['actions'];goals=z['goals'];initial=np.linalg.norm(goals-pos[0],axis=-1).sum();final=np.linalg.norm(goals-pos[-1],axis=-1).sum();ii,jj=np.triu_indices(4,1);dd=np.linalg.norm(pos[:,:,None]-pos[:,None,:],axis=-1)[:,ii,jj];speed=np.linalg.norm(acts,axis=-1);f[names.index(c['controller_name']),c['state_index'],[10,13].index(c['eta_index'])]=[(initial-final)/initial,z['projection_active'].mean(),dd.min(),speed.mean(),(speed.mean(-1)<.01).sum()]
        else:unknown.append(r)
        index.append(dict(case=c,raw_sha256=sha(path),trace_sha256=r['trace_sha256']))
    np.savez_compressed(HERE/'R5_FEATURES.npz',features=f.reshape(len(names),160),controller_names=names);write(HERE/'R5_RAW_INDEX.json',index);write(HERE/'probe_CACHE_POSTFLIGHT.json',dict(utc=utc(),complete_records=len(index),task_labels=0,unknown=unknown,semantic='unlabeled probes; do not insert synthetic task outcomes into rollout DB'));seal('R5_FROZEN.receipt.json',[HERE/'R5_FEATURES.npz',HERE/'R5_RAW_INDEX.json',HERE/'probes.py'],before_evaluation_labels=True)
if __name__=='__main__':
    if sys.argv[1]=='worker':worker(int(sys.argv[2]),int(sys.argv[3]))
    else:globals()[sys.argv[1]]()
