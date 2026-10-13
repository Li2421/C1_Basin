"""Static FF encounter responses, with no physical rollout or future preview."""
import argparse,json,time
from pathlib import Path
import numpy as np
from diagnostics.controller_ff_action_mediation_v1.study import HERE as LOCAL,FOUR,FULL,INDICES,read,write,freeze,sha

HERE=Path(__file__).resolve().parent

def prepare():
    p=read(LOCAL/'protocol.json');p.update(schema='ff_static_encounter_mediation_v1',
        hypothesis='Current FF action is insufficient. Test whether binding controller response to map/agent constraints, through synthetic static encounter probes, repairs the transferable mechanism.',
        conditions=['current_action_only','current_and_encounter'],
        architecture='Same per-agent fused encoder,64operator+5current+14encounter channels; encounter channels zeroed in control.',
        action_input='Reuse independent current-action features. Add per-agent map-contact and agent-contact FF query mean2/variance2/validfraction1/displacement2. Probe poses depend only on physical geometry, never critic predictions or labels.',
        keys=dict(root=2026101205,fold_in=[0,1],independent_of_label_seed_roots=True),
        map_probe='First near-contact with a map boundary along the focal goal ray; if none, nearest map-boundary contact. Clearance=radius+wall extent+margin+0.05m. Other robots stay at their original positions. Reject any invalid joint pose.',
        agent_probe='Move the closest other robot0.05m beyond the two-radius contact distance, ahead of focal robot along its goal direction. Preserve velocities. Reject any invalid joint pose.',
        probe_semantics='Counterfactual static poses, not simulated future states. Native environment distances/outside validate each pose. Environment.step forbidden. Only focal action is recorded.',
        numerical='All attempted query failures and masks retained; never silently drop a physical test state.',
        status='DEVELOPMENT privileged mechanistic diagnostic, not final interface or confirmation.')
    freeze(HERE/'protocol.json',p)


def map_clearance(points,obstacles,r):
    values=[]
    for o in obstacles:
        a,b=np.asarray(o['a']),np.asarray(o['b']);rad=o['radius']
        if o['curvature']:
            values.append(o['interior_sign']*(np.linalg.norm(points-a,axis=-1)-rad)-r)
        else:
            u=np.clip(np.sum((points-a)*(b-a),axis=-1)/max(np.sum((b-a)**2),1e-30),0,1)
            q=a+u[...,None]*(b-a);values.append(np.linalg.norm(points-q,axis=-1)-rad-r)
    return np.min(np.stack(values,-1),axis=-1)


def valid(env):
    wall,pair=env.distances()
    cfg=env.config;wm=getattr(cfg,'wall_collision_margin',getattr(cfg,'collision_margin',0.))
    return bool(not env.outside(env.positions).any() and np.min(wall)>wm+.005 and
                np.min(pair)>getattr(cfg,'agent_collision_margin',0.)+.005)


def probes(env,obstacles):
    p=env.positions.copy();g=env.goals.copy();v=env.velocities.copy();n=len(p);r=env.config.agent_radius
    wm=getattr(env.config,'wall_collision_margin',getattr(env.config,'collision_margin',0.));gap=.05+wm
    answer=[]
    for i in range(n):
        e=g[i]-p[i];e/=max(np.linalg.norm(e),1e-12);lat=np.array([-e[1],e[0]])
        ray=p[i]+np.linspace(0,1,257)[:,None]*(g[i]-p[i]);distance=map_clearance(ray,obstacles,r)
        hit=np.flatnonzero(distance<=gap)
        if len(hit):
            j=int(hit[0]);lo=max(j-1,0)/256;hi=j/256
            for _ in range(35):
                mid=(lo+hi)/2;q=p[i]+mid*(g[i]-p[i])
                if float(map_clearance(q[None],obstacles,r)[0])<=gap:hi=mid
                else:lo=mid
            q=p[i]+lo*(g[i]-p[i])
        else:
            choices=[]
            for o in obstacles:
                a,b=np.asarray(o['a']),np.asarray(o['b']);rad=o['radius']
                if o['curvature']:
                    normal=(p[i]-a)/max(np.linalg.norm(p[i]-a),1e-12)
                    q=a+rad*normal+o['interior_sign']*normal*(r+gap)
                else:
                    u=np.clip(np.dot(p[i]-a,b-a)/max(np.sum((b-a)**2),1e-30),0,1);closest=a+u*(b-a)
                    normal=(p[i]-closest)/max(np.linalg.norm(p[i]-closest),1e-12)
                    q=closest+normal*(r+rad+gap)
                choices.append((float(np.linalg.norm(q-p[i])),q))
            q=min(choices,key=lambda z:z[0])[1]
        pose=p.copy();pose[i]=q;env.positions[:]=pose;env.velocities[:]=v
        displacement=np.array([(q-p[i])@e,(q-p[i])@lat])/4
        answer.append(dict(agent=i,type=0,positions=pose.copy(),valid=valid(env),displacement=displacement))
        j=min((j for j in range(n) if j!=i),key=lambda j:(float(np.linalg.norm(p[j]-p[i])),j))
        pose=p.copy();pose[j]=p[i]+(2*r+.05+getattr(env.config,'agent_collision_margin',0.))*e
        env.positions[:]=pose
        displacement=np.array([(pose[j]-p[j])@e,(pose[j]-p[j])@lat])/4
        answer.append(dict(agent=i,type=1,positions=pose.copy(),valid=valid(env),displacement=displacement))
    env.positions[:]=p;env.velocities[:]=v
    return answer


def features(index,workers):
    import jax
    from diagnostics.controller_q_diversity_v1.runtime import native_runtime
    from diagnostics.controller_response_families_v1.runtime import EndpointMatchedField,FAMILIES
    from diagnostics.physical_state_critic_formal_v1.physical_entities import physical_scene
    p=read(FOUR/'protocol.json');bases={};runtimes={};stats=[];t0=time.monotonic()
    for si in range(index,len(p['states']),workers):
        state=p['states'][si];sc=state['scenario'];sid=state['state_uid']
        if sc not in bases:bases[sc]=native_runtime(sc)
        base=bases[sc];jax.config.update('jax_enable_x64',False)
        def forbidden_step(*a,**kw):raise RuntimeError('Physical step forbidden in encounter descriptor')
        base.env_cls.step=forbidden_step
        origin=base.make_env(state);poses=probes(origin,physical_scene(sc,state['physical'])['obstacles']);n=len(origin.positions)
        e=origin.goals-origin.positions;e/=np.maximum(np.linalg.norm(e,axis=-1,keepdims=True),1e-12);lat=np.stack((-e[:,1],e[:,0]),-1)
        for name in FAMILIES:
            path=HERE/'features'/sid/(name+'.npz')
            if path.exists():continue
            if (sc,name) not in runtimes:runtimes[sc,name]=EndpointMatchedField(base,name)
            rt=runtimes[sc,name];env=rt.make_env(state);values=np.zeros((16,n,2,7),np.float32)
            raw=np.full((16,n,2,2,2),np.nan);success=np.zeros((16,n,2,2),bool);errors=[]
            for probe in poses:
                if not probe['valid']:continue
                i,t=probe['agent'],probe['type'];env.positions[:]=probe['positions'];env.velocities[:]=origin.velocities
                assert valid(env)
                for j in INDICES:
                    for k in range(2):
                        key=jax.random.fold_in(jax.random.PRNGKey(2026101205),k)
                        try:
                            a,_,_=rt.action(env,key,np.asarray(p['eta'][j]),'field');raw[j,i,t,k]=a[i];success[j,i,t,k]=True
                        except Exception as ex:errors.append(dict(agent=i,type=t,eta=j,key=k,error=type(ex).__name__,message=str(ex)[:240]))
                    if success[j,i,t].any():
                        a=raw[j,i,t,success[j,i,t]];local=np.stack((a@e[i],a@lat[i]),-1)/rt.config.max_speed
                        values[j,i,t,:2]=local.mean(0);values[j,i,t,2:4]=local.var(0)
                    values[j,i,t,4]=success[j,i,t].mean();values[j,i,t,5:]=probe['displacement']
            path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,encounter_features=values.reshape(16,n,14),raw_actions=raw,query_success=success,physical_steps=0,
                probe_positions=np.asarray([z['positions'] for z in poses]),probe_valid=np.asarray([z['valid'] for z in poses]))
            stats.append(dict(state_uid=sid,controller=name,valid_probes=sum(z['valid'] for z in poses),queries=sum(z['valid'] for z in poses)*8,errors=errors))
    write(HERE/f'features_worker{index}.json',dict(complete=True,physical_steps=0,seconds=time.monotonic()-t0,rows=stats))


def assemble():
    old=np.load(LOCAL/'dataset.npz');data={k:old[k] for k in old.files};values=[];query_count=0;failures=0;valid_count=0
    for sid,name in zip(data['state_uid'],data['controller']):
        x=np.load(HERE/'features'/str(sid)/(str(name)+'.npz'));assert int(x['physical_steps'])==0
        v=x['encounter_features'];values.append(np.pad(v,((0,0),(0,4-v.shape[1]),(0,0))))
        valid_count+=int(x['probe_valid'].sum());query_count+=int(x['probe_valid'].sum())*8
        failures+=int(x['probe_valid'].sum())*8-int(x['query_success'].sum())
    data['action_feature']=np.concatenate((data['action_feature'],np.asarray(values)),-1)
    np.savez_compressed(HERE/'dataset.npz',**data)
    freeze(HERE/'dataset_audit.json',dict(dataset_sha256=sha(HERE/'dataset.npz'),rows=len(values),valid_counterfactual_probes=valid_count,
        static_queries=query_count,query_failures=failures,physical_steps=0,new_rollouts=0,old_labels_unchanged=True))
    print(json.dumps(read(HERE/'dataset_audit.json')),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','features','assemble']);p.add_argument('--index',type=int,default=0);p.add_argument('--workers',type=int,default=16);a=p.parse_args()
    if a.stage=='features':features(a.index,a.workers)
    else:globals()[a.stage]()
