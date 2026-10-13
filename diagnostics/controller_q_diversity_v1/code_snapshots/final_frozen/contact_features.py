"""Static, geometrically valid encounter queries; never advance a simulator."""
import argparse
import numpy as np
import jax
from .experiment import HERE,read,write,sha
from .runtime import native_runtime,AnalyticField


def valid_positions(p,cfg,scene):
    r=cfg.agent_radius
    distances=np.linalg.norm(p[:,None]-p[None,:],axis=-1)+np.eye(len(p))*100
    if distances.min()<2*r+.01:return False
    if scene=='toy_give_way':
        pad=r+cfg.wall_radius+cfg.wall_collision_margin
        return bool(np.all(np.abs(p[:,0])+pad<=cfg.corridor_half_length) and np.all(np.abs(p[:,1])+pad<=cfg.corridor_width/2))
    radius=np.linalg.norm(p,axis=-1)
    return bool(np.all(radius>=cfg.obstacle_radius+r+.01) and np.all(radius<=cfg.outer_radius-r-.01))


def measure(rt,state):
    env=rt.make_env(state);p=env.positions.copy();v=env.velocities.copy();g=env.goals.copy()
    heading=g-p;heading/=np.maximum(np.linalg.norm(heading,axis=1,keepdims=True),1e-12)
    lateral=np.stack((-heading[:,1],heading[:,0]),-1);result=[]
    keys=[jax.random.fold_in(jax.random.PRNGKey(2026100942),i) for i in range(4)]
    calls=valid_count=0
    for i in range(len(p)):
        values=[]
        for distance in (2*rt.config.agent_radius+.05,.65):
            for speed in (0.,.5,-.5):
                env.positions[:]=p;env.velocities[:]=v;env.velocities[i]=speed*rt.config.max_speed*heading[i]
                base=np.asarray([rt.nominal(env,key)[1][i] for key in keys]);calls+=4
                responses=[]
                for j in range(len(p)):
                    if i==j:continue
                    probe=p.copy();probe[j]=p[i]+distance*heading[i]
                    if not valid_positions(probe,rt.config,state['scenario']):continue
                    env.positions[:]=probe
                    near=np.asarray([rt.nominal(env,key)[1][i] for key in keys]);calls+=4;valid_count+=1
                    difference=np.mean(near-base,axis=0)/rt.config.max_speed
                    responses.append([difference@heading[i],difference@lateral[i]])
                response=np.mean(responses,axis=0) if responses else np.zeros(2)
                values.extend([float(response[0]),float(response[1]),len(responses)/(len(p)-1)])
        result.append(values)
    out=np.asarray(result,np.float32);assert out.shape==(len(p),18) and np.isfinite(out).all()
    return out,dict(function_calls=calls,valid_probe_positions=valid_count,simulator_steps=0)


def run(probe_only=False):
    p=read(HERE/'protocol.json');bases={};runtimes={};count=0;stats=[]
    if not probe_only and (HERE/'extra_controller_profiles.json').exists():
        for scene,profiles in read(HERE/'extra_controller_profiles.json').items():p['controllers'][scene].update(profiles)
    for s in p['states']:
        if probe_only:
            if s['study_split']!='train' or s['study_index']>=64:continue
            names=[f'analytic{i}' for i in range(4)]
        elif s['study_split']=='final':names=['analytic6','analytic7']
        elif s['study_split']=='validation':names=[f'analytic{i}' for i in range(6)]+['flow']
        else:
            names=[f'analytic{i}' for i in range(4)]+['flow']
            if s['study_index']<64:names+=[f'analytic{i}' for i in range(8,12)]
        scene=s['scenario']
        if scene not in bases:bases[scene]=native_runtime(scene)
        for name in names:
            dest=HERE/'contact_features'/s['state_uid']/(name+'.npz')
            if dest.exists():continue
            ctl=p['controllers'][scene][name];jax.config.update('jax_enable_x64',bool(ctl['payload']['jax_enable_x64']))
            if (scene,name) not in runtimes:runtimes[scene,name]=bases[scene] if name=='flow' else AnalyticField(bases[scene],ctl['payload']['controller_parameters'])
            value,info=measure(runtimes[scene,name],s)
            dest.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest,interaction=value,code_sha256=np.array(sha(__file__)))
            count+=1;stats.append(dict(state_uid=s['state_uid'],controller=name,**info))
    write(HERE/('contact_probe_features_complete.json' if probe_only else 'contact_features_complete.json'),dict(
        measurements=count,feature_code_sha256=sha(__file__),simulator_steps=0,labels_used=False,
        definition='Move one other agent to0.37or0.65m ahead along focal goal heading; reject invalid geometry; query0,+half,-half focal velocity; mean four common-random-seed nominal differences; mask fraction retained',rows=stats))


def load(d):
    values=[]
    for sid,name in zip(d['state_uid'],d['controller']):
        f=np.load(HERE/'contact_features'/str(sid)/(str(name)+'.npz'))
        assert str(f['code_sha256'])==sha(__file__)
        x=f['interaction'];values.append(np.pad(x,((0,4-len(x)),(0,0))))
    return np.asarray(values)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--probe-only',action='store_true');a=ap.parse_args();run(a.probe_only)
