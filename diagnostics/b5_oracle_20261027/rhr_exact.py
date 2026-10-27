"""Historical RHR inserted after safety-only second projection, active agents only."""
import numpy as np
from runtime import RN,SOURCE,clean

anchor=';_,done,info=env.step(a)'
assert SOURCE.count(anchor)==1
source=SOURCE.replace('def run(c):','def run(c, post):').replace(anchor,
    '\n            a,second=post(env,a,filters,t,second)\n            _,done,info=env.step(a)')
ns=dict(vars(RN));exec(compile(source,__file__,'exec'),ns)

def run(c):
    hist={}
    def post(env,a,filters,t,second):
        d=np.linalg.norm(env.goals-env.positions,axis=1);hist[t]=d.copy();hist.pop(t-41,None)
        active=(hist[t-40]-d<.05)&(d>env.config.goal_tolerance)if t>=40 else np.zeros(len(d),bool)
        if not active.any():return a,second
        q=env.goals-env.positions if c['spec']['mode']=='exact_rhr_goal'else env.positions
        target=a.copy()
        for i in np.flatnonzero(active):
            target[i]+=.35*np.array([q[i,1],-q[i,0]])/max(np.linalg.norm(q[i]),1e-30)
            second[i]=filters[i].project(target[i:i+1]);a[i]=second[i][0][0]
        return a,second
    result,tr=ns['run'](c,post);cfg=c['scenario_payload']['config']
    d=np.linalg.norm(tr['goals'][None]-tr['positions'],axis=-1).sum(-1);q=np.flatnonzero(d<=d[0]-.1)
    result.update(local_progress_time=float(q[0]*cfg['dt'])if len(q)else None,
        completion_time=result['episode_length']*cfg['dt']if result['success']else None,
        truncated_time=result['episode_length']*cfg['dt']if result['success']else cfg['dt']*cfg['max_steps'],horizon_time=cfg['dt']*cfg['max_steps'])
    return clean(result),tr
