"""Additional arms only; frozen pipeline logic is untouched."""
from core import *
from pipeline import LocalSafety,decentralized_action

def nominal_only(rt,env,key,eta,counts):
    action=rt.nominal(env,key)[1]
    return action,dict(projection_calls=0,minimum_local_linear_residual=np.nan,neighbor_counts=[0]*4,rhr_active=np.zeros(4,bool))
def right_hand_local(rt,env,key,eta,counts):
    nominal=np.stack([rt.nominal(env,key)[1][i] for i in range(4)])
    filters=[LocalSafety(env.positions,i,env.config,1.5) for i in range(4)];safe=np.stack([f.project(nominal[i:i+1])[0][0] for i,f in enumerate(filters)])
    target=safe.copy();active=np.zeros(4,bool)
    for i in range(4):
        if counts[i]>=40 and np.linalg.norm(env.goals[i]-env.positions[i])>env.config.goal_tolerance:
            q=env.positions[i];target[i]+=.15*np.array([q[1],-q[0]])/max(np.linalg.norm(q),1e-30);active[i]=True
    executed=np.stack([f.project(target[i:i+1])[0][0] for i,f in enumerate(filters)])
    return executed,dict(projection_calls=sum(f.calls for f in filters),minimum_local_linear_residual=min(f.certificate(executed[i:i+1])['min_linear_residual'] for i,f in enumerate(filters)),neighbor_counts=[len(f.neighbor_indices) for f in filters],rhr_active=active)
def dispatch(rt,env,key,eta,kind,counts):
    if kind=='nominal':return nominal_only(rt,env,key,eta,counts)
    if kind=='rhr_local':return right_hand_local(rt,env,key,eta,counts)
    return decentralized_action(rt,env,key,eta,kind,counts)
