"""Time-shift fixed verified paths; certify pairs with original local CBFs.

This is an auxiliary expert trajectory, not a new controller or rollout.
No environment, nominal or safety solver is stepped/called.
"""
from core import *

OUT=HERE/'zero_coverage_audit'


def profile_at(profile,start,t):
    p,u=profile;k=t-start
    return p[np.clip(k,0,len(p)-1)],u[np.clip(k,0,len(u)-1)]*((k>=0)&(k<len(u)))[:,None]


def pair_check(a,sa,b,sb,dt,safe,gamma):
    t=np.arange(max(sa+len(a[1]),sb+len(b[1]))+1)
    p,u=profile_at(a,sa,t);q,v=profile_at(b,sb,t);d=p-q;squared=(d*d).sum(1)
    # Exact source: swap_safety.local_problem, pair responsibility 0.5.
    lower=-.5*gamma*(squared-safe*safe);active=squared<=.5**2
    residual=min(float(np.min((2*d*u).sum(1)[active]-lower[active],initial=np.inf)),
                 float(np.min((-2*d*v).sum(1)[active]-lower[active],initial=np.inf)))
    delta=dt*(u-v);den=(delta*delta).sum(1)
    fraction=np.clip(-(d*delta).sum(1)/np.maximum(den,1e-30),0,1)
    distance=np.linalg.norm(d+fraction[:,None]*delta,axis=1).min()
    return residual>=-1e-10 and distance>=safe-1e-10,residual,float(distance-safe)


def schedule(source,s):
    serial=read(source);trace=np.load(serial['trace']);pos=trace['positions'];actions=trace['actions'];dt=s['config']['dt']
    n=pos.shape[1];profiles=[];order=[]
    for i in range(n):
        inds=np.flatnonzero(np.linalg.norm(actions[:,i],axis=1)>1e-14)
        assert len(inds) and inds[-1]-inds[0]+1==len(inds)
        profiles.append((pos[inds[0]:inds[-1]+2,i].copy(),actions[inds[0]:inds[-1]+1,i].copy()))
        order.append((int(inds[0]),i))
    safe=2*s['config']['agent_radius']+s['config']['agent_collision_margin']+RN.HardProjectionConfig().separation_buffer
    gamma=RN.HardProjectionConfig().gamma;starts={};order=[i for _,i in sorted(order)]
    # Preserve the geometric move ordering; search earliest legal start times.
    # One-step resolution, explicit finite temporal search, no future labels.
    for i in order:
        static=(profiles[i][0][:1],np.zeros((1,2)))
        for j in range(n):
            if j==i or j in starts:continue
            other=(profiles[j][0][:1],np.zeros((1,2)))
            assert pair_check(profiles[i],0,other,0,dt,safe,gamma)[0]
        latest=max([0,*[starts[j]+len(profiles[j][1]) for j in starts]])
        for candidate in range(latest+1):
            if all(pair_check(profiles[i],candidate,profiles[j],starts[j],dt,safe,gamma)[0]for j in starts):
                starts[i]=candidate;break
        else:raise RuntimeError('No legal timing even at serial boundary')
    horizon=max(starts[i]+len(profiles[i][1])for i in range(n));t=np.arange(horizon+1)
    pp=[];uu=[]
    for i in range(n):
        p,u=profile_at(profiles[i],starts[i],t);pp.append(p);uu.append(u[:-1])
    p=np.stack(pp,axis=1);u=np.stack(uu,axis=1)
    residual=np.inf;clearance=np.inf
    for i in range(n):
        for j in range(i):
            ok,r,c=pair_check(profiles[i],starts[i],profiles[j],starts[j],dt,safe,gamma)
            assert ok;residual=min(residual,r);clearance=min(clearance,c)
    dynamics=float(np.max(abs(np.diff(p,axis=0)-dt*u),initial=0));assert dynamics<1e-12
    assert np.array_equal(p[-1],np.asarray(s['states'][serial['state_index']]['goals']))
    path=source.with_name(source.stem+'_scheduled.npz');np.savez_compressed(path,positions=p,actions=u)
    return dict(type_id=serial['type_id'],level=serial['level'],state_index=serial['state_index'],
        serial_source=str(source),trace=str(path),steps=horizon,seconds=horizon*dt,
        horizon_seconds=s['config']['max_steps']*dt,within_original_horizon=horizon<=s['config']['max_steps'],
        start_steps=starts,original_CBF_rows_passed=True,minimum_pair_CBF_residual=residual,
        minimum_continuous_pair_clearance=clearance,minimum_wall_clearance=serial['minimum_wall_clearance'],
        dynamics_residual=dynamics,original_solver_replayed=False,environment_steps=0,
        scope='Numerical explicit joint velocity witness under original local CBF/speed constraints and continuous geometric collision checks. No policy/solver rollout; excluded from performance counts.')


def main():
    scenes={(e['type_id'],e['level']):e['scene']for e in read(HERE/'SCENES.json')['entries']};rows=[]
    for source in sorted(OUT.glob('t08_l*_s[0-9]_cbf.json')):
        c=read(source)
        if not c['original_CBF_rows_passed']:continue
        target=source.with_name(source.stem+'_scheduled.json')
        if target.exists():result=read(target)
        else:
            result=schedule(source,scenes[(c['type_id'],c['level'])]);dump(target,result)
        rows.append(result);print('JOINT_CBF_SCHEDULE',c['level'],c['state_index'],result['seconds'],result['within_original_horizon'],flush=True)
        dump(OUT/'SCHEDULED_RESULTS.json',dict(cases=rows,complete=False))
    dump(OUT/'SCHEDULED_RESULTS.json',dict(cases=rows,complete=True,environment_steps=0,solver_calls=0))


if __name__=='__main__':main()
