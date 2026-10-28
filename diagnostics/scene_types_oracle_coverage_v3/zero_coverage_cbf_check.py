"""Construct velocities for diagnostic paths; check original CBF rows directly.

No env.step, safety projection, nominal-controller call or Oracle evaluation.
The original distributed joint domain is the product of its local domains.
"""
from core import *
import time

OUT=HERE/'zero_coverage_audit'


def retime(s,state,witness):
    env=RN.SwapEnv(state,RN.LiteratureConfig(**s['config']));snap=env.snapshot()
    cfg=s['config'];dt=cfg['dt'];p=np.asarray(state['positions']).copy()
    positions=[p.copy()];actions=[];min_residual=float('inf');max_speed=0.
    start=time.monotonic()
    for move in witness['moves']:
        i=move['agent'];path=np.asarray(move['path']);assert np.array_equal(p[i],path[0])
        for goal in path[1:]:
            while np.linalg.norm(goal-p[i])>1e-12:
                if len(actions)>30000:raise RuntimeError('Finite diagnostic retiming limit')
                displacement=goal-p[i];length=np.linalg.norm(displacement);direction=displacement/length
                snap['positions']=p.copy();filters=[RN.SwapSafety(snap,j)for j in range(len(p))]
                f=filters[i];slopes=f.A@direction
                if any(np.max(x.lower,initial=-1.)>1e-10 for x in filters):
                    raise RuntimeError('Zero action not original-feasible at a constructed configuration')
                negative=slopes < -1e-12
                limit=min(cfg['max_speed']*.999,float(np.min(f.lower[negative]/slopes[negative],initial=np.inf))*.999)
                if limit<=1e-10:raise RuntimeError('Path direction stalled in original CBF domain')
                u=np.zeros_like(p);u[i]=direction*min(limit,length/dt)
                residual=min(float(np.min(x.A@u[j]-x.lower,initial=np.inf))for j,x in enumerate(filters))
                assert residual>=-1e-10
                min_residual=min(min_residual,residual);max_speed=max(max_speed,float(np.linalg.norm(u,axis=1).max()))
                p=p+dt*u
                positions.append(p.copy());actions.append(u.copy())
            # Roundoff-only endpoint snap is explicitly bounded, not a motion.
            assert np.linalg.norm(p[i]-goal)<1e-12
            p[i]=goal
            positions[-1]=p.copy()
    pos=np.asarray(positions);u=np.asarray(actions)
    dynamics_residual=float(np.max(abs(np.diff(pos,axis=0)-dt*u),initial=0.))
    assert dynamics_residual<1e-12
    terminal=float(np.max(np.linalg.norm(p-np.asarray(state['goals']),axis=1)))
    assert terminal<cfg['goal_tolerance'] and max_speed<=cfg['max_speed']+1e-12
    duration=len(u)*dt
    return dict(original_CBF_rows_passed=True,minimum_CBF_residual=min_residual,max_speed=max_speed,
        dynamics_residual=dynamics_residual,terminal_max_goal_distance=terminal,
        steps=len(u),seconds=duration,horizon_seconds=cfg['max_steps']*dt,within_original_horizon=len(u)<=cfg['max_steps'],
        continuous_geometric_sweeps_passed=True,minimum_pair_clearance=witness['minimum_pair_clearance'],
        minimum_wall_clearance=witness['minimum_wall_clearance'],
        original_solver_replayed=False,environment_steps=0,retiming_compute_seconds=time.monotonic()-start),pos,u


def main():
    scenes={(e['type_id'],e['level']):e['scene']for e in read(HERE/'SCENES.json')['entries']}
    output=[]
    for source in sorted(OUT.glob('t08_l*_s[0-9].json')):
        x=read(source)
        if not x['found']:continue
        target=source.with_name(source.stem+'_cbf.json')
        if target.exists():output.append(read(target));continue
        s=scenes[(x['type_id'],x['level'])];state=s['states'][x['state_index']]
        result=dict(type_id=x['type_id'],level=x['level'],state_index=x['state_index'],geometric_source=str(source),
            scope='Diagnostic explicit velocity sequence satisfying every original local CBF row and speed bound, with continuous geometric sweeps. No nominal-controller/solver execution; never added to Oracle or paper statistics.')
        try:
            r,p,u=retime(s,state,x['witness']);result.update(r)
            trace=target.with_suffix('.npz');np.savez_compressed(trace,positions=p,actions=u);result['trace']=str(trace)
        except Exception as e:result.update(original_CBF_rows_passed=False,error=str(e))
        dump(target,result);output.append(result)
        print('ORIGINAL_CBF_PATH',x['level'],x['state_index'],result.get('original_CBF_rows_passed'),result.get('seconds'),result.get('within_original_horizon'),flush=True)
        dump(OUT/'CBF_RESULTS.json',dict(cases=output,complete=False,environment_steps=0,solver_calls=0))
    dump(OUT/'CBF_RESULTS.json',dict(cases=output,complete=True,environment_steps=0,solver_calls=0,
        meaning='Numerical constructive feasibility witnesses under actual original distributed CBF rows; horizon checked separately. Not executed C1 policy success or a mathematical proof about every initial state.'))


if __name__=='__main__':main()
