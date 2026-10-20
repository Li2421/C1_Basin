"""Hard-stop physical equivalence gates, before demonstration collection."""
from common import frozen_config
import json
from pathlib import Path
import numpy as np
from bottleneck_family.scenario import Config, Geometry, build_instance
from bottleneck_family.environment import BottleneckEnv
from single_integrator.environment import segment_distance
from geometry import GeneratedGeometry, GeneratedEnv, make_geometry, task
from single_agent import SingleAgentGeneratedEnv

HERE=Path(__file__).resolve().parent
class FixtureError(RuntimeError):pass
def legal(condition,message):
    if not condition:raise FixtureError(message)
def identical(a,b):
    a=np.asarray(a);b=np.asarray(b)
    assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()

def main():
    frozen_config()
    assert GeneratedEnv.step is BottleneckEnv.step
    assert GeneratedEnv.reset is BottleneckEnv.reset
    assert GeneratedEnv.snapshot is BottleneckEnv.snapshot
    for name in ['valid_points','visible','path']:
        assert getattr(GeneratedGeometry,name) is getattr(Geometry,name)
    c=Config(num_agents=4,seed=2026103901,max_steps=100)
    instance=build_instance(c)
    legal(instance.geometry.valid_points(np.vstack([instance.positions,instance.goals])).all(),'Gap fixture invalid')
    original=BottleneckEnv(c)
    geo=GeneratedGeometry(c,original.instance.geometry.rectangles)
    generated=GeneratedEnv(geo,original.positions,original.goals)
    identical(original.walls,generated.walls)
    rng=np.random.default_rng(2026103901)
    for t in range(100):
        u=rng.uniform(-.25,.25,(4,2))
        a=original.step(u,diagnose_stalls=False);b=generated.step(u,diagnose_stalls=False)
        identical(original.positions,generated.positions)
        identical(original.velocities,generated.velocities)
        assert a[1]==b[1]
        for k in a[2]:
            if isinstance(a[2][k],np.ndarray):identical(a[2][k],b[2][k])
            else:assert a[2][k]==b[2][k]
        if a[1]:break
    records=[]
    for family in ['A','D','E']:
        geo,_=make_geometry(family,2026103902,n=2,H=100)
        p=np.array([[-6.,-2.],[-6.,2.]]) if family!='D' else np.array([[-6.,0.],[6.,0.]])
        for kind in ['success','collision']:
            u=np.array([[.5,0.],[.5,0.]]) if kind=='success' else np.array([[-.5,0.],[-.5,0.]])
            if family=='D' and kind=='collision':u[1]=[.5,0.]
            g=p+2*u if kind=='success' else p+np.array([[0.,1.],[0.,1.]])
            if family=='D' and kind=='collision':g=p+np.array([[.5,0.],[-.5,0.]])
            legal(geo.valid_points(np.vstack([p,g])).all(),f'{family} {kind} invalid starts/goals')
            legal(np.linalg.norm(p[0]-p[1])>2.,'initial separation must exceed2m')
            if kind=='success':
                legal(all(geo.path(a,b) is not None for a,b in zip(p,g)),'success fixture lacks path')
            else:
                future=p.copy(); minimum=np.inf
                for _ in range(100):
                    after=future+geo.config.dt*u
                    minimum=min(minimum,float(segment_distance(future[:,None],after[:,None],geo.walls[None,:,0],geo.walls[None,:,1]).min()-geo.margin))
                    future=after
                legal(minimum<0.,'collision fixture does not sweep through a wall')
            env=GeneratedEnv(geo,p,g)
            solos=[SingleAgentGeneratedEnv(geo,p[i:i+1],g[i:i+1]) for i in range(2)]
            separation=[];wall_minimum=np.inf
            for t in range(100):
                before=env.positions.copy();_,done,info=env.step(u,diagnose_stalls=False)
                sep=float(np.linalg.norm(env.positions[0]-env.positions[1]));separation.append(sep);assert sep>2.
                for i,solo in enumerate(solos):
                    assert not solo.done, 'test inputs must compare every original physical step'
                    _,_,single=solo.step(u[i:i+1],diagnose_stalls=False)
                    identical(env.positions[i:i+1],solo.positions)
                    clearance=segment_distance(before[i:i+1,None],env.positions[i:i+1,None],
                          env.walls[None,:,0],env.walls[None,:,1]).min()-geo.margin
                    wall_minimum=min(wall_minimum,float(clearance))
                    identical(np.asarray(clearance),np.asarray(single['min_swept_wall_clearance']))
                    collision=bool(clearance<=0 or not geo.valid_points(env.positions[i:i+1]).all())
                    success=bool(np.linalg.norm(g[i]-env.positions[i])<=env.config.goal_tolerance)
                    expected='collision' if collision else 'success' if success else 'timeout' if t+1>=100 else 'running'
                    assert single['termination']==expected and single['stalled_groups']==[]
                if done:break
            assert env.termination==kind
            if kind=='collision':legal(wall_minimum<0.,'executed fixture did not have negative swept wall clearance')
            records.append(dict(family=family,kind=kind,steps=env.step_count,
                                minimum_separation=min(separation),minimum_wall_clearance=wall_minimum,passed=True))
    reproducibility=[]
    for family in 'ABCDEFG':
        a,_=make_geometry(family,2026103999);b,_=make_geometry(family,2026103999)
        identical(a.walls,b.walls);identical(a.nodes,b.nodes)
        reproducibility.append(dict(family=family,passed=True))
    try:Config(openings=(((0.,10.),),))
    except ValueError:pass
    else:raise AssertionError('invalid wide opening accepted')
    invalid=GeneratedGeometry(Config(),[(-.5,-4.,.5,4.)])
    assert not invalid.valid_points(np.array([[0.,0.]])).any()
    assert invalid.path(np.array([-2.,0.]),np.array([2.,0.])) is None
    (HERE/'GEOMETRY_EQUIVALENCE.json').write_text(json.dumps(dict(passed=True,
          multiagent_step_inherited=True, gap_equivalence=True,
          single_agent_equivalence=records, reproducibility=reproducibility,
          invalid_samples_rejected=True),indent=2)+'\n')

if __name__=='__main__':
    try:main()
    except Exception as error:
        name='GEOMETRY_EQUIVALENCE_FIXTURE_ERROR_V2.json' if isinstance(error,FixtureError) else 'GEOMETRY_EQUIVALENCE_FAILURE_V2.json'
        (HERE/name).write_text(json.dumps(dict(
            passed=False,type=type(error).__name__,message=str(error),hard_stop=not isinstance(error,FixtureError)),indent=2)+'\n')
        raise
