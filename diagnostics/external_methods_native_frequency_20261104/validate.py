"""Saved-action replay and clock/termination regressions, zero new controller calls."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.sched_setaffinity(0,{23})
from clock_plant import *

def main():
    checks=[]
    # Every native interval covers precisely the same physical horizon, including
    # horizons not divisible by the controller period. No extra final callback.
    for period in PERIODS.values():
        for score,n in [(.05,839),(.1,747),(.15,194),(.02,500)]:
            scene=dict(config=dict(dt=score,max_steps=n));grid,controls,scores,h=timeline(scene,period)
            assert grid[0]==0 and grid[-1]==h and h not in controls
            assert all(t==k*frac(period) for t,k in controls.items())
            assert sum((b-a for a,b in zip(grid,grid[1:])),Fraction(0))==h
    checks.append(dict(name='native clocks, original score clocks, exact horizons, final partial holds',passed=True))
    # Real saved trajectories at equal clocks must reproduce every stored byte.
    cases=read(OLD/'ACTIVE_COMPARISON_CASES.json');selected=[]
    for dt in [.05,.1,.15]:
        for c in [x for x in cases if x['scenario']['config']['dt']==dt][:4]:selected.append(c)
    steps=0
    for c in selected:
        raw=read(OLD/'raw'/(c['rollout_uid']+'.json'))
        with np.load(raw['trace']) as z:
            env=ClockPlant(c['scenario'],c['state'],c['scenario']['config']['dt']);p=[env.positions.copy()];cl=[]
            for k,u in enumerate(z['actions']):
                env.set_control();end=(k+1)*frac(env.config.dt);info=env.advance(u,end,True)
                p.append(env.positions.copy());cl.append(info['swept_clearance'])
            assert np.asarray(p).tobytes()==z['positions'].tobytes(),c['name']
            assert np.asarray(cl).tobytes()==z['swept_clearance'].tobytes(),c['name']
            assert env.termination==raw['outcome']
            steps+=len(cl)
    checks.append(dict(name='equal-clock replay of actual historical trajectories',passed=True,traces=len(selected),steps=steps,positions_and_clearance='bytewise'))
    # Crossing agents collide even though safe at both sampled endpoints.
    cfg=dict(dt=.1,max_steps=2,num_agents=2,agent_radius=.1,agent_collision_margin=0.,wall_radius=0.,wall_collision_margin=0.,max_speed=3.,goal_tolerance=.01)
    state=dict(positions=[[-.2,0],[.2,0]],velocities=[[0,0],[0,0]],goals=[[.2,0],[-.2,0]])
    env=ClockPlant(dict(config=cfg,walls=[]),state,.1);env.set_control();info=env.advance(np.array([[4.,0],[-4.,0]]),frac(.1),True)
    assert env.collided and env.termination=='collision' and info['pair']<0
    checks.append(dict(name='swept crossing collision takes precedence over sampled goal success',passed=True))
    # Hitting the goal on a control-only event does not grant early score success.
    cfg.update(num_agents=1,agent_radius=.01,goal_tolerance=.001)
    state=dict(positions=[[0,0]],velocities=[[0,0]],goals=[[.02,0]])
    env=ClockPlant(dict(config=cfg,walls=[]),state,.02);env.set_control();env.advance(np.array([[1.,0]]),frac(.02),False)
    assert env.termination=='running'
    checks.append(dict(name='goal success restricted to original common score clock',passed=True))
    dump(D/'CLOCK_PLANT_CERTIFICATE.json',dict(passed=True,checks=checks,new_controller_calls=0,new_rollouts=0,scope='clock and SI propagator; no assertion of performance or safety guarantees'))
    print('clock/plant passed',len(selected),steps)

if __name__=='__main__':main()
