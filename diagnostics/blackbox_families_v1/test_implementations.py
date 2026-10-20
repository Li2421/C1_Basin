from common import *
from controllers import CrossEntropy
from legacy_controllers import Nominal
from spacetime_astar import SpaceTimeGrid,PrioritizedAStar
from progress import ProgressSwitch,offline_switch
import jax
p=read(D/'other_controller_classes_v1/main_protocol.json');s=p['states'][0];manifest=p['native_manifest'];checks=[]
for i,params in enumerate(read(HERE/'DESIGN_CONFIG.json')['CEM']):
 rt=CrossEntropy(params,manifest);env=rt.make_env(s);a=rt.nominal(env,jax.random.PRNGKey(7))[1];b=rt.nominal(env,jax.random.PRNGKey(7))[1];assert a.tobytes()==b.tobytes();assert a.shape==(4,2) and np.isfinite(a).all() and np.linalg.norm(a,axis=-1).max()<=env.config.max_speed+1e-12;checks.append('cross_entropy_'+str(i))
grid=SpaceTimeGrid(.1,.2,max_time=80);path=grid.search([0,0],[1,0]);assert path is not None and np.abs(path[:,1]).max()<=.1;assert np.linalg.norm(path[-1]-[1,0])<1e-12
grid=SpaceTimeGrid(.1,.2,separation=.15,max_time=80);a=grid.search([0,0],[1,0]);b=grid.search([.5,-.5],[.5,.5]);assert a is not None and b is not None
from spacetime_astar import segment_distance
for t in range(max(len(a),len(b))-1):assert segment_distance(a[min(t,len(a)-1)]-b[min(t,len(b)-1)],a[min(t+1,len(a)-1)]-b[min(t+1,len(b)-1)])>=.15-1e-12
# Test the real replan dispatch without costly geometry search or physical rollouts.
class Dispatch(PrioritizedAStar):
 def plan(self,env):self.last_replan=env.step_count;self.replan_steps.append(env.step_count);self.paths={i:None for i in range(4)};self.grid=type('Grid',(),{'dt':.1})()
for params in read(HERE/'DESIGN_CONFIG.json')['ASTAR']:
 rt=Dispatch(params,manifest);env=rt.make_env(s)
 for t in range(2*params[1]+1):env.step_count=t;rt.nominal(env,jax.random.PRNGKey(0))
 assert rt.replan_steps==[0,params[1],2*params[1]]
pos=np.array([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]]);goal=-pos;rule=ProgressSwitch(pos,goal,700,.9681620884830543);assert rule.eta_index(0,pos,goal)==13;assert rule.eta_index(1,goal,goal)==10;assert rule.eta_index(2,pos,goal)==10
rule=ProgressSwitch(pos,goal,700,.9681620884830543);assert rule.eta_index(599,pos,goal)==13;assert rule.eta_index(600,pos,goal)==10;assert offline_switch(np.repeat(pos[None],701,axis=0),goal,700,.9681620884830543)==600
write(HERE/'UNIT_TESTS.json',dict(utc=utc(),passed=True,cross_entropy=checks,astar_single_robot_straight=True,astar_priority_continuous_reservations=True,astar_replan_dispatch=True,progress_threshold_fallback_latching=True,no_task_rollouts=True))
print('new controller and progress unit tests PASS')
