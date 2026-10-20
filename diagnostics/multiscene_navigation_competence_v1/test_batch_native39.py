from common import *
from dataclasses import replace
from types import SimpleNamespace
from geometry import GeneratedGeometry
from single_agent import SingleAgentGeneratedEnv
from bottleneck_family.scenario import Config
from dagger39 import collect_episode
from batch_collect39 import collect_batch
import jax

def main():
 geo=GeneratedGeometry(Config(num_agents=4),[(-.3,-4,.3,-.3),(-.3,.3,.3,4)])
 fixtures=[([-0.5,1.],[1.,1.],[.5,0.],10),([-2.,0.],[-1.9,0.],[.5,0.],10),([-2.,-1.],[-1.,-1.],[0.,0.],3)]
 def env(f):
  p,g,u,H=f;assert geo.valid_points(np.array([p,g])).all();assert geo.path(np.asarray(p),np.asarray(g)) is not None
  return SingleAgentGeneratedEnv(geo,[p],[g],replace(geo.config,max_steps=H))
 class Constant:
  def __init__(self,u):self.u=np.asarray([u])
  def __call__(self,env,fields,key):return self.u.copy()
  def from_observation(self,env,observed,key):return self.u.copy()
 policies=[Constant(f[2]) for f in fixtures];keys=[jax.random.PRNGKey(i) for i in range(3)]
 a=[collect_episode(env(f),p,k) for f,p,k in zip(fixtures,policies,keys)];b=collect_batch([env(f) for f in fixtures],policies,keys)
 checks=[]
 for i,(old,new) in enumerate(zip(a,b)):
  assert old['outcome']==new['outcome'];assert new['error'] is None
  for name in ['observations','labels','actions','positions','wall_clearance']:assert old[name].dtype==new[name].dtype and old[name].tobytes()==new[name].tobytes(),(i,name)
  checks.append(dict(case=i,outcome=new['outcome'],steps=len(new['actions']),all_arrays_bitwise=True))
 assert [r['outcome'] for r in b]==['collision','success','timeout'];assert b[0]['wall_clearance'][-1]<0
 write(HERE/'BATCH_NATIVE39_TEST.json',dict(passed=True,checks=checks,each_environment_native_step=True,independent_termination=True,no_postterminal_labels=True))
 print(checks)
if __name__=='__main__':main()
