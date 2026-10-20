"""New nominal sources, inheriting the protected centralized safety plant."""
from common import *
from historical_controllers import Observed,GaussianNominal,NovelNominal
from historical_controllers import make as historical_make
from legacy_controllers import Nominal,bound,shooting_cost
import jax
jax.config.update('jax_enable_x64',False)
def make(name,p,cache):
 kind=p['profiles'][name]['payload'].get('controller_class','Flow')
 if kind not in ['CEM','ASTAR','deterministic','mixture','diffusion']:
  return historical_make(name,p,cache)
 if name not in cache:
  v=p['profiles'][name]['payload']
  if kind=='CEM':cache[name]=CrossEntropy(v['nominal_parameters'],p['native_manifest'])
  elif kind=='ASTAR':
   from spacetime_astar import PrioritizedAStar
   cache[name]=PrioritizedAStar(v['nominal_parameters'],p['native_manifest'])
  else:
   from learned import LearnedNominal
   cache[name]=LearnedNominal(v)
 return cache[name]

def cross_entropy(env,key,parameters):
 horizon,iterations,samples=parameters;cfg=env.config
 token=np.asarray(key,np.uint32).tobytes();rng=np.random.default_rng(int.from_bytes(hashlib.sha256(token).digest()[:8],'little'))
 mean=np.broadcast_to(bound(env.goals-env.positions,cfg.max_speed),(3,4,2)).copy();std=np.full_like(mean,.8*cfg.max_speed)
 for _ in range(iterations):
  knots=bound(mean[None]+rng.normal(size=(samples,3,4,2))*std[None],cfg.max_speed)
  actions=knots[:,np.minimum(np.arange(horizon)*3//horizon,2)]
  positions=env.positions[None,None]+cfg.dt*np.cumsum(actions,axis=1)
  cost=shooting_cost(positions,actions,env.goals,cfg,100,env.velocities)
  elite=np.argsort(cost,kind='stable')[:int(np.ceil(.1*samples))]
  mean=knots[elite].mean(0);std=np.maximum(knots[elite].std(0),.001)
 return actions[int(np.argmin(cost)),0].copy()
class CrossEntropy(Nominal):
 def __init__(self,parameters,manifest):super().__init__('CEM',parameters,manifest)
 def nominal(self,env,key,traced=False):
  assert not traced;return None,cross_entropy(env,key,self.parameters),None
