from common import *
from controllers import NovelNominal,GaussianNominal
from astar import AStarNominal
from fair_rhr import FairRuntime
from pipeline import LocalSafety
class NeighborRightRuntime(FairRuntime):
 def action(self,env,key,eta,chain='old'):
  nominal=np.stack([self.base.nominal(env,key)[1][i] for i in range(4)]);filters=[LocalSafety(env.positions,i,env.config,1.5) for i in range(4)];safe=np.stack([f.project(nominal[i:i+1])[0][0] for i,f in enumerate(filters)]);target=safe.copy();active=np.zeros(4,bool)
  for i in range(4):
   if np.linalg.norm(env.goals[i]-env.positions[i])<=env.config.goal_tolerance:continue
   bump=np.zeros(2)
   for j in range(4):
    if i==j:continue
    q=env.positions[j]-env.positions[i];r=np.linalg.norm(q)
    if 0<r<=self.rule['distance']:bump+=self.rule['speed']*np.array([q[1],-q[0]])/r;active[i]=True
   target[i]+=bump*min(1.,self.rule['speed']/max(np.linalg.norm(bump),1e-30))
  executed=np.stack([f.project(target[i:i+1])[0][0] for i,f in enumerate(filters)]);self.active_log.append(active);self.neighbor_log.append([len(f.neighbor_indices) for f in filters])
  return executed,dict(min_linear_residual=min(f.certificate(executed[i:i+1])['min_linear_residual'] for i,f in enumerate(filters))),sum(f.calls for f in filters)
def make(name,p,cache):
 if name not in cache:
  v=p['profiles'][name]['payload'];kind=v['controller_class']
  if kind=='G':rt=GaussianNominal(v)
  elif kind=='DM':rt=NovelNominal(kind,v['nominal_parameters'],p['native_manifest'])
  else:rt=AStarNominal(v['nominal_parameters'],p['native_manifest'])
  cache[name]=rt
 guard();return cache[name]
def fair(rt,rule):return (NeighborRightRuntime if rule['mode']=='neighbor_right' else FairRuntime)(rt,rule)
