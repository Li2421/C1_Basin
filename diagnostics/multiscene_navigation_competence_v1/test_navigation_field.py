from common import *
from geometry import make_geometry,GeneratedGeometry
from navigation_field import NavigationGrid
from single_agent import SingleAgentGeneratedEnv
from observation import observe as old_observe
from observation39 import observe
from bottleneck_family.scenario import Config

def main():
 checks=[]
 # These are static navigation-function tests, never simulator rollouts.
 for family in ['B','F','D']:
  geo,resources=make_geometry(family,202640900000+ord(family));grid=NavigationGrid(geo)
  candidates=np.array([[5.,2.2],[5.,-2.2],[6.,0.],[-6.,-2.8],[0.,0.]])
  candidates=candidates[geo.valid_points(candidates)];goal=candidates[0];f=grid.field(goal)
  assert np.array_equal(f.waypoint(goal[None],.5)[0],goal)
  assert np.allclose(f.waypoint(np.repeat(goal[None],3,0),1.5),goal)
  # Door mouth, nearest legal corner, and representative starts.
  probes=[candidates[-1]]
  for k,(x,y,w) in enumerate(resources):
   q=np.array([x,y-.7]) if family=='F' and k>=2 else np.array([x-.7,y])
   if geo.valid_points(q[None])[0]:probes.append(q)
  for rect in geo.rectangles[:2]:
   a=np.array([rect[0]-geo.margin-.035,rect[1]-geo.margin-.035])
   if geo.valid_points(a[None])[0]:probes.append(a)
  for p in probes:
   path=geo.path(p,goal)
   if path is None:continue
   out=f.waypoint(p[None],.5)[0];assert np.isfinite(out).all();assert geo.valid_points(out[None])[0]
   pathlen=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum());assert np.linalg.norm(out-p)<=.5+1e-8
   checks.append(dict(family=family,point=p.tolist(),geometry_path_length=pathlen,waypoint_displacement=float(np.linalg.norm(out-p)),finite_reachable_cost=bool(np.isfinite(f.cost).any())))
  batch=np.asarray(probes);vector=f.waypoint(batch,.5);scalar=np.concatenate([f.waypoint(p[None],.5) for p in probes]);assert np.allclose(vector,scalar,atol=1e-12,rtol=0)
  env=SingleAgentGeneratedEnv(geo,candidates[-1:],[goal]);obs=observe(env,[f]);assert obs.shape==(1,39);assert np.array_equal(obs[:,:35],old_observe(env))
  expected=np.r_[f.waypoint(env.positions,.5)[0]-env.positions[0],f.waypoint(env.positions,1.5)[0]-env.positions[0]]
  assert np.allclose(obs[0,35:],expected)
 write(HERE/'NAVIGATION_UNIT_TESTS.json',dict(passed=True,checks=checks,observation_world_coordinates=True,original_35_dimensions_bitwise_equal=True,static_only=True))
 print('static tests passed',len(checks))
if __name__=='__main__':main()
