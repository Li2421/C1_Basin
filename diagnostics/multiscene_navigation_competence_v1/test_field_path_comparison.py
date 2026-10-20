"""Static field integration versus inherited geometric path; no simulator rollout."""
from common import *
from geometry import make_geometry
from navigation_field import NavigationGrid

def main():
 checks=[]
 for family in ['B','F','D']:
  geo,resources=make_geometry(family,202640900000+ord(family));grid=NavigationGrid(geo)
  start=np.array([-6.,-2.8]) if family!='D' else np.array([-6.,0.]);goal=np.array([5.,2.2]) if family!='D' else np.array([5.,0.])
  assert geo.valid_points(np.array([start,goal])).all();reference=geo.path(start,goal);assert reference is not None
  field=grid.field(goal);p=start.copy();points=[p.copy()]
  for _ in range(2000):
   p=field.waypoint(p[None],.05)[0];points.append(p.copy())
   if np.linalg.norm(p-goal)<1e-8:break
  points=np.array(points);arc=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum());length=float(np.linalg.norm(np.diff(reference,axis=0),axis=1).sum())
  check=dict(family=family,geometry_path_length=length,field_following_arc_length=arc,ratio=arc/length,steps=len(points)-1,reached=bool(np.linalg.norm(p-goal)<1e-8),all_points_valid=bool(geo.valid_points(points).all()),static_only=True);checks.append(check)
  assert check['reached'] and check['all_points_valid'],check
  assert arc>=np.linalg.norm(goal-start)-1e-8
 write(HERE/'FIELD_PATH_COMPARISON_TEST.json',dict(passed=True,checks=checks));print(checks)
if __name__=='__main__':main()
