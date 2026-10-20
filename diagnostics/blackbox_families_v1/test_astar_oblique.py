"""Required empty-map straightness, including non-axis-aligned paths."""
from common import *
from fast_astar import FastGrid
checks=[]
for res in [.05,.1]:
 for target in [[1.,.4],[2.,.8],[1.,.7]]:
  goal=np.asarray(target);path=FastGrid(res,res/.52,max_time=200).search([0.,0.],goal);t=np.clip(path@goal/(goal@goal),0,1);deviation=float(np.linalg.norm(path-t[:,None]*goal,axis=1).max());length=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum());checks.append(dict(resolution=res,goal=target,max_lateral_deviation_m=deviation,max_lateral_deviation_cells=deviation/res,path_extra_length_cells=(length-float(np.linalg.norm(goal)))/res,passed=deviation<=res+1e-12))
write(HERE/'ASTAR_OBLIQUE_UNIT_TEST.json',dict(utc=utc(),passed=all(x['passed'] for x in checks),checks=checks,physical_steps=0,evaluation_outcomes_read=False,initial_axis_aligned_test_insufficient=True,geometry_and_controller_parameters_not_changed=True))
assert not all(x['passed'] for x in checks),'unexpectedly passed'
print('REQUIRED STRAIGHTNESS FAIL; preserve family as implementation gap')
