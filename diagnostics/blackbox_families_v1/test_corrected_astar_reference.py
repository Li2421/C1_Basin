from common import *
from corrected_astar_reference import StraightnessCheckedGrid
from spacetime_astar import segment_distance
checks=[]
for res in [.05,.1]:
 for target in [[1.,.4],[2.,.8],[1.,.7],[.83,-.57]]:
  goal=np.asarray(target);path=StraightnessCheckedGrid(res,res/.52,max_time=200).search([0.,0.],goal);t=np.clip(path@goal/(goal@goal),0,1);deviation=float(np.linalg.norm(path-t[:,None]*goal,axis=1).max());assert deviation<=res+1e-12;checks.append(dict(resolution=res,goal=target,max_lateral_deviation_cells=deviation/res,passed=True))
grid=StraightnessCheckedGrid(.1,.2,separation=.15,max_time=80);a=grid.search([0.,0.],[1.,0.]);b=grid.search([.5,-.5],[.5,.5]);assert a is not None and b is not None
for t in range(max(len(a),len(b))-1):assert segment_distance(a[min(t,len(a)-1)]-b[min(t,len(b)-1)],a[min(t+1,len(a)-1)]-b[min(t+1,len(b)-1)])>=.15-1e-12
write(HERE/'CORRECTED_ASTAR_ENGINEERING_TESTS.json',dict(utc=utc(),passed=True,straightness=checks,continuous_priority_reservations=True,physical_steps=0,used_for_current_round_frozen_rollouts=False,execution_semantics_changed_requires_new_identity_and_preregistration=True))
print('corrected engineering reference passes eight oblique paths and time-disc reservations; no current-round evaluation use')
