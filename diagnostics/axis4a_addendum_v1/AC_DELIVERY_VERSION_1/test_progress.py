from core import *
from progress import *
for H in [7,500,700,1000]:
 goals=np.ones((4,2))*10;p0=np.zeros((4,2));path=np.asarray([p0+(goals-p0)*min(t/H,1) for t in range(H+1)])
 for threshold in [0,.2,.5,.9728120313305867,1.]:
  online=ProgressSwitch(path[0],goals,H,threshold)
  for t in range(H):online.eta_index(t,path[t],goals)
  assert online.switched_at==offline_switch(path,goals,H,threshold)
# No action at the terminal snapshot: it must not invent a switch after termination.
p=np.array([np.zeros((4,2)),np.ones((4,2)),np.ones((4,2))*8]);g=np.ones((4,2))*10;sw=ProgressSwitch(p[0],g,500,.7)
for t in range(2):sw.eta_index(t,p[t],g)
assert sw.switched_at is None and offline_switch(p,g,500,.7) is None
# An exact threshold crossing and fallback boundary both use eta10 at that action index.
p=np.zeros((4,2));g=np.ones((4,2))*2;sw=ProgressSwitch(p,g,7,.5);assert sw.eta_index(0,p,g)==13 and sw.eta_index(1,np.ones((4,2)),g)==10 and sw.switched_at==1
sw=ProgressSwitch(p,g,7,1);assert sw.eta_index(5,p,g)==13 and sw.eta_index(6,p,g)==10
print('Progress online/offline, equality, fallback, and terminal-action tests PASS')
