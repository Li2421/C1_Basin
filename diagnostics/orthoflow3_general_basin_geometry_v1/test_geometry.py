#!/usr/bin/env python3
"""Offline mathematical/unit checks only: no policy load, GPU or rollouts."""
from families import *
import importlib.util
def run():
 sys.path.insert(0,str(ROOT))
 from shared_control.basis_families import get_basis_family
 from diagnostics.double_bottleneck_eta_basis_redesign.tools.bases import correction
 rng=np.random.default_rng(2026092901);tests=[]
 for n in (2,4,6):
  for _ in range(20):
   p=rng.normal(size=(n,2));goal=rng.normal(size=(n,2));safe=rng.normal(size=(n,2));v=rng.normal(size=3)
   a=get_basis_family('orthoflow3').compute(p,goal,safe,.5).correction(v)
   b=correction('P1-OrthoFlow3',v,p,goal,safe,.5,3.303687238760696)
   assert np.array_equal(a,b)
 tests.append('60 exact authoritative/shared basis equivalence checks for2/4/6 agents')
 # Native-domain reference corners and gap coordinates.
 assert inside((np.array([[0,0,0],[.1,0,.02],[1.25,.5,.75]])-AFF)/SCALE).all()
 assert not inside((np.array([[0,.1,0],[1.3,0,.3]])-AFF)/SCALE).any()
 tests.append('continuous E_bridge includes zero-to-active bridge; rejects invalid corners')
 Z=rng.uniform([-1.16,-.5,-.5],[.5,.5,.5],size=(2000,3));Z=Z[inside(Z)]
 models=json.load(open(HERE/'family_models_initial.json'));checked=0
 for item in models:
  m=item['model'];full=contains(m,Z);ret=contains(m,Z,True)
  assert np.all(~ret|full),item['state_id']
  if m['family'] in ('semialgebraic','common_core_minus_exclusions'):
   for d in np.vstack([np.eye(3),-np.eye(3)]):
    assert contains(m,Z[ret]+.02499*d).all(),(item['state_id'],m['family'],'Euclidean erosion')
  checked+=1
 tests.append(f'{checked} fitted models: retained subset of full; spherical/quadratic .025 erosion checked')
 # Erosion constraint reaches equality on a simple plane and sphere cap.
 m={'family':'semialgebraic','w':[0,1,0,0,0,0,0,0,0,0],'caps':[]}
 assert contains(m,[[.025,0,0]],True)[0]
 assert not contains(m,[[.024,0,0]],True)[0]
 assert contains(m,[[0,0,0]])[0]
 tests.append('analytic quadratic erosion plane boundary/interior/exterior')
 dump('offline_unit_tests.json',{'passed':True,'checks':tests,'GPU_jobs':0,'new_eta_rollouts':0})
 print(json.dumps(tests))
if __name__=='__main__':run()
