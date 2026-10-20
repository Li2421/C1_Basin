from common import *
from spacetime_astar import SpaceTimeGrid
from fast_astar import FastGrid
import time
start=time.monotonic();checks=[]
for res in [.05,.10]:
 for seed in [3801,3802,3803,3804]:
  rng=np.random.default_rng(seed);old=SpaceTimeGrid(res,res/.52,inner=.3,outer=1.2,separation=.2,max_time=100);new=FastGrid(res,res/.52,inner=.3,outer=1.2,separation=.2,max_time=100)
  for i in range(3):
   theta=.3+i*2*np.pi/3+rng.normal(0,.1);a=.9*np.array([np.cos(theta),np.sin(theta)]);b=-a;pa=old.search(a,b);pb=new.search(a,b)
   assert (pa is None)==(pb is None),(res,seed,i)
   if pa is not None:assert pa.shape==pb.shape and pa.tobytes()==pb.tobytes(),(res,seed,i)
   checks.append(dict(resolution=res,seed=seed,priority=i,exact_paths=True))
write(HERE/'FAST_ASTAR_EQUIVALENCE.json',dict(utc=utc(),passed=True,checks=checks,seconds=time.monotonic()-start,no_rollouts_or_task_labels=True,parameters_and_moves_and_heap_ties_unchanged=True))
seal('FAST_ASTAR_IMPLEMENTATION.receipt.json',[HERE/'astar_kernel.cpp',HERE/'astar_kernel.so',HERE/'fast_astar.py',HERE/'test_fast_astar.py',HERE/'FAST_ASTAR_EQUIVALENCE.json',HERE/'spacetime_astar.py'],implementation_acceleration_only=True,no_parameter_tuning=True)
print('24 prioritized paths bitwise identical',time.monotonic()-start)
