import unittest
import numpy as np
from new_benchmark_common.final_diagnostics import load_nominal
from new_benchmark_common.final_diagnostics import Adapter, NominalTrajectory, run
class GuardTests(unittest.TestCase):
 def test_test_is_rejected_before_manifest_access(self):
  with self.assertRaises(PermissionError): load_nominal('/definitely/not/opened','test',frozen_test=False)
 def test_synthetic_dev_smoke_emits_k_ood_and_mode_fields(self):
  class Agent:
   config={'max_speed':1.0}
   def sample_actions(self,obs,key): return np.zeros((len(obs),4,2),dtype=np.float32)
  class Env:
   def __init__(self): self.p=np.zeros((4,2)); self.n=0
   def reset(self,s): self.p=np.asarray(s['positions'],float);self.n=0
   def summary(self): return {'collision_free_success':self.n>=2,'episode_steps':self.n,'wall_collision':False,'obstacle_collision':False,'agent_collision':False}
  def reset(e,s): e.reset(s)
  def step(e,u):
   e.p=e.p+u;e.n+=1;return e.n>=2,{'termination':'success' if e.n>=2 else 'running','wall_collision':False,'agent_collision':False}
  def snap(e): return {'positions':e.p.copy(),'n':e.n}
  def restore(e,s): e.p=np.asarray(s['positions']).copy();e.n=s['n']
  adapter=Adapter('synthetic',Env,reset,lambda e:np.zeros((4,3),np.float32),step,snap,restore,lambda s:np.asarray(s['positions']),lambda p:'synthetic',1.)
  t=NominalTrajectory('dev_nominal',{'positions':np.zeros((4,2))},np.zeros((101,4,2)),np.zeros((101,4,3),np.float32),np.zeros((100,4,2),np.float32),{})
  result=run(adapter,Agent(),(t,),np.zeros((8,4,3),np.float32),np.zeros((8,4,3),np.float32))
  self.assertIn('k_step',result['aggregate']);self.assertIn('mode_signature',result['rollouts'][0]);self.assertIn('rollout_ood',result['rollouts'][0])
if __name__=='__main__': unittest.main()
