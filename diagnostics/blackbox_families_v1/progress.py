"""Progress switch is evaluated before each physical action."""
import numpy as np

def progress(positions,goals,initial_distance):
 d0=np.asarray(initial_distance,float);assert np.all(d0>0)
 return float(np.clip(1-np.linalg.norm(np.asarray(goals)-positions,axis=-1)/d0,0,1).mean())
class ProgressSwitch:
 def __init__(self,positions,goals,H,threshold):
  self.initial_distance=np.linalg.norm(np.asarray(goals)-positions,axis=-1);assert np.all(self.initial_distance>0);self.threshold=float(threshold);self.fallback=round(6*H/7);self.switched_at=None
 def eta_index(self,t,positions,goals):
  if self.switched_at is None and (progress(positions,goals,self.initial_distance)>=self.threshold or t>=self.fallback):self.switched_at=int(t)
  return 13 if self.switched_at is None else 10

def offline_switch(positions,goals,H,threshold):
 p=np.asarray(positions);d0=np.linalg.norm(np.asarray(goals)-p[0],axis=-1)
 for t in range(min(H,len(p)-1)):
  if progress(p[t],goals,d0)>=threshold or t>=round(6*H/7):return t
 return None
