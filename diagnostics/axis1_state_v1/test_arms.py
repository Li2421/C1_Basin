from arms import action
import numpy as np
class Fake:
 def __init__(self):self.log=[]
 def nominal(self,env,key,traced=False):self.log.append(('nominal',traced));return None,np.array([[.1,.2]]),None
 def action(self,env,key,eta,chain):self.log.append((eta.tolist(),chain));return np.array([[.3,.4]]),None,7
r=Fake();a,n=action(r,None,12,dict(arm='nominal_only'),0);assert n==0 and np.array_equal(a,[[.1,.2]]) and r.log==[('nominal',False)]
r=Fake();c=dict(arm='safety_baseline',eta_a=[0,0,0],eta_b=[0,0,0],switch_step=700);a,n=action(r,None,0,c,0);assert n==7 and r.log==[([0.,0.,0.],'field')]
r=Fake();c=dict(arm='S300',eta_a=[1,2,3],eta_b=[4,5,6],switch_step=300)
for i in [299,300]:action(r,None,0,c,i)
assert r.log==[([1.,2.,3.],'field'),([4.,5.,6.],'field')]
print('PASS: nominal bypass; safety zero FF; switch boundary')
