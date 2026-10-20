"""New arms, using original Runtime APIs and unchanged simulator."""
import numpy as np

def nominal_only(rt,env,key):
 _,flow,_=rt.nominal(env,key)
 return flow,0

def action(rt,env,key,case,step):
 if case['arm']=='nominal_only':return nominal_only(rt,env,key)
 eta=np.asarray(case['eta_a'] if step<case['switch_step'] else case['eta_b'],np.float64)
 flow,_,calls=rt.action(env,key,eta,'field')
 return flow,calls
