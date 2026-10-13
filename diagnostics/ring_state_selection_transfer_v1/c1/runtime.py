"""Continuous affine FF operator anchored on the exact frozen native Flow endpoint."""
import numpy as np
from diagnostics.controller_flow_endpoint_probe_v1.runtime import FlowEndpointMatchedField
class ContinuousFlowField(FlowEndpointMatchedField):
 def __init__(self,base,kappa):
  super().__init__(base,'isotropic');self.kappa=float(kappa)
 def matrix(self,env):
  j=np.array([[0.,-1.],[1.,0.]])
  return 2.5*(np.eye(2*len(env.positions))+self.kappa*np.kron(np.eye(len(env.positions)),j))
def make(base,name):
 if name in ['isotropic','rotate_positive','rotate_negative']:return FlowEndpointMatchedField(base,name)
 return ContinuousFlowField(base,float(name))
