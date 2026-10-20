from exact_reuse import compatible
import copy
base=dict(N=2,H=2000,kind='Flow',state_uid='s',seed=0,evaluation_config={'fixed':True},arm='F13',eta_a=[1.,0.,0.],eta_b=[1.,0.,0.]);target=dict(base,arm='S13t1',eta_b=[2.,0.,0.],switch_step=857)
p=dict(case=base,physical_steps=857,outcome='success',error=None);assert compatible(p,target)
p.update(outcome='collision');assert compatible(p,target)
p.update(outcome='numerical_failure',error={'step':857});assert not compatible(p,target)
p.update(physical_steps=856,error={'step':856});assert compatible(p,target)
p.update(physical_steps=2000,outcome='timeout',error=None);assert not compatible(p,target)
p.update(physical_steps=900,outcome='success');assert not compatible(p,target)
p.update(physical_steps=500,outcome='success');assert not compatible(p,dict(target,seed=1));assert not compatible(p,dict(target,eta_a=[1.00000000001,0.,0.]));assert not compatible(p,dict(target,H=3000))
print('Pre-switch terminal/unknown boundary and identity tests PASS')
