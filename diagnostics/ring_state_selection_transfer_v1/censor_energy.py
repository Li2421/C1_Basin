"""Conservative interval propagation for independent-half controller contrast energies."""
import numpy as np
from study import OUT,read,write

def bounds(y):
 x=(y==1).astype(float);u=(y<0).astype(float);parts=[]
 for a in [slice(0,8),slice(8,16)]:
  d=x[1,:,:,a].mean(-1)-x[0,:,:,a].mean(-1);lo=d-u[0,:,:,a].mean(-1);hi=d+u[1,:,:,a].mean(-1)
  ce=(lo.mean(0)-hi.mean(),hi.mean(0)-lo.mean());sce=(lo-hi.mean(0)-hi.mean(1,keepdims=True)+lo.mean(),hi-lo.mean(0)-lo.mean(1,keepdims=True)+hi.mean());parts.append([ce,sce])
 result=[]
 for j in range(2):
  a,b=parts[0][j],parts[1][j];products=np.stack([aa*bb for aa in a for bb in b]);result.append([float(products.min(0).mean()),float(products.max(0).mean())])
 return result
r=read(OUT/'C0_FF_TT_FOCUS.json');y=np.load(OUT/'controller_panels'/(r['panel']+'.npz'))['Y'];rng=np.random.default_rng(2026101103);boot=np.array([bounds(y[:,rng.integers(y.shape[1],size=y.shape[1])]) for _ in range(5000)]);out=dict(names=['controller_eta','state_controller_eta'],energy_censor_bounds=bounds(y),paired_state_bootstrap_censor95=[[float(np.quantile(boot[:,j,0],.025)),float(np.quantile(boot[:,j,1],.975))] for j in range(2)],definition='Interval arithmetic on centered contrasts and independent seed-half products; conservative with unknown outcomes. Negative unbiased energy estimates permitted.');write(OUT/'C0_CENSOR_ENERGY.json',out);print(out)
