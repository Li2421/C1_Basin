from pathlib import Path
import numpy as np
from . import factor_data,learn,metrics
from .factor_analysis import specs
from .main_analysis import write
HERE=Path(__file__).resolve().parent

def main():
 D=factor_data.load();trains,tests=specs(D);result={};test=list(range(8,16));bs=np.random.default_rng(20261011).integers(0,8,(2000,8))
 for reg in ['FAMILY','COUNT','TIME']:
  tr,y=learn.rows(D,[(g,list(range(8)),[0,1]) for g in trains[reg]],allow_unknown=True);prior=learn.prior(tr,y,2,4);ps=[learn.fit(D,tr,y,'ENT_mech_phys',s,HERE/'models/normalization'/reg) for s in learn.SEEDS];gs=tests[reg];yy=D['Y'][gs][:,test]
  point,aa=metrics.evaluate(yy,np.mean(ps,0)[gs][:,test],prior);record={'ensemble':point,'per_seed':{str(s):metrics.stats(yy,p[gs][:,test],prior)[0] for s,p in zip(learn.SEEDS,ps)},'comparisons':{}}
  for arm in ['ENT','ENT_mech']:
   ps0=[np.load(HERE/'models/factors'/reg/f'{arm}_{s}.npz')['pred'] for s in learn.SEEDS];base,bb=metrics.stats(yy,np.mean(ps0,0)[gs][:,test],prior);diff=aa['success']-bb['success'];dq=aa['q_mse']-bb['q_mse'];record['comparisons'][arm]={'delta_success':diff.mean(),'ci95':np.percentile(diff[bs].mean(1),[2.5,97.5]),'delta_q_mse':dq.mean(),'q_mse_ci95':np.percentile(dq[bs].mean(1),[2.5,97.5])}
  result[reg]=record;print(reg,point['success'],record['comparisons'],flush=True)
 write(HERE/'normalization_results.json',{'scope':'exploratory label-free fixed physical-unit normalization diagnostic','results':result,'new_rollouts':0,'fits':9})
if __name__=='__main__':main()
