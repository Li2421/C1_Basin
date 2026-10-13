"""Label-free feature scaling audit; reads only input arrays and source indices."""
from pathlib import Path
import json
import numpy as np
from .main_analysis import write
HERE=Path(__file__).resolve().parent

def main():
 d=np.load(HERE/'data/factors.npz');ctx=json.loads((HERE/'data/factor_contexts.json').read_text());out={}
 tests={'FAMILY':[i for i,c in enumerate(ctx) if c['topology']=='parallel'],'COUNT':[i for i,c in enumerate(ctx) if c['N']==4],'TIME':[i for i,c in enumerate(ctx) if c['target_horizon']==850]}
 names={'raw':['width','N','H_seconds','passage_count','holes'],'mech':['clearance_ratio','euclidean_time_margin','detour_ratio','holes','passage_count','demand_per_opening','approach_imbalance','wall_clearance']}
 for reg,tg in tests.items():
  tr=np.load(HERE/'models/factors'/reg/'ENT_17.npz')['tr'];si=tr[:,0]*16+tr[:,1];ti=np.array([g*16+s for g in tg for s in range(8,16)])
  out[reg]={}
  for kind in ['raw','mech']:
   x=d[kind];mu=x[si].mean(0);sd=x[si].std(0);z=(x[ti]-mu)/np.maximum(sd,.01)
   out[reg][kind]={'names':names[kind],'source_sd':sd,'max_abs_target_standardized':abs(z).max(0),'source_constant_but_target_changes':[names[kind][j] for j in range(len(mu)) if sd[j]<1e-8 and np.max(abs(x[ti,j]-mu[j]))>1e-6]}
 write(HERE/'feature_shift_audit.json',{'uses_outcome_labels':False,'audit':out})
 print(out)
if __name__=='__main__':main()
