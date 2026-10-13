"""Exploratory empirical η-regime coverage from existing training labels."""
import json
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent

def main():
    p=json.loads((BASE/'protocol.json').read_text());extra=json.loads((BASE/'stage_b/protocol.json').read_text())
    a=np.load(BASE/'LABELS.npz')['outcomes'][:,:128,:,:4];b=np.load(BASE/'stage_b/LABELS.npz')['outcomes'];rows=[]
    panels=[(name,x,'original') for name,x in zip(p['controller_order'],a)]+[(name,x,'added') for name,x in zip(extra['controller_order'],b)]
    for name,x,group in panels:
        mask=(x!=-2).any(-1);assert np.all(mask.sum(0)==64)
        lower=np.array([(x[:,j]==1).mean(-1)[mask[:,j]].mean() for j in range(16)])
        upper=lower+np.array([(x[:,j]==-1).mean(-1)[mask[:,j]].mean() for j in range(16)])
        s=(x==1).sum(-1);v=((x==1)|(x==0)).sum(-1)
        observed=np.nanmean(np.divide(s,v,out=np.full_like(s,np.nan,dtype=float),where=v>0),axis=0)
        j=int(lower.argmax())
        rows.append(dict(controller=name,group=group,observed_lower_bound_best_eta=j,
                         observed_valid_best_eta=int(observed.argmax()),best_lower=float(lower[j]),best_upper=float(upper[j]),
                         eta13_lower=float(lower[13]),eta13_upper=float(upper[13]),
                         success_curve_lower=lower.tolist(),success_curve_upper=upper.tolist()))
    result=dict(status='EXPLORATORY TRAIN LABEL SUMMARY',states=128,per_eta_states=64,seeds=4,
                missing_eta_allocation='balanced random eight heads per state; only requested cells included',
                unknown_policy='reported lower and upper success bounds, no unknown recoded as failure',rows=rows)
    (BASE/'SOURCE_REGIME_COVERAGE.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
