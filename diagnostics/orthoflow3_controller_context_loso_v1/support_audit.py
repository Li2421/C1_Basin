"""Post-freeze source/target context support, label-free and non-selective."""
import csv
import numpy as np
from scipy.spatial.distance import cdist
from .train import OUT,FOLDS,KINDS,load,dump,data
from .evaluate import normalized
def main():
 assert (OUT/'target_predictions.json').exists()
 result=[]
 for fold in FOLDS:
  raw=np.load(OUT/f'target_context_{fold}.npz')
  for kind in ('C1_full','C3_full'):
   rows,x,e,c,si,s,f,w,g=data(fold,kind);tr=np.array([r['split']=='train' for r in rows]);train=c[tr]
   if kind=='C1_full':train=np.unique(train,axis=0)
   target=normalized(raw['C1'] if kind=='C1_full' else raw['C3'],load(OUT/fold/kind/'normalization.json'))
   near=[]
   for start in range(0,len(target),64):near.extend(cdist(target[start:start+64],train).min(1))
   lo=train.min(0);hi=train.max(0);outside=((target<lo-1e-6)|(target>hi+1e-6))
   result.append({'fold':fold,'kind':kind,'source_vectors':len(train),'target_vectors':len(target),
     'normalized_context_NN_p50':float(np.median(near)),'NN_p95':float(np.quantile(near,.95)),
     'outside_any_source_coordinate_range_fraction':float(outside.any(1).mean()),
     'outside_coordinate_count_mean':float(outside.sum(1).mean()),'max_abs_source_z_target':float(abs(target[:,:-1]).max()),
     'target_invalid_contexts':int((target[:,-1]==0).sum())})
 with (OUT/'context_support_audit.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
 dump(OUT/'context_support_scope.json',{'posthoc_only':True,'used_for_selection':False,'distance_is_not_identifiability_proof':True})
if __name__=='__main__':main()
