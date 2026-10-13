import numpy as np
from study import OUT,CONF,read,write
from analyze_learning import ci
p=np.load(OUT/'N3_predictions.npz');y=np.load(CONF/'fresh_rollouts.npz')['outcomes'][:,:,8:,0].mean(-1);tasks=read(OUT/'splits.json')['tasks'];rows=[]
for region in [0,1]:
 for arm in ['scalar','heads','knn','prior']:
  vv={}
  for mode in ['shift','random']:
   ts=[t for t in tasks if t['phase']=='N3' and t['source_region']==region and t['mode']==mode];ix=np.array(ts[0]['test'])-608
   z=np.concatenate([p[t['key']+'_'+arm] for t in ts]);choice=z[:,ix].argmax(-1);v=y[ix][np.arange(len(ix)),choice];gain=v.mean(0)-y[ix,13];vv[mode]=v.mean(0)
   rows.append(dict(source_region=region,target_region=1-region,arm=arm,mode=mode,test_states=len(ix),success=float(v.mean()),gain=float(gain.mean()),gain_ci95=ci(gain),hindsight_fixed_eta=int(y[ix].mean(0).argmax()),hindsight_fixed_success=float(y[ix].mean(0).max())))
  diff=vv['shift']-vv['random'];rows.append(dict(source_region=region,arm=arm,comparison='shift_minus_random',mean=float(diff.mean()),ci95=ci(diff)))
write(OUT/'N3_SUMMARY.json',rows)
print(rows)
