import numpy as np
from study import OUT,OLD,CONF,read,write,sha
from analyze_learning import ci

def run():
 a=np.load(OUT/'N1_predictions.npz');b=np.load(CONF/'frozen_predictions.npz');y=np.load(CONF/'fresh_rollouts.npz');Y=y['outcomes'][:,:,8:,0].mean(-1);rows=np.arange(256);rep={}
 for n in [32,512]:
  z=a[f'N1_o0_n{n}_scalar'];ref=b[f'physical23_eta_n{n}_logits'];err=float(abs(z-ref).max());assert err<1e-5
  rep[str(n)]=dict(max_logit_abs_error=err,exact_choices=bool(np.array_equal(z.argmax(-1),ref.argmax(-1))),success_by_fit=Y[rows,z.argmax(-1)].mean(1).tolist())
 def values(n,arm):
  zz=np.concatenate([a[f'N1_o{i}_n{n}_{arm}'] for i in ([0] if n==512 else [0,1,2])]);return Y[rows,zz.argmax(-1)].mean(0)
 compare=[]
 for arm,n,refarm,refn in [('heads',128,'scalar',128),('heads',256,'scalar',256),('heads',128,'scalar',512),('heads',256,'scalar',512),('heads',512,'scalar',512),('knn',64,'scalar',64),('knn',512,'scalar',512)]:
  d=values(n,arm)-values(refn,refarm);compare.append(dict(model=arm,n=n,reference=refarm,reference_n=refn,mean=float(d.mean()),ci95=ci(d)))
 rec=[read(f) for f in (OUT/'fits').glob('*/complete.json')];n1=[r for r in rec if r['task']['phase']=='N1'];n3=[r for r in rec if r['task']['phase']=='N3'];src=read(OLD/'models_frozen.json');checks=[]
 for arm in ['physical23_eta','physical23_heads']:
  for n in [32,64,128,256,512]:
   for ck in src['selections'][f'{arm}_n{n}']['models']:assert sha(ck['path'])==ck['sha256'];checks.append(ck)
 write(OUT/'REPRODUCTION.json',dict(status='PASSED FROZEN CHECKPOINT INFERENCE AND NUMERICAL RESULT REPRODUCTION',not_retraining_original512=True,architecture='Physical23+eta3 scalar 64x64 SiLU MLP; comparatorPhysical23->16heads',predictions=rep,verified_checkpoints=checks,new_N1_fits=sum(3 for r in n1 if not r['reused']),reused_N1_checkpoints=sum(3 for r in n1 if r['reused']),partial_N3_completed_fits=len(n3)*3,limitation='Old256 state confirmation retains its historical confirmation status; all new comparisons DEVELOPMENT.'))
 write(OUT/'N1_PAIRED_COMPARISONS.json',compare);print(rep,compare)
if __name__=='__main__':run()
