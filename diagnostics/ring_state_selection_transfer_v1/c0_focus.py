"""Prespecified controller audit reliability diagnostics: censoring and tied Q maxima."""
import numpy as np
from study import OUT,read,write
from analyze_learning import ci

def run():
 r=next(r for r in read(OUT/'C0_RESULTS.json')['pair_panels'] if set(r['families'])=={'FF','TT'})
 p=np.load(r['file']);y=p['Y'];lo=(y==1).astype(float);u=(y<0).astype(float);C,S,E,K=y.shape;records=[];pooled={}
 for tie in ['bank_order','discovery_global_prior','fixed13_tie']:
  per=[]
  for flip in [0,1]:
   a=slice(0,8) if flip==0 else slice(8,16);b=slice(8,16) if flip==0 else slice(0,8);q=lo[:,:,:,a].mean(-1);v=lo[:,:,:,b].mean(-1);uv=u[:,:,:,b].mean(-1);ch=[]
   for c in range(2):
    if tie=='bank_order':sel=q[c].argmax(-1)
    else:
     prior=q[c].mean(0);priority=prior if tie=='discovery_global_prior' else (np.arange(E)==13).astype(float)
     sel=np.array([np.lexsort((np.arange(E),-priority,-a))[0] for a in q[c]])
    ch.append(sel)
   rows=[]
   for c in range(2):
    d=v[c,np.arange(S),ch[c]]-v[c,np.arange(S),ch[1-c]];du=uv[c,np.arange(S),ch[c]];dv=uv[c,np.arange(S),ch[1-c]]
    rows.append(dict(controller=r['families'][c],matched_minus_mismatched=float(d.mean()),ci95=ci(d),censor_bounds=[float((d-dv).mean()),float((d+du).mean())],worst_unknown_ci95=ci(d-dv),discovery_tied_max_fraction=float((np.sum(q[c]==q[c].max(-1,keepdims=True),axis=-1)>1).mean()),own_choices=ch[c].tolist()))
   records.append(dict(tie=tie,half=flip,rows=rows));per.append([v[c,np.arange(S),ch[c]]-v[c,np.arange(S),ch[1-c]] for c in range(2)])
  av=np.mean(per,axis=0);pooled[tie]=[dict(controller=r['families'][c],effect=float(av[c].mean()),ci95=ci(av[c])) for c in range(2)]
 # Conservative Q-level bounds on exact same support.
 diff=lo[1].mean((1,2))-lo[0].mean((1,2));low=diff-u[0].mean((1,2));high=diff+u[1].mean((1,2))
 # Independent seed contrast energy after dropping NO outcomes: censor uncertainty explicitly; complete-label sensitivity separately.
 eligible=np.all(y>=0,axis=(0,2,3));z=y[:,eligible];energy=[]
 for arr,label in [(y,'all_states_lower_endpoint'),(z,'complete_state_sensitivity')]:
  l=(arr==1).astype(float);da=l[1,:,:,:8].mean(-1)-l[0,:,:,:8].mean(-1);db=l[1,:,:,8:].mean(-1)-l[0,:,:,8:].mean(-1)
  def decompose(d):
   ce=d.mean(0)-d.mean();sce=d-d.mean(0,keepdims=True)-d.mean(1,keepdims=True)+d.mean();return ce,sce
  aa,bb=decompose(da),decompose(db);rng=np.random.default_rng(2026101103);est=[]
  for _ in range(5000):
   ix=rng.integers(len(da),size=len(da));a,b=decompose(da[ix]),decompose(db[ix]);est.append([np.mean(a[j]*b[j]) for j in range(2)])
  energy.append(dict(scope=label,states=int(arr.shape[1]),point=[float(np.mean(aa[j]*bb[j])) for j in range(2)],ci95=np.quantile(est,[.025,.975],axis=0).T.tolist()))
 write(OUT/'C0_FF_TT_FOCUS.json',dict(panel=r['panel'],states=S,etas=E,seeds=K,unknown_by_controller=(y<0).sum((1,2,3)).tolist(),q_TT_minus_FF_bounds=[float(low.mean()),float(high.mean())],q_level_censor_ci95=[ci(low)[0],ci(high)[1]],tie_diagnostics=records,pooled_seed_swapped=pooled,energies=energy,interpretation='Matched oracle dependence can be inflated by arbitrary ties; global fixed optimum unchanged. Neither oracle nor tie sensitivity establishes a deployable held-controller selector.'))
 print(pooled)
if __name__=='__main__':run()
