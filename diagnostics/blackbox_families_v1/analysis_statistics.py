"""Paired controller then state-cluster intervals, with unknown envelopes."""
import numpy as np
ARMS=['nominal_only','safety_baseline','F10','F13','S500','Sprog','RHR','RHR_fair']
MENU=[2,3,4,5,6]

def oracle(lo,hi,state_indices=None,menu=None):
 menu=MENU if menu is None else menu
 nc,ns,_,nq=lo.shape;orig=np.broadcast_to(np.arange(ns),(nc,ns)) if state_indices is None else state_indices;low=np.zeros((nc,ns,nq));high=low.copy();point=low.copy();choice=np.zeros((nc,ns),int)
 for c in range(nc):
  for fold in [0,1]:
   train=np.flatnonzero(orig[c]%2!=fold);test=np.flatnonzero(orig[c]%2==fold)
   if not len(test):continue
   tl=lo[c,train][:,menu].mean((0,2)) if len(train) else np.zeros(len(menu));th=hi[c,train][:,menu].mean((0,2)) if len(train) else np.ones(len(menu));best=menu[int(tl.argmax())];possible=[best] if np.array_equal(tl,th) else np.asarray(menu)[th>=tl.max()]
   choice[c,test]=best;point[c,test]=lo[c,test,best];low[c,test]=lo[c,test][:,possible].min(1);high[c,test]=hi[c,test][:,possible].max(1)
 return low,high,point,choice

def values(lo,hi,choices,orig=None):
 nc,ns=lo.shape[:2];cc=np.arange(nc)[:,None];ss=np.arange(ns)[None,:];out={};selected={}
 for name,ix in choices.items():
  ix=np.asarray(ix);ix=np.full(nc,int(ix)) if ix.ndim==0 else ix;selected[name]=np.broadcast_to(ix[:,None],(nc,ns));out[name]=(lo[cc,ss,ix[:,None]],hi[cc,ss,ix[:,None]])
 a,b,point,choice=oracle(lo,hi,orig);out['crossfit_oracle']=(a,b);selected['crossfit_oracle']=choice
 a,b,_,choice=oracle(lo,hi,orig,[2,3]);out['fixed_oracle']=(a,b);selected['fixed_oracle']=choice
 nc,ns=lo.shape[:2];scores=lo[:,:,MENU].mean((1,3));idx=np.asarray(MENU)[scores.argmax(-1)];selected['in_sample_oracle']=np.broadcast_to(idx[:,None],(nc,ns));out['in_sample_oracle']=(lo[cc,ss,idx[:,None]],hi[cc,ss,np.asarray(MENU)[hi[:,:,MENU].mean((1,3)).argmax(-1)][:,None]])
 return out,selected

def summarize(label,lo,hi,events,choices,seed):
 nc,ns=lo.shape[:2];metrics,selected=values(lo,hi,choices);rng=np.random.default_rng(seed);boots={k:[] for k in metrics};spec={'schedule_minus_safety':('S500','safety_baseline'),'right_hand_minus_schedule':('RHR','S500'),'fair_right_hand_minus_schedule':('RHR_fair','S500'),'progress_minus_proportion':('Sprog','S500'),'expanded_regret':('crossfit_oracle','expanded'),'expanded_regret_minus_always_schedule':('S500','expanded'),'pilot4_regret':('crossfit_oracle','pilot_4')};diffs={k:[] for k in spec}
 for _ in range(5000):
  cc=rng.integers(nc,size=nc);ss=rng.integers(ns,size=(nc,ns));ch={k:(np.full(nc,int(v)) if np.asarray(v).ndim==0 else np.asarray(v)[cc]) for k,v in choices.items()};mm,_=values(lo[cc[:,None],ss],hi[cc[:,None],ss],ch,ss)
  for k,(a,b) in mm.items():boots[k].append([a.mean()*100,b.mean()*100])
  for k,(a,b) in spec.items():diffs[k].append([(mm[a][0]-mm[b][1]).mean()*100,(mm[a][1]-mm[b][0]).mean()*100])
 def ci(x):return [float(np.quantile(np.asarray(x)[:,0],.025)),float(np.quantile(np.asarray(x)[:,1],.975))]
 out=dict(unit=label,controllers=nc,states=ns,seeds=lo.shape[-1],bootstrap_replicates=5000,methods={},paired={},evidence_level='CONFIRMATORY');cc=np.arange(nc)[:,None];ss=np.arange(ns)[None,:]
 for name,(a,b) in metrics.items():
  ref=metrics['crossfit_oracle'];rl=(ref[0]-b).mean((1,2))*100;rh=(ref[1]-a).mean((1,2))*100;safe=metrics['safety_baseline'];f10=metrics['F10'];sel=selected[name]
  out['methods'][name]=dict(success_pct=[float(a.mean()*100),float(b.mean()*100)],success_CI95_pct=ci(boots[name]),mean_regret_pp=[float(rl.mean()),float(rh.mean())],max_regret_pp=[float(rl.max()),float(rh.max())],harm_certain=int(((safe[0]-b).mean((1,2))>=.1-1e-12).sum()),harm_possible=int(((safe[1]-a).mean((1,2))>=.1-1e-12).sum()),catastrophe_certain=int(((f10[0]-b).mean((1,2))>=.1-1e-12).sum()),catastrophe_possible=int(((f10[1]-a).mean((1,2))>=.1-1e-12).sum()),events_pct={k:float(v[cc,ss,sel].mean()*100) for k,v in events.items()})
 for k,(a,b) in spec.items():out['paired'][k]=dict(estimate_pp=[float((metrics[a][0]-metrics[b][1]).mean()*100),float((metrics[a][1]-metrics[b][0]).mean()*100)],CI95_pp=ci(diffs[k]))
 def outcomes(arm):return np.where(events['unknown'][:,:,arm]>0,'unknown',np.where(events['collision'][:,:,arm]>0,'collision',np.where(lo[:,:,arm]>0,'success','timeout')))
 a=outcomes(0);b=outcomes(1);out['nominal_to_safety']={}
 for event in ['success','collision','timeout','unknown']:
  mask=a==event;den=int(mask.sum());counts={e:int(((b==e)&mask).sum()) for e in ['success','collision','timeout','unknown']};out['nominal_to_safety'][event]=dict(denominator=den,counts=counts,fractions={k:v/den if den else None for k,v in counts.items()})
 ci1=out['paired']['schedule_minus_safety']['CI95_pp'];out['endpoint1']='SUPPORTED' if ci1[0]>0 else 'REJECTED' if ci1[1]<=0 else 'INCONCLUSIVE'
 return out
