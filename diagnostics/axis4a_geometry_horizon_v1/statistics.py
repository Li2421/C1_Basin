"""Paired hierarchical bootstrap with state parity cross-fitting and unknown bounds."""
from common import *

def oracle(lo,hi,menu,orig=None):
 nc,ns,_,nq=lo.shape;low=np.zeros((nc,ns,nq));high=low.copy();selected=np.zeros((nc,ns),int);parity=np.broadcast_to(np.arange(ns) if orig is None else orig,(nc,ns))%2
 for c in range(nc):
  for fold in range(2):
   train=np.flatnonzero(parity[c]!=fold);test=np.flatnonzero(parity[c]==fold)
   if not len(test):continue
   a=lo[c,train][:,menu].mean((0,2)) if len(train) else np.zeros(len(menu));b=hi[c,train][:,menu].mean((0,2)) if len(train) else np.ones(len(menu));best=menu[int(a.argmax())];possible=np.array([best]) if np.array_equal(a,b) else np.array(menu)[b>=a.max()];selected[c,test]=best;low[c,test]=lo[c,test][:,possible].min(1);high[c,test]=hi[c,test][:,possible].max(1)
 return (low,high),selected

def evaluate(lo,hi,choices,full,fixed,orig=None):
 nc,ns=lo.shape[:2];cc=np.arange(nc)[:,None];ss=np.arange(ns)[None,:];out={};sel={}
 for name,ix in choices.items():
  ix=np.asarray(ix);ix=np.full(nc,int(ix)) if ix.ndim==0 else ix;sel[name]=np.broadcast_to(ix[:,None],(nc,ns));out[name]=(lo[cc,ss,ix[:,None]],hi[cc,ss,ix[:,None]])
 for name,menu in [('best fixed oracle',fixed),('full menu oracle',full)]:out[name],sel[name]=oracle(lo,hi,menu,orig)
 return out,sel

def ci(v):
 z=np.asarray(v);return [float(np.quantile(z[:,0],.025)*100),float(np.quantile(z[:,1],.975)*100)]
def delta(a,b):return [(a[0]-b[1]).mean()*100,(a[1]-b[0]).mean()*100]
def summarize(lo,hi,events,arms,choices,seed):
 full=[i for i,a in enumerate(arms) if a not in ['nominal_only','safety_baseline']];fixed=[arms.index(a) for a in ['F10','F13']];metrics,sel=evaluate(lo,hi,choices,full,fixed);nc,ns=lo.shape[:2];pairs={}
 for a,b in [('Srel','best fixed oracle'),('Srel','Sabs'),('Srel','F13'),('A3','safety baseline'),('S500','safety baseline'),('S500','best fixed oracle'),('A3','A3 absolute')]:
  if a in metrics and b in metrics:pairs[a+' minus '+b]=(a,b)
 pairs['oracle minus A3']=('full menu oracle','A3');boots={k:[] for k in metrics};dboot={k:[] for k in pairs};rng=np.random.default_rng(seed)
 for _ in range(5000):
  cc=rng.integers(nc,size=nc);ss=rng.integers(ns,size=(nc,ns));ch={k:np.asarray(v)[cc] if np.asarray(v).ndim else v for k,v in choices.items()};m,_=evaluate(lo[cc[:,None],ss],hi[cc[:,None],ss],ch,full,fixed,ss)
  for k,(a,b) in m.items():boots[k].append([a.mean(),b.mean()])
  for k,(a,b) in pairs.items():dboot[k].append([(m[a][0]-m[b][1]).mean(),(m[a][1]-m[b][0]).mean()])
 out=dict(methods={},paired={},bootstrap_replicates=5000,controllers=nc,states=ns,seeds=lo.shape[-1]);cc=np.arange(nc)[:,None];ss=np.arange(ns)[None,:];orc=metrics['full menu oracle'];safe=metrics['safety baseline'];f10=metrics['F10']
 for name,(a,b) in metrics.items():
  rl=(orc[0]-b).mean((1,2))*100;rh=(orc[1]-a).mean((1,2))*100;out['methods'][name]=dict(success_pct=[a.mean()*100,b.mean()*100],success_95CI_pct=ci(boots[name]),mean_regret_pp=[rl.mean(),rh.mean()],max_regret_pp=[rl.max(),rh.max()],harm_certain=int(((safe[0]-b).mean((1,2))>=.1-1e-12).sum()),harm_possible=int(((safe[1]-a).mean((1,2))>=.1-1e-12).sum()),catastrophes_certain=int(((f10[0]-b).mean((1,2))>=.1-1e-12).sum()),catastrophes_possible=int(((f10[1]-a).mean((1,2))>=.1-1e-12).sum()),events_pct={e:v[cc,ss,sel[name]].mean()*100 for e,v in events.items()},controller_success_bounds_pct=np.stack([a.mean((1,2)),b.mean((1,2))],1)*100)
 for k,(a,b) in pairs.items():out['paired'][k]=dict(estimate_pp=delta(metrics[a],metrics[b]),CI_pp=ci(dboot[k]))
 out['in_sample_oracle_pct']={name:[lo[:,:,menu].mean((1,3)).max(1).mean()*100,hi[:,:,menu].mean((1,3)).max(1).mean()*100] for name,menu in [('fixed',fixed),('full',full)]}
 def outcomes(i):return np.where(events['unknown'][:,:,i]>0,'unknown',np.where(events['collision'][:,:,i]>0,'collision',np.where(lo[:,:,i]>0,'success','timeout')))
 nominal=outcomes(arms.index('nominal_only'));safety=outcomes(arms.index('safety_baseline'));out['nominal_to_safety']={}
 for x in ['success','collision','timeout','unknown']:
  mask=nominal==x;den=int(mask.sum());counts={y:int(((safety==y)&mask).sum()) for y in ['success','collision','timeout','unknown']};out['nominal_to_safety'][x]=dict(denominator=den,counts=counts,fractions={k:v/den if den else None for k,v in counts.items()})
 return clean(out)
