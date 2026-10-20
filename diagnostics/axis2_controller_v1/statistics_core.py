"""Preregistered paired inference; no simulator or controller imports."""
from core import *

def oracle_bounds(lo,hi,indices,original_indices=None):
 nc,ns,_,nq=lo.shape;low=np.zeros((nc,ns,nq));high=low.copy();point=low.copy();selected=np.zeros((nc,ns),int)
 parity=np.broadcast_to(np.arange(ns) if original_indices is None else original_indices,(nc,ns))%2
 for c in range(nc):
  for fold in range(2):
   train=np.flatnonzero(parity[c]!=fold);test=np.flatnonzero(parity[c]==fold)
   if not len(test):continue
   if not len(train):tl=np.zeros(len(indices));th=np.ones(len(indices))
   else:tl=lo[c,train][:,indices].mean((0,2));th=hi[c,train][:,indices].mean((0,2))
   best=indices[int(tl.argmax())];possible=np.array([best]) if np.array_equal(tl,th) else np.asarray(indices)[th>=tl.max()]
   selected[c,test]=best;point[c,test]=lo[c,test,best];low[c,test]=lo[c,test][:,possible].min(1);high[c,test]=hi[c,test][:,possible].max(1)
 return low,high,point,selected

def evaluate(lo,hi,choices,full,fixed,orig=None):
 nc,ns=lo.shape[:2];cc=np.arange(nc)[:,None];ss=np.arange(ns)[None,:];out={};selected={}
 for name,ix in choices.items():
  ix=np.asarray(ix);ix=np.full(nc,int(ix)) if ix.ndim==0 else ix;selected[name]=np.broadcast_to(ix[:,None],(nc,ns));out[name]=(lo[cc,ss,ix[:,None]],hi[cc,ss,ix[:,None]])
 for name,menu in [('best fixed oracle',fixed),('full menu oracle',full)]:
  a,b,p,s=oracle_bounds(lo,hi,menu,orig);out[name]=(a,b);selected[name]=s
 return out,selected

def interval(samples):
 z=np.asarray(samples);return [float(np.quantile(z[:,0],.025)*100),float(np.quantile(z[:,1],.975)*100)]
def difference(a,b):return [float((a[0]-b[1]).mean()*100),float((a[1]-b[0]).mean()*100)]
def summarize(label,lo,hi,events,choices,full,fixed,seed):
 nc,ns=lo.shape[:2]
 if not nc:return dict(cohort=label,status='INCONCLUSIVE',reason='No eligible controllers',methods={})
 metrics,selected=evaluate(lo,hi,choices,full,fixed);boots={k:[] for k in metrics};diffspec={}
 if 'A3 full' in metrics:
  diffspec={'A3_minus_safety':('A3 full','safety baseline'),'CT2_regret_A3_minus_A0':('A0 fixed2','A3 fixed2'),'CT3_regret_mixed_minus_A3':('A3 full','mixed full'),'CT1_regret':('full menu oracle','A3 full')}
 if 'original frozen method' in metrics:diffspec.update(original_minus_nominal=('original frozen method','nominal only'),original_minus_safety=('original frozen method','safety baseline'))
 diffboots={k:[] for k in diffspec};rng=np.random.default_rng(seed)
 for _ in range(5000):
  cc=rng.integers(nc,size=nc);ss=rng.integers(ns,size=(nc,ns));ch={k:(np.full(nc,int(v)) if np.asarray(v).ndim==0 else np.asarray(v)[cc]) for k,v in choices.items()};mm,_=evaluate(lo[cc[:,None],ss],hi[cc[:,None],ss],ch,full,fixed,ss)
  for k,(a,b) in mm.items():boots[k].append([a.mean(),b.mean()])
  for k,(a,b) in diffspec.items():diffboots[k].append([(mm[a][0]-mm[b][1]).mean(),(mm[a][1]-mm[b][0]).mean()])
 out=dict(cohort=label,controllers=nc,states=ns,seeds=lo.shape[-1],bootstrap_replicates=5000,methods={},paired={})
 cc=np.arange(nc)[:,None];ss=np.arange(ns)[None,:]
 for name,(a,b) in metrics.items():
  orc=metrics['full menu oracle'];rl=(orc[0]-b).mean((1,2));rh=(orc[1]-a).mean((1,2));f10=metrics['F10'];safe=metrics['safety baseline'];sel=selected[name]
  out['methods'][name]=dict(success_pct=[a.mean()*100,b.mean()*100],success_95CI_pct=interval(boots[name]),mean_regret_pp=[rl.mean()*100,rh.mean()*100],max_regret_pp=[rl.max()*100,rh.max()*100],catastrophes_certain=int(((f10[0]-b).mean((1,2))>=.10-1e-12).sum()),catastrophes_possible=int(((f10[1]-a).mean((1,2))>=.10-1e-12).sum()),harm_certain=int(((safe[0]-b).mean((1,2))>=.10-1e-12).sum()),harm_possible=int(((safe[1]-a).mean((1,2))>=.10-1e-12).sum()),events_pct={e:v[cc,ss,sel].mean()*100 for e,v in events.items()})
 for k,(a,b) in diffspec.items():out['paired'][k]=dict(estimate_pp=difference(metrics[a],metrics[b]),CI_pp=interval(diffboots[k]))
 out['in_sample_oracle_pct']={k:[lo[:,:,idx].mean((1,3)).max(1).mean()*100,hi[:,:,idx].mean((1,3)).max(1).mean()*100] for k,idx in [('fixed',fixed),('full',full)]}
 if 'A3 full' in metrics:
  rg=out['paired']['CT1_regret'];gain=out['paired']['A3_minus_safety']['CI_pp'];m=out['methods']['A3 full'];status='SUPPORTED' if rg['estimate_pp'][1]<=3 and gain[0]>0 and m['harm_possible']==0 else 'REJECTED' if rg['estimate_pp'][0]>3 or gain[1]<=0 or m['harm_certain']>0 else 'INCONCLUSIVE'
  f10=metrics['F10'];f13=metrics['F13'];n10=int(((f10[0]-f13[1]).mean((1,2))>=.10-1e-12).sum());n13=int(((f13[0]-f10[1]).mean((1,2))>=.10-1e-12).sum());ci2=out['paired']['CT2_regret_A3_minus_A0']['CI_pp'];ci3=out['paired']['CT3_regret_mixed_minus_A3']['CI_pp']
  out['CT1']=dict(status=status,regret_pp=rg['estimate_pp'],A3_minus_safety_CI_pp=gain,harm_certain=m['harm_certain'],harm_possible=m['harm_possible'])
  out['CT2']=dict(status='INCONCLUSIVE' if min(n10,n13)<3 else 'SUPPORTED' if ci2[1]<0 else 'REJECTED',prefers_F10_by_10pp=n10,prefers_F13_by_10pp=n13,regret_difference_CI_pp=ci2,interpretation='Insufficient preference disagreement' if min(n10,n13)<3 else 'Preregistered superiority criterion; failed significance is not evidence of zero effect')
  out['CT3']=dict(status='SUPPORTED' if ci3[1]<0 else 'REJECTED',regret_difference_CI_pp=ci3,interpretation='Preregistered superiority criterion; failed significance is not evidence of zero effect')
 def outcomes(arm):return np.where(events['unknown'][:,:,arm]>0,'unknown',np.where(events['collision'][:,:,arm]>0,'collision',np.where(lo[:,:,arm]>0,'success','timeout')))
 nominal=outcomes(int(choices['nominal only']));safe=outcomes(int(choices['safety baseline']));out['nominal_to_safety']={}
 for x in ['success','collision','timeout','unknown']:
  mask=nominal==x;den=int(mask.sum());counts={y:int(((safe==y)&mask).sum()) for y in ['success','collision','timeout','unknown']};out['nominal_to_safety'][x]=dict(denominator=den,counts=counts,fractions={k:v/den if den else None for k,v in counts.items()})
 return clean(out)
