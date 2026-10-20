"""Paired state-fold selection and bounded hierarchical bootstrap."""
from core import *
FIXED=[2,3,4];CORRECTION=list(range(2,9));FULL=list(range(9));SCHEDULE=[5,6,7,8]
def pick(a,b,menu):
 best=menu[int(np.argmax(a))];possible=np.array([best]) if np.array_equal(a,b) else np.asarray(menu)[b>=np.max(a)];return best,possible

def oracle(lo,hi,menu,orig=None):
 nc,ns,_,nq=lo.shape;out=(np.zeros((nc,ns,nq)),np.zeros((nc,ns,nq)));selected=np.zeros((nc,ns),int);parity=np.broadcast_to(np.arange(ns) if orig is None else orig,(nc,ns))%2
 for c in range(nc):
  for fold in [0,1]:
   train=np.flatnonzero(parity[c]!=fold);test=np.flatnonzero(parity[c]==fold)
   if not len(test):continue
   a=lo[c,train][:,menu].mean((0,2)) if len(train) else np.zeros(len(menu));b=hi[c,train][:,menu].mean((0,2)) if len(train) else np.ones(len(menu));best,possible=pick(a,b,menu);selected[c,test]=best;out[0][c,test]=lo[c,test][:,possible].min(1);out[1][c,test]=hi[c,test][:,possible].max(1)
 return out,selected

def source_choices(source):
 out=dict(per_controller={},agnostic={})
 for fold in [0,1]:
  train=np.flatnonzero(np.arange(32)%2!=fold)
  for c,kind in enumerate(source['classes']):
   a=source['lower'][c,train][:,CORRECTION].mean((0,2));b=source['upper'][c,train][:,CORRECTION].mean((0,2));best,possible=pick(a,b,CORRECTION);out['per_controller'].setdefault(kind,{})[str(fold)]=dict(point=best,possible=possible.tolist())
  a=source['lower'][:,train][:,:,CORRECTION].mean((0,1,3));b=source['upper'][:,train][:,:,CORRECTION].mean((0,1,3));best,possible=pick(a,b,CORRECTION);out['agnostic'][str(fold)]=dict(point=best,possible=possible.tolist())
 return out

def transfer(lo,hi,classes,source,orig=None,agnostic=False,source_controller_sample=None):
 nc,ns=lo.shape[:2];out=(np.zeros((nc,ns,2)),np.zeros((nc,ns,2)));selected=np.zeros((nc,ns),int);indices=np.broadcast_to(np.arange(ns) if orig is None else orig,(nc,ns));sc=np.arange(len(source['classes'])) if source_controller_sample is None else source_controller_sample
 for c,kind in enumerate(classes):
  for fold in [0,1]:
   test=np.flatnonzero(indices[c]%2==fold);train=indices[c][indices[c]%2!=fold]
   if not len(test):continue
   if not len(train):a=np.zeros(len(CORRECTION));b=np.ones(len(CORRECTION))
   elif agnostic:a=source['lower'][sc][:,train][:,:,CORRECTION].mean((0,1,3));b=source['upper'][sc][:,train][:,:,CORRECTION].mean((0,1,3))
   else:
    j=source['classes'].index(kind);a=source['lower'][j,train][:,CORRECTION].mean((0,2));b=source['upper'][j,train][:,CORRECTION].mean((0,2))
   best,possible=pick(a,b,CORRECTION);selected[c,test]=best;out[0][c,test]=lo[c,test][:,possible].min(1);out[1][c,test]=hi[c,test][:,possible].max(1)
 return out,selected

def evaluate(lo,hi,classes,source=None,orig=None,source_controller_sample=None):
 nc,ns=lo.shape[:2];m={};sel={};mapping={'nominal only':0,'safety baseline':1,'F10':2,'F13':3,'F15':4,'S13t1':5,'S13t2':6,'S15t1':7,'S15t2':8,'S500 class':6}
 for name,i in mapping.items():m[name]=(lo[:,:,i],hi[:,:,i]);sel[name]=np.full((nc,ns),i)
 for name,menu in [('best fixed oracle',FIXED),('full menu oracle',FULL),('best schedule oracle',SCHEDULE)]:m[name],sel[name]=oracle(lo,hi,menu,orig)
 if source:
  m['source mean transfer'],sel['source mean transfer']=transfer(lo,hi,classes,source,orig,True,source_controller_sample)
  if all(k in source['classes'] for k in classes):m['source per-controller transfer'],sel['source per-controller transfer']=transfer(lo,hi,classes,source,orig,False,source_controller_sample)
 return m,sel

def ci(v):
 z=np.asarray(v);return [float(np.quantile(z[:,0],.025)*100),float(np.quantile(z[:,1],.975)*100)]
def difference(a,b):return [(a[0]-b[1]).mean()*100,(a[1]-b[0]).mean()*100]
def summarize(data,source,seed):
 lo,hi,classes=data['lower'],data['upper'],data['classes'];nc,ns=lo.shape[:2];metrics,selected=evaluate(lo,hi,classes,source);pairs={'best_schedule_minus_fixed':('best schedule oracle','best fixed oracle')}
 if source:
  pairs['transfer_mean_minus_safety']=('source mean transfer','safety baseline');pairs['transfer_mean_regret']=('full menu oracle','source mean transfer')
  if 'source per-controller transfer' in metrics:pairs.update(transfer_controller_minus_safety=('source per-controller transfer','safety baseline'),transfer_controller_regret=('full menu oracle','source per-controller transfer'))
 boots={k:[] for k in metrics};diffs={k:[] for k in pairs};rng=np.random.default_rng(seed)
 for _ in range(5000):
  cc=rng.integers(nc,size=nc);ss=rng.integers(ns,size=(nc,ns));source_cc=rng.integers(len(source['classes']),size=len(source['classes'])) if source else None;mm,_=evaluate(lo[cc[:,None],ss],hi[cc[:,None],ss],[classes[c] for c in cc],source,ss,source_cc)
  for k,(a,b) in mm.items():boots[k].append([a.mean(),b.mean()])
  for k,(a,b) in pairs.items():diffs[k].append([(mm[a][0]-mm[b][1]).mean(),(mm[a][1]-mm[b][0]).mean()])
 out=dict(N=data['N'],classes=classes,controllers=nc,states=ns,seeds=2,bootstrap_replicates=5000,methods={},paired={});cc=np.arange(nc)[:,None];ss=np.arange(ns)[None,:];orc=metrics['full menu oracle'];safe=metrics['safety baseline'];f10=metrics['F10']
 for name,(a,b) in metrics.items():
  rl=(orc[0]-b).mean((1,2))*100;rh=(orc[1]-a).mean((1,2))*100;out['methods'][name]=dict(success_pct=[a.mean()*100,b.mean()*100],success_95CI_pct=ci(boots[name]),mean_regret_pp=[rl.mean(),rh.mean()],max_regret_pp=[rl.max(),rh.max()],harm_certain=int(((safe[0]-b).mean((1,2))>=.1-1e-12).sum()),harm_possible=int(((safe[1]-a).mean((1,2))>=.1-1e-12).sum()),catastrophes_certain=int(((f10[0]-b).mean((1,2))>=.1-1e-12).sum()),catastrophes_possible=int(((f10[1]-a).mean((1,2))>=.1-1e-12).sum()),events_pct={e:v[cc,ss,selected[name]].mean()*100 for e,v in data['events'].items()})
 for k,(a,b) in pairs.items():out['paired'][k]=dict(estimate_pp=difference(metrics[a],metrics[b]),CI_pp=ci(diffs[k]))
 out['in_sample_oracle_pct']={name:[lo[:,:,menu].mean((1,3)).max(1).mean()*100,hi[:,:,menu].mean((1,3)).max(1).mean()*100] for name,menu in [('fixed',FIXED),('full',FULL)]};rate=lo.mean((1,3));out['winning_arms']={kind:dict(fixed=FIXED[int(rate[i,FIXED].argmax())],schedule=SCHEDULE[int(rate[i,SCHEDULE].argmax())],full=FULL[int(rate[i,FULL].argmax())]) for i,kind in enumerate(classes)}
 nominal=np.where(data['events']['unknown'][:,:,0]>0,'unknown',np.where(data['events']['collision'][:,:,0]>0,'collision',np.where(lo[:,:,0]>0,'success','timeout')));safety=np.where(data['events']['unknown'][:,:,1]>0,'unknown',np.where(data['events']['collision'][:,:,1]>0,'collision',np.where(lo[:,:,1]>0,'success','timeout')));out['nominal_to_safety']={}
 for x in ['success','collision','timeout','unknown']:
  mask=nominal==x;den=int(mask.sum());counts={y:int(((safety==y)&mask).sum()) for y in ['success','collision','timeout','unknown']};out['nominal_to_safety'][x]=dict(denominator=den,counts=counts,fractions={k:v/den if den else None for k,v in counts.items()})
 return clean(out)
