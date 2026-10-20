from common import *
from statistics_core import evaluate,interval

def main():
 results=read(HERE/'results.json');frozen=read(HERE/'FROZEN_CHOICES.json');units=[]
 for scene in SCENES:
  path=HERE/f'{scene}_LABELS.npz'
  if not path.exists():continue
  z=np.load(path);names=z['controller_names'].tolist();ix=np.asarray([ARMS.index(frozen['controllers'][scene][n]) for n in names]);ch={'nominal only':0,'safety baseline':1,'F10':2,'S500':6,'Ring default schedule':6,'LOSO critic':ix,'source mean action':ARMS.index(frozen['source_mean'][scene]['chosen'])};units.append((z['lower'],z['upper'],ch))
 names=list(results[next(s for s in SCENES if results[s].get('methods'))]['methods']);boots={k:[] for k in names};rng=np.random.default_rng(202610153420)
 for _ in range(5000):
  rates={k:[] for k in names}
  for lo,hi,ch in units:
   nc,ns=lo.shape[:2];cc=rng.integers(nc,size=nc);ss=rng.integers(ns,size=(nc,ns));ccs={k:np.asarray(v)[cc] if np.asarray(v).ndim else v for k,v in ch.items()};m,_=evaluate(lo[cc[:,None],ss],hi[cc[:,None],ss],ccs,list(range(2,9)),[2,3,4],ss)
   for k,(a,b) in m.items():rates[k].append([a.mean(),b.mean()])
  for k in names:boots[k].append(np.mean(rates[k],axis=0))
 out={}
 for name in names:
  source=[results[s]['methods'][name] for s in SCENES if results[s].get('methods')];v=dict(success_pct=np.mean([m['success_pct'] for m in source],0),success_95CI_pct=interval(boots[name]),mean_regret_pp=np.mean([m['mean_regret_pp'] for m in source],0),max_regret_pp=np.max([m['max_regret_pp'] for m in source],0),events_pct={e:float(np.mean([m['events_pct'][e] for m in source])) for e in ['collision','timeout','unknown','deadlock']})
  for k in ['catastrophes_certain','catastrophes_possible','harm_certain','harm_possible']:v[k]=sum(m[k] for m in source)
  out[name]=v
 results['pooled']['methods']=out;results['pooled']['success_scope']='Equal available-scene weights, no estimate for unavailableToy';write(HERE/'results.json',results)
if __name__=='__main__':main()
