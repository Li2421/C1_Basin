from common import *
from statistics_core import oracle_bounds,interval,summarize
from analyze_D import baseline
from analyze_A_v2 import raw_for

def reduced(label,names,lo,hi,events,seed):
 # Available correction menu only; absent safety baseline is explicitly unavailable.
 f=read(HERE/'FROZEN_REFERENCE_CHOICES.json')['controllers'];idx={key:np.array([1 if v[n]=='F10' else 2 for n in names]) for key,v in f.items()};idx.update(**{'nominal only':np.zeros(len(names),int),'F10':np.ones(len(names),int),'S500 class':np.full(len(names),2)})
 def ev(a,b,cc=None,ss=None):
  nc,ns=a.shape[:2];c=np.arange(nc)[:,None];s=np.arange(ns)[None,:];ix={k:v if cc is None else v[cc] for k,v in idx.items()};m={k:(a[c,s,v[:,None]],b[c,s,v[:,None]])for k,v in ix.items()};sel={k:np.broadcast_to(v[:,None],(nc,ns))for k,v in ix.items()}
  for key,menu in [('best fixed oracle',[1]),('full menu oracle',[1,2])]:x,y,_,z=oracle_bounds(a,b,menu,ss);m[key]=(x,y);sel[key]=z
  return m,sel
 metrics,selected=ev(lo,hi);rng=np.random.default_rng(seed);boots={k:[]for k in metrics};nc,ns=lo.shape[:2]
 for _ in range(5000):
  cc=rng.integers(nc,size=nc);ss=rng.integers(ns,size=(nc,ns));m,_=ev(lo[cc[:,None],ss],hi[cc[:,None],ss],cc,ss)
  for k,(a,b)in m.items():boots[k].append([a.mean(),b.mean()])
 out=dict(cohort=label,methods={},unavailable_methods={'safety baseline':'No label in the required positive control or old geometry gate; no unnecessary additional rollout','expert':'No privileged Ring reference for this cohort'},evidence='DESCRIPTIVE available-menu reanalysis',available_menu=['F10','S500'],in_sample_oracle_pct={k:[lo[:,:,ix].mean((1,3)).max(1).mean()*100,hi[:,:,ix].mean((1,3)).max(1).mean()*100]for k,ix in [('fixed',[1]),('full',[1,2])]})
 for k,(a,b)in metrics.items():
  o=metrics['full menu oracle'];f10=metrics['F10'];rl=(o[0]-b).mean((1,2))*100;rh=(o[1]-a).mean((1,2))*100;c=np.arange(nc)[:,None];s=np.arange(ns)[None,:];out['methods'][k]=dict(success_pct=[a.mean()*100,b.mean()*100],success_95CI_pct=interval(boots[k]),mean_regret_pp=[rl.mean(),rh.mean()],max_regret_pp=[rl.max(),rh.max()],catastrophes_certain=int(((f10[0]-b).mean((1,2))>=.1-1e-12).sum()),catastrophes_possible=int(((f10[1]-a).mean((1,2))>=.1-1e-12).sum()),harm_certain=None,harm_possible=None,events_pct={e:v[c,s,selected[k]].mean()*100 for e,v in events.items()})
 return out

def main():
 out={};cases=read(HERE/'geometry_controls_CASES.json');names=read(HERE/'HISTORICAL_INVENTORY.json')['controller_names'];oldstates={s['state_uid']:s['study_index']for s in read(R35/'pilot_protocol.json')['states']}
 for j,(label,r,cohort)in enumerate([('A1_positive',.9,'positive')]+[(f'A_old_gate_r{r}',r,'old_geometry_nominal')for r in [.8,.85,.95,1.]]):
  rows=[c for c in cases if c['cohort']==cohort and c['r']==r]
  if cohort=='old_geometry_nominal':
   for c in read(R35/'pilot_CASES.json'):
    if c['r']==r:rows.append(dict(c,state_index=oldstates[c['state_uid']],reference_path=str(R35/'pilot/raw'/f"{c['rollout_uid']}.json")))
  shape=(14,16,3,1);data={k:np.zeros(shape)for k in ['lower','upper','collision','timeout','unknown']};seen=np.zeros(shape,bool)
  for c in rows:
   a=['nominal_only','F10','S500'].index(c['arm']);i=(names.index(c['base_controller_name']),c['state_index'],a,0);assert not seen[i];seen[i]=True;raw,_=raw_for(c);u=raw['numerical_failure'];data['lower'][i]=raw['success'] if not u else 0;data['upper'][i]=data['lower'][i]+u
   for k in ['collision','timeout','unknown']:data[k][i]=raw['numerical_failure' if k=='unknown' else k]
  assert seen.all(),(label,int(seen.sum()));out[label]=reduced(label,names,data['lower'],data['upper'],{k:data[k]for k in ['collision','timeout','unknown']},202610163710+j);write(HERE/'A_STANDARD_PARTIAL.json',out)
 for j,r in enumerate([.8,.85,.95,1.]):
  label=f'D4_r{r}';data,names,arms,choices,full,fixed,schedule=baseline(label);choices.update(**{'nominal only':0,'safety baseline':1,'F10':2,'S500 class':3});out[f'A2_r{r}']=summarize(f'A2_r{r}',data['lower'],data['upper'],{k:data[k]for k in ['collision','timeout','unknown']},choices,full,fixed,202610163711+j)
 write(HERE/'A_STANDARD_RESULTS.json',out);seal('A_STANDARD_RESULTS.receipt.json',[HERE/'A_STANDARD_RESULTS.json'],additional_physics=0,full_menu_is_available_subset_only=True)
if __name__=='__main__':main()
