from common import *
import csv
MENU=['F10','F13','S300','S500'];ARMS=['nominal_only','safety_baseline']+MENU
METHODS=['nominal only','safety baseline','F10','S500','A3 zero-shot','A0','G_E','best fixed oracle','full menu oracle','A3 per state','A0 per state','state oracle','F13','S300']

def oracle_bounds(lo,hi,indices,by_seed=False,original_indices=None):
 nc,ns,na,nq=lo.shape;point=np.zeros((nc,ns,nq));low=point.copy();high=point.copy()
 if by_seed:
  for q in range(2):
   trainlo=lo[:,:,indices,1-q];trainhi=hi[:,:,indices,1-q];evlo=lo[:,:,indices,q];evhi=hi[:,:,indices,q];best=trainlo.argmax(-1)
   point[:,:,q]=np.take_along_axis(evlo,best[...,None],-1)[...,0];possible=trainhi>=trainlo.max(-1,keepdims=True)
   exact=(trainlo==trainhi).all(-1)
   possible=np.where(exact[...,None],np.arange(len(indices))==best[...,None],possible)
   low[:,:,q]=np.where(possible,evlo,np.inf).min(-1);high[:,:,q]=np.where(possible,evhi,-np.inf).max(-1)
 else:
  parity=(np.arange(ns) if original_indices is None else np.asarray(original_indices))%2
  if parity.ndim==1:parity=np.broadcast_to(parity,(nc,ns))
  for c in range(nc):
   for fold in range(2):
    train=np.flatnonzero(parity[c]!=fold);test=np.flatnonzero(parity[c]==fold)
    if not len(test):continue
    if not len(train):
     # Empty train fold in bootstrap: all menu arms possible, deterministic first arm for point.
     tl=np.zeros(len(indices));th=np.ones(len(indices))
    else:tl=lo[c,train][:,indices].mean((0,2));th=hi[c,train][:,indices].mean((0,2))
    best=indices[int(tl.argmax())];possible=np.array([best]) if np.array_equal(tl,th) else np.asarray(indices)[th>=tl.max()]
    point[c,test]=lo[c,test,best];low[c,test]=lo[c,test][:,possible].min(1);high[c,test]=hi[c,test][:,possible].max(1)
 return point,low,high

def evaluate(lo,hi,choices,orig=None):
 nc,ns,_,nq=lo.shape;ci=np.arange(nc)[:,None];si=np.arange(ns)[None,:];out={}
 def chosen(indices):return lo[ci,si,indices],hi[ci,si,indices]
 for name,a in [('nominal only',0),('safety baseline',1),('F10',2),('S500',5),('F13',3),('S300',4)]:out[name]=(lo[:,:,a],hi[:,:,a])
 for model in ['A3','A0']:
  for kind,label in [('controller',model+' zero-shot' if model=='A3' else 'A0'),('state',model+' per state')]:
   idx=choices[model][kind]+2
   if kind=='controller':idx=idx[:,None]
   out[label]=chosen(idx)
 out['G_E']=chosen(choices['G_E'][:,None]+2)
 extras={}
 for name,idx,state in [('best fixed oracle',[2,3],False),('full menu oracle',[2,3,4,5],False),('state oracle',[2,3,4,5],True)]:
  point,lower,upper=oracle_bounds(lo,hi,idx,state,orig);out[name]=(lower,upper);extras[name]=point
 return out,extras

def summarize(cohort,lo,hi,events,choice,seed):
 nc,ns=lo.shape[:2];nq=lo.shape[-1];metrics,oraclepoints=evaluate(lo,hi,choice);rng=np.random.default_rng(seed);boots={k:[] for k in METHODS};diffs=[];state_diff=[];regret=[];potential=[]
 for b in range(5000):
  cc=rng.integers(nc,size=nc);ss=rng.integers(ns,size=(nc,ns));ll=lo[cc[:,None],ss];hh=hi[cc[:,None],ss];ch={m:{'controller':choice[m]['controller'][cc],'state':choice[m]['state'][cc[:,None],ss]} for m in ['A3','A0']};ch['G_E']=choice['G_E'][cc]
  mm,pp=evaluate(ll,hh,ch,ss)
  for name,(a,z) in mm.items():boots[name].append([a.mean(),z.mean()])
  a3=mm['A3 zero-shot'];safe=mm['safety baseline'];st=mm['A3 per state'];orc=mm['full menu oracle'];sorc=mm['state oracle']
  diffs.append([(a3[0]-safe[1]).mean(),(a3[1]-safe[0]).mean()]);state_diff.append([(st[0]-a3[1]).mean(),(st[1]-a3[0]).mean()]);regret.append([(orc[0]-a3[1]).mean(),(orc[1]-a3[0]).mean()]);potential.append([(sorc[0]-orc[1]).mean(),(sorc[1]-orc[0]).mean()])
 def ci(x):return [float(np.quantile(np.asarray(x)[:,0],.025)*100),float(np.quantile(np.asarray(x)[:,1],.975)*100)]
 result={'cohort':cohort,'controllers':nc,'states':ns,'bootstrap_replicates':5000,'methods':{}}
 for name,(low,high) in metrics.items():
  fixed=metrics['full menu oracle'];difflo=(fixed[0]-high).mean((1,2));diffhi=(fixed[1]-low).mean((1,2));arm=ARMS.index(name.replace(' ','_')) if name in ['nominal only','safety baseline','F10','S500','F13','S300'] else None
  # Outcome rates for a selected method are computed using exactly its frozen choices.
  if arm is not None:event={e:float(v[:,:,arm].mean()*100) for e,v in events.items()}
  elif name in ['A3 zero-shot','A0','G_E','A3 per state','A0 per state']:
   if name=='G_E':ix=choice['G_E'][:,None]+2
   else:
    m='A3' if name.startswith('A3') else 'A0';ix=choice[m]['state']+2 if name.endswith('per state') else choice[m]['controller'][:,None]+2
   event={e:float(v[np.arange(nc)[:,None],np.arange(ns)[None,:],ix].mean()*100) for e,v in events.items()}
  else:
   indices=[2,3] if name=='best fixed oracle' else [2,3,4,5]
   selected=np.zeros((nc,ns,nq),int)
   if name=='state oracle':
    for q in range(nq):selected[:,:,q]=np.asarray(indices)[lo[:,:,indices,1-q].argmax(-1)]
   else:
    for c in range(nc):
     for fold in range(2):
      train=np.arange(ns)%2!=fold;test=~train;ix=indices[int(lo[c,train][:,indices].mean((0,2)).argmax())];selected[c,test]=ix
   event={e:float(v[np.arange(nc)[:,None,None],np.arange(ns)[None,:,None],selected,np.arange(nq)[None,None,:]].mean()*100) for e,v in events.items()}
   event['note']='deterministic oracle trained on known-success lower bounds, evaluated on disjoint fold'
  result['methods'][name]={'success_pct':[float(low.mean()*100),float(high.mean()*100)],'success_95CI_pct':ci(boots[name]),'mean_regret_pp':[float(difflo.mean()*100),float(diffhi.mean()*100)],'max_regret_pp':[float(difflo.max()*100),float(diffhi.max()*100)],'catastrophes_certain':int(((metrics['F10'][0]-high).mean((1,2))>=.10-1e-12).sum()),'catastrophes_possible':int(((metrics['F10'][1]-low).mean((1,2))>=.10-1e-12).sum()),'harm_certain':int(((metrics['safety baseline'][0]-high).mean((1,2))>=.10-1e-12).sum()),'harm_possible':int(((metrics['safety baseline'][1]-low).mean((1,2))>=.10-1e-12).sum()),'events_pct':event}
 a3=metrics['A3 zero-shot'];safe=metrics['safety baseline'];st=metrics['A3 per state'];orc=metrics['full menu oracle'];rg=[float((orc[0]-a3[1]).mean()*100),float((orc[1]-a3[0]).mean()*100)];dc=ci(diffs);stci=ci(state_diff)
 status='SUPPORTED' if rg[1]<=3 and dc[0]>0 else 'REJECTED' if rg[0]>3 or dc[1]<=0 else 'INCONCLUSIVE'
 result['endpoint']={'status':status,'A3_vs_crossfit_oracle_regret_pp':rg,'regret_CI_pp':ci(regret),'A3_minus_safety_CI_pp':dc}
 result['ST3']={'status':'gain' if stci[0]>0 else 'harm' if stci[1]<0 else 'undetected','state_minus_controller_pp':[float((st[0]-a3[1]).mean()*100),float((st[1]-a3[0]).mean()*100)],'CI_pp':stci,'crossfit_state_oracle_potential_gain_CI_pp':ci(potential)}
 result['in_sample_oracle_pct']={k:[float(lo[:,:,idx].mean((1,3)).max(1).mean()*100),float(hi[:,:,idx].mean((1,3)).max(1).mean()*100)] for k,idx in [('fixed',[2,3]),('full',[2,3,4,5])]}
 result['crossfit_oracle_point_pct']={k:float(v.mean()*100) for k,v in oraclepoints.items()}
 # Episode-level transitions include all outcomes; nominal collision row is explicitly reported.
 transitions={};nominal=np.where(events['unknown'][:,:,0]>0,'unknown',np.where(events['collision'][:,:,0]>0,'collision',np.where(lo[:,:,0]>0,'success','timeout')));safety=np.where(events['unknown'][:,:,1]>0,'unknown',np.where(events['collision'][:,:,1]>0,'collision',np.where(lo[:,:,1]>0,'success','timeout')))
 for x in ['success','collision','timeout','unknown']:
  mask=nominal==x;den=int(mask.sum());transitions[x]={'denominator':den,'counts':{y:int(((safety==y)&mask).sum()) for y in ['success','collision','timeout','unknown']}}
 result['nominal_to_safety']=transitions
 return result

def main():
 prereg();rows=read(HERE/'main_CASES.json');assert all((HERE/'main/raw'/f"{c['rollout_uid']}.json").exists() for c in rows)
 seal('ENDPOINT_OPENING.receipt.json',[HERE/'PREREGISTRATION.md',HERE/'CHOICES_FROZEN.receipt.json',HERE/'main_CASES.json'],all_16128_raw_present=True)
 states=read(HERE/'EVAL_STATES.json');names=read(HERE/'protocol.json')['controller_order'];ids=[s['state_uid'] for s in states];shape=(14,96,6,2);lo=np.zeros(shape);unknown=np.zeros(shape);events={k:np.zeros(shape) for k in ['collision','timeout','unknown']};seen=np.zeros(shape,bool)
 for c in rows:
  r=read(HERE/'main/raw'/f"{c['rollout_uid']}.json");ix=(names.index(c['base_controller_name']),ids.index(c['state_uid']),ARMS.index(c['arm']),c['seed']);assert not seen[ix];seen[ix]=True;lo[ix]=r['success'];unknown[ix]=r['numerical_failure'];events['collision'][ix]=r['collision'];events['timeout'][ix]=r['timeout'];events['unknown'][ix]=r['numerical_failure']
 assert seen.all();hi=lo+unknown;np.savez_compressed(HERE/'LABELS.npz',lower=lo,upper=hi,**events,controller_names=names,state_uids=ids,arms=ARMS)
 frozen=read(HERE/'FROZEN_CHOICES.json')['choices'];results={}
 for i,cohort in enumerate(['1a','1b']):
  ii=[i for i,s in enumerate(states) if s['cohort']==cohort];ch={m:{k:np.asarray(v) for k,v in frozen[m+'_'+cohort].items() if k in ['controller','state']} for m in ['A3','A0']};ch['G_E']=np.asarray(frozen['G_E']);results[cohort]=summarize(cohort,lo[:,ii],hi[:,ii],{k:v[:,ii] for k,v in events.items()},ch,202610153200+i)
  write(HERE/f'{cohort}_RESULTS.json',results[cohort]);print(cohort,'analyzed',flush=True)
 # Stratification of shift states uses preregistered physical thresholds and frozen controller A3.
 strata={};ii=[i for i,s in enumerate(states) if s['cohort']=='1b'];ch=np.asarray(frozen['A3_1b']['controller'])+2
 for j,idx in enumerate(ii):
  s=states[idx];p=np.asarray(s['physical']['positions']);g=np.asarray(s['physical']['goals']);v=np.asarray(s['physical']['velocities']);ang=np.unwrap(np.arctan2(p[:,1],p[:,0])-np.arange(4)*np.pi/2);conditions={'radius_below_2.3':min(np.linalg.norm(p,axis=1).min(),np.linalg.norm(g,axis=1).min())<2.3,'speed_above_.11':np.linalg.norm(v,axis=1).max()>.11,'angular_spread_above_.4':np.ptp(ang)>.4}
  for key,val in conditions.items():strata.setdefault(key+'='+str(bool(val)),[]).append(idx)
 results['shift_strata']={k:{'states':len(ix),'A3_success_pct':[float(lo[np.arange(14)[:,None],np.asarray(ix)[None,:],ch[:,None]].mean()*100),float(hi[np.arange(14)[:,None],np.asarray(ix)[None,:],ch[:,None]].mean()*100)]} for k,ix in strata.items()}
 pooledlo=[];pooledhi=[]
 for cohort in ['1a','1b']:
  ix=[i for i,s in enumerate(states) if s['cohort']==cohort];ch=frozen['A3_'+cohort];cix=np.asarray(ch['controller'])[:,None]+2;six=np.asarray(ch['state'])+2;cc=np.arange(14)[:,None];ss=np.asarray(ix)[None,:]
  pooledlo.append(lo[cc,ss,six]-hi[cc,ss,cix]);pooledhi.append(hi[cc,ss,six]-lo[cc,ss,cix])
 pl=np.concatenate(pooledlo,1).mean(-1);ph=np.concatenate(pooledhi,1).mean(-1);rng=np.random.default_rng(202610153202);rep=[]
 for _ in range(5000):
  cc=rng.integers(14,size=14);ss=rng.integers(96,size=(14,96));rep.append([pl[cc[:,None],ss].mean()*100,ph[cc[:,None],ss].mean()*100])
 interval=[float(np.quantile(np.asarray(rep)[:,0],.025)),float(np.quantile(np.asarray(rep)[:,1],.975))]
 results['ST3_pooled']={'state_minus_controller_pp':[float(pl.mean()*100),float(ph.mean()*100)],'CI_pp':interval,'status':'gain' if interval[0]>0 else 'harm' if interval[1]<0 else 'undetected'}
 results['state_shift_change']={m:results['1b']['methods'][m]['success_pct'][0]-results['1a']['methods'][m]['success_pct'][0] for m in METHODS};results['evidence']='CONFIRMATORY';write(HERE/'results.json',results)
if __name__=='__main__':main()
