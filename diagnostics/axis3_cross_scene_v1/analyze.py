from common import *
from statistics_core import summarize,oracle_bounds,interval

def contrasts(lo,hi,choices,possible,seed):
 nc,ns=lo.shape[:2];cc=np.arange(nc)[:,None];ss=np.arange(ns)[None,:];ix=np.asarray(choices);critic=(lo[cc,ss,ix[:,None]],hi[cc,ss,ix[:,None]]);source=(lo[:,:,possible].min(2),hi[:,:,possible].max(2));schedule=oracle_bounds(lo,hi,list(range(5,9)))[:2];fixed=oracle_bounds(lo,hi,[2,3,4])[:2];full=oracle_bounds(lo,hi,list(range(2,9)))[:2];safe=(lo[:,:,1],hi[:,:,1]);pairs={'XS1_regret_difference':(source,critic),'XS3_schedule_minus_fixed':(schedule,fixed),'XS5_best_correction_minus_safety':(full,safe)};samples={k:[] for k in pairs};rng=np.random.default_rng(seed)
 for _ in range(5000):
  c=rng.integers(nc,size=nc);s=rng.integers(ns,size=(nc,ns));a=lo[c[:,None],s];b=hi[c[:,None],s];ii=ix[c];cr=(a[np.arange(nc)[:,None],np.arange(ns)[None,:],ii[:,None]],b[np.arange(nc)[:,None],np.arange(ns)[None,:],ii[:,None]]);src=(a[:,:,possible].min(2),b[:,:,possible].max(2));sh=oracle_bounds(a,b,list(range(5,9)),s)[:2];fx=oracle_bounds(a,b,[2,3,4],s)[:2];fo=oracle_bounds(a,b,list(range(2,9)),s)[:2];sa=(a[:,:,1],b[:,:,1])
  for k,(x,y) in {'XS1_regret_difference':(src,cr),'XS3_schedule_minus_fixed':(sh,fx),'XS5_best_correction_minus_safety':(fo,sa)}.items():samples[k].append([(x[0]-y[1]).mean(),(x[1]-y[0]).mean()])
 out={k:dict(estimate_pp=[(x[0]-y[1]).mean()*100,(x[1]-y[0]).mean()*100],CI_pp=interval(samples[k])) for k,(x,y) in pairs.items()};return clean(out),samples

def main():
 prereg()
 for p,h in read(HERE/'ANALYSIS_IMPLEMENTATION_FROZEN.receipt.json')['files'].items():assert sha(p)==h,p
 rows=read(HERE/'main_CASES.json');assert all((HERE/'main/raw'/f"{c['rollout_uid']}.json").exists() for c in rows);seal('ENDPOINT_OPENING.receipt.json',[HERE/'CHOICES_FROZEN.receipt.json',HERE/'ANALYSIS_IMPLEMENTATION_FROZEN.receipt.json',HERE/'main_CASES.json'],all_main_present=True);frozen=read(HERE/'FROZEN_CHOICES.json');states=read(HERE/'STATES.json');results={};pooled=[];definite=0;possible_count=0
 for j,sc in enumerate(SCENES):
  names=frozen['eligible'][sc]
  if not names:results[sc]=dict(status='GAP',methods={},reason=read(HERE/'CAPABILITY_GATE.json')['scene_status'][sc]);possible_count+=1;continue
  ids=[s['state_uid'] for s in states[sc]['eval']];shape=(len(names),32,9,2);lo=np.zeros(shape);events={k:np.zeros(shape) for k in ['collision','timeout','unknown','deadlock']};seen=np.zeros(shape,bool)
  for c in rows:
   if c['scene']!=sc:continue
   name=c['base_controller_name'].split('::',1)[1];ix=(names.index(name),ids.index(c['state_uid']),ARMS.index(c['arm']),c['seed']);assert not seen[ix];seen[ix]=True;r=read(HERE/'main/raw'/f"{c['rollout_uid']}.json");lo[ix]=r['success']
   for k in events:events[k][ix]=r['numerical_failure' if k=='unknown' else k]
  assert seen.all();hi=lo+events['unknown'];np.savez_compressed(HERE/f'{sc}_LABELS.npz',lower=lo,upper=hi,**events,controller_names=names,state_uids=ids,arms=ARMS);ix=np.asarray([ARMS.index(frozen['controllers'][sc][n]) for n in names]);source=frozen['source_mean'][sc];ch={'nominal only':0,'safety baseline':1,'F10':2,'S500':6,'Ring default schedule':6,'LOSO critic':ix,'source mean action':ARMS.index(source['chosen'])};r=summarize(sc,lo,hi,events,ch,list(range(2,9)),[2,3,4],202610153400+j);paired,draws=contrasts(lo,hi,ix,[ARMS.index(a) for a in source['possible']],202610153410+j);r['paired']=paired;rg=r['methods']['LOSO critic']['mean_regret_pp'];r['XS2']=dict(regret_pp=rg,definitely_le5=rg[1]<=5,possibly_le5=rg[0]<=5);definite+=rg[1]<=5;possible_count+=rg[0]<=5;pooled.append(draws);r['evidence']='CONFIRMATORY new labels/fixed LOSO; historical source descriptors qualified';r['source_mean']=source;r['in_sample_winning_family']={a:float(lo[:,:,ARMS.index(a)].mean()*100) for a in MENU};results[sc]=r;write(HERE/f'{sc}_RESULTS.json',r);print(sc,'analyzed',flush=True)
 n=len(pooled);b=np.mean([np.asarray(x['XS1_regret_difference']) for x in pooled],axis=0);ci=interval(b);available_status='SUPPORTED' if ci[1]<0 else 'REJECTED';results['pooled']=dict(target_scenes_available=n,target_scenes_required=4,XS1=dict(status=available_status if n==4 else 'INCONCLUSIVE',available_scene_status=available_status,CI_pp=ci,scope='Equal scene weighting; unavailable target prevents four-target endpoint'),XS2=dict(status='SUPPORTED' if definite>=3 else 'REJECTED' if possible_count<3 else 'INCONCLUSIVE',definite_scene_count=int(definite),possible_scene_count_including_missing=int(possible_count),threshold=3),evidence='Available-target partial summary; no full4scene claim whenany GAP');write(HERE/'results.json',results)
if __name__=='__main__':main()
