"""Read-only closed-loop evidence analysis; no simulator calls or label acquisition."""
from core import *
import csv,collections
ARMS=['F10','F13','S300','S500'];COLS=[(m,a) for m in ['shared','local','central'] for a in ARMS]+[('rhr','B-RHR'),('rhr_f10','B-RHR+F10')]
BASE=202610152800;B=5000

def interval(x):return [float(v) for v in np.quantile(x,[.025,.975],axis=0)]
def status_positive(worst,best):return 'SUPPORTED' if worst>0 else 'REJECTED' if best<=0 else 'INCONCLUSIVE'
def stall_stats(mask):
 counts=[];dur=[]
 for agent in range(4):
  v=np.r_[False,mask[:,agent],False].astype(int);d=np.diff(v);lengths=np.flatnonzero(d==-1)-np.flatnonzero(d==1);lengths=lengths[lengths>=40];counts.append(len(lengths));dur.extend(lengths.tolist())
 return dict(stalled_robots=sum(v>0 for v in counts),stall_runs=sum(counts),stall_total_seconds=sum(dur)*.05,stall_max_seconds=max(dur,default=0)*.05)
def bootstrap(x,group):
 """x [controller,state,column], pair all arms/seeds within sampled state."""
 rng=np.random.default_rng(BASE+group);out=[]
 for begin in range(0,B,100):
  n=min(100,B-begin);c=rng.integers(0,14,(n,14));s=rng.integers(0,32,(n,14,32));out.append(x[c[:,:,None],s].mean(axis=2))
 return np.concatenate(out) # [draw, sampled_controller, column]
def gain_stats(point,worst,best,drawpoint,drawworst,drawbest):
 return dict(estimate=float(np.mean(point)),identification_interval=[float(np.mean(worst)),float(np.mean(best))],ci_lower_label_bound=interval(drawpoint.mean(axis=1)),lower_ci_endpoint_range=[float(np.quantile(drawworst.mean(axis=1),.025)),float(np.quantile(drawbest.mean(axis=1),.025))],ci_unknown_envelope=[float(np.quantile(drawworst.mean(axis=1),.025)),float(np.quantile(drawbest.mean(axis=1),.975))])
def main():
 prereg();assert read(HERE/'MERGE_RECEIPT.json')['postflight']['main']['records']==12544;p=read(HERE/'main_protocol.json');names=p['controller_order'];ci={n:i for i,n in enumerate(names)};ix={v:i for i,v in enumerate(COLS)};shape=(14,32,14,2);Y=np.full(shape,np.nan);U=np.zeros(shape);C=np.zeros(shape);T=np.zeros(shape);ST=np.zeros(shape);SR=np.zeros(shape);SD=np.zeros(shape);SM=np.zeros(shape);records=[];csvrows=[];errors=collections.Counter();seen=set()
 for f in sorted((HERE/'main/raw').glob('*.json')):
  r=read(f);c=r['case'];a=(c['pipeline'],c['arm']);i=(ci[c['base_controller_name']],c['state_index'],ix[a],c['seed']);assert i not in seen;seen.add(i);Y[i]=r['success'];U[i]=r['numerical_failure'];C[i]=r['collision'];T[i]=r['timeout'];assert not r['deadlock'];assert not r['jax_enable_x64'];assert sha(r['trace_path'])==r['trace_sha256']
  with np.load(r['trace_path'],allow_pickle=False) as z:
   ss=stall_stats(z['slow_mask']);assert len(z['positions'])==r['physical_steps']+1;assert len(z['swept_clearance'])==r['physical_steps']
   if r['success']:assert not r['collision'] and z['swept_clearance'].min()>=0
  ST[i]=ss['stalled_robots'];SR[i]=ss['stall_runs'];SD[i]=ss['stall_total_seconds'];SM[i]=ss['stall_max_seconds']
  records.append(r);csvrows.append(dict(rollout_uid=c['rollout_uid'],controller=c['base_controller_name'],pipeline=c['pipeline'],arm=c['arm'],state_index=c['state_index'],seed=c['seed'],success=r['success'],unknown=r['numerical_failure'],collision=r['collision'],timeout=r['timeout'],physical_steps=r['physical_steps'],elapsed_seconds=r['elapsed_seconds'],minimum_swept_clearance=r['minimum_swept_clearance'],**ss))
 assert len(seen)==12544 and np.isfinite(Y).all();assert not np.any(Y*U)
 errors=collections.Counter((r['case']['pipeline'],r['error']['type'],r['error']['message']) for r in records if r['error'])
 with (HERE/'rollout_metrics.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
 lo=Y.mean(axis=3);hi=(Y+U).mean(axis=3);point=lo.mean(axis=1);upper=hi.mean(axis=1);bootlo=bootstrap(lo,0);boothi=bootstrap(hi,0);meanboot=bootlo.mean(axis=1);rates={};rateboots={0:(meanboot,boothi.mean(axis=1)),1:(bootstrap(lo,1).mean(axis=1),bootstrap(hi,1).mean(axis=1)),2:(bootstrap(lo,2).mean(axis=1),bootstrap(hi,2).mean(axis=1))}
 for j,(m,a) in enumerate(COLS):
  ratebl,ratebh=rateboots[1 if m=='local' else 2 if m=='central' else 0]
  rates[m+'/'+a]=dict(success_lower=float(lo[:,:,j].mean()),success_upper=float(hi[:,:,j].mean()),ci_lower_bound=interval(ratebl[:,j]),ci_unknown_envelope=[float(np.quantile(ratebl[:,j],.025)),float(np.quantile(ratebh[:,j],.975))],n=896,successes=int(Y[:,:,j].sum()),unknown=int(U[:,:,j].sum()),collisions=int(C[:,:,j].sum()),timeouts=int(T[:,:,j].sum()),stall_episode_rate=float((ST[:,:,j]>0).mean()),mean_stalled_robots=float(ST[:,:,j].mean()),stall_runs=int(SR[:,:,j].sum()),mean_stall_total_seconds=float(SD[:,:,j].mean()),maximum_stall_seconds=float(SM[:,:,j].max()))
 # DC1: independently adversarial unknown assignment to schedule vs fixed arms is exact by monotonicity.
 sp=point[:,2:4].max(axis=1);fp=point[:,:2].max(axis=1);sw=point[:,2:4].max(axis=1)-upper[:,:2].max(axis=1);sb=upper[:,2:4].max(axis=1)-point[:,:2].max(axis=1)
 dp=bootlo[:,:,2:4].max(axis=2)-bootlo[:,:,:2].max(axis=2);dw=bootlo[:,:,2:4].max(axis=2)-boothi[:,:,:2].max(axis=2);db=boothi[:,:,2:4].max(axis=2)-bootlo[:,:,:2].max(axis=2)
 dc1=gain_stats(sp-fp,sw,sb,dp,dw,db);definite=int((sw>=-.01).sum());possible=int((sb>=-.01).sum());dc1.update(dominance_definite=definite,dominance_possible=possible,dominance_total=14,required=10)
 g=dc1['lower_ci_endpoint_range'];dc1['status']='SUPPORTED' if g[0]>0 and definite>=10 else 'REJECTED' if g[1]<=0 or possible<10 else 'INCONCLUSIVE'
 # DC2 exact regret bounds: exclude S500 self contribution to avoid assigning same unknown inconsistently.
 rp=point[:,:4].max(axis=1)-point[:,3];rw=np.maximum(0,point[:,:3].max(axis=1)-upper[:,3]);rb=np.maximum(0,upper[:,:3].max(axis=1)-point[:,3]);rboot=bootlo[:,:,:4].max(axis=2)-bootlo[:,:,3]
 dc2=dict(mean_regret=float(rp.mean()),worst_controller_regret=float(rp.max()),regret_identification_interval=[float(rw.mean()),float(rb.mean())],ci_lower_label_bound=interval(rboot.mean(axis=1)),threshold=.03,status='SUPPORTED' if rb.mean()<=.03 else 'REJECTED' if rw.mean()>.03 else 'INCONCLUSIVE')
 dc2['worst_controller_regret_identification_interval']=[float(rw.max()),float(rb.max())]
 dc2['best_fixed_mean_regret_lower_labels']=float((point[:,:4].max(axis=1)-fp).mean())
 dc2['S500_minus_best_fixed']=gain_stats(point[:,3]-fp,point[:,3]-upper[:,:2].max(axis=1),upper[:,3]-fp,bootlo[:,:,3]-bootlo[:,:,:2].max(axis=2),bootlo[:,:,3]-boothi[:,:,:2].max(axis=2),boothi[:,:,3]-bootlo[:,:,:2].max(axis=2))
 dc4={}
 for j,label in [(12,'B-RHR'),(13,'B-RHR+F10')]:
  v=gain_stats(point[:,3]-point[:,j],point[:,3]-upper[:,j],upper[:,3]-point[:,j],bootlo[:,:,3]-bootlo[:,:,j],bootlo[:,:,3]-boothi[:,:,j],boothi[:,:,3]-bootlo[:,:,j]);v['status']=status_positive(*v['lower_ci_endpoint_range']);dc4[label]=v
 dc4status='SUPPORTED' if all(v['status']=='SUPPORTED' for v in dc4.values()) else 'REJECTED' if any(v['status']=='REJECTED' for v in dc4.values()) else 'INCONCLUSIVE'
 differences={}
 for label,offset1,offset2,group in [('DC3_shared_minus_central',0,8,2),('DC5_local_minus_shared',4,0,1)]:
  bl=bootstrap(lo,group);bh=bootstrap(hi,group);details={}
  for a in range(4):
   j,k=offset1+a,offset2+a;v=gain_stats(point[:,j]-point[:,k],point[:,j]-upper[:,k],upper[:,j]-point[:,k],bl[:,:,j]-bl[:,:,k],bl[:,:,j]-bh[:,:,k],bh[:,:,j]-bl[:,:,k]);v['descriptive_metrics']={}
   for key,arr in [('collision_rate',C),('numerical_failure_rate',U),('stall_episode_rate',(ST>0).astype(float)),('stalled_robots',ST),('stall_total_seconds',SD)]:
    x=arr.mean(axis=3);d=x[:,:,j]-x[:,:,k];bd=bootstrap(d[:,:,None],group)[:,:,0].mean(axis=1);v['descriptive_metrics'][key]=dict(difference=float(d.mean()),ci=interval(bd))
   details[ARMS[a]]=v
  differences[label]=details
 expert=[read(f) for f in sorted((HERE/'expert/raw').glob('*.json'))];assert len(expert)==32;times=np.array([r['planning_seconds'] for r in expert]);es=sum(r['success'] for r in expert);eu=sum(r['numerical_failure'] for r in expert)
 percontroller=[]
 for c,n in enumerate(names):
  percontroller.append(dict(controller=n,rates={m+'/'+a:[float(point[c,j]),float(upper[c,j])] for j,(m,a) in enumerate(COLS)},best_schedule_gain_lower=float(sp[c]-fp[c]),always_S500_regret_lower=float(rp[c]),always_S500_regret_identification_interval=[float(rw[c]),float(rb[c])],best_fixed=ARMS[int(np.argmax(point[c,:2]))],best_schedule=ARMS[2+int(np.argmax(point[c,2:4]))]))
 result=dict(utc=utc(),main_n=12544,bootstrap=dict(draws=B,seed=BASE,unit='controller then state clusters; Q2 and arms paired; oracle reselected per draw'),unknown_policy='preserved, no retries; success identification lower0 upper1; not known failures',any_collision=bool(C.any()),collisions=int(C.sum()),unknown=int(U.sum()),rates=rates,DC1=dc1,DC2=dc2,DC4=dc4,DC4_status=dc4status,**differences,per_controller=percontroller,expert=dict(n=32,success=es,unknown=eu,success_lower=es/32,success_upper=(es+eu)/32,collisions=sum(r['collision'] for r in expert),planning_seconds=dict(mean=float(times.mean()),median=float(np.median(times)),min=float(times.min()),max=float(times.max()),total=float(times.sum()))),numerical_failure_types=[dict(pipeline=k[0],type=k[1],message=k[2],count=v) for k,v in errors.items()],source_hashes=dict(prereg=sha(HERE/'PREREGISTRATION.md'),protocol=sha(HERE/'main_protocol.json'),pipeline=sha(HERE/'pipeline.py'),analysis=sha(__file__),metrics=sha(HERE/'rollout_metrics.csv')))
 write(HERE/'results.json',result);np.savez_compressed(HERE/'analysis_arrays.npz',success=Y,unknown=U,collision=C,stall_robots=ST,stall_total_seconds=SD)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':10});out=HERE/'figures';out.mkdir(exist_ok=True)
 fig,ax=plt.subplots(figsize=(9,4));x=np.arange(4)
 for g,m in enumerate(['shared','local','central']):
  vals=[rates[m+'/'+a]['success_lower'] for a in ARMS];ups=[rates[m+'/'+a]['success_upper'] for a in ARMS];ax.bar(x+(g-1)*.24,vals,.24,label=m);ax.vlines(x+(g-1)*.24,vals,ups,color='black')
 ax.set(xticks=x,xticklabels=ARMS,ylabel='Success proportion (unknown lower; line to upper)',ylim=(0,1));ax.legend();fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(out/f'pipeline_success.{ext}',dpi=180)
 plt.close(fig)
 fig,ax=plt.subplots(figsize=(7,4));keys=['shared/F10','shared/F13','shared/S300','shared/S500','rhr/B-RHR','rhr_f10/B-RHR+F10'];ax.bar(np.arange(6),[rates[k]['success_lower'] for k in keys]);ax.set(xticks=np.arange(6),xticklabels=['F10','F13','S300','S500','RHR','RHR+F10'],ylabel='Success proportion',ylim=(0,1));fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(out/f'baseline_comparison.{ext}',dpi=180)
 plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(11,4))
 for g,m in enumerate(['shared','local','central']):
  for ax,key in zip(axs,['stall_episode_rate','mean_stall_total_seconds']):ax.bar(x+(g-1)*.24,[rates[m+'/'+a][key] for a in ARMS],.24,label=m);ax.set(xticks=x,xticklabels=ARMS)
 axs[0].set_ylabel('Episodes with diagnostic stall >=2s');axs[1].set_ylabel('Mean total stalled robot-seconds');axs[0].legend();fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(out/f'stall_diagnostics.{ext}',dpi=180)
 plt.close(fig);write(HERE/'NUMERICAL_ANALYSIS_COMPLETE.json',dict(utc=utc(),results_sha256=sha(HERE/'results.json'),formal_batch_complete=False,reason='video acceptance separately required'));print('Analysis complete',dc1['status'],dc2['status'],dc4status)
if __name__=='__main__':main()
