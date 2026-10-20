import os,sys,json,hashlib,csv,datetime
from pathlib import Path
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];os.environ['MPLCONFIGDIR']=str(HERE/'mplconfig');sys.path.insert(0,str(ROOT/'diagnostics/controller_loco_real_flow_v1/plot_vendor'))
import numpy as np
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
assert json.load(open(HERE/'RECONSTRUCTION_GATE.json'))['passed'];assert hashlib.sha256((HERE/'PREREGISTRATION.md').read_bytes()).hexdigest()==json.load(open(HERE/'PREREGISTRATION.receipt.json'))['sha256'];inventory=json.load(open(HERE/'DATA_INVENTORY.json'));rng=np.random.default_rng(202610152100);B=5000;datasets={};results={};tables={k:[] for k in 'ABDEF'}
def read(p):return json.load(open(p))
def ci(a):return np.quantile(a,[.025,.975],axis=0).tolist()
def group(n):return n.rsplit('__',1)[0] if n.endswith(('__best','__final')) else n
def pred_choices(folder,prefix,model,names,arms,available):
 files=[Path(folder)/'predictions'/f'{prefix}{model}_seed{s}.npz' for s in (17,23,41)]
 if not all(f.exists() for f in files):return None
 zz=[np.load(f,allow_pickle=False) for f in files];pn=zz[0]['controller_names'].tolist();pa=zz[0]['arms'].tolist();pr=np.mean([z['probability'] for z in zz],axis=0).mean(1);out=np.full((len(names),len(arms)),np.nan)
 for i,n in enumerate(names):
  assert n in pn
  for j,a in enumerate(arms):
   aa='F'+str(4 if n.startswith('base_u_v8_local') else 2) if a=='Freg' else a
   if aa in pa and available[i,j]:out[i,j]=pr[pn.index(n),pa.index(aa)]
 return out
for tag,meta in inventory.items():
 z=np.load(HERE/(tag+'.npz'),allow_pickle=False);datasets[tag]=dict(y=z['success'],u=z['unknown'],names=z['names'].tolist(),arms=z['arms'].tolist(),states=z['states'].tolist(),available=z['available'],meta=meta)
for sub,ix in [('R17N2',[i for i,n in enumerate(datasets['R17']['names']) if n.endswith('__arch512')]),('R17N1',[i for i,n in enumerate(datasets['R17']['names']) if not n.endswith('__arch512')])]:
 d=datasets['R17'];datasets[sub]=dict(d,y=d['y'][ix],u=d['u'][ix],names=[d['names'][i] for i in ix],available=d['available'][ix])
for tag,d in datasets.items():
 y=d['y'];u=d['u'];names=d['names'];arms=d['arms'];av=d['available'];C,S,A,Q=y.shape;lo=np.where(av,y.mean((1,3)),np.nan);hi=np.where(av,(y+u).mean((1,3)),np.nan);fixed=[i for i,a in enumerate(arms) if a.startswith('F')];sched=[i for i,a in enumerate(arms) if a.startswith('S')];bf=np.nanmax(lo[:,fixed],1);bs=np.nanmax(lo[:,sched],1);oracle=np.nanmax(lo,1);defarm='S15to10_786' if tag=='R19b' else 'S15_t2' if d['meta']['scene']=='double_bottleneck' else 'S13_t1' if d['meta']['scene']=='toy_give_way' else 'S500'
 if d['meta']['scene']=='four_way_intersection':defarm='S15to10_786' if 'S15to10_786' in arms else 'S15_t2'
 if defarm=='S500' and 'S500' not in arms:defarm='S13_t2'
 if tag=='R9':defarm='S13to10_500'
 default=arms.index(defarm);strategies={'F10':np.full(C,arms.index('F10')),'best_fixed':np.nanargmax(lo[:,fixed],1),'default_schedule':np.full(C,default),'oracle':np.nanargmax(lo,1)};strategies['best_fixed']=np.array(fixed)[strategies['best_fixed']]
 for label,menu in [('source_fixed',fixed),('source_full',list(range(A)))]:
  ch=[]
  for i,n in enumerate(names):
   src=[j for j,m in enumerate(names) if group(m)!=group(n)];allowed=[j for j in menu if av[i,j]];means=np.nanmean(lo[src],0);allowed=[j for j in allowed if np.isfinite(means[j])];ch.append(allowed[int(np.argmax(means[allowed]))])
  strategies[label]=np.array(ch)
 p=Path(d['meta']['folder'])
 if tag in ('R12','R14A'):
  q=read(p/'SELECTOR_MODELS.json');g={r['controller']:r['primary_action'] for r in q['choices']};strategies['G_E']=np.array([arms.index(g[n]) for n in names])
 predspec=[]
 if tag in ('R14A','R10','R12'):predspec=[(ROOT/'diagnostics/ring_controller_critic_v1',{'R14A':'T1_','R10':'T2_','R12':'T3_'}[tag],m,'R16_'+m) for m in ('A0','A3')]
 if tag.startswith('R17'):predspec=[(p,'',m,m) for m in ('A0','A3','A3_all')]
 if tag=='R18TEST':predspec=[(p,'',m,m) for m in ('C0','C1','C2','C3','A0')]
 if tag=='R15':
  cvfolder=ROOT/'diagnostics/ring_controller_critic_v1/predictions'
  for model in ('A0','A3'):
   probs=np.full((C,A),np.nan);seen_names=set()
   for fold in range(7):
    zz=[np.load(cvfolder/f'CV_fold{fold}_{model}_seed{seed}.npz',allow_pickle=False) for seed in (17,23,41)];pn=zz[0]['controller_names'].tolist();pa=zz[0]['arms'].tolist();pr=np.mean([z['probability'] for z in zz],axis=0).mean(1)
    for ii,n in enumerate(pn):
     assert n not in seen_names;seen_names.add(n);i=names.index(n)
     for j,a in enumerate(arms):probs[i,j]=pr[ii,pa.index(a)]
   assert seen_names==set(names)
   for key,menu in [('fixed',fixed),('full',list(range(A)))]:
    mask=np.full_like(probs,-np.inf);mask[:,menu]=probs[:,menu];strategies['R16_CV_'+model+'_'+key]=np.argmax(mask,1)
 for folder,prefix,m,label in predspec:
  pred=pred_choices(folder,prefix,m,names,arms,av)
  if pred is None:continue
  for key,menu in [('fixed',fixed),('full',list(range(A)))]:
   mask=np.full_like(pred,-np.inf);mask[:,menu]=np.nan_to_num(pred[:,menu],nan=-np.inf);strategies[label+'_'+key]=np.argmax(mask,1)
 # Hierarchical controller then independent within-controller state cluster resampling.
 blo=np.empty((B,C,A));bhi=blo.copy();indices=np.empty((B,C),int)
 for start in range(0,B,100):
  n=min(100,B-start);cc=rng.integers(0,C,(n,C));ss=rng.integers(0,S,(n,C,S));indices[start:start+n]=cc;yy=y[cc[:,:,None],ss];uu=u[cc[:,:,None],ss];blo[start:start+n]=np.where(av[cc],yy.mean((2,4)),np.nan);bhi[start:start+n]=np.where(av[cc],(yy+uu).mean((2,4)),np.nan)
 bootbf=np.nanmax(blo[:,:,fixed],2);bootbs=np.nanmax(blo[:,:,sched],2);bo=np.nanmax(blo,2);bohi=np.nanmax(bhi,2);diff=bs-bf;bdiff=(bootbs-bootbf).mean(1);point={};bootstrap_policy={}
 for key,ch in strategies.items():
  value=lo[np.arange(C),ch];upper=hi[np.arange(C),ch];pick=ch[indices];bv=np.take_along_axis(blo,pick[:,:,None],2)[:,:,0];bvu=np.take_along_axis(bhi,pick[:,:,None],2)[:,:,0]
  if key=='best_fixed':bv=bootbf
  if key=='oracle':bv=bo
  bootv=bv.mean(1);br=(bo-bv).mean(1);bootstrap_policy[key]=bootv;point[key]=dict(choices=[arms[i] for i in ch],success=float(value.mean()),success_upper=float(upper.mean()),success_CI=ci(bootv),mean_regret=float((oracle-value).mean()),regret_to_fixed_oracle=float((bf-value).mean()),max_regret=float((oracle-value).max()),regret_CI=ci(br),unknown_regret_bounds=[float(np.maximum(0,oracle-upper).mean()),float((np.nanmax(hi,1)-value).mean())]);tables['B'].append(dict(dataset=tag,policy=key,success=point[key]['success'],mean_regret=point[key]['mean_regret'],fixed_oracle_regret=point[key]['regret_to_fixed_oracle'],ci_low=point[key]['regret_CI'][0],ci_high=point[key]['regret_CI'][1]))
 comparisons={}
 for one,two in [('default_schedule','best_fixed'),('best_fixed','F10'),('default_schedule','F10')]:comparisons[one+'-'+two]=dict(difference=point[one]['success']-point[two]['success'],CI=ci(bootstrap_policy[one]-bootstrap_policy[two]))
 Aresult=dict(mean=float(diff.mean()),median=float(np.median(diff)),CI=ci(bdiff),median_CI=ci(np.median(bootbs-bootbf,1)),schedule_better=int((diff>=.01-1e-14).sum()),fixed_better=int((diff<=-.01+1e-14).sum()),tie=int((abs(diff)<.01-1e-14).sum()),per_controller=diff.tolist(),comparisons=comparisons)
 tables['A'].append(dict(dataset=tag,n=C,states=S,seeds=Q,mean=Aresult['mean'],median=Aresult['median'],ci_low=Aresult['CI'][0],ci_high=Aresult['CI'][1],schedule_better=Aresult['schedule_better'],fixed_better=Aresult['fixed_better'],tie=Aresult['tie']))
 theory=[]
 for j in sched:
  arm=arms[j]
  if 'to' in arm:
   first,last=arm[1:].split('to');a=int(first);b=int(last.split('_')[0])
  elif arm.startswith(('S13_','S15_')):a=int(arm[1:3]);b=10
  else:a=13;b=10
  fa='F'+str(a);fb='F'+str(b)
  if fa not in arms or fb not in arms:continue
  ia=arms.index(fa);ib=arms.index(fb);effect=lo[:,j]-lo[:,ib];late=lo[:,j]-lo[:,ia];net=lo[:,ia]-lo[:,ib];dominance=lo[:,j]>=np.maximum(lo[:,ia],lo[:,ib])-.01-1e-14;ordering=(effect>=-1e-14)&(late>=-1e-14);bootdom=(blo[:,:,j]>=np.maximum(blo[:,:,ia],blo[:,:,ib])-.01-1e-14).mean(1);bootorder=((blo[:,:,j]>=blo[:,:,ia]-1e-14)&(blo[:,:,j]>=blo[:,:,ib]-1e-14)).mean(1);row=dict(arm=arm,endpoints=[fa,fb],dominance_fraction=float(dominance.mean()),dominance_CI=ci(bootdom),ordering_fraction=float(ordering.mean()),ordering_CI=ci(bootorder),E=effect.tolist(),L=late.tolist(),N=net.tolist(),N_positive=int((net>1e-14).sum()),N_negative=int((net<-1e-14).sum()),N_tie=int((abs(net)<=1e-14).sum()));theory.append(row);tables['D'].append(dict(dataset=tag,arm=arm,dominance=row['dominance_fraction'],ordering=row['ordering_fraction'],N_positive=row['N_positive'],N_negative=row['N_negative'],N_tie=row['N_tie']))
 sensitivity=None
 grid=[a for a in ('S200','S300','S400','S500','S600') if a in arms]
 if len(grid)==5:
  rates=lo[:,[arms.index(a) for a in grid]];best=rates.max(1);reg=best[:,None]-rates;platform=[list(np.array([200,300,400,500,600])[r<=.02+1e-14]) for r in reg];sensitivity=dict(best_step=np.array([200,300,400,500,600])[rates.argmax(1)].tolist(),regret=reg.tolist(),platform_steps=platform,platform_span=[max(x)-min(x) for x in platform],coverage_each_step=(reg<=.02+1e-14).mean(0).tolist(),mean_regret=reg.mean(0).tolist(),grid=grid)
  for i,n in enumerate(names):tables['E'].append(dict(dataset=tag,controller=n,best_step=sensitivity['best_step'][i],platform_steps=str(platform[i]),span=sensitivity['platform_span'][i]))
 ladder=['F10','best_fixed','default_schedule'];crit=next((k for k in ('G_E','A3_full','C2_full','R16_A3_full','R16_CV_A3_full') if k in point),None)
 if crit:ladder.append(crit)
 ladder.append('oracle');F=[]
 for idx,k in enumerate(ladder):
  row=dict(policy=k,success=point[k]['success'],CI=point[k]['success_CI'],increment=None if idx==0 else point[k]['success']-point[ladder[idx-1]]['success'],increment_CI=None if idx==0 else ci(bootstrap_policy[k]-bootstrap_policy[ladder[idx-1]]));F.append(row);tables['F'].append(dict(dataset=tag,policy=k,success=row['success'],ci_low=row['CI'][0],ci_high=row['CI'][1],increment=row['increment'],increment_CI=str(row['increment_CI'])))
 cross_order=[]
 for t in ('t1','t2'):
  if 'S13_'+t in arms and 'S15_'+t in arms:
   i=arms.index('S13_'+t);j=arms.index('S15_'+t);cross_order.append(dict(time=t,delta13_minus15=float((lo[:,i]-lo[:,j]).mean()),CI=ci((blo[:,:,i]-blo[:,:,j]).mean(1))))
 results[tag]=dict(n_controller=C,n_state=S,n_seed=Q,unknown=int(u.sum()),menu=arms,default=defarm,label_evidence=d['meta']['label_evidence'],analysis_evidence='DEVELOPMENT unified retrospective',A=Aresult,B=point,D=theory,E=sensitivity,F=F,cross_scene_order=cross_order,rates_lower=lo.tolist(),rates_upper=hi.tolist(),bootstrap= dict(draws=B,seed=202610152100,controller_then_state=True,source_policy_ci_conditional=True));print(tag,'A gain',float(diff.mean()),flush=True)
 fig,ax=plt.subplots(figsize=(8,4));xs=np.arange(len(F));ax.plot(xs,[x['success'] for x in F],'-o');ax.set_xticks(xs,[x['policy'] for x in F],rotation=25,ha='right');ax.set_ylabel('Success lower bound');ax.set_ylim(0,1.05);ax.set_title(tag+' / DEVELOPMENT');fig.tight_layout()
 for ext in ('png','pdf'):fig.savefig(HERE/(tag+'_ladder.'+ext),dpi=170)
 plt.close(fig)
# Across scene LOSO reselected from immutable point source scene means.
for prefix in ('R13_','R20_'):
 tags=[t for t in datasets if t.startswith(prefix)];scene_rates=np.array([np.array(results[t]['rates_lower']).mean(0) for t in tags]);arms=datasets[tags[0]]['arms'];out=[]
 for i,t in enumerate(tags):
  sources=[j for j,s in enumerate(tags) if (j!=i if prefix=='R13_' else datasets[s]['meta']['scene']!=datasets[t]['meta']['scene'])];mean=scene_rates[sources].mean(0);row=dict(target=t,sources=[tags[j] for j in sources])
  for key,menu in [('fixed',[0,1,2]),('full',list(range(7)))]:
   ch=menu[int(np.argmax(mean[menu]))];row[key]=dict(choice=arms[ch],regret=float(scene_rates[i].max()-scene_rates[i,ch]))
  out.append(row)
 results[prefix+'LOSO']=out
for k,rows in tables.items():
 with (HERE/('ANALYSIS_'+k+'.csv')).open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
(HERE/'STATISTICAL_RESULTS.json').write_text(json.dumps(results,indent=2,allow_nan=True,default=lambda x:np.asarray(x).tolist())+'\n')
tags=[t for t in results if isinstance(results[t],dict)];fig,ax=plt.subplots(figsize=(16,5));ax.boxplot([results[t]['A']['per_controller'] for t in tags],tick_labels=tags,showfliers=True);ax.axhline(0,color='gray');ax.set_ylabel('Best schedule minus best fixed');ax.tick_params(axis='x',rotation=70);fig.tight_layout()
for ext in ('png','pdf'):fig.savefig(HERE/('schedule_fixed_distribution.'+ext),dpi=170)
plt.close(fig)
fig,ax=plt.subplots(figsize=(10,5))
for t in ('R15','R17N2','R18TEST'):
 e=results[t]['E'];ax.plot([200,300,400,500,600],e['mean_regret'],'-o',label=t)
ax.set_ylabel('Mean regret to best schedule');ax.set_xlabel('Switch step');ax.legend();fig.tight_layout()
for ext in ('png','pdf'):fig.savefig(HERE/('switch_sensitivity.'+ext),dpi=170)
plt.close(fig)
# R9 curve shape, by controller and direction.
r9=results['R9'];curves=[]
for i,n in enumerate(datasets['R9']['names']):
 for direction in ('10to13','13to10'):
  armgrid=[f'S{direction}_{t}' for t in (100,200,300,400,500)];values=[r9['rates_lower'][i][r9['menu'].index(a)] for a in armgrid];delta=np.diff(values);shape='plateau' if np.all(abs(delta)<=.01) else 'monotone_increasing' if np.all(delta>=-1e-14) else 'monotone_decreasing' if np.all(delta<=1e-14) else 'nonmonotonic';curves.append(dict(controller=n,direction=direction,t=[100,200,300,400,500],rates=values,shape=shape))
(HERE/'R9_SWITCH_CURVES.json').write_text(json.dumps(curves,indent=2)+'\n');print('Statistics A/B/D/E/F complete',flush=True)
