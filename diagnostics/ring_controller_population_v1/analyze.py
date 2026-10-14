from core import *
import math,csv
from scipy.stats import spearmanr,pearsonr
sys.path.insert(0,str(R9/'vendor'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ARMS=['F10','F13','F2','S200','S300','S400','S500','S600'];SW=np.array([200,300,400,500,600]);REPS=['R5-PCA2','R6-Delta','R5-PCA2+R6-Delta'];TARGETS=['E','L','N','t*']
def rho(a,b):
 if len(a)<3 or np.std(a)<1e-12 or np.std(b)<1e-12:return None
 v=float(spearmanr(a,b).statistic);return v if np.isfinite(v) else None
def ridge(x,y,t):
 y=np.asarray(y,float);xm=x.mean(0);ym=y.mean(0);beta=np.linalg.solve((x-xm).T@(x-xm)+np.eye(x.shape[1]),(x-xm).T@(y-ym));return (t-xm)@beta+ym
def prepared(raw5,raw6,train,test,rep):
 def scale(a):
  src=a[train].astype(float);tar=a[test].astype(float);keep=np.isfinite(src).any(0);src=src[:,keep];tar=tar[:,keep];mu=np.nanmean(src,axis=0);src=np.where(np.isfinite(src),src,mu);tar=np.where(np.isfinite(tar),tar,mu);sd=src.std(0);sd[sd<1e-6]=1;return (src-mu)/sd,(tar-mu)/sd
 if rep=='R6-Delta':return scale(raw6)
 a,b=scale(raw5);_,_,vt=np.linalg.svd(a,full_matrices=False);pc=vt[:2].copy()
 for v in pc:
  if v[np.argmax(abs(v))]<0:v*=-1
 a=a@pc.T;b=b@pc.T
 if rep=='R5-PCA2':return a,b
 c,d=scale(raw6);return np.c_[a,c],np.c_[b,d]
def effect(rate):
 e=rate[:,ARMS.index('S500')]-rate[:,ARMS.index('F10')];l=rate[:,ARMS.index('S500')]-rate[:,ARMS.index('F13')];n=rate[:,ARMS.index('F13')]-rate[:,ARMS.index('F10')]
 t=SW[::-1][np.argmax(rate[:,[ARMS.index('S'+str(x)) for x in SW]][:,::-1],axis=1)]
 return np.c_[e,l,n,t]
def metrics(y,p,baseline):
 ans={}
 for j,name in enumerate(TARGETS):
  truth=y[:,j];pred=p[:,j];rmse=float(np.sqrt(np.mean((truth-pred)**2)));base=float(np.sqrt(np.mean((truth-baseline[:,j])**2)))
  ans[name]=dict(Spearman=rho(truth,pred),RMSE=rmse,source_mean_RMSE=base,RMSE_ratio_vs_source_mean=rmse/base if base>0 else None)
  if j<3:ans[name]['sign_accuracy']=float(np.mean(np.sign(truth)==np.sign(pred)))
  else:ans[name]['within_100_steps']=float(np.mean(abs(truth-pred)<=100))
 return ans
def cv(raw5,raw6,rate,profiles,plan,waves,mode,k,rep):
 names=list(profiles);recipe=np.array([profiles[n]['parent_best_name'] for n in names]);wave=np.array([profiles[n]['wave'] for n in names]);available=wave<=waves;y=effect(rate);pred=np.full_like(y,np.nan);base=np.full_like(y,np.nan);chosen=np.full((len(names),4),-1,int);folds=[]
 if mode=='recipe':splits=[(np.flatnonzero(available&~np.isin(recipe,pair)&(wave<=k)),np.flatnonzero(available&np.isin(recipe,pair)),pair) for pair in plan['recipe_folds']]
 else:splits=[]
 if mode=='seed':
  for s in range(1,waves+1):
   offsets=[v for v in range(1,waves+1) if v!=s][:k]
   if len(offsets)==k:splits.append((np.flatnonzero(np.isin(wave,offsets)),np.flatnonzero(wave==s),[s]))
 for fi,(tr,te,held) in enumerate(splits):
  if not len(tr) or not len(te):continue
  x,z=prepared(raw5,raw6,tr,te,rep);pred[te]=ridge(x,y[tr],z);base[te]=y[tr].mean(0);armhat=ridge(x,rate[tr],z);source_arm=int(np.argmax(rate[tr].mean(0)))
  for ii,i in enumerate(te):
   eh=pred[i,0];th=pred[i,3];near=SW[np.flatnonzero(abs(SW-th)==np.min(abs(SW-th)))[-1]];chosen[i]=[source_arm,ARMS.index('F10') if eh<.05 else ARMS.index('S500'),ARMS.index('F10') if eh<.05 else ARMS.index('S'+str(near)),int(np.argmax(armhat[ii]))]
  folds.append(dict(fold=fi,held_out=held,train_count=len(tr),test_count=len(te),test_indices=te.tolist()))
 test=np.flatnonzero(np.isfinite(pred[:,0]));oracle=rate[test].max(1);regret=np.array([[oracle[z]-rate[i,chosen[i,j]] for j in range(4)] for z,i in enumerate(test)]);foldreg=[]
 for f in folds:
  idx=[np.where(test==i)[0][0] for i in f['test_indices']];foldreg.append(np.mean(regret[idx],axis=0).tolist())
 return dict(mode=mode,k=k,representation=rep,test_indices=test.tolist(),metrics=metrics(y[test],pred[test],base[test]),mean_regret=dict(zip(['source_mean','G_E','G_EL','multioutput_ridge'],np.mean(regret,axis=0).tolist())),folds=folds,fold_regret=foldreg,predictions=pred[test].tolist(),baseline=base[test].tolist())
def boot_ci(values,seed=20261014):
 a=np.asarray(values,float);rng=np.random.default_rng(seed);draw=rng.integers(0,len(a),(10000,len(a)));means=a[draw].mean(1);return [float(np.quantile(means,.025)),float(np.quantile(means,.975))]
def icc(a,recipe):
 groups=[a[recipe==r] for r in sorted(set(recipe))];groups=[g for g in groups if len(g)]
 if len(groups)<2 or min(map(len,groups))<2:return None
 n=len(groups[0]);m=np.array([g.mean() for g in groups]);grand=np.mean(m);msb=n*np.sum((m-grand)**2)/(len(groups)-1);msw=sum(np.sum((g-g.mean())**2) for g in groups)/(len(groups)*(n-1));den=msb+(n-1)*msw
 return float((msb-msw)/den) if den>0 else None
def external_e(raw5,source_e,waves):
 sources=[];root=ROOT/'diagnostics'
 specs=[('round10_best14',root/'controller_behavior_repr_v1/FEATURES.npz',root/'controller_phase_decomposition_v1/results.json','R5','E_S500'),('round12_siblings8',root/'controller_new_checkpoint_confirmation_v1/fp32_n8/R5_FINAL_FEATURES.npz',root/'controller_new_checkpoint_confirmation_v1/fp32_n8/results.json','features','E'),('round14A_independent14',root/'ring_independent_flow_confirmation_v2/R5_FINAL_FEATURES.npz',root/'ring_independent_flow_confirmation_v2/results.json','features','E')]
 for title,feat,out,key,label in specs:
  if not feat.exists() or not out.exists():sources.append(dict(group=title,status='UNAVAILABLE'));continue
  z=np.load(feat);names=z['controller_names'].tolist();v=z[key].astype(float);keep=np.ones(160,bool);keep[105:110]=False;v=v[:,keep];effects=read(out).get('effects',{});selected=[i for i,n in enumerate(names) if n in effects and label in effects[n]]
  if not selected:sources.append(dict(group=title,status='UNAVAILABLE_LABELS'));continue
  src=raw5.copy();tar=v[selected];valid=np.isfinite(src).any(0);src=src[:,valid];tar=tar[:,valid];mu=np.nanmean(src,0);src=np.where(np.isfinite(src),src,mu);tar=np.where(np.isfinite(tar),tar,mu);sd=src.std(0);sd[sd<1e-6]=1;a=(src-mu)/sd;b=(tar-mu)/sd;_,_,vt=np.linalg.svd(a,full_matrices=False);pc=vt[:2].copy()
  for row in pc:
   if row[np.argmax(abs(row))]<0:row*=-1
  pred=ridge(a@pc.T,source_e,b@pc.T);truth=np.array([effects[names[i]][label]['point'] for i in selected]);sources.append(dict(group=title,status='DESCRIPTIVE',count=len(selected),Spearman=rho(truth,pred),RMSE=float(np.sqrt(np.mean((truth-pred)**2))),rows=[dict(controller=names[i],truth=float(t),predicted=float(p)) for i,t,p in zip(selected,truth,pred)]))
 return sources
def main():
 labels_allowed();plan=read(HERE/'analysis_plan.json');profiles=read(HERE/'FINAL_BASE_PROFILES.json')['profiles'];waves=0
 for k in range(1,5):
  f=HERE/f'wave{k}_VERIFY.json'
  if not f.exists():break
  assert read(f)['complete'];waves=k
 assert waves>=1,'No complete wave; no main analysis'
 names=[n for n in profiles if profiles[n]['wave']<=waves];assert len(names)==14*waves
 z5=np.load(HERE/'R5_FEATURES.npz');z6=np.load(HERE/'R6_FEATURES.npz');all5=z5['controller_names'].tolist();all6=z6['controller_names'].tolist();keep=np.ones(160,bool);keep[105:110]=False;raw5=z5['features'][[all5.index(n) for n in names]][:,keep].astype(float);raw6=z6['delta'][[all6.index(n) for n in names]].astype(float)
 assert np.isfinite(raw5).any(0).all() and np.isfinite(raw6).all(1).all(),'A controller lacks a complete paired R6 descriptor; R6 analyses are INCONCLUSIVE'
 current={n:profiles[n] for n in names};rate=np.zeros((len(names),8));upper=np.zeros_like(rate);half=np.zeros((len(names),8,2));counts=np.zeros((len(names),8,3),int);dataset=HERE/'dataset.jsonl';case_index={}
 for k in range(1,waves+1):
  for c in read(HERE/f'wave{k}_MISSING_CASES.json'):case_index.setdefault((c['base_controller_name'],c['arm']),[]).append((k,c))
 with dataset.open('w') as f:
  for i,n in enumerate(names):
   for j,arm in enumerate(ARMS):
    records=[]
    for k,c in case_index.get((n,arm),[]):
      r=read(HERE/f'wave{k}'/'raw'/f"{c['rollout_uid']}.json");unknown=bool(r['numerical_failure']);records.append((c,r,unknown));f.write(json.dumps(dict(controller=n,recipe=current[n]['parent_best_name'],training_seed=current[n]['training_seed'],wave=k,state_uid=c['state_uid'],state_index=c['state_index'],arm=arm,future_seed=c['seed'],rollout_uid=c['rollout_uid'],success=int(r['success']),unknown=unknown,outcome=r['outcome'],episode_length=r['episode_length'],raw_path=str(HERE/f'wave{k}'/'raw'/f"{c['rollout_uid']}.json")),sort_keys=True)+'\n')
    assert len(records)==128,(n,arm,len(records))
    success=sum(r['success'] for _,r,_ in records);unk=sum(u for _,_,u in records);counts[i,j]=[success,unk,128-success-unk];rate[i,j]=success/128;upper[i,j]=(success+unk)/128
    for part in range(2):half[i,j,part]=sum(r['success'] for c,r,u in records if c['seed']//2==part)/64
 y=effect(rate);intervals={}
 for label,a,b in [('E','S500','F10'),('L','S500','F13'),('N','F13','F10')]:
  ai=ARMS.index(a);bi=ARMS.index(b);intervals[label]=np.c_[rate[:,ai]-upper[:,bi],upper[:,ai]-rate[:,bi]].tolist()
 recipes=np.array([current[n]['parent_best_name'] for n in names]);results=dict(status='COMPLETE_4_WAVES' if waves==4 else 'PARTIAL_COMPLETE_WAVES',utc=utc(),complete_waves=waves,controllers=len(names),dataset_sha256=sha(dataset),unknown_total=int(counts[:,:,1].sum()),rates_lower=rate.tolist(),rates_upper=upper.tolist(),controller_names=names,label_columns=TARGETS,labels=clean(y),a_star=[ARMS[i] for i in np.argmax(rate,axis=1)],label_intervals=intervals,upper_choice_labels=clean(effect(upper)),CV={},CV_upper_sensitivity={},split_half={},variance={},hypotheses={},evidence_level='DEVELOPMENT',video_exemption='Pure Q function data generation/evaluation; preregistered before results')
 for j,arm in enumerate(ARMS):results['split_half'][arm]=dict(Spearman=rho(half[:,j,0],half[:,j,1]),Pearson=float(pearsonr(half[:,j,0],half[:,j,1]).statistic) if np.std(half[:,j,0]) and np.std(half[:,j,1]) else None)
 yh=effect(half[:,:,0]);zh=effect(half[:,:,1]);
 for j,label in enumerate(TARGETS):results['split_half'][label]=dict(Spearman=rho(yh[:,j],zh[:,j]),Pearson=float(pearsonr(yh[:,j],zh[:,j]).statistic) if np.std(yh[:,j]) and np.std(zh[:,j]) else None)
 for label in TARGETS:
  r=results['split_half'][label]['Pearson'];full=2*r/(1+r) if r is not None and r>0 else None;results['split_half'][label]['Spearman_Brown_full_reliability']=full;results['split_half'][label]['correlation_noise_ceiling']=math.sqrt(min(1.,full)) if full is not None else None
 results['split_half']['a*']=dict(agreement=float(np.mean(np.argmax(half[:,:,0],axis=1)==np.argmax(half[:,:,1],axis=1))))
 results['split_half']['t*']['agreement']=float(np.mean(yh[:,3]==zh[:,3]))
 rng=np.random.default_rng(20261014)
 for j,label in enumerate(TARGETS):
  v=icc(y[:,j],recipes);boot=[];groups=sorted(set(recipes))
  if waves>=2:
   for _ in range(10000):
    sampled=rng.integers(0,len(groups),len(groups));a=np.concatenate([y[recipes==groups[t],j] for t in sampled]);g=np.repeat(np.arange(len(groups)),waves);val=icc(a,g)
    if val is not None:boot.append(val)
  results['variance'][label]=dict(ICC=v,CI=[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))] if boot else None,between_fraction=max(0,v) if v is not None else None,within_fraction=1-max(0,v) if v is not None else None)
 if waves==4:results['variance']['discordant_N_sign_recipes']=sum(len(set(np.sign(y[recipes==r,2])))>1 for r in set(recipes))
 else:results['variance']['discordant_N_sign_recipes']=None
 for mode in ['recipe','seed']:
  for k in range(1,(waves if mode=='recipe' else waves-1)+1):
   for rep in REPS:
    ans=cv(raw5,raw6,rate,current,plan,waves,mode,k,rep);key=f'{mode}|k{k}|{rep}';results['CV'][key]=ans
    sensitivity=cv(raw5,raw6,upper,current,plan,waves,mode,k,rep);results['CV_upper_sensitivity'][key]=dict(metrics=sensitivity['metrics'],mean_regret=sensitivity['mean_regret'])
 if waves<4:
  results['hypotheses']={x:'INCONCLUSIVE' for x in ['H_cov_L','H_cov_E','H_switch']}
 else:
  a=results['CV'];get=lambda k,r:a[f'recipe|k{k}|{r}'];l=[get(4,r)['metrics']['L'] for r in REPS[1:]]
  results['hypotheses']['H_cov_L']='SUPPORTED' if any(x['Spearman'] is not None and x['Spearman']>=.6 and x['RMSE_ratio_vs_source_mean'] is not None and x['RMSE_ratio_vs_source_mean']<=.7 for x in l) else 'REJECTED'
  e1=get(1,'R5-PCA2');e4=get(4,'R5-PCA2');d=[]
  for f in range(7):
   ii=e1['folds'][f]['test_indices'];p1=np.asarray(e1['predictions'])[np.isin(e1['test_indices'],ii),0];p4=np.asarray(e4['predictions'])[np.isin(e4['test_indices'],ii),0];truth=y[ii,0];d.append(((truth-p4)**2,(truth-p1)**2))
  rng_boot=np.random.default_rng(20261014);draw=rng_boot.integers(0,7,(10000,7));sq4=np.asarray([v[0] for v in d]);sq1=np.asarray([v[1] for v in d]);delta=np.sqrt(sq4[draw].mean(axis=(1,2)))-np.sqrt(sq1[draw].mean(axis=(1,2)));ci=[float(np.quantile(delta,.025)),float(np.quantile(delta,.975))];results['hypotheses']['H_cov_E']='SUPPORTED' if ci[1]<0 else 'REJECTED';results['H_cov_E_CI']=ci
  el=get(4,'R5-PCA2+R6-Delta');d=np.asarray(el['fold_regret'])[:,2]-np.asarray(el['fold_regret'])[:,1];ci=boot_ci(d);results['hypotheses']['H_switch']='SUPPORTED' if ci[1]<0 else 'REJECTED';results['H_switch_CI']=ci
 results['external_E_validation']=external_e(raw5,y[:,0],waves)
 gate_audit=read(HERE/'CONSISTENCY_GATE.json');preflight=read(HERE/'main_CACHE_PREFLIGHT.json');postflight=read(HERE/'completed_POSTFLIGHT.json');results['provenance']=dict(preregistration_sha256=sha(HERE/'PREREGISTRATION.md'),r5_receipt_sha256=sha(HERE/'R5_FROZEN.receipt.json'),r6_receipt_sha256=sha(HERE/'R6_FROZEN.receipt.json'),analysis_plan_sha256=sha(HERE/'analysis_plan.json'),consistency_gate_passed=gate_audit['passed'],consistency_cases=len(gate_audit['checks']),preflight_summary=preflight['summary'],completed_postflight_summary=postflight['summary'])
 write(HERE/'results.json',results)
 fig,ax=plt.subplots(1,3,figsize=(14,4))
 for rep in REPS:
  ks=range(1,waves+1);ax[0].plot(list(ks),[results['CV'][f'recipe|k{k}|{rep}']['metrics']['E']['RMSE'] for k in ks],marker='o',label=rep);ax[1].plot(list(ks),[results['CV'][f'recipe|k{k}|{rep}']['metrics']['L']['RMSE'] for k in ks],marker='o',label=rep)
 ax[0].set(title='E learning curve',xlabel='seeds per source recipe',ylabel='RMSE');ax[1].set(title='L learning curve',xlabel='seeds per source recipe',ylabel='RMSE');ax[2].scatter(y[:,0],y[:,1],s=20);ax[2].set(title='E vs L',xlabel='E',ylabel='L');ax[0].legend(fontsize=7);fig.tight_layout();fig.savefig(HERE/'LEARNING_E_L.png',dpi=150);plt.close(fig)
 fig,ax=plt.subplots(1,2,figsize=(10,4))
 for rep in REPS:
  ks=range(1,waves+1);ax[0].plot(list(ks),[results['CV'][f'recipe|k{k}|{rep}']['metrics']['E']['RMSE'] for k in ks],marker='o',label=rep);ax[1].plot(list(ks),[results['CV'][f'recipe|k{k}|{rep}']['metrics']['L']['RMSE'] for k in ks],marker='o',label=rep)
 ax[0].set(title='E',xlabel='seeds per source recipe',ylabel='RMSE');ax[1].set(title='L',xlabel='seeds per source recipe',ylabel='RMSE');ax[0].legend(fontsize=7);fig.tight_layout();fig.savefig(HERE/'LEARNING_CURVES.png',dpi=150);plt.close(fig)
 fig,ax=plt.subplots(figsize=(5,5));ax.scatter(y[:,0],y[:,1],s=24);ax.set(xlabel='E',ylabel='L',title='Early vs late effects');fig.tight_layout();fig.savefig(HERE/'E_L_SCATTER.png',dpi=150);plt.close(fig)
 fig,ax=plt.subplots(figsize=(7,4));ax.bar(TARGETS,[results['variance'][x]['ICC'] or 0 for x in TARGETS]);ax.set(ylabel='ICC',title='Recipe variance fraction');fig.tight_layout();fig.savefig(HERE/'VARIANCE_DECOMPOSITION.png',dpi=150);plt.close(fig)
 def fmt(x):return 'undefined' if x is None else f'{x:.3f}'
 hl=results['hypotheses'];regime=('配方差异为主' if results['variance']['N']['ICC'] is not None and results['variance']['N']['ICC']>.5 else 'seed 差异为主') if waves==4 else 'INCONCLUSIVE';lead=f"H_cov_L {hl['H_cov_L']}，H_switch {hl['H_switch']}；H_cov_E {hl['H_cov_E']}。controller regime 判定：{regime}。"+(f"N 的配方 ICC={fmt(results['variance']['N']['ICC'])}；14 个配方中有 {results['variance']['discordant_N_sign_recipes']} 个配方，其四个训练 seed 的 N 符号不一致。" if waves==4 else '仅完成不足四个训练 seed 波次，四 seed regime 判定不可完成。')
 lines=[lead,'',f"证据级别：DEVELOPMENT。完整波次 {waves}/4，controller {len(names)}/56；新 outcome case 的 unknown 数 {results['unknown_total']}。下界与上界均存于 results.json；unknown 保留，不重跑，timeout 未称为 deadlock。",'', '## 事实与门槛','',f"H_cov_L：{hl['H_cov_L']}；H_cov_E：{hl['H_cov_E']}；H_switch：{hl['H_switch']}。门槛及特征/分析收据见 PREREGISTRATION.md、ANALYSIS_FROZEN.receipt.json、ANALYSIS_DETAIL.receipt.json、R5_FROZEN.receipt.json、R6_FROZEN.receipt.json。",'', '## 历史证据、兼容性与前瞻范围','', '历史代码与配方来自 round14A 的 GPU 复现门；本轮 56 个 checkpoint 的训练、参数新颖性和兼容性审计见 TRAINING_AUDIT.json、CHECKPOINT_NOVELTY_AUDIT.json、COMPATIBILITY_AUDIT.json；32 个新 state 的去重见 STATE_DEDUP_AUDIT.json。新 controller 和新 state 的 Q4 标签为前瞻采集；交叉验证同属该数据集，故学习曲线为 DEVELOPMENT。', '', '## 数值结果','', '下表 RMSE 与遗憾用成功率比例表示，乘 100 即百分点；未知结果的上界敏感性在 results.json。', '', '|特征|k|E RMSE|L Spearman|L RMSE 比|G_E 遗憾|G_EL 遗憾|','|---|---:|---:|---:|---:|---:|---:|']
 unknown_audit=read(HERE/'POSTFLIGHT_UNKNOWN_AUDIT.json')
 lines.insert(11,f"缓存：预检 {preflight['summary']}；完整波次合并后标准复检 {postflight['summary']}。标准复检把 {unknown_audit['unknown_present_in_db']} 条已在 DB 中的 numerical_failure 按可复用性标为 missing；POSTFLIGHT_UNKNOWN_AUDIT.json 逐条核对 DB、raw 和 journal，真实缺失={unknown_audit['true_absent']}，这些案例保留为 unknown，不重跑。32 例一致性门通过={gate_audit['passed']}。")
 if waves==4:lines.insert(7,f"H_cov_E 折 bootstrap 差值 95% CI={results['H_cov_E_CI']}；H_switch 配对遗憾差 95% CI={results['H_switch_CI']}。")
 for rep in REPS:
  for k in range(1,waves+1):
   r=results['CV'][f'recipe|k{k}|{rep}'];lines.append(f"|{rep}|{k}|{fmt(r['metrics']['E']['RMSE'])}|{fmt(r['metrics']['L']['Spearman'])}|{fmt(r['metrics']['L']['RMSE_ratio_vs_source_mean'])}|{fmt(r['mean_regret']['G_E'])}|{fmt(r['mean_regret']['G_EL'])}|")
 lines+=['','### 新训练 seed 泛化','', '|特征|训练 seed 数|E RMSE|L Spearman|L RMSE 比|t* ±100 比例|','|---|---:|---:|---:|---:|---:|']
 for rep in REPS:
  for k in range(1,waves):
   r=results['CV'][f'seed|k{k}|{rep}'];lines.append(f"|{rep}|{k}|{fmt(r['metrics']['E']['RMSE'])}|{fmt(r['metrics']['L']['Spearman'])}|{fmt(r['metrics']['L']['RMSE_ratio_vs_source_mean'])}|{fmt(r['metrics']['t*']['within_100_steps'])}|")
 lines+=['','### 方差分解','', '|标签|配方 ICC|95% CI|配方占比|同配方 seed 占比|','|---|---:|---:|---:|---:|']
 for label in TARGETS:
  v=results['variance'][label];ci=v['CI'];lines.append(f"|{label}|{fmt(v['ICC'])}|{fmt(ci[0])+' ～ '+fmt(ci[1]) if ci else 'undefined'}|{fmt(v['between_fraction'])}|{fmt(v['within_fraction'])}|")
 if waves==4:lines+=['',f"预注册描述性预期为至少 3/14 配方的四个训练 seed 对 N 符号不一致；实际为 {results['variance']['discordant_N_sign_recipes']}/14，此项不是判定门。"]
 lines+=['','### 外部 E 验证（描述性）','', '|组|状态|n|Spearman|RMSE|','|---|---|---:|---:|---:|']
 for x in results['external_E_validation']:lines.append(f"|{x['group']}|{x['status']}|{x.get('count','—')}|{fmt(x.get('Spearman'))}|{fmt(x.get('RMSE'))}|")
 lines+=['','## Split-half 信度及噪声参照','', '|标签|Split-half Spearman|Split-half Pearson|推算相关上限|','|---|---:|---:|---:|']
 for label in ['E','L','N','t*']:
  r=results['split_half'][label];lines.append(f"|{label}|{fmt(r['Spearman'])}|{fmt(r['Pearson'])}|{fmt(r['correlation_noise_ceiling'])}|")
 lines+=['','相关上限采用正的 half-Pearson 经 Spearman–Brown 修正后开方；负值或常数标签不推算。它是独立噪声假设下的参照，不是门槛。所有详细折、seed 泛化、符号准确率、t* 误差、四种规则及各动作 split-half 信度在 results.json。图见 LEARNING_CURVES.png、E_L_SCATTER.png 和 VARIANCE_DECOMPOSITION.png。', '', '## 解释与限制','', 'H 假设仅依据预冻结门槛；若少于四波，k=4 门无法检验。Q4 是低预算数据集而非 B15 认证。纯 Q 数据生成与评估按 AGENTS.md 的预注册例外免视频入口。']
 if hl['H_cov_L']=='SUPPORTED' and hl['H_switch']=='SUPPORTED':lines+=['','解释（DEVELOPMENT）：增加 controller 后，后期效应与切换规则均通过冻结门。下一步可用本数据集训练完整 critic Q(s, probe, a)，再在新 state 和新 controller 上做前瞻确认。']
 elif hl['H_cov_E']=='SUPPORTED' and hl['H_cov_L']=='REJECTED':lines+=['','解释（DEVELOPMENT）：覆盖改善前期效应预测，后期效应未通过冻结门；应与上面的 split-half 噪声参照一起判断表示与标签噪声的贡献。']
 if waves==4 and results['variance']['discordant_N_sign_recipes']>=3:lines+=['',f"解释（描述性）：有 {results['variance']['discordant_N_sign_recipes']}/14 个配方的不同训练 seed 对 N 的符号判断不一致，单凭配方无法确定每个 seed 的偏好；总体 N 方差仍以配方之间为主。"]
 resource=[]
 if (HERE/'RESOURCE_USAGE.jsonl').exists():resource=[json.loads(x) for x in (HERE/'RESOURCE_USAGE.jsonl').read_text().splitlines() if x.strip()]
 for k in range(1,waves+1):
  resource += [read(p) for p in (HERE/f'wave{k}').glob('RESOURCE_worker_*.json')]
 if resource:
  cpu=max(x.get('total_allocated_cpus',x.get('cpu_shards',0)) for x in resource);gpu=max(x.get('running_gpu_shards',x.get('gpu_shards',0)) for x in resource)
  lines+=['','## 资源记录','',f'提交前、训练中及各波 worker 启动时的 squeue/进程快照保存在 RESOURCE_USAGE.jsonl 和 wave*/RESOURCE_worker_*.json；快照最高 CPU={cpu}、GPU={gpu} shard。各波开始的占用与时间还保存在 START.receipt.json。08:30 前上限 24/6，08:30 起上限 18/2。']
 (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
 hand=['# Round15 handoff','',f"UTC: {utc()}",f"Complete waves: {waves}/4; controllers: {len(names)}/56",f"Hypotheses: {hl}",'','## Design and evidence','', 'Four independent GPU training seeds per each of 14 Ring Flow recipes are original seed +30001..+30004. All 56 controllers share 32 new physical Ring states. Q4 uses future seeds 0..3 and F10/F13/F2/S200/S300/S400/S500/S600. R5 and R6 are outcome-blind probes frozen before the first new full-rollout outcome. Core physics, safety, controller and success code was not changed; the only action intervention changes eta at the selected step.','', 'PREREGISTRATION.md, ANALYSIS_DETAIL.md and their receipts fix analysis definitions. HISTORICAL_EVIDENCE_AUDIT.json, HISTORICAL_CODE_AUDIT.json, CHECKPOINT_NOVELTY_AUDIT.json, COMPATIBILITY_AUDIT.json and STATE_DEDUP_AUDIT.json document reuse and compatibility. CONSISTENCY_GATE.json records 32 exact historical replays. The rollout cache index is shared_rollout_db/rollout.sqlite; completed_POSTFLIGHT.json is the standard reuse-oriented postflight, and POSTFLIGHT_UNKNOWN_AUDIT.json confirms all standard missing cases are present numerical failures retained as unknown.','', '## Data and results','', 'dataset.jsonl has one row per controller × state × action × future seed in complete waves, including unknown flags and raw-record links. Raw traces and per-case journal shards are retained under wave*/ and journals/. results.json has lower/upper rates, labels, split-half reliability, ICC intervals, grouped CV predictions, regrets and hypotheses. REPORT.md starts with the three required answers. All learning curves are DEVELOPMENT. The pure-Q video exemption was frozen before results.','', '## Reproduce','',f'Python: {ROOT.parent/"02_C1_Toy_GiveWay/.venv-c1/bin/python"}','', 'From the repository root, run `python -B diagnostics/ring_controller_population_v1/analyze.py` only after complete-wave verification and DB postflight. Do not regenerate unknown rollout labels.','', 'Hard local-time deadlines were 07:30 for a new wave, 10:00 for all rollouts/probes, and 10:30 for REPORT/HANDOFF and compute release. Review WAVE_PLAN.json, wave*/START.receipt.json and RESOURCE_USAGE.jsonl for timing and resource evidence.']
 (HERE/'HANDOFF.md').write_text('\n'.join(hand)+'\n')
 print(json.dumps(dict(waves=waves,hypotheses=hl,unknown=results['unknown_total'])))
if __name__=='__main__':main()
