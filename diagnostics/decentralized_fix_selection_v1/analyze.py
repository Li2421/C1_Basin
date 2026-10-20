from support import *
G=guard()
ARM_FULL={'safety baseline':'只加安全层，不修正','F13':'全程η13','S500':'前500步η13、之后η10的时间表','RHR-A':'右手规则甲（1秒无进展、0.35m/s）','RHR-B':'右手规则乙（邻居0.8m以内、0.10m/s）','RHR-C':'右手规则丙（2秒无进展、0.25m/s）'}
TYPE=np.array([0,1,1,2,2,2]);TYPE_FULL=['不修正','时间表类','右手规则类']
def interval(x):return np.quantile(x,[.025,.975]).tolist()
def main():
 prereg();z=np.load(HERE/'LABELS.npz');labels=z['labels'];names=list(z['controller_names']);fam=z['families'];lo=(labels==1).mean(-1);hi=(labels!=0).mean(-1);rates=lo.mean(-1);upper=hi.mean(-1);ci={n:i for i,n in enumerate(names)};pol={};folds={}
 for kind,file in [('rule','RULE_SELECTION.json'),('critic','CRITIC_SELECTION.json')]:
  choice=np.full(46,-1,int)
  for f in FAMILIES:
   dest=HERE/'folds'/f;receipt=read(dest/'CHOICES_FROZEN.receipt.json')
   for p,h in receipt['files'].items():assert sha(p)==h,p
   r=read(dest/file)
   for n,a in zip(r['held_names'],r['choices']):choice[ci[n]]=ARMS.index(a)
   folds.setdefault(f,{})[kind]=r
  assert (choice>=0).all();pol[kind]=choice
 for a in ARMS:pol[a]=np.full(46,ARMS.index(a),int)
 oracle=np.argmax(rates,axis=1);family_choice=np.empty(46,int)
 for f in FAMILIES:family_choice[fam==f]=np.argmax(rates[fam==f].mean(0))
 pol['controller_oracle']=oracle;pol['family_oracle']=family_choice
 rng=np.random.default_rng(2026101708);groups={f:np.flatnonzero(fam==f) for f in FAMILIES};draws={k:[] for k in pol};family_draws={f:{k:[] for k in pol} for f in FAMILIES};diff={k:[] for k in ['rule','critic']};diff_bounds={k:[[],[]] for k in ['rule','critic']};diff_family_equal={k:[] for k in ['rule','critic']}
 for b in range(5000):
  ids=np.concatenate([rng.choice(groups[f],len(groups[f]),replace=True) for f in FAMILIES]);ss=rng.integers(0,32,(46,32));sampled=lo.transpose(0,2,1)[ids[:,None],ss].mean(1);supper=hi.transpose(0,2,1)[ids[:,None],ss].mean(1);fixed=sampled.mean(0);fixedupper=supper.mean(0);offset=0;family_fixed=[]
  for f in FAMILIES:
   ix=slice(offset,offset+len(groups[f]));family_fixed.append(sampled[ix].mean(0));offset+=len(groups[f])
  for k,choice in pol.items():
   values=sampled[np.arange(46),choice[ids]];uvalues=supper[np.arange(46),choice[ids]];draws[k].append(values.mean());offset=0;fvals=[]
   for f in FAMILIES:
    ix=slice(offset,offset+len(groups[f]));v=values[ix].mean();family_draws[f][k].append(v);fvals.append(v);offset+=len(groups[f])
   if k in diff:
    diff[k].append(values.mean()-fixed.max());diff_bounds[k][0].append(values.mean()-fixedupper.max());diff_bounds[k][1].append(uvalues.mean()-fixed.max());diff_family_equal[k].append(np.mean(fvals)-np.mean(family_fixed,axis=0).max())
 summary={}
 for k,ch in pol.items():
  v=rates[np.arange(46),ch];u=upper[np.arange(46),ch];summary[k]=dict(success_lower=float(v.mean()),success_upper=float(u.mean()),ci=interval(draws[k]),family_equal_success=float(np.mean([v[fam==f].mean() for f in FAMILIES])),controller_oracle_regret_lower=float(np.mean(rates.max(1)-v)),controller_oracle_regret_ci=interval(np.asarray(draws['controller_oracle'])-np.asarray(draws[k])),controller_oracle_regret_bounds=[float(np.mean(np.maximum(rates.max(1)-u,0))),float(np.mean(upper.max(1)-v))],families={})
  for f in FAMILIES:
   ix=groups[f];bestlower=rates[ix].mean(0).max();bestupper=upper[ix].mean(0).max();bestindex=int(np.argmax(rates[ix].mean(0)));tset=np.unique(TYPE[np.isclose(rates[ix].mean(0),bestlower,atol=1e-12,rtol=0)]);summary[k]['families'][f]=dict(success_lower=float(v[ix].mean()),success_upper=float(u[ix].mean()),ci=interval(family_draws[f][k]),oracle_regret=float(np.mean(rates[ix].max(1)-v[ix])),oracle_regret_ci=interval(np.asarray(family_draws[f]['controller_oracle'])-np.asarray(family_draws[f][k])),catastrophe_count=int(((bestlower-v[ix])>.10+1e-12).sum()),catastrophe_count_bounds=[int(((bestlower-u[ix])>.10+1e-12).sum()),int(((bestupper-v[ix])>.10+1e-12).sum())],type_accuracy=float(np.mean(TYPE[ch[ix]]==TYPE[bestindex])),type_accuracy_allow_tied_best=float(np.mean(np.isin(TYPE[ch[ix]],tset))),best_fixed_arm=ARMS[bestindex],best_fixed_type=TYPE_FULL[TYPE[bestindex]],tied_best_types=[TYPE_FULL[i] for i in tset],choices={names[i]:ARMS[ch[i]] for i in ix})
 endpoints={k:dict(difference=float(summary[k]['success_lower']-rates.mean(0).max()),ci=interval(diff[k]),unknown_adversarial_ci=[interval(diff_bounds[k][0])[0],interval(diff_bounds[k][1])[1]],status='PASS' if interval(diff[k])[0]>0 else 'NOT_PASS',unknown_robust_status='PASS' if interval(diff_bounds[k][0])[0]>0 else 'NOT_PASS',family_equal_ci=interval(diff_family_equal[k]),best_pooled_fixed_arm=ARMS[int(np.argmax(rates.mean(0)))]) for k in diff}
 old=read(HERE/'FROZEN_OLD_CRITIC_REUSE.json');oldrows=[]
 for r in old:
  i=ci[r['name']];a=ARMS.index(r['intersection_choice']);row=dict(name=r['name'],original_choice=r['original_choice'],intersection_choice=r['intersection_choice'],intersection_success_lower=float(rates[i,a]),intersection_success_upper=float(upper[i,a]));
  if r['original_choice'] in ARMS:row.update(original_success_lower=float(rates[i,ARMS.index(r['original_choice'])]),original_success_upper=float(upper[i,ARMS.index(r['original_choice'])]))
  else:
   history=read(D/'teacher_student_line_closeout_v1/MANIFEST.json') if r['name'].startswith('ST') else read(D/'decentralized_blackbox_planners_v1/ANALYSIS_MANIFEST.json');index=int(r['name'][2:]) if r['name'].startswith('ST') else None;rr=[q for q in history if q['arm']==r['original_choice'] and ((q.get('group')=='V0' and q.get('controller_index')==index) if index is not None else q.get('name')==r['name'])];row.update(original_success_lower=float(sum(q['success'] and not q['unknown'] for q in rr)/64),original_success_upper=float(sum(q['success'] or q['unknown'] for q in rr)/64),available_records=len(rr))
  oldrows.append(row)
 features=read(HERE/'DIAGNOSTIC_FEATURES.json');fd={}
 for f in FAMILIES:
  rr=[r for r in features if r['family']==f];fd[f]={key:dict(mean=float(np.mean([r[key] for r in rr])),min=float(np.min([r[key] for r in rr])),max=float(np.max([r[key] for r in rr]))) for key in ['freeze_proportion','moving_no_progress_robot_seconds','progress300','intervention','mixed_proportion']}
 # Prespecified single features: ranks, without fitted classification thresholds.
 def auc(a,b):return float(np.mean((a[:,None]>b[None,:])+.5*(a[:,None]==b[None,:])))
 pairwise={}
 X=np.load(HERE/'DIAGNOSTIC_FEATURES.npz')['features']
 for f in ['ST','TD','G','AS']:
  pairwise[f]=dict(freeze_DM_greater_AUC=auc(X[fam=='DM',0],X[fam==f,0]),moving_other_greater_AUC=auc(X[fam==f,1],X[fam=='DM',1]))
 results=dict(evidence_level='DEVELOPMENT',utc=utc(),controllers=46,records=int(labels.size),unknown=int((labels<0).sum()),arms=ARMS,arm_full_names=ARM_FULL,family_full_names=FULL,bootstrap=dict(replicates=5000,seed=2026101708,controller_stratified_by_family=True,states_clustered=True,seeds_clustered=True,paired=True,reselect_best_fixed_each_bootstrap=True),endpoints=endpoints,policies=summary,folds=folds,diagnostic_distributions=fd,single_feature_descriptive_auc=pairwise,old_frozen_critic=dict(available_controllers=len(oldrows),missing_family='TD',rows=oldrows),limitations=['All DEVELOPMENT; prior labels opened.','Right-hand menu was tuned using family-labelled historical outcomes.','No frozen universal critic/R5 with exact distributed-teacher probe protocol; missing not filled.','Uncertainty conditional on frozen LOFO fits; bootstrap does not refit selectors.','Timeouts are not identified as deadlocks.'],guard=guard());write(HERE/'results.json',results);np.savez_compressed(HERE/'BOOTSTRAP_DRAWS.npz',**{k:np.asarray(v) for k,v in draws.items()},rule_difference=diff['rule'],critic_difference=diff['critic']);source_inventory('imports_analysis.json');seal('ANALYSIS.receipt.json',[HERE/'results.json',HERE/'BOOTSTRAP_DRAWS.npz'],held_labels_evaluated_after_all_fold_choices_frozen=True,guard=guard());print(endpoints,flush=True)
if __name__=='__main__':main()
