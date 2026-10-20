from support import *
import csv
G=guard()
ARM_FULL={'safety baseline':'只加安全层，不修正','F13':'全程η13','S500':'前500步η13、之后η10的时间表','RHR-A':'右手规则甲（1秒无进展、0.35m/s）','RHR-B':'右手规则乙（邻居0.8m内、0.10m/s）','RHR-C':'右手规则丙（2秒无进展、0.25m/s）','F10':'全程η10','S300':'前300步η13、之后η10的时间表'}
TYPE=np.array([0,1,1,2,2,2]);TYPE_FULL=['不修正','时间表类','右手规则类']
def CI(x):return np.quantile(x,[.025,.975]).tolist()
def main():
 prereg();receipt=read(HERE/'ALL_CHOICES_FROZEN.receipt.json')
 for path,h in receipt['files'].items():assert sha(path)==h,path
 p=read(HERE/'main_protocol.json');names=p['controller_order'];N=len(names);assert N==58;families=np.array([next(f for f in FAMILIES if n.startswith(f)) for n in names]);groups={f:np.flatnonzero(families==f) for f in FAMILIES};ci={n:i for i,n in enumerate(names)};labels=np.full((N,6,32,2),-9,np.int8);lookup={};manifest=[]
 for c in read(HERE/'main_CASES.json'):
  path=HERE/'main/raw'/f"{c['rollout_uid']}.json";r=read(path);assert sha(r['trace_path'])==r['trace_sha256'];n=c['base_controller_name'];a=c['arm'];i=ci[n];s=c['state_index'];k=c['seed'];label=-1 if r['numerical_failure'] else r['success'];lookup[n,a,s,k]=label
  if a in ARMS:assert labels[i,ARMS.index(a),s,k]==-9;labels[i,ARMS.index(a),s,k]=label
  manifest.append(dict(controller=n,family=families[i],arm=a,state_index=s,seed=k,rollout_uid=c['rollout_uid'],raw_path=str(path),raw_sha256=sha(path),trace_path=r['trace_path'],trace_sha256=r['trace_sha256'],label=label,outcome=r['outcome'],primary_six_arm=a in ARMS))
 assert not (labels==-9).any();np.savez_compressed(HERE/'LABELS.npz',labels=labels,controller_names=names,arms=ARMS,families=families);write(HERE/'ANALYSIS_MANIFEST.json',manifest)
 rates=(labels==1).mean((2,3));upper=(labels!=0).mean((2,3));fbest={f:int(np.argmax(rates[ix].mean(0))) for f,ix in groups.items()};oracle=np.argmax(rates,axis=1);choices=read(HERE/'NEW_INSTANCE_CHOICES.json')['controllers'];policy_labels={};policy_arms={}
 for method in ['primary','secondary','old_critic']:
  yy=np.full((N,32,2),-1,np.int8);aa=[]
  for i,n in enumerate(names):
   arm=choices[n][method];aa.append(arm)
   if arm is not None:
    for s in range(32):
     for k in [0,1]:yy[i,s,k]=lookup[n,arm,s,k]
  policy_labels[method]=yy;policy_arms[method]=aa
 for a,arm in enumerate(ARMS):policy_labels[arm]=labels[:,a];policy_arms[arm]=[arm]*N
 for method,inds in [('controller_oracle',oracle),('family_oracle',np.array([fbest[f] for f in families]))]:policy_labels[method]=labels[np.arange(N),inds];policy_arms[method]=[ARMS[j] for j in inds]
 keys=list(policy_labels);lo=np.stack([(policy_labels[k]==1).mean(-1) for k in keys],-1);hi=np.stack([(policy_labels[k]!=0).mean(-1) for k in keys],-1);idx={k:i for i,k in enumerate(keys)};draw={k:[] for k in keys};fdraw={f:{k:[] for k in keys} for f in FAMILIES};eqdraw={k:[] for k in keys};family_best_draws={f:[] for f in FAMILIES};diff={k:[] for k in ['confirmation1','confirmation2','secondary_minus_primary','secondary_minus_primary_DM']};bounds={k:[[],[]] for k in diff};eqdiff={k:[] for k in ['confirmation1','confirmation2','secondary_minus_primary']};rng=np.random.default_rng(2026101710)
 for b in range(5000):
  ids=np.concatenate([rng.choice(ix,len(ix),replace=True) for ix in groups.values()]);ss=rng.integers(0,32,(N,32));v=lo[ids[:,None],ss].mean(1);u=hi[ids[:,None],ss].mean(1);m=v.mean(0);mu=u.mean(0);fm={};fu={};offset=0
  for f,ix in groups.items():
   block=slice(offset,offset+len(ix));fm[f]=v[block].mean(0);fu[f]=u[block].mean(0);offset+=len(ix)
  em=np.mean(list(fm.values()),0);eu=np.mean(list(fu.values()),0)
  for k in keys:
   j=idx[k];draw[k].append(m[j]);eqdraw[k].append(em[j])
   for f in FAMILIES:fdraw[f][k].append(fm[f][j])
  fixed=[idx[k] for k in ARMS];primary=idx['primary'];secondary=idx['secondary'];schedule=idx['S500'];vals=dict(confirmation1=m[primary]-m[fixed].max(),confirmation2=m[primary]-m[schedule],secondary_minus_primary=m[secondary]-m[primary],secondary_minus_primary_DM=fm['DM'][secondary]-fm['DM'][primary]);lower=dict(confirmation1=m[primary]-mu[fixed].max(),confirmation2=m[primary]-mu[schedule],secondary_minus_primary=m[secondary]-mu[primary],secondary_minus_primary_DM=fm['DM'][secondary]-fu['DM'][primary]);upp=dict(confirmation1=mu[primary]-m[fixed].max(),confirmation2=mu[primary]-m[schedule],secondary_minus_primary=mu[secondary]-m[primary],secondary_minus_primary_DM=fu['DM'][secondary]-fm['DM'][primary]);ev=dict(confirmation1=em[primary]-em[fixed].max(),confirmation2=em[primary]-em[schedule],secondary_minus_primary=em[secondary]-em[primary])
  for f in FAMILIES:family_best_draws[f].append(fm[f][fixed].max())
  for k in diff:diff[k].append(vals[k]);bounds[k][0].append(lower[k]);bounds[k][1].append(upp[k])
  for k in ev:eqdiff[k].append(ev[k])
 policies={};point={k:float(lo[...,idx[k]].mean()) for k in keys};fpoint={f:{k:float(lo[groups[f],:,idx[k]].mean()) for k in keys} for f in FAMILIES}
 for k in keys:
  j=idx[k];v=lo[...,j].mean(1);u=hi[...,j].mean(1);r=dict(success_lower=point[k],success_upper=float(u.mean()),CI=CI(draw[k]),unknown=int((policy_labels[k]<0).sum()),family_equal_success=float(np.mean([fpoint[f][k] for f in FAMILIES])),family_equal_CI=CI(eqdraw[k]),oracle_regret=float(np.mean(rates.max(1)-v)),oracle_regret_CI=CI(np.asarray(draw['controller_oracle'])-np.asarray(draw[k])),families={})
  for f,ix in groups.items():
   best=rates[ix].mean(0).max();bestu=upper[ix].mean(0).max();ties=np.flatnonzero(np.isclose(rates[ix].mean(0),best,atol=1e-12,rtol=0));types=np.unique(TYPE[ties]);selected_type=np.array([0 if a=='safety baseline' else 1 if a in ['F10','F13','S300','S500'] else 2 if a in RULES else -1 for a in policy_arms[k]]);familydiff=np.asarray(fdraw[f][k])-np.asarray(family_best_draws[f]);r['families'][f]=dict(success_lower=float(v[ix].mean()),success_upper=float(u[ix].mean()),CI=CI(fdraw[f][k]),best_fixed_arm=ARMS[fbest[f]],best_fixed_tied_arms=[ARMS[a] for a in ties],best_fixed_success=float(best),difference_from_family_best=float(v[ix].mean()-best),difference_from_family_best_CI=CI(familydiff),type_accuracy=float(np.mean(selected_type[ix]==TYPE[fbest[f]])),type_accuracy_allow_tied_best=float(np.mean(np.isin(selected_type[ix],types))),catastrophe_count=int((best-v[ix]>.10+1e-12).sum()),catastrophe_count_bounds=[int((best-u[ix]>.10+1e-12).sum()),int((bestu-v[ix]>.10+1e-12).sum())],oracle_regret=float(np.mean(rates[ix].max(1)-v[ix])),choices={names[i]:policy_arms[k][i] for i in ix})
  policies[k]=r
 fixed_best=ARMS[int(np.argmax([point[a] for a in ARMS]))];points=dict(confirmation1=point['primary']-max(point[a] for a in ARMS),confirmation2=point['primary']-point['S500'],secondary_minus_primary=point['secondary']-point['primary'],secondary_minus_primary_DM=fpoint['DM']['secondary']-fpoint['DM']['primary']);endpoints={};eqpoint={k:float(np.mean([fpoint[f][k] for f in FAMILIES])) for k in keys};eqpointdiff=dict(confirmation1=eqpoint['primary']-max(eqpoint[a] for a in ARMS),confirmation2=eqpoint['primary']-eqpoint['S500'],secondary_minus_primary=eqpoint['secondary']-eqpoint['primary'])
 for k in diff:endpoints[k]=dict(difference=points[k],CI=CI(diff[k]),unknown_adversarial_CI=[CI(bounds[k][0])[0],CI(bounds[k][1])[1]],status='PASS' if CI(diff[k])[0]>0 else 'NOT_PASS',unknown_robust_status='PASS' if CI(bounds[k][0])[0]>0 else 'NOT_PASS',evidence_level='CONFIRMATORY' if k.startswith('confirmation') else 'EXPLORATORY',family_equal_CI=CI(eqdiff[k]) if k in eqdiff else None,family_equal_difference=eqpointdiff.get(k))
 table=[]
 for i,n in enumerate(names):
  for a,arm in enumerate(ARMS):table.append(dict(controller=n,family=FULL[families[i]],arm=ARM_FULL[arm],arm_code=arm,success_lower=float(rates[i,a]),success_upper=float(upper[i,a]),unknown=int((labels[i,a]<0).sum()),primary_choice=choices[n]['primary'],secondary_choice=choices[n]['secondary'],old_critic_choice=choices[n]['old_critic']))
 write(HERE/'CONTROLLER_SIX_ARM_TABLE.json',table)
 with (HERE/'CONTROLLER_SIX_ARM_TABLE.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
 X=np.load(HERE/'DIAGNOSTIC_FEATURES.npz')['features'];rule=read(HERE/'FROZEN_RULES.json');diagnostics={f:{FEATURES[j]:dict(mean=float(X[groups[f],j].mean()),min=float(X[groups[f],j].min()),max=float(X[groups[f],j].max())) for j in range(7)} for f in FAMILIES};diagnostic_branches={f:dict(right_hand=int((X[groups[f],0]>rule['primary']['A']).sum()),no_correction=int(((X[groups[f],0]<=rule['primary']['A'])&(X[groups[f],4]>rule['primary']['B'])).sum()),schedule=int(((X[groups[f],0]<=rule['primary']['A'])&(X[groups[f],4]<=rule['primary']['B'])).sum()),within_002_freeze_threshold=int((np.abs(X[groups[f],0]-rule['primary']['A'])<=.02).sum()),below_secondary_C=int((X[groups[f],0]<=rule['secondary']['C']).sum())) for f in FAMILIES}
 conclusion='BOTH_CONFIRMATIONS_PASS' if all(endpoints[k]['status']=='PASS' for k in ['confirmation1','confirmation2']) else 'ONLY_DEFAULT_BEATEN' if endpoints['confirmation2']['status']=='PASS' else 'PRIMARY_CONFIRMATION_NOT_SUPPORTED';results=dict(evidence_level='CONFIRMATORY',secondary_evidence_level='EXPLORATORY',utc=utc(),controllers=N,primary_six_arm_records=int(labels.size),total_task_records=len(manifest),unknown_six_arm=int((labels<0).sum()),unknown_all_tasks=sum(r['label']<0 for r in manifest),arms=ARMS,arm_full_names=ARM_FULL,family_full_names=FULL,source_rules=rule,best_pooled_fixed_arm=fixed_best,endpoints=endpoints,policies=policies,diagnostic_distributions=diagnostics,diagnostic_branch_counts=diagnostic_branches,bootstrap=dict(replicates=5000,seed=2026101710,controller_stratified_by_family=True,states_clustered=True,seeds_clustered=True,paired=True,reselect_best_single_fixed_arm=True,primary_weighting='each controller equally'),conclusion=conclusion,limitations=['Students were previously evaluated on old states; novelty is relative to round30h and new evaluation states.','Related teacher/student lineages; families not independent random samples.','Right-hand menu parameters carry historical family information.','Old critic reuses native-protocol teacher R5; domain mismatch explicit, descriptive only.','No-correction branch may remain unused; no thresholds changed.','Timeout is not deadlock.'],guard=guard());write(HERE/'results.json',results);np.savez_compressed(HERE/'BOOTSTRAP_DRAWS.npz',**{k:np.asarray(v) for k,v in draw.items()},**{k:np.asarray(v) for k,v in diff.items()});source_inventory('imports_analysis.json');seal('ANALYSIS.receipt.json',[HERE/'results.json',HERE/'LABELS.npz',HERE/'ANALYSIS_MANIFEST.json',HERE/'CONTROLLER_SIX_ARM_TABLE.csv',HERE/'BOOTSTRAP_DRAWS.npz'],task_labels_read_after_all_choices_frozen=True,guard=guard());print(endpoints,flush=True)
if __name__=='__main__':main()
