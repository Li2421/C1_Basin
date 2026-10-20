from support import *
import csv
G=guard()
def main():
 prereg();assert (HERE/'DIAGNOSTICS_FROZEN.receipt.json').exists();p=read(HERE/'CONTROLLERS.json');names=p['controller_order'];ci={n:i for i,n in enumerate(names)};labels=np.full((46,6,32,2),-9,np.int8);manifest=[]
 for r in read(HERE/'HISTORY_REUSE.json'):
  i=ci[r['name']];a=ARMS.index(r['arm']);s=r['state_index'];k=r['seed'];assert labels[i,a,s,k]==-9;labels[i,a,s,k]=-1 if r['unknown'] else r['success'];manifest.append(r)
 for c in read(HERE/'supplement_CASES.json'):
  path=HERE/'supplement/raw'/f"{c['rollout_uid']}.json";r=read(path);assert sha(r['trace_path'])==r['trace_sha256'];i=ci[c['base_controller_name']];a=ARMS.index(c['arm']);s=c['state_index'];k=c['seed'];assert labels[i,a,s,k]==-9;unknown=bool(r['numerical_failure']);labels[i,a,s,k]=-1 if unknown else r['success'];manifest.append(dict(name=c['base_controller_name'],arm=c['arm'],state_index=s,seed=k,rollout_uid=c['rollout_uid'],raw_path=str(path),raw_sha256=sha(path),trace_path=r['trace_path'],trace_sha256=r['trace_sha256'],source_round='decentralized_fix_selection_v1',unknown=unknown,success=r['success'],outcome=r['outcome']))
 assert not (labels==-9).any();z=np.load(HERE/'DIAGNOSTIC_FEATURES.npz');assert list(z['controller_names'])==names;X=z['features'];families=np.array([next(f for f in FAMILIES if n.startswith(f)) for n in names]);write(HERE/'ANALYSIS_MANIFEST.json',manifest);np.savez_compressed(HERE/'LABELS.npz',labels=labels,controller_names=names,arms=ARMS,families=families)
 table=[]
 for i,n in enumerate(names):
  for a,arm in enumerate(ARMS):
   r=[r for r in manifest if r['name']==n and r['arm']==arm];row=dict(controller=n,family=FULL[families[i]],arm=arm,success_lower=float((labels[i,a]==1).mean()),success_upper=float((labels[i,a]!=0).mean()),unknown=int((labels[i,a]==-1).sum()),source_rounds=','.join(sorted(set(v['source_round'] for v in r))));table.append(row)
 legacy=[]
 for q in read(D/'teacher_student_line_closeout_v1/MANIFEST.json'):
  if q['group'] in ['V0','TD'] and q['arm'] in ['F10','S300']:
   legacy.append(dict(q,name=('ST' if q['group']=='V0' else 'TD')+f"{q['controller_index']:02}",family='ST' if q['group']=='V0' else 'TD'))
 for q in read(D/'decentralized_blackbox_planners_v1/ANALYSIS_MANIFEST.json'):
  if q['group'] in ['G','DM','AS'] and q['arm'] in ['F10','S300']:legacy.append(dict(q,family=q['group']))
 lt=[]
 for name in names:
  for arm in ['F10','S300']:
   rr=[q for q in legacy if q['name']==name and q['arm']==arm];lt.append(dict(controller=name,arm=arm,records=len(rr),success_lower=sum(q['success'] and not q['unknown'] for q in rr)/64 if rr else None,success_upper=sum(q['success'] or q['unknown'] for q in rr)/64 if rr else None,unknown=sum(q['unknown'] for q in rr),not_in_selection_menu=True))
 write(HERE/'DESCRIPTIVE_LEGACY_ARMS.json',lt)
 with (HERE/'DESCRIPTIVE_LEGACY_ARMS.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(lt[0]));w.writeheader();w.writerows(lt)
 write(HERE/'CONTROLLER_SIX_ARM_TABLE.json',table)
 with (HERE/'CONTROLLER_SIX_ARM_TABLE.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
 for family in FAMILIES:
  ix=np.flatnonzero(families!=family);hx=np.flatnonzero(families==family);dest=HERE/'folds'/family;dest.mkdir(parents=True,exist_ok=True)
  # This is the only task-label file a selector process can load. Held labels excluded.
  np.savez_compressed(dest/'SOURCE_ONLY.npz',features=X[ix],labels=labels[ix],families=families[ix],names=np.array(names)[ix],held_features=X[hx],held_names=np.array(names)[hx]);seal(f'folds/{family}/SOURCE_ONLY.receipt.json',[dest/'SOURCE_ONLY.npz',HERE/'DIAGNOSTICS_FROZEN.receipt.json'],excluded_family=family,source_controllers=list(np.array(names)[ix]),held_labels_present=False)
 seal('LABELS_FROZEN.receipt.json',[HERE/'ANALYSIS_MANIFEST.json',HERE/'LABELS.npz',HERE/'CONTROLLER_SIX_ARM_TABLE.csv'],records=int(labels.size),unknown=int((labels==-1).sum()),guard=guard());source_inventory('imports_pack.json');print('labels',labels.size,'unknown',int((labels==-1).sum()),flush=True)
if __name__=='__main__':main()
