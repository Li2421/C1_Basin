from support import *
import csv,subprocess
G=guard()
def main():
 prereg();p=read(HERE/'CONTROLLERS.json');names=p['controller_order'];rows=[];X=[];rawindex=[];gaps=[];R5=[];r5index=[];sourcehash={};td=np.load(D/'ring_independent_flow_confirmation_v2/R5_FINAL_FEATURES.npz');st=np.load(D/'teacher_student_macflow_v1f/R5_FEATURES.npz')
 for n in names:
  rr=[read(HERE/'diagnostic/raw'/f'diagnostic_{n}_{i:02}.json') for i in range(16)];good=all(r['complete'] for r in rr)
  if not good:gaps.append(dict(controller=n,reason='Incomplete safety-only diagnostic',errors=[r['error'] for r in rr if r['error']]))
  for r in rr:
   assert not r['terminal_labels_read'] and not r['task_journal_written'];assert sha(r['trace_path'])==r['trace_sha256'];rawindex.append(dict(case=r['case'],trace_path=r['trace_path'],trace_sha256=r['trace_sha256']))
  x=np.array([[r.get('features',{}).get(k,np.nan) for k in FEATURES] for r in rr]).mean(0);X.append(x);f=next(f for f in FAMILIES if n.startswith(f));rows.append(dict(controller=n,family=f,family_full=FULL[f],complete=good,**dict(zip(FEATURES,x))))
  v=p['profiles'][n]['payload']
  if n.startswith('TD'):
   name=v['source_name'];source=D/'ring_independent_flow_confirmation_v2/R5_FINAL_FEATURES.npz';j=list(td['controller_names']).index(name);r5=td['features'][j];r5index.append(dict(controller=n,source=str(source),source_row=j,protocol='Historical native joint-flow R5, not new D-local probes; original frozen-critic descriptor protocol retained.'))
  elif n.startswith('ST'):
   name=f"{v['variant']}__{v['source_index']}";source=D/'teacher_student_macflow_v1f/R5_FEATURES.npz';j=list(st['controller_names']).index(name);r5=st['features'][j];r5index.append(dict(controller=n,source=str(source),source_row=j,protocol='Existing exact final-checkpoint distributed R5; reused without new probes.'))
  else:
   ff=np.full((16,2,5),np.nan,np.float32)
   for i in range(16):
    for a,j in enumerate([10,13]):
     path=HERE/'probe/raw'/f'R5_{n}_{i:02}_{j}.json';r=read(path);assert not r['terminal_labels_read'] and not r['task_journal_written'];assert sha(r['trace_path'])==r['trace_sha256'];r5index.append(dict(controller=n,case=r['case'],raw_path=str(path),raw_sha256=sha(path),trace_sha256=r['trace_sha256']))
     if r['complete']:
      z=np.load(r['trace_path']);pos=z['positions'];acts=z['actions'];goals=z['goals'];initial=np.linalg.norm(goals-pos[0],axis=-1).sum();final=np.linalg.norm(goals-pos[-1],axis=-1).sum();ii,jj=np.triu_indices(4,1);dd=np.linalg.norm(pos[:,:,None]-pos[:,None,:],axis=-1)[:,ii,jj];speed=np.linalg.norm(acts,axis=-1);ff[i,a]=[(initial-final)/initial,z['projection_active'].mean(),dd.min(),speed.mean(),(speed.mean(-1)<.01).sum()]
   r5=ff.reshape(160);source=None
  if source is not None:sourcehash[str(source)]=sha(source)
  R5.append(r5)
 write(HERE/'DIAGNOSTIC_FEATURES.json',rows);write(HERE/'DIAGNOSTIC_RAW_INDEX.json',rawindex);write(HERE/'DIAGNOSTIC_GAPS.json',gaps);np.savez_compressed(HERE/'DIAGNOSTIC_FEATURES.npz',features=X,controller_names=names,feature_names=FEATURES);np.savez_compressed(HERE/'R5_FEATURES.npz',features=R5,controller_names=names);write(HERE/'R5_RAW_INDEX.json',dict(rows=r5index,reused_source_hashes=sourcehash,new_probes=512,reused_controllers=42))
 with (HERE/'DIAGNOSTIC_FEATURES.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(clean(rows))
 seal('DIAGNOSTICS_FROZEN.receipt.json',[HERE/n for n in ['DIAGNOSTIC_FEATURES.json','DIAGNOSTIC_FEATURES.npz','DIAGNOSTIC_FEATURES.csv','DIAGNOSTIC_RAW_INDEX.json','DIAGNOSTIC_GAPS.json','R5_FEATURES.npz','R5_RAW_INDEX.json']],before_any_new_instance_choices=True,before_new_task_labels=True,main_diagnostics=928,additional_R5_probes=512,guard=guard());seal('R5_FROZEN.receipt.json',[HERE/'R5_FEATURES.npz',HERE/'R5_RAW_INDEX.json'],before_new_task_labels=True)
 # Frozen universal model inference only, with unchanged source transform/weights.
 assert (HERE/'NORMALIZER_STAGE_BEGIN.json').exists();r=subprocess.run([sys.executable,'-B',str(HERE/'freeze_choices.py')],capture_output=True,text=True);print(r.stdout+r.stderr,flush=True);assert r.returncode==0
 rules=read(HERE/'FROZEN_RULES.json');X=np.asarray(X);choices={};valid=np.isfinite(X).all(1);primary=np.where(valid,select(X,rules['primary']),2);secondary=np.where(valid,select(X,rules['secondary'],True),2);old=read(HERE/'FROZEN_CHOICES.json')['controllers']
 for i,n in enumerate(names):choices[n]=dict(primary=ARMS[primary[i]] if valid[i] else None,secondary=ARMS[secondary[i]] if valid[i] else None,old_critic=old[n]['A3_full'],feature_complete=bool(valid[i]))
 write(HERE/'NEW_INSTANCE_CHOICES.json',dict(utc=utc(),controllers=choices,rules=rules,evaluation_labels_read=False,diagnostic_gap_fallback=None));seal('ALL_CHOICES_FROZEN.receipt.json',[HERE/n for n in ['NEW_INSTANCE_CHOICES.json','FROZEN_RULES.json','DIAGNOSTICS_FROZEN.receipt.json','CHOICES_FROZEN.receipt.json','FROZEN_CHOICES.json']],before_any_new_evaluation=True,guard=guard());source_inventory('imports_features_choices.json');print('ALL CHOICES FROZEN',len(names),'diagnostic gaps',len(gaps),flush=True)
if __name__=='__main__':main()
