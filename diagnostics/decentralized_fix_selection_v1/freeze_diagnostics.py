from support import *
from run import FEATURES
import csv
G=guard()
def main():
 prereg();p=read(HERE/'CONTROLLERS.json');rows=[];X=[];index=[];gaps=[]
 for name in p['controller_order']:
  rr=[read(HERE/'diagnostic/raw'/f'diagnostic_{name}_{i:02}.json') for i in range(16)]
  for r in rr:assert sha(r['trace_path'])==r['trace_sha256'] and not r['terminal_labels_read'] and not r['task_journal_written']
  valid=all(r['complete'] for r in rr)
  if not valid:gaps.append(dict(name=name,errors=[r['error'] for r in rr if r['error']]))
  xx=np.array([[r.get('features',{}).get(f,np.nan) for f in FEATURES] for r in rr]);X.append(xx.mean(0));index.extend(dict(path=str(HERE/'diagnostic/raw'/f'diagnostic_{name}_{i:02}.json'),sha256=sha(HERE/'diagnostic/raw'/f'diagnostic_{name}_{i:02}.json'),trace_path=r['trace_path'],trace_sha256=r['trace_sha256']) for i,r in enumerate(rr));rows.append(dict(controller=name,family=next(f for f in FAMILIES if name.startswith(f)),family_full=FULL[next(f for f in FAMILIES if name.startswith(f))],complete=valid,**dict(zip(FEATURES,xx.mean(0)))))
 write(HERE/'DIAGNOSTIC_FEATURES.json',rows);write(HERE/'DIAGNOSTIC_RAW_INDEX.json',index);write(HERE/'DIAGNOSTIC_GAPS.json',gaps);np.savez_compressed(HERE/'DIAGNOSTIC_FEATURES.npz',features=np.asarray(X),controller_names=p['controller_order'],feature_names=FEATURES)
 with (HERE/'DIAGNOSTIC_FEATURES.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(clean(rows))
 teacher_names=[r['name'] for r in read(HERE/'TEACHERS.json')];r5=[];pred=[];mapping=[]
 for name in p['controller_order']:
  if name.startswith('TD'):mapping.append(dict(name=name,available=False,reason='No R5 measured under this distributed teacher diagnostic execution; original DEVELOPMENT_TEACHER_CHOICES is not an exact protocol match. No replacement generated.'));continue
  if name.startswith('ST'):
   i=int(name[2:]);n=teacher_names[i];directory=D/('teacher_student_macflow_v1' if i in [8,10,12,13] else 'teacher_student_macflow_v1b' if i in [1,7] else 'teacher_student_macflow_v1c')
  else:n=name;directory=D/'decentralized_blackbox_planners_v1'
  z=np.load(directory/'R5_FEATURES.npz');j=list(z['controller_names']).index(n);row=z['features'][j];r5.append(row);mapping.append(dict(name=name,available=True,feature_row=len(r5)-1,source=str(directory/'R5_FEATURES.npz'),source_row=j,sha256=sha(directory/'R5_FEATURES.npz')))
  q=np.load(directory/'BLIND_PREDICTIONS.npz');k=list(q['controller_names']).index(n);prob=q['A3'][:,k].mean(axis=(0,1));arms=list(q['arms']);choice=arms[int(np.argmax(prob))];keep=[arms.index(a) for a in ['F13','S500']];inter=['F13','S500'][int(np.argmax(prob[keep]))];pred.append(dict(name=name,source=str(directory/'BLIND_PREDICTIONS.npz'),sha256=sha(directory/'BLIND_PREDICTIONS.npz'),original_arms=arms,probability=prob,original_choice=choice,intersection_choice=inter))
 np.savez_compressed(HERE/'RETAINED_R5.npz',features=np.array(r5));write(HERE/'RETAINED_R5_MAP.json',mapping);write(HERE/'FROZEN_OLD_CRITIC_REUSE.json',pred);source_inventory('imports_feature_freeze.json');seal('DIAGNOSTICS_FROZEN.receipt.json',[HERE/n for n in ['DIAGNOSTIC_FEATURES.json','DIAGNOSTIC_FEATURES.csv','DIAGNOSTIC_FEATURES.npz','DIAGNOSTIC_RAW_INDEX.json','DIAGNOSTIC_GAPS.json','RETAINED_R5.npz','RETAINED_R5_MAP.json','FROZEN_OLD_CRITIC_REUSE.json']],before_any_selector=True,complete_controllers=len(rows)-len(gaps),total_controllers=len(rows),task_labels_read=False,guard=guard());print('FROZEN',len(rows),len(gaps),flush=True)
if __name__=='__main__':main()
