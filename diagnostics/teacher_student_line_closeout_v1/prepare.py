"""Resolve saved evidence and provenance; never invokes an experiment runner."""
from core import *
import collections,csv
precision()
def main():
 prereg();start=read(HERE/'INTEGRITY_START.json');roots=[pathlib.Path(p) for p in start['roots']];ts=read(D/'teacher_student_macflow_v1f/TEACHERS.json');names=[t['name'] for t in ts];states=read(D/'teacher_student_macflow_v1f/EVAL_STATES.json');index=collections.defaultdict(list)
 for path in start['files']:
  p=pathlib.Path(path)
  if p.suffix=='.json' and p.parent.name=='raw':index[p.stem].append(p)
 rows=[];gaps=[];sources=set();checks=[]
 def add(group,c,origin):
  rid=c['rollout_uid'];candidates=index.get(rid,[]);preferred=[p for p in candidates if str(p).startswith(str(origin)+'/')];candidates=preferred+[p for p in candidates if p not in preferred]
  if not candidates:gaps.append(dict(group=group,rollout_uid=rid,reason='No immutable raw JSON found in audited histories'));return
  chosen=None
  for p in candidates:
   r=read(p);trace=pathlib.Path(r.get('trace_path',str(p.with_suffix('.npz'))))
   if trace.exists() and str(trace) in start['files']:chosen=(p,r,trace);break
  if not chosen:gaps.append(dict(group=group,rollout_uid=rid,reason='No audited trace available',candidate_paths=[str(p) for p in candidates]));return
  p,r,trace=chosen;expected=r.get('trace_sha256');actual=start['files'][str(trace)];assert expected is None or expected==actual,(trace,expected,actual)
  if group=='expert':i=0
  elif group=='Z':i=0
  else:i=c.get('student_index') if c.get('student_index') is not None else names.index(c.get('teacher_name',c['base_controller_name']))
  row=dict(group=group,group_full=FULL[group],controller_index=i,teacher_name=names[i] if group not in ['expert','Z'] else None,state_index=c['state_index'],seed=c['seed'],arm='expert' if group=='expert' else c['arm'],rollout_uid=rid,raw_path=str(p),raw_sha256=start['files'][str(p)],trace_path=str(trace),trace_sha256=actual,round=origin.name,evidence_level='DEVELOPMENT',success=int(r.get('success',0)),collision=int(r.get('collision',0)),timeout=int(r.get('timeout',0)),unknown=int(bool(r.get('numerical_failure',0) or r.get('error'))),outcome=r.get('outcome'),episode_length=r.get('physical_steps',r.get('episode_length')),case=c)
  rows.append(row);sources.add(str(p));sources.add(str(trace))
 croot=D/'teacher_student_macflow_v1c'
 for c in read(croot/'main_CASES.json'):
  if c['arm'] in ARMS:add('V0',c,croot)
 for c in read(D/'teacher_student_macflow_v1d/fair_CASES.json'):
  if c.get('kind')=='student':add('V0',c,D/'teacher_student_macflow_v1d')
 for c in read(D/'teacher_student_macflow_v1f/main_CASES.json'):add(c['variant'],c,D/'teacher_student_macflow_v1f')
 for c in read(D/'teacher_student_macflow_v1/teacher_CASES.json'):
  add('TD',c,D/'teacher_student_macflow_v1')
  if c['arm']=='nominal only':add('TC',c,D/'teacher_student_macflow_v1')
 for c in read(D/'decentralized_pipeline_v1/main_CASES.json'):
  if c['pipeline'] in ['central','local'] and c['arm'] in ['F10','F13','S300','S500']:add('TC' if c['pipeline']=='central' else 'TD',c,D/'decentralized_pipeline_v1')
 for c in read(D/'teacher_student_macflow_v1d/A_CASES.json'):
  if c['kind']=='Z' and c['pipeline']=='local':add('Z',c,D/'teacher_student_macflow_v1d')
 for c in read(D/'decentralized_pipeline_v1/expert_protocol.json')['cases']:add('expert',c,D/'decentralized_pipeline_v1')
 keys=[(r['group'],r['controller_index'],r['state_index'],r['seed'],r['arm']) for r in rows];assert len(keys)==len(set(keys))
 # Resolve pairing independently of legacy state_uid normalization.
 for path in [D/'teacher_student_macflow_v1/teacher_protocol.json',croot/'main_protocol.json',D/'decentralized_pipeline_v1/main_protocol.json',D/'teacher_student_macflow_v1d/A_protocol.json']:
  p=read(path)
  for a,b in zip(states,p['states']):
   assert a['uid']==b['uid']
   for k in ['positions','velocities','goals']:assert np.asarray(a['physical'][k],np.float64).tobytes()==np.asarray(b['physical'][k],np.float64).tobytes()
  checks.append(dict(path=str(path),sha256=sha(path),physical_bytes_and_rng_alias_match=True))
 choices={}
 for folder in ['teacher_student_macflow_v1','teacher_student_macflow_v1b','teacher_student_macflow_v1c']:
  p=D/folder/'FROZEN_CHOICES.json';choices.update(read(p)['controllers']);sources.add(str(p))
 frozen={'V0':{str(i):choices[n]['A3_full'] for i,n in enumerate(names)}};new=read(D/'teacher_student_macflow_v1f/FROZEN_CHOICES.json')['controllers'];frozen.update({v:{str(i):new[f'{v}__{i}']['A3_full'] for i in range(14)} for v in ['V1','V3']})
 write(HERE/'FROZEN_CRITIC_REUSE.json',frozen);write(HERE/'TEACHER_NAMES.json',names);write(HERE/'EVAL_STATES_REFERENCE.json',dict(path=str(D/'teacher_student_macflow_v1f/EVAL_STATES.json'),sha256=sha(D/'teacher_student_macflow_v1f/EVAL_STATES.json')));write(HERE/'MANIFEST.json',rows);write(HERE/'HISTORY_AUDIT.json',dict(utc=utc(),sources={p:start['files'][p] for p in sorted(sources)},source_roots=start['roots'],pairing=checks,records_by_group=dict(collections.Counter(r['group'] for r in rows)),unknown_by_group=dict(collections.Counter(r['group'] for r in rows if r['unknown'])),gaps=gaps,compatibility='Exact frozen checkpoint generation and execution arms from saved manifests; no invalid round30 global students; state physical bytes, RNG alias, seed paired',evidence_gap='Directional, collision geometry and timing metrics absent from prior reports; recompute from saved trajectories only. No new model inference needed.',DB_opened=False,rollouts_generated=0,training_steps=0))
 for group in ['TC','TD','Z']:
  for arm in ARMS:
   if not any(r['group']==group and r['arm']==arm for r in rows):gaps.append(dict(group=group,arm=arm,reason='No exact compatible historical arm; report missing without acquiring labels'))
 write(HERE/'GAPS.json',gaps);seal('EVIDENCE_RESOLVED.receipt.json',[HERE/'MANIFEST.json',HERE/'HISTORY_AUDIT.json',HERE/'FROZEN_CRITIC_REUSE.json'],before_metrics=True,precision=precision());print('resolved',len(rows),dict(collections.Counter(r['group'] for r in rows)),'gaps',len(gaps),flush=True)
if __name__=='__main__':main()
