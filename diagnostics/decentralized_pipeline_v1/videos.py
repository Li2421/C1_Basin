from core import *
import subprocess,csv
from new_benchmark_common import batch_videos as official

def main():
 prereg();rows=[read(f) for f in (HERE/'main/raw').glob('*.json')];rows.sort(key=lambda r:(r['case']['base_controller_name'],r['case']['state_uid'],r['case']['seed'],r['case']['arm']))
 metrics={r['rollout_uid']:r for r in csv.DictReader((HERE/'rollout_metrics.csv').open())};index={(r['case']['base_controller_name'],r['case']['state_uid'],r['case']['seed'],r['case']['pipeline'],r['case']['arm']):r for r in rows};chosen=[];used=set()
 for r in rows:
  c=r['case'];s=c['state_uid']
  if s in used:continue
  found=None
  for mode,arm in [('rhr','B-RHR'),('rhr_f10','B-RHR+F10'),('shared','F10'),('shared','F13')]:
   base=index[(c['base_controller_name'],s,c['seed'],mode,arm)]
   if base['success'] or base['collision'] or base['numerical_failure'] or float(metrics[base['case']['rollout_uid']]['stalled_robots'])<1:continue
   for schedule in ['S500','S300']:
    success=index[(c['base_controller_name'],s,c['seed'],'shared',schedule)]
    if success['success'] and success['minimum_swept_clearance']>=0:found=(success,base);break
   if found:break
  if found:chosen.append(found);used.add(s)
  if len(chosen)==2:break
 safety=next((r for r in rows if r['case']['pipeline']=='shared' and r['success'] and r['case']['state_uid'] not in used and r['minimum_swept_clearance']>=0),None)
 entries=[]
 if safety:entries.append(dict(role='safety_success',trace=str(Path(safety['trace_path']).relative_to(HERE))))
 for i,(r,b) in enumerate(chosen):entries.append(dict(role=['deadlock_show','deadlock_resolution'][i],trace=str(Path(r['trace_path']).relative_to(HERE)),baseline=str(Path(b['trace_path']).relative_to(HERE))))
 plan=dict(batch_id='exp_decentralized_pipeline_v1',scenarios={'ring_exchange':entries});write(HERE/'video_plan.json',plan)
 cmd=[sys.executable,'-B','-m','new_benchmark_common.batch_videos','--plan',str(HERE/'video_plan.json'),'--output',str(HERE/'official_videos')];attempt=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True);(HERE/'official_video_entry.log').write_text(attempt.stdout+attempt.stderr)
 # Scope-only second check leaves all official role/deadlock checks intact; never patches historical code.
 scope_error=None;previous=official.SCENARIOS
 try:
  official.SCENARIOS=('ring_exchange',);official.validate_plan(plan,HERE)
 except Exception as e:scope_error=f'{type(e).__name__}: {e}'
 finally:official.SCENARIOS=previous
 out=HERE/'videos';out.mkdir(exist_ok=True);records=[];render_errors=[]
 todo=[('safety_success',safety,None)] if safety else []
 todo += [(f'stall_timeout_comparison_{i+1}',r,b) for i,(r,b) in enumerate(chosen)]
 for role,r,b in todo:
  try:
   trace=Path(r['trace_path']);meta,data=official.load_trace(trace);baseline=None
   if b:
    bp=Path(b['trace_path']);bm,bd=official.load_trace(bp);assert bm['seed']==meta['seed'] and bm['config']==meta['config'] and bm['initial_state_sha256']==meta['initial_state_sha256'];assert np.array_equal(bd['positions'][0],data['positions'][0]);assert np.array_equal(bd['goals'],data['goals']);baseline=(bp,bm,bd)
   video=out/f'{role}.mp4';labels=['STALLED/TIMEOUT DIAGNOSTIC BASELINE','SCHEDULE SUCCESS'] if b else ['SAFETY SUCCESS'];frames=official.render_video(video,meta,data,baseline,panel_labels=labels)
   records.append(dict(role=role,scenario='ring_exchange',file=video.name,sha256=sha(video),frames=frames,dt=meta['config']['dt'],playback_speed=1,trace=str(trace),trace_sha256=sha(trace),provenance=meta,baseline=None if b is None else dict(trace=str(baseline[0]),sha256=sha(baseline[0]),provenance=baseline[1],stall_metrics=metrics[b['case']['rollout_uid']]),formal_deadlock_role=False))
  except Exception as e:render_errors.append(dict(role=role,error=f'{type(e).__name__}: {e}'))
 result=dict(batch_id=plan['batch_id'],utc=utc(),complete=False,requested_ring_illustrations_complete=len(records)==3,formal_role_gaps=['deadlock_show','deadlock_resolution'],reason='Ring has no formal deadlock detector; timeout/stall are diagnostics. Official four-scene gate also exceeds Ring-only task scope.',official_entry=dict(command=cmd,exit_code=attempt.returncode,log=str(HERE/'official_video_entry.log'),ring_scope_validation_error=scope_error),videos=records,render_errors=render_errors,selection='frozen lexicographic controller/state/seed; baseline RHR,RHR+F10,F10,F13 priority; schedule S500,S300 priority; independent safety state',code_sha256=sha(__file__))
 write(out/'manifest.json',result);write(HERE/'VIDEO_ACCEPTANCE.json',dict(utc=utc(),complete=False,exit_code=2,rendered=len(records),formal_role_gaps=result['formal_role_gaps']));print('Video illustrations',len(records),'formal acceptance remains incomplete',flush=True);return 2
if __name__=='__main__':sys.exit(main())
