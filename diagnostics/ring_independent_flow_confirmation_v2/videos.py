"""Ring-scoped official video validation, honest missing-deadlock delivery."""
from core import *
from collections import Counter
import subprocess,traceback
sys.path.insert(0,str(R9/'vendor'))
from new_benchmark_common import batch_videos as official

def history_audit():
    prereg();files=sorted(set(ROOT.glob('diagnostics/*/videos*/manifest.json'))|set(ROOT.glob('diagnostics/*/*videos*/manifest.json')));rows=[];candidates=[]
    def visit(x,path):
        if isinstance(x,dict):
            if x.get('scenario')=='ring_exchange' or x.get('scene')=='ring_exchange':
                rows.append(dict(source=str(path),sha256=sha(path),outcome=x.get('outcome',x.get('termination')),deadlock_detected=x.get('deadlock_detected'),deadlock_criterion=x.get('deadlock_criterion'),trace=x.get('trace')))
                if x.get('deadlock_detected') is True and x.get('deadlock_criterion'):candidates.append(rows[-1])
            for value in x.values():visit(value,path)
        elif isinstance(x,list):
            for value in x:visit(value,path)
    for f in files:
        if f.is_relative_to(HERE):continue
        visit(read(f),f)
    write(HERE/'HISTORICAL_VIDEO_AUDIT.json',dict(utc=utc(),manifests_scanned=[dict(path=str(f),sha256=sha(f)) for f in files if not f.is_relative_to(HERE)],ring_rows=rows,declared_deadlock_candidates=candidates,original_runtime_has_deadlock_event=False,scope='Historical video manifests; no new-cohort label access. Declared candidates still require exact full-state/seed/config and real detector validation.'))
    return candidates

def main():
    prereg();gate();assert (HERE/'ENDPOINT_OPENING.receipt.json').exists();assert read(HERE/'main_POSTFLIGHT.json')['status']=='PASS';history=read(HERE/'HISTORICAL_VIDEO_AUDIT.json');records=[];lookup={}
    for c in read(HERE/'main_CASES.json'):
        r=read(HERE/'main/raw'/f"{c['rollout_uid']}.json");r={k:r[k] for k in ['case','success','numerical_failure','collision','minimum_swept_clearance','outcome','trace_path','trace_sha256','checkpoint_sha256']};records.append(r);lookup[c['base_controller_name'],c['arm'],c['state_index'],c['seed']]=r
    def valid_success(r):return bool(r['success'] and not r['numerical_failure'] and not r['collision'] and r['minimum_swept_clearance'] is not None and r['minimum_swept_clearance']>=0)
    candidates=[]
    for r in records:
        c=r['case']
        if c['arm']!='S500' or not valid_success(r):continue
        b=lookup[c['base_controller_name'],'F13',c['state_index'],c['seed']]
        if b['outcome']=='timeout' and not b['numerical_failure']:candidates.append((r,b))
    candidates.sort(key=lambda pair:pair[0]['case']['rollout_uid']);pairs=[];used=set()
    used_controllers=set()
    for r,b in candidates:
        name=r['case']['base_controller_name']
        if name in used_controllers:continue
        pairs.append((r,b));used.add(r['case']['rollout_uid']);used_controllers.add(name)
        if len(pairs)==2:break
    used_initial_seeds={(r['case']['state_uid'],r['case']['seed']) for r,b in pairs}
    safety=next((r for r in sorted(records,key=lambda r:r['case']['rollout_uid']) if r['case']['arm']=='S500' and valid_success(r) and r['case']['rollout_uid'] not in used and (r['case']['state_uid'],r['case']['seed']) not in used_initial_seeds),None)
    exported=[]
    def load(r):
        meta,data=official.load_trace(Path(r['trace_path']));assert meta['batch_id']==EXP and sha(r['trace_path'])==r['trace_sha256'];name=r['case']['base_controller_name']; original=read(HERE/'FINAL_BASE_PROFILES.json')['profiles'][name]['parent_best_name']; seed=next(x['new_seed'] for x in read(HERE/'TRAINING_PLAN.json') if x['run']==original); short=original.replace('base_u_','').replace('_macflow','').replace('plus_w_','w_').replace('train_targeted','targeted');meta['display_label']=f"Ring {short} {r['case']['arm']} train{seed} cp{r['checkpoint_sha256'][:8]}"
        initial_candidates=sorted((HERE/'actual_initial_states').glob(f"{r['case']['state_uid']}_{r['checkpoint_sha256'][:12]}_*.json"));assert initial_candidates
        initial_path=next((path for path in initial_candidates if hashlib.sha256(json.dumps(read(path),sort_keys=True,separators=(',',':')).encode()).hexdigest()==meta['initial_state_sha256']),None);assert initial_path is not None
        meta['augmented_initial_state_path']=str(initial_path);meta['augmented_initial_state_file_sha256']=sha(initial_path);meta['initial_state_provenance']='Actual simulator augmented initial state captured before any physical action; recorded hash verified'
        return meta,data
    if safety:
        meta,data=load(safety);path=HERE/'videos/ring_exchange/safety_success.mp4';frames=official.render_video(path,meta,data);exported.append(dict(role='safety_success',formal_role=True,file=str(path),sha256=sha(path),frames=frames,dt=meta['config']['dt'],trace=safety['trace_path'],trace_sha256=safety['trace_sha256'],provenance=meta))
    for i,(r,b) in enumerate(pairs,1):
        meta,data=load(r);bm,bd=load(b);assert bm['initial_state_sha256']==meta['initial_state_sha256'] and bm['seed']==meta['seed'] and bm['config']==meta['config'];path=HERE/f'videos/ring_exchange/timeout_to_success_{i}.mp4';frames=official.render_video(path,meta,data,(Path(b['trace_path']),bm,bd),panel_labels=('TIMEOUT BASELINE','SUCCESS'));exported.append(dict(role=f'timeout_to_success_{i}',formal_role=False,counts_as_deadlock_resolution=False,file=str(path),sha256=sha(path),frames=frames,dt=meta['config']['dt'],trace=r['trace_path'],trace_sha256=r['trace_sha256'],provenance=meta,baseline=dict(trace=b['trace_path'],trace_sha256=b['trace_sha256'],provenance=bm)))
    # User explicitly scopes this evaluation to Ring. No historical source edit.
    official.SCENARIOS=('ring_exchange',)
    plan=dict(batch_id=EXP,scenarios={'ring_exchange':[] if safety is None else [dict(role='safety_success',trace=safety['trace_path'])]});write(HERE/'video_plan.json',plan)
    try:
        official.main(['--plan',str(HERE/'video_plan.json'),'--output',str(HERE/'videos/strict_gate')]);gate_status=dict(exit_code=0,status='PASS')
    except Exception as exc:gate_status=dict(exit_code=2,status='FAIL',type=type(exc).__name__,message=str(exc))
    missing=['deadlock_show','deadlock_resolution']+([] if safety else ['safety_success']);manifest=dict(batch_id=EXP,complete=False,strict_gate=gate_status,required_scope=['ring_exchange'],required_roles=list(official.ROLES),missing_roles=missing,videos=exported,historical_audit_sha256=sha(HERE/'HISTORICAL_VIDEO_AUDIT.json'),reason='Frozen native Ring runtime has no deadlock criterion/event. No compatible real recorded deadlock pair found; timeout comparisons are explicitly diagnostic and do not fill deadlock roles.',numerical_results_unaffected=True)
    write(HERE/'videos/manifest.json',manifest);write(HERE/'VIDEO_GATE.json',dict(utc=utc(),**gate_status,missing_roles=missing,complete=False,unified_entry='new_benchmark_common.batch_videos.main',scope_binding='SCENARIOS=(ring_exchange,) in isolated wrapper; unchanged ROLES/load_trace/render/validator source',source_sha256=sha(ROOT/'new_benchmark_common/batch_videos.py')))
    print(json.dumps(dict(video_count=len(exported),complete=False,missing_roles=missing)));return 2
if __name__=='__main__':sys.exit(main())
