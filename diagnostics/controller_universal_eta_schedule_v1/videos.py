"""Ring-scoped official video validation, honest missing-deadlock delivery."""
from core import *
from collections import Counter
import subprocess,traceback
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
        r=read(HERE/'main/raw'/f"{c['rollout_uid']}.json");records.append(r);lookup[c['base_controller_name'],c['arm'],c['state_index'],c['seed']]=r
    def valid_success(r):return bool(r['success'] and not r['numerical_failure'] and not r['collision'])
    candidates=[]
    for r in records:
        c=r['case']
        if c['arm']!='S500' or not valid_success(r):continue
        b=lookup[c['base_controller_name'],'F13',c['state_index'],c['seed']]
        if b['outcome']=='timeout' and not b['numerical_failure']:candidates.append((r,b))
    candidates.sort(key=lambda pair:pair[0]['case']['rollout_uid']);pairs=[];used=set()
    for target in ('dev','held_out'):
        for r,b in candidates:
            isdev=r['case']['base_controller_name'] in DEV
            if isdev!=(target=='dev') or r['case']['rollout_uid'] in used:continue
            pairs.append((r,b));used.add(r['case']['rollout_uid']);break
    used_initial_seeds={(r['case']['state_uid'],r['case']['seed']) for r,b in pairs}
    safety=next((r for r in sorted(records,key=lambda r:r['case']['rollout_uid']) if r['case']['arm']=='S500' and valid_success(r) and r['case']['rollout_uid'] not in used and (r['case']['state_uid'],r['case']['seed']) not in used_initial_seeds),None)
    exported=[]
    def load(r):
        meta,data=official.load_trace(Path(r['trace_path']));assert meta['batch_id']==EXP and sha(r['trace_path'])==r['trace_sha256'];meta['display_label']=f"Ring | {r['case']['base_controller_name'].replace('_macflow','')} {r['case']['arm']} | cp {r['checkpoint_sha256'][:8]}"
        # Deterministic true-t0 reconstruction, no env.step/action or rollout.
        from ring_exchange.environment import Config,RingExchangeEnv
        env=RingExchangeEnv(Config(**meta['config']));env.reset(data['positions'][0],velocities=data['velocities'][0],goals=data['goals']);initial=clean(env.augmented_state());initial_hash=hashlib.sha256(json.dumps(initial,sort_keys=True,separators=(',',':')).encode()).hexdigest();assert initial_hash==meta['initial_state_sha256']
        initial_path=HERE/'videos/initial_states'/f"{r['case']['rollout_uid']}.json";write(initial_path,initial);meta['augmented_initial_state_path']=str(initial_path);meta['augmented_initial_state_file_sha256']=sha(initial_path);meta['initial_state_reconstruction']='Original deterministic empty-monitor true-t0 reset from saved initial positions, velocities, goals and config; exact recorded augmented-state hash verified; zero simulation steps'
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
