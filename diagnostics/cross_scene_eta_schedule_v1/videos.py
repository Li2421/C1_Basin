"""Mandatory four-scene full-simulation videos, no fabricated deadlocks."""
from core import *
from new_benchmark_common import batch_videos as official

def main():
    prereg();gate();assert read(HERE/'main_POSTFLIGHT.json')['status']=='PASS';cases=read(HERE/'main_CASES.json');records=[]
    for c in cases:
        r=read(HERE/'main/raw'/f"{c['rollout_uid']}.json");records.append({k:r[k] for k in ['case','success','collision','numerical_failure','minimum_swept_clearance','outcome','trace_path','trace_sha256']})
    lookup={(r['case']['scene'],r['case']['base_controller_name'],r['case']['state_index'],r['case']['seed'],r['case']['arm']):r for r in records};formal={};exported=[];missing={};selected=[]
    def safe(r):return bool(r['success'] and not r['collision'] and not r['numerical_failure'] and r['minimum_swept_clearance'] is not None and r['minimum_swept_clearance']>=0)
    def load(r):
        assert sha(r['trace_path'])==r['trace_sha256'];meta,data=official.load_trace(Path(r['trace_path']));assert sha(meta['augmented_initial_state_path'])==meta['augmented_initial_state_file_sha256'];initial=read(meta['augmented_initial_state_path']);assert hashlib.sha256(json.dumps(initial,sort_keys=True,separators=(',',':')).encode()).hexdigest()==meta['initial_state_sha256'];meta['display_label']=f"{r['case']['scene']} {r['case']['base_controller_name']} {r['case']['arm']}";return meta,data
    for sc in SCENES:
        public='toy_giveway' if sc=='toy_give_way' else sc;scene=sorted([r for r in records if r['case']['scene']==sc],key=lambda r:r['case']['rollout_uid']);pairs=[];used=set();pairs_uids=set()
        for r in scene:
            c=r['case']
            if c['arm'] not in ARMS[3:] or not safe(r):continue
            for fixed in ARMS[:3]:
                b=lookup[sc,c['base_controller_name'],c['state_index'],c['seed'],fixed]
                if b['outcome']=='deadlock' and not b['collision'] and not b['numerical_failure'] and (c['state_uid'],c['seed']) not in used:
                    pairs.append((r,b));used.add((c['state_uid'],c['seed']));pairs_uids.add(c['rollout_uid']);break
            if len(pairs)==2:break
        safety=next((r for r in scene if r['case']['arm'] in ARMS[3:] and safe(r) and (r['case']['state_uid'],r['case']['seed']) not in used and r['case']['rollout_uid'] not in pairs_uids),None);formal[public]=[]
        for i,(r,b) in enumerate(pairs):
            role=['deadlock_show','deadlock_resolution'][i];meta,data=load(r);bm,bd=load(b);assert meta['seed']==bm['seed'] and meta['config']==bm['config'] and meta['initial_state_sha256']==bm['initial_state_sha256'];assert bm['deadlock_detected'] and bm['deadlock_criterion'];path=HERE/'videos'/public/f'{role}.mp4';frames=official.render_video(path,meta,data,(Path(b['trace_path']),bm,bd));formal[public].append(dict(role=role,trace=r['trace_path'],baseline=b['trace_path']));exported.append(dict(scenario=public,role=role,formal_role=True,file=str(path),sha256=sha(path),frames=frames,dt=meta['config']['dt'],trace=r['trace_path'],trace_sha256=r['trace_sha256'],provenance=meta,baseline=dict(trace=b['trace_path'],sha256=b['trace_sha256'],provenance=bm)));selected.append(c['rollout_uid'])
        if safety:
            meta,data=load(safety);path=HERE/'videos'/public/'safety_success.mp4';frames=official.render_video(path,meta,data);formal[public].append(dict(role='safety_success',trace=safety['trace_path']));exported.append(dict(scenario=public,role='safety_success',formal_role=True,file=str(path),sha256=sha(path),frames=frames,dt=meta['config']['dt'],trace=safety['trace_path'],trace_sha256=safety['trace_sha256'],provenance=meta))
        missing[public]=[role for role in official.ROLES if role not in {x['role'] for x in formal[public]}]
        # Honest diagnostic pairs for scenes without formal deadlock evidence.
        if len(pairs)<2:
            diagnostic=[];diag_used=set(used)
            for r in scene:
                c=r['case']
                if c['arm'] not in ARMS[3:] or not safe(r) or (c['state_uid'],c['seed']) in diag_used:continue
                if safety and (c['state_uid'],c['seed'])==(safety['case']['state_uid'],safety['case']['seed']):continue
                b=next((lookup[sc,c['base_controller_name'],c['state_index'],c['seed'],a] for a in ARMS[:3] if lookup[sc,c['base_controller_name'],c['state_index'],c['seed'],a]['outcome']=='timeout' and not lookup[sc,c['base_controller_name'],c['state_index'],c['seed'],a]['numerical_failure']),None)
                if b:
                    diagnostic.append((r,b));diag_used.add((c['state_uid'],c['seed']))
                    if len(diagnostic)==2-len(pairs):break
            for i,(r,b) in enumerate(diagnostic):
                meta,data=load(r);bm,bd=load(b);assert meta['seed']==bm['seed'] and meta['config']==bm['config'] and meta['initial_state_sha256']==bm['initial_state_sha256'];role=f'timeout_to_success_{i+1}';path=HERE/'videos'/public/f'{role}.mp4';frames=official.render_video(path,meta,data,(Path(b['trace_path']),bm,bd),panel_labels=('TIMEOUT BASELINE','SUCCESS'));exported.append(dict(scenario=public,role=role,formal_role=False,counts_as_deadlock_resolution=False,file=str(path),sha256=sha(path),frames=frames,dt=meta['config']['dt'],trace=r['trace_path'],trace_sha256=r['trace_sha256'],provenance=meta,baseline=dict(trace=b['trace_path'],sha256=b['trace_sha256'],provenance=bm)))
    plan=dict(batch_id=EXP,scenarios=formal);write(HERE/'video_plan.json',plan)
    try:official.main(['--plan',str(HERE/'video_plan.json'),'--output',str(HERE/'videos/strict_gate')]);status=dict(exit_code=0,status='PASS')
    except Exception as ex:status=dict(exit_code=2,status='FAIL',error_type=type(ex).__name__,message=str(ex))
    complete=status['exit_code']==0;write(HERE/'videos/manifest.json',dict(batch_id=EXP,complete=complete,required_scenes=list(official.SCENARIOS),required_roles=list(official.ROLES),missing_roles=missing,videos=exported,strict_gate=status,reason='Ring/Four frozen native runtimes have no deadlock detector. Missing roles never filled with timeout. Other scene gaps reported explicitly.',unified_source_sha256=sha(ROOT/'new_benchmark_common/batch_videos.py')));write(HERE/'VIDEO_GATE.json',dict(utc=utc(),**status,complete=complete,missing_roles=missing,unified_entry='new_benchmark_common.batch_videos.main; unmodified SCENARIOS and ROLES'));print(json.dumps(dict(video_count=len(exported),complete=complete,missing=missing)));return status['exit_code']
if __name__=='__main__':sys.exit(main())
