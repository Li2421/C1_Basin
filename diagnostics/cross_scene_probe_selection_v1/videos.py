"""Full-simulation video evidence with strict final entry and honest gaps."""
from core import *
from new_benchmark_common import batch_videos as official

def main():
    prereg();assert read(HERE/'main_POSTFLIGHT.json')['status']=='PASS';active=set(read(HERE/'PROBE_SELECTION_FREEZE.json')['active_variants']);cases=[c for c in read(HERE/'main_CASES.json') if c['variant'] in active];records=[]
    for c in cases:
        r=read(HERE/'main/raw'/f"{c['rollout_uid']}.json");records.append(r)
    lookup={(r['case']['variant'],r['case']['base_controller_name'],r['case']['state_index'],r['case']['seed'],r['case']['arm']):r for r in records};formal={sc:[] for sc in official.SCENARIOS};exported=[];missing={};used={sc:set() for sc in official.SCENARIOS}
    def safe(r):return bool(r['success'] and not r['collision'] and not r['numerical_failure'] and r['minimum_swept_clearance'] is not None and r['minimum_swept_clearance']>=0)
    def load(r):
        assert sha(r['trace_path'])==r['trace_sha256'];meta,data=official.load_trace(Path(r['trace_path']));assert sha(meta['augmented_initial_state_path'])==meta['augmented_initial_state_file_sha256'];return meta,data
    def render(sc,role,r,b=None):
        meta,data=load(r);baseline=None
        if b:
            bm,bd=load(b);assert meta['seed']==bm['seed'] and meta['config']==bm['config'] and meta['initial_state_sha256']==bm['initial_state_sha256'];assert bm['deadlock_detected'] and bm['deadlock_criterion'];baseline=(Path(b['trace_path']),bm,bd)
        path=HERE/'videos'/sc/f'{role}.mp4';frames=official.render_video(path,meta,data,baseline);row=dict(scenario=sc,role=role,file=str(path),sha256=sha(path),frames=frames,trace=r['trace_path'],trace_sha256=r['trace_sha256'],provenance=meta)
        if b:row['baseline']=dict(trace=b['trace_path'],trace_sha256=b['trace_sha256'])
        exported.append(row);formal[sc].append(dict(role=role,trace=r['trace_path'],**({'baseline':b['trace_path']} if b else {})))
    for sc in ('toy_give_way','double_bottleneck'):
        public='toy_giveway' if sc=='toy_give_way' else sc;scene=sorted([r for r in records if r['case']['scene']==sc],key=lambda r:r['case']['rollout_uid']);pairs=[]
        for r in scene:
            c=r['case'];key=c['state_uid']
            if c['arm'] not in ARMS[3:] or not safe(r) or key in used[public]:continue
            for fixed in ARMS[:3]:
                b=lookup[c['variant'],c['base_controller_name'],c['state_index'],c['seed'],fixed]
                if b['outcome']=='deadlock' and not b['collision'] and not b['numerical_failure']:
                    pairs.append((r,b));used[public].add(key);break
            if len(pairs)==2:break
        for i,(r,b) in enumerate(pairs):render(public,('deadlock_show','deadlock_resolution')[i],r,b)
        safety=next((r for r in scene if safe(r) and r['case']['state_uid'] not in used[public]),None)
        if safety:render(public,'safety_success',safety)
    # Original-geometry Ring and Four safety evidence is reused for context only.
    # No old timeout is promoted to a formal deadlock role.
    old=read(ROOT/'diagnostics/cross_scene_eta_schedule_v1/videos/manifest.json')
    for sc in ('ring_exchange','four_way_intersection'):
        row=next(x for x in old['videos'] if x['scenario']==sc and x['role']=='safety_success');source=Path(row['trace']);meta,data=official.load_trace(source);meta=dict(meta,batch_id=EXP,source_record=meta['source_record'],cache_reuse_from_batch=old['batch_id'],cache_reuse_trace_sha256=sha(source));dest=HERE/'historical_video_traces'/sc/'safety_success.npz';dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**data,metadata_json=np.array(json.dumps(meta)));m,d=official.load_trace(dest);path=HERE/'videos'/sc/'safety_success.mp4';frames=official.render_video(path,m,d);formal[sc].append(dict(role='safety_success',trace=str(dest)));exported.append(dict(scenario=sc,role='safety_success',file=str(path),sha256=sha(path),frames=frames,trace=str(dest),trace_sha256=sha(dest),historical_cache_reuse=True,source_trace=str(source),source_trace_sha256=sha(source),provenance=m))
    for sc in official.SCENARIOS:missing[sc]=[role for role in official.ROLES if role not in {x['role'] for x in formal[sc]}]
    plan=dict(batch_id=EXP,scenarios=formal);write(HERE/'video_plan.json',plan)
    try:official.main(['--plan',str(HERE/'video_plan.json'),'--output',str(HERE/'videos/strict_gate')]);status=dict(exit_code=0,status='PASS')
    except Exception as ex:status=dict(exit_code=2,status='FAIL',type=type(ex).__name__,message=str(ex))
    complete=status['exit_code']==0;write(HERE/'videos/manifest.json',dict(batch_id=EXP,complete=complete,missing_roles=missing,videos=exported,strict_gate=status,unified_source_sha256=sha(ROOT/'new_benchmark_common/batch_videos.py')));write(HERE/'VIDEO_GATE.json',dict(utc=utc(),**status,complete=complete,missing_roles=missing));print(json.dumps(dict(video_count=len(exported),complete=complete,missing=missing)),flush=True);return status['exit_code']
if __name__=='__main__':sys.exit(main())
