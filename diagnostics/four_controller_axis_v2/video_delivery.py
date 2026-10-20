"""Render true Four safety traces and run the strict uniform video gate."""
from __future__ import annotations
import datetime,hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from new_benchmark_common import batch_videos as official
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def main():
    assert read(HERE/'FIXED_POSTFLIGHT.json')['status']=='PASS'
    cases=read(HERE/'FIXED_CASES.json');eligible=[];used=set()
    for c in cases:
        r=read(HERE/'stage2/raw'/f"{c['rollout_uid']}.json")
        if r['success'] and not r['collision'] and not r['numerical_failure'] and r['minimum_swept_clearance'] is not None and r['minimum_swept_clearance']>=0:
            key=(c['state_uid'],c['seed'])
            if key not in used:eligible.append(r);used.add(key)
    eligible.sort(key=lambda r:(r['episode_length'],r['case']['rollout_uid']))
    chosen=eligible[:3];assert chosen,'No collision-free full success trace exists'
    videos=HERE/'videos';videos.mkdir(exist_ok=True)
    exported=[]
    for i,r in enumerate(chosen):
        role='safety_success' if i==0 else f'diagnostic_success_{i}'
        trace=Path(r['trace_path']);assert sha(trace)==r['trace_sha256']
        meta,data=official.load_trace(trace)
        path=videos/f'{role}.mp4';frames=official.render_video(path,meta,data)
        exported.append(dict(role=role,formal_role=role=='safety_success',file=str(path),sha256=sha(path),frames=frames,
                             trace=str(trace),trace_sha256=sha(trace),provenance=meta))
    plan=dict(batch_id='exp_round19b_four_controller_axis_v2',scenarios={'four_way_intersection':[
        dict(role='safety_success',trace=chosen[0]['trace_path'])]})
    write(HERE/'video_plan.json',plan)
    original=official.SCENARIOS;official.SCENARIOS=('four_way_intersection',)
    try:
        official.main(['--plan',str(HERE/'video_plan.json'),'--output',str(videos/'strict_gate')])
        rc=0;error=None
    except Exception as exc:
        rc=2;error=f'{type(exc).__name__}: {exc}'
    finally:official.SCENARIOS=original
    missing=[role for role in official.ROLES if role!='safety_success']
    status=dict(batch_id=plan['batch_id'],complete=rc==0,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        videos=exported,missing_roles=missing,official_exit_code=rc,official_error=error,
        uniform_entry='new_benchmark_common.batch_videos.main; Four-only scenario scope',
        uniform_entry_source_sha256=sha(ROOT/'new_benchmark_common/batch_videos.py'),plan_sha256=sha(HERE/'video_plan.json'),
        reason='Four runtime has no recorded deadlock criterion; timeout cannot fill deadlock roles' if missing else None)
    write(videos/'manifest.json',status);write(HERE/'VIDEO_GATE.json',status)
    print(json.dumps(dict(complete=status['complete'],exported=len(exported),missing_roles=missing,official_error=error)),flush=True)
if __name__=='__main__':main()
