"""Render full frozen-simulator MC1 audit episodes; report missing roles honestly."""
import json
from pathlib import Path

from new_benchmark_common.batch_videos import load_trace, render_video, sha256


ROOT=Path('diagnostics/gap_flow_safety_audit_mc1_v1')


def pick(mode, *, n=2, terminal=None):
    folder=ROOT/f'n{n}'/mode
    summary=folder/'summary.json'
    if not summary.exists():return None
    rows=json.loads(summary.read_text())['rollouts']
    matches=[r for r in rows if terminal is None or r['termination']==terminal]
    return folder/'traces'/f"{matches[0]['rollout_id']}.npz" if matches else None


def main():
    selected={
        'single_agent_success':pick('solo_even',terminal='success'),
        'single_agent_navigation_failure':pick('solo_odd_remote',terminal='timeout'),
        'one_way_success':pick('one_way',terminal='success'),
        'one_way_navigation_failure':pick('one_way',terminal='timeout'),
        'temporally_separated_success':pick('temporal_even_first',terminal='success'),
        'n2_opposing_failure':pick('opposing',terminal='timeout'),
        'n10_one_way_failure':pick('one_way',n=10,terminal='timeout'),
    }
    diagnostics=ROOT/'all_rollout_diagnostics.json'
    deadlocked=[]
    if diagnostics.exists():
        deadlocked=[r for r in json.loads(diagnostics.read_text())
                    if r['N']==2 and r['mode']=='opposing' and r.get('failure_category')=='A']
        mixed=[r for r in json.loads(diagnostics.read_text())
               if r['N']==10 and r['mode']=='opposing' and r.get('failure_category')=='E']
        selected['n10_mixed_opposing_failure']=(ROOT/'n10/opposing/traces'/
            f"{mixed[0]['rollout_id'].removesuffix('_opposing')}.npz") if mixed else None
    selected['canonical_n2_deadlock']=(ROOT/'n2/opposing/traces'/
        f"{deadlocked[0]['rollout_id'].removesuffix('_opposing')}.npz") if deadlocked else None
    folder=ROOT/'videos';folder.mkdir(exist_ok=True)
    records=[];missing=[]
    for label,path in selected.items():
        if path is None:
            missing.append(label);continue
        meta,data=load_trace(path)
        target=folder/f'{label}.mp4'
        frames=render_video(target,meta,data)
        records.append(dict(label=label,video=str(target),video_sha256=sha256(target),
            trace=str(path),trace_sha256=sha256(path),frames=frames,
            termination=meta['termination'],controller=meta['controller'],
            checkpoint=meta['checkpoint'],seed=meta['seed'],dt=meta['config']['dt']))
        print(label,frames,flush=True)
    (folder/'manifest.json').write_text(json.dumps(dict(videos=records,
        missing_roles=missing,complete_full_rollouts=True),indent=2)+'\n')
    print('Missing roles:',missing,flush=True)


if __name__=='__main__':main()
