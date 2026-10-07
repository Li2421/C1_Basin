"""Validate the three complete Gap-family development videos per requested N."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from new_benchmark_common.batch_videos import load_trace, sha256


ROLES=('deadlock_show','deadlock_resolution','safety_success')


def validate(root: Path, sizes=(2,10,50)):
    marker=root/'manifest.json'
    marker.unlink(missing_ok=True)
    records=[]
    for count in sizes:
        used=set()
        for role in ROLES:
            matches=sorted(root.glob(f'n{count}_{role}_*.json'))
            if len(matches)!=1:
                raise ValueError(f'N={count} {role}: expected one video manifest, got {len(matches)}')
            item=json.loads(matches[0].read_text())
            if item['role']!=role or item['formal_four_scene_delivery'] is not False:
                raise ValueError('incorrect role or formal-delivery label')
            trace=Path(item['trace']);video=Path(item['video'])
            if sha256(trace)!=item['trace_sha256'] or sha256(video)!=item['video_sha256']:
                raise ValueError(f'changed trace or video: {matches[0]}')
            meta,data=load_trace(trace)
            if (meta['config']['num_agents']!=count or meta['termination']!='success' or
                    meta['collision'] or meta['rollout_id'] in used):
                raise ValueError('wrong N, non-success, or reused intervention rollout')
            used.add(meta['rollout_id'])
            baseline=item['baseline']
            if role.startswith('deadlock'):
                if baseline is None or item['deadlock_resolution_claim'] is not True:
                    raise ValueError('missing deadlock baseline')
                baseline_trace=Path(baseline['trace'])
                if sha256(baseline_trace)!=baseline['trace_sha256']:
                    raise ValueError('changed deadlock baseline')
                bm,bd=load_trace(baseline_trace)
                if (bm['termination']!='deadlock' or bm.get('deadlock_detected') is not True or
                        not bm.get('deadlock_criterion') or bm['collision'] or
                        bm['seed']!=meta['seed'] or bm['config']!=meta['config'] or
                        bm['initial_state_sha256']!=meta['initial_state_sha256'] or
                        not np.array_equal(bd['positions'][0],data['positions'][0]) or
                        not np.array_equal(bd['goals'],data['goals'])):
                    raise ValueError('deadlock baseline does not match intervention')
                expected=max(len(data['positions']),len(bd['positions']))
            else:
                if baseline is not None or item['deadlock_resolution_claim'] is not False or not meta['safety_enabled']:
                    raise ValueError('invalid independent safety success')
                expected=len(data['positions'])
            probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames',
                '-select_streams','v:0','-show_entries','stream=nb_read_frames','-of','json',str(video)]))
            frames=int(probe['streams'][0]['nb_read_frames'])
            if frames!=expected or frames!=item['frames']:
                raise ValueError(f'video frame count mismatch: {video}')
            records.append(dict(num_agents=count,role=role,video=str(video),
                                video_sha256=item['video_sha256'],frames=frames,
                                intervention_rollout=meta['rollout_id'],
                                baseline_rollout=None if baseline is None else bm['rollout_id']))
    result=dict(schema='gap_development_video_bundle_v1',complete=True,
                formal_four_scene_delivery=False,sizes=list(sizes),videos=records)
    marker.write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--sizes',type=int,nargs='+',default=[2,10,50])
    args=parser.parse_args()
    print(json.dumps({'validated_videos':len(validate(args.root,args.sizes)['videos'])}))


if __name__=='__main__':main()
