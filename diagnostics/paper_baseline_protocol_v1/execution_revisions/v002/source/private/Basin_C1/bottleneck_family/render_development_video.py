"""Render complete successful Gap-family development rollouts for inspection.

These videos show the full simulator trajectory; this single-scene utility is
not the mandatory four-scene deadlock-video delivery gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from new_benchmark_common.batch_videos import load_trace, render_video, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--role', choices=('deadlock_show','deadlock_resolution','safety_success'),
                        required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    meta, data = load_trace(args.trace)
    if meta['scenario'] != 'bottleneck_family' or meta['termination'] != 'success' or meta['collision']:
        raise ValueError('requires a complete collision-free Gap-family success trace')
    baseline = None
    if args.role.startswith('deadlock'):
        if args.baseline is None:
            raise ValueError('deadlock video requires an actual baseline trace')
        bm, bd = load_trace(args.baseline)
        if (bm['scenario'] != meta['scenario'] or bm['termination'] != 'deadlock' or
                bm.get('deadlock_detected') is not True or not bm.get('deadlock_criterion') or
                bm['collision'] or bm['seed'] != meta['seed'] or bm['config'] != meta['config'] or
                bm.get('initial_state_sha256') != meta.get('initial_state_sha256') or
                not np.array_equal(bd['positions'][0], data['positions'][0]) or
                not np.array_equal(bd['goals'], data['goals']) or
                not np.array_equal(bd['walls'], data['walls'])):
            raise ValueError('deadlock and intervention must share full initial state and configuration')
        baseline = (args.baseline, bm, bd)
    elif args.baseline is not None or not meta['safety_enabled']:
        raise ValueError('safety success needs an enabled safety controller and no baseline')
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / f"n{meta['config']['num_agents']}_{args.role}_{meta['rollout_id']}.mp4"
    record = args.output / f'{target.stem}.json'
    record.unlink(missing_ok=True)
    frames = render_video(target, meta, data, baseline=baseline)
    result = {'schema': 'gap_development_full_video_v1',
              'formal_four_scene_delivery': False, 'role': args.role,
              'deadlock_resolution_claim': baseline is not None,
              'video': str(target), 'video_sha256': sha256(target), 'trace': str(args.trace),
              'trace_sha256': sha256(args.trace), 'frames': frames,
              'dt': meta['config']['dt'], 'provenance': meta,
              'baseline': None if baseline is None else
                  {'trace': str(args.baseline), 'trace_sha256': sha256(args.baseline),
                   'provenance': baseline[1]}}
    record.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'video': str(target), 'frames': frames, 'sha256': result['video_sha256']}))


if __name__ == '__main__':
    main()
