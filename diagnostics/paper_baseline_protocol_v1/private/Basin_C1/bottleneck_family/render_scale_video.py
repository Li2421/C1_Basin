"""Render a complete Gap-family simulator trace with provenance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from new_benchmark_common.batch_videos import load_trace, render_video, sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    meta, data = load_trace(args.trace)
    if meta['scenario'] != 'bottleneck_family':
        raise ValueError('expected a Gap-family simulator trace')
    if args.output.exists():
        raise FileExistsError(args.output)
    frames = render_video(args.output, meta, data)
    manifest = {
        'schema': 'gap_scale_full_rollout_video_v1',
        'label': args.label,
        'trace': str(args.trace),
        'trace_sha256': sha256(args.trace),
        'video': str(args.output),
        'video_sha256': sha256(args.output),
        'frames': frames,
        'simulator_steps': meta['episode_steps'],
        'simulator_dt': meta['config']['dt'],
        'N': meta['config']['num_agents'],
        'geometry': {'barrier_x': meta['config']['barrier_x'],
                     'openings': meta['config']['openings']},
        'rollout_id': meta['rollout_id'],
        'controller': meta['controller'],
        'checkpoint': meta['checkpoint'],
        'seed': meta['seed'],
        'termination': meta['termination'],
        'collision': meta['collision'],
        'min_swept_clearance': float(np.min(data['swept_clearance'])),
    }
    destination = args.output.with_name(args.output.stem + '_manifest.json')
    destination.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
