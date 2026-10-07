"""Full-rollout MP4 evidence and a mandatory four-scene delivery gate.

Only reads exported simulation traces; does not run or change a controller.
See docs/batch_videos.md for the trace and batch manifest contracts.
"""
from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

SCENARIOS = ('toy_giveway', 'double_bottleneck', 'four_way_intersection', 'ring_exchange')
ROLES = ('deadlock_show', 'deadlock_resolution', 'safety_success')


def sha256(path):
    with Path(path).open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
        return digest.hexdigest()


def load_trace(path):
    """Reject partial/relabelled traces before any video is written."""
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    meta = json.loads(str(arrays.pop('metadata_json').item()))
    p = arrays['positions']
    scenario = meta['scenario']
    count = (meta.get('config', {}).get('num_agents') if scenario == 'bottleneck_family'
             else 2 if scenario == 'toy_giveway' else 4)
    if type(count) is not int or count < 2:
        raise ValueError('invalid num_agents')
    if scenario not in (*SCENARIOS, 'bottleneck_family') or p.ndim != 3 or p.shape[1:] != (count, 2) or len(p) < 2:
        raise ValueError('invalid scenario or trajectory shape')
    if not np.isfinite(p).all():
        raise ValueError('nonfinite positions')
    if (meta.get('start_step') != 0 or meta.get('episode_steps') != len(p) - 1
            or not np.array_equal(arrays['steps'], np.arange(len(p)))
            or meta.get('complete') is not True):
        raise ValueError('full episode required: initial state and every step through termination')
    for key in ('rollout_id', 'controller', 'checkpoint', 'seed', 'termination', 'source_record', 'config'):
        if key not in meta or meta[key] in (None, ''):
            raise ValueError(f'missing provenance: {key}')
    if meta['termination'] not in ('success', 'deadlock', 'timeout', 'collision', 'numerical_failure'):
        raise ValueError('unknown terminal event')
    for key in ('collision', 'safety_enabled'):
        if type(meta.get(key)) is not bool:
            raise ValueError(f'{key} must be a simulator boolean')
    cfg = meta['config']
    if not all(np.isfinite(cfg[k]) and cfg[k] > 0 for k in ('dt', 'agent_radius')):
        raise ValueError('invalid simulation dt or radius')
    if arrays['goals'].shape != (count, 2) or not np.isfinite(arrays['goals']).all():
        raise ValueError('invalid goals')
    walls = arrays['walls']
    if walls.shape != (len(walls), 2, 2) or not np.isfinite(walls).all():
        raise ValueError('walls must have shape [W,2,2]')
    if scenario == 'ring_exchange':
        if not 0 < cfg['obstacle_radius'] < cfg['outer_radius']:
            raise ValueError('invalid annular geometry')
    elif not len(walls):
        raise ValueError('scenario wall geometry is required')
    # Swept clearance must come from the simulator, not recomputation on frames.
    clearance = arrays['swept_clearance']
    if clearance.shape != (len(p) - 1,) or not np.isfinite(clearance).all():
        raise ValueError('one finite simulator swept clearance per transition is required')
    if meta['termination'] == 'success' and (meta['collision'] or np.min(clearance) < 0):
        raise ValueError('successful video must be collision-free for the entire episode')
    return meta, arrays


def validate_plan(plan, root):
    if not isinstance(plan.get('batch_id'), str) or not plan['batch_id'].strip():
        raise ValueError('batch_id required')
    entries = []
    used = set()
    scenarios = SCENARIOS + (('bottleneck_family',) if 'bottleneck_family' in plan['scenarios'] else ())
    unknown = set(plan['scenarios']) - set(scenarios)
    if unknown:
        raise ValueError(f'unknown scenarios: {sorted(unknown)}')
    for scenario in scenarios:
        rows = plan['scenarios'].get(scenario, [])
        if {row['role'] for row in rows} != set(ROLES) or len(rows) != len(ROLES):
            raise ValueError(f'{scenario}: need exactly one selected video for each role {ROLES}')
        for row in rows:
            path = (root / row['trace']).resolve()
            meta, data = load_trace(path)
            if meta['scenario'] != scenario or meta['termination'] != 'success':
                raise ValueError(f'{scenario}: expected a successful scenario trace')
            if meta.get('batch_id') != plan['batch_id']:
                raise ValueError('trace must belong to this batch (record exact cache reuse explicitly)')
            identity = (scenario, meta['rollout_id'])
            trajectory_hash = hashlib.sha256(data['positions'].tobytes()).hexdigest()
            if identity in used or (scenario, trajectory_hash) in used:
                raise ValueError('duplicate rollout cannot fill multiple video roles')
            used.update((identity, (scenario, trajectory_hash)))
            baseline = None
            if row['role'].startswith('deadlock'):
                baseline_path = (root / row['baseline']).resolve()
                bm, bd = load_trace(baseline_path)
                if (bm['scenario'] != scenario or bm['termination'] != 'deadlock'
                        or bm.get('deadlock_detected') is not True or not bm.get('deadlock_criterion')
                        or bm['collision'] or bm.get('batch_id') != plan['batch_id']):
                    raise ValueError('deadlock video requires a real, documented deadlock baseline')
                if (bm['seed'] != meta['seed'] or bm['config'] != meta['config']
                        or not np.array_equal(bd['positions'][0], data['positions'][0])
                        or not np.array_equal(bd['goals'], data['goals'])
                        or not np.array_equal(bd['walls'], data['walls'])
                        or not bm.get('initial_state_sha256')
                        or bm['initial_state_sha256'] != meta.get('initial_state_sha256')):
                    raise ValueError('baseline and intervention require identical full initial state, geometry and seed')
                baseline = (baseline_path, bm, bd)
            elif not meta['safety_enabled']:
                raise ValueError('safety_success requires the safety controller enabled')
            entries.append((scenario, row['role'], path, meta, data, baseline))
    return entries


def render_video(path, meta, data, baseline=None):
    """Every stored simulation state is rendered once, at native simulation time."""
    from PIL import Image, ImageDraw

    panels = [(meta, data)] if baseline is None else [(baseline[1], baseline[2]), (meta, data)]
    width, height = 720 * len(panels), 640
    frames = max(len(d['positions']) for _, d in panels)
    count = data['positions'].shape[1]
    colors = (('#0072B2', '#D55E00', '#009E73', '#CC79A7') if count <= 4 else
              tuple(tuple(round(255*x) for x in colorsys.hsv_to_rgb((i*.61803398875)%1,.75,.75))
                    for i in range(count)))
    cfg = meta['config']
    points = [d['positions'].reshape(-1, 2) for _, d in panels]
    points += [data['goals'], data['walls'].reshape(-1, 2)]
    if meta['scenario'] == 'ring_exchange':
        r = cfg['outer_radius']
        points.append(np.array([[-r, -r], [r, r]]))
    all_points = np.concatenate(points)
    lo, hi = all_points.min(axis=0) - .35, all_points.max(axis=0) + .35
    scale = min(640 / (hi[0] - lo[0]), 420 / (hi[1] - lo[1]))
    center = (lo + hi) / 2
    path.parent.mkdir(parents=True, exist_ok=True)
    def xy(panel, point):
        x, y = (np.asarray(point) - center) * scale
        return (720*panel + 360 + x, 345 - y)
    def circle(draw, panel, point, radius, fill=None, outline='black', line=2):
        x, y = xy(panel, point); r = radius * scale
        draw.ellipse((x-r, y-r, x+r, y+r), fill=fill, outline=outline, width=line)
    static = Image.new('RGB', (width, height), 'white')
    static_draw = ImageDraw.Draw(static)
    for panel, (pm, pd) in enumerate(panels):
        if pm['scenario'] == 'ring_exchange':
            circle(static_draw, panel, (0,0), cfg['outer_radius'], fill='#eeeeee')
            circle(static_draw, panel, (0,0), cfg['obstacle_radius'], fill='#777777')
        for wall in pd['walls']:
            static_draw.line([xy(panel, wall[0]), xy(panel, wall[1])], fill='#333333', width=3)
    trails = Image.new('RGBA', (width, height), (0,0,0,0))
    trail_draw = ImageDraw.Draw(trails)
    with tempfile.TemporaryDirectory(dir=path.parent) as temporary:
        target = Path(temporary) / 'video.mp4'
        cmd = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', f'{width}x{height}', '-r', str(1 / cfg['dt']), '-i', '-', '-an',
               '-c:v', 'libx264', '-threads', '1', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(target)]
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=errors)
            try:
                for frame in range(frames):
                    for panel, (_, pd) in enumerate(panels):
                        if 0 < frame < len(pd['positions']):
                            for agent, color in enumerate(colors[:pd['positions'].shape[1]]):
                                trail_draw.line([xy(panel, pd['positions'][frame-1, agent]),
                                                 xy(panel, pd['positions'][frame, agent])],
                                                fill=color, width=2)
                    image = static.copy()
                    image.paste(trails, mask=trails.getchannel('A'))
                    draw = ImageDraw.Draw(image)
                    for panel, (pm, pd) in enumerate(panels):
                        offset = 720 * panel
                        k = min(frame, len(pd['positions']) - 1)
                        label = (pm['termination'].upper() if baseline is None else
                                 'SUCCESS' if panel == len(panels)-1 else 'DEADLOCK BASELINE')
                        draw.text((offset+24, 18), f'{pm["scenario"]} | {label}', fill='black')
                        draw.text((offset+24, 40), f'controller: {pm["controller"]} | seed: {pm["seed"]}', fill='black')
                        draw.text((offset+24, 62), f'step {k}/{len(pd["positions"])-1} | t={k*cfg["dt"]:.2f}s | 1x playback', fill='black')
                        status = pm['termination'] if k == len(pd['positions'])-1 else 'running'
                        draw.text((offset+24, 84), f'status: {status} | safety enabled: {pm["safety_enabled"]}', fill='black')
                        for agent, color in enumerate(colors[:pd['positions'].shape[1]]):
                            circle(draw, panel, pd['goals'][agent], cfg['agent_radius'], outline=color)
                            draw.text(xy(panel, pd['goals'][agent]), 'goal', fill=color)
                            circle(draw, panel, pd['positions'][k, agent], cfg['agent_radius'], fill=color, outline=color)
                        if k:
                            draw.text((offset+24, 585), f'min swept clearance so far: {pd["swept_clearance"][:k].min():.5f} m', fill='black')
                        draw.text((offset+24, 607), f'rollout: {pm["rollout_id"]}', fill='black')
                    process.stdin.write(image.tobytes())
                process.stdin.close()
                code = process.wait()
            except BaseException:
                process.kill(); process.wait()
                raise
            if code:
                errors.seek(0)
                raise RuntimeError(errors.read().decode())
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
            '-select_streams', 'v:0', '-show_entries', 'stream=nb_read_frames', '-of', 'json', str(target)]))
        if int(probe['streams'][0]['nb_read_frames']) != frames:
            raise RuntimeError('encoded frame count differs from full simulation')
        target.replace(path)
    return frames


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args(argv)
    # Invalidate prior success even when preflight fails on a rerun.
    manifest = args.output / 'manifest.json'
    if not args.validate_only:
        manifest.unlink(missing_ok=True)
    plan = json.loads(args.plan.read_text())
    entries = validate_plan(plan, args.plan.resolve().parent)
    if args.validate_only:
        print(f'Validated {len(entries)} full-rollout video inputs; MP4 delivery still pending.')
        return
    for tool in ('ffmpeg', 'ffprobe'):
        if not shutil.which(tool):
            raise RuntimeError(f'{tool} required')
    records = []
    for scenario, role, trace, meta, data, baseline in entries:
        path = args.output / scenario / f'{role}.mp4'
        frames = render_video(path, meta, data, baseline)
        records.append(dict(scenario=scenario, role=role, file=str(path.relative_to(args.output)),
            sha256=sha256(path), frames=frames, dt=meta['config']['dt'],
            trace=str(trace), trace_sha256=sha256(trace), provenance=meta,
            baseline=None if baseline is None else dict(trace=str(baseline[0]),
                sha256=sha256(baseline[0]), provenance=baseline[1])))
        print(f'Wrote {path}', flush=True)
    result = dict(batch_id=plan['batch_id'], complete=True, videos=records,
                  plan_sha256=sha256(args.plan))
    temporary = manifest.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(manifest)


if __name__ == '__main__':
    main()
