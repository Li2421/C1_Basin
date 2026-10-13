"""Render every stored LOSO simulator state once, within 20 s per MP4."""
from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess

import numpy as np

from new_benchmark_common.batch_videos import load_trace, render_video, sha256

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
HIST = ROOT/'diagnostics/orthoflow3_loso_root_cause_v1'


def main():
    cases = json.loads((OUT/'selection.json').read_text())['cases']
    videos = OUT/'videos';videos.mkdir(exist_ok=True)
    rows = []
    for case in cases:
        trace = OUT/'traces'/(case['name']+'.npz')
        meta, data = load_trace(trace)
        assert meta['rollout_id'] == case['expected']['rollout_uid']
        frames = len(data['positions'])
        fps = max(24, math.ceil(frames/18.5))
        target = videos/(case['name']+'.mp4')
        if not target.exists():
            actual_frames = render_video(target, meta, data, playback_fps=fps)
            assert actual_frames == frames
        probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
            '-show_entries','stream=nb_read_frames,duration,avg_frame_rate','-of','json',str(target)]))['streams'][0]
        duration = float(probe['duration'])
        assert int(probe['nb_read_frames']) == frames
        assert duration < 20.0, (target,duration)
        robust = None
        q16 = None
        if case['selected_index'] is not None:
            truth = (HIST/case['scene']/'target_truth.json' if case['scene']!='ring'
                     else ROOT/'diagnostics/orthoflow3_cross_scene_zero_shot_v1/target_truth.json')
            all_truth = json.loads(truth.read_text())
            row = all_truth[case['state_index']]
            assert row['state_uid'] == case['state_uid']
            robust = row['robust'][case['selected_index']]
            q16 = row['q'][case['selected_index']]
        rows.append({'name':case['name'],'scenario':meta['scenario'],'panel':case['panel'],
                     'state_index':case['state_index'],'state_uid':case['state_uid'],
                     'selected_eta_index':case['selected_index'],'eta':case['eta'],
                     'seed':case['future_index'],'outcome':meta['termination'],
                     'Q16':q16,'robust_15_of_16':robust,
                     'simulator_steps':frames-1,'simulator_seconds':(frames-1)*meta['config']['dt'],
                     'video_frames':frames,'playback_fps':fps,'playback_duration_seconds':duration,
                     'all_states_rendered':True,'min_swept_clearance_m':float(np.min(data['swept_clearance'])),
                     'collision':meta['collision'],'video':str(target.relative_to(ROOT)),
                     'video_sha256':sha256(target),'trace':str(trace.relative_to(ROOT)),
                     'trace_sha256':sha256(trace), 'checkpoint_sha256':meta['checkpoint_sha256'],
                     'cached_rollout_uid':case['expected']['rollout_uid'],
                     'source_record':case['expected']['original_source_file']})
        print(f"{case['name']}: {frames} frames, {duration:.2f}s, {meta['termination']}",flush=True)
    by_name = {r['name']:r for r in rows}
    pair_specs = [
        ('toy_state15_within_vs_loso', 'toy_within_state15_success', 'toy_loso_state15_deadlock'),
        ('toy_state20_within_vs_loso', 'toy_within_state20_deadlock', 'toy_loso_state20_success'),
        ('db_state06_within_vs_loso', 'db_within_state06_success', 'db_loso_state06_timeout'),
        ('four_state00_within_vs_loso', 'four_within_state00_success', 'four_loso_state00_success'),
        ('four_state00_safety_vs_loso', 'four_safety_baseline_state00_timeout', 'four_loso_state00_success'),
        ('ring_state12_within_vs_loso', 'ring_within_state12_timeout', 'ring_loso_state12_success'),
        ('ring_state16_within_vs_loso', 'ring_within_state16_success', 'ring_loso_state16_timeout'),
    ]
    comparisons = []
    for name, left, right in pair_specs:
        lm, ld = load_trace(ROOT/by_name[left]['trace'])
        rm, rd = load_trace(ROOT/by_name[right]['trace'])
        assert lm['scenario'] == rm['scenario'] and lm['state_uid'] == rm['state_uid']
        assert lm['seed'] == rm['seed']
        np.testing.assert_array_equal(ld['positions'][0],rd['positions'][0])
        np.testing.assert_array_equal(ld['goals'],rd['goals'])
        frames = max(len(ld['positions']),len(rd['positions']))
        fps = max(24,math.ceil(frames/18.5))
        target = videos/(name+'.mp4')
        if not target.exists():
            render_video(target,rm,rd,baseline=(ROOT/by_name[left]['trace'],lm,ld),
                         playback_fps=fps,panel_labels=[lm['panel'].upper() if 'panel' in lm else by_name[left]['panel'].upper(),
                                                        by_name[right]['panel'].upper()])
        probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
            '-show_entries','stream=nb_read_frames,duration','-of','json',str(target)]))['streams'][0]
        duration = float(probe['duration'])
        assert int(probe['nb_read_frames']) == frames and duration < 20
        comparisons.append({'name':name,'left':left,'right':right,'same_state':True,
                            'same_seed':True,'frame_count':frames,'duration_seconds':duration,
                            'video':str(target.relative_to(ROOT)),'sha256':sha256(target)})
        print(f'{name}: {frames} frames, {duration:.2f}s',flush=True)
    manifest = {'schema':'strict_loso_complete_video_v1','complete':len(rows)==len(cases),
                'historical_evidence':str((ROOT/'diagnostics/physical_state_critic_minimal_v1/strict/REPORT.md').relative_to(ROOT)),
                'new_Q_rollouts':0,'trace_replay_count':len(rows),
                'note':'Historical exact continuations replayed solely to export full traces. Video FPS changes viewing time only; no simulator steps dropped.',
                'eta_success_showcase':[r['name'] for r in rows if r['name'] in (
                    'toy_within_state15_success','db_within_state06_success',
                    'four_within_state00_success','ring_within_state16_success')],
                'limitations':['Within-scene Double-Bottleneck selected eta succeeds 24/24, so its failure example is the safety-only baseline.',
                               'Four-Way selected eta continuations have no observed valid failure; its failure example is the safety-only baseline.',
                               'Timeout is labelled timeout, not deadlock.'],
                'videos':rows,'paired_comparisons':comparisons}
    (videos/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'complete':manifest['complete'],'videos':len(rows),
                      'paired_comparisons':len(comparisons),
                      'max_duration':max([r['playback_duration_seconds'] for r in rows]
                                     +[r['duration_seconds'] for r in comparisons])},indent=2))


if __name__=='__main__':
    main()
