"""Render complete simulator evidence for the frozen Gap1 competence audit."""
from __future__ import annotations

import json
from pathlib import Path

from new_benchmark_common.batch_videos import load_trace, render_video, sha256


ROOT=Path('diagnostics/gap_flow_competence_v3_recovery')


def main():
    selections={
        'single_lr_success':ROOT/'acceptance_basic/traces/test_nav_0000_solo_LR_swap0.npz',
        'single_rl_success':ROOT/'acceptance_basic/traces/test_nav_0000_solo_RL_swap0.npz',
        'one_way_lr_success':ROOT/'acceptance_basic/traces/test_nav_0000_one_way_LR_swap0.npz',
        'canonical_n2_safe_gridlock':ROOT/'frozen_eval/opposing/traces/test_gap_n2_0011.npz',
        'n2_delayed_progress_timeout':ROOT/'frozen_eval/opposing/traces/test_gap_n2_0005.npz',
        'n2_spontaneous_opposing_success':ROOT/'replication_n2/opposing/traces/fresh_control_0005.npz',
        'n10_single_navigation_failure':Path('diagnostics/gap_flow_competence_n10_v2_recovery/full_solo_test/traces/test_n10_nav_0000_solo_LR_swap0.npz'),
        'n10_one_way_failure':Path('diagnostics/gap_flow_competence_n10_v2_recovery/full_one_way_test/traces/test_gap_n10_0000.npz'),
    }
    folder=ROOT/'videos';folder.mkdir(exist_ok=True)
    prior_path=folder/'manifest.json'
    prior={row['label']:row for row in json.loads(prior_path.read_text())['videos']} if prior_path.exists() else {}
    records=[]
    for label,trace in selections.items():
        meta,data=load_trace(trace)
        target=folder/f'{label}.mp4'
        old=prior.get(label)
        reuse=(target.exists() and old is not None and
               old['trace_sha256']==sha256(trace) and
               old['video_sha256']==sha256(target) and
               old['frames']==len(data['positions']))
        frames=len(data['positions']) if reuse else render_video(target,meta,data)
        records.append(dict(label=label,video=str(target),video_sha256=sha256(target),
            trace=str(trace),trace_sha256=sha256(trace),frames=frames,
            simulator_steps=meta['episode_steps'],termination=meta['termination'],
            collision=meta['collision'],controller=meta['controller'],
            checkpoint=meta['checkpoint'],seed=meta['seed'],dt=meta['config']['dt'],
            complete=True))
        print(label,frames,flush=True)
    manifest={'schema':'gap1_competence_full_video_manifest_v1','videos':records,
              'renders_all_simulator_steps_from_initial_state':True,
              'opposing_videos_are_nominal_only_no_intervention':True}
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
