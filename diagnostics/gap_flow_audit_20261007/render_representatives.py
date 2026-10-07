"""Render complete simulator traces selected by the frozen-controller audit."""
import json
from pathlib import Path

from new_benchmark_common.batch_videos import load_trace, render_video, sha256


ROOT = Path('diagnostics/gap_flow_audit_20261007')
SOURCES = {
    'n2_nominal_wall_collision': Path('diagnostics/gap_flow_v1/n2_recovery_wide/eval_test_mc16_raw/traces/test_gap_n2_0000.npz'),
    'n10_nominal_wall_collision': Path('diagnostics/gap_flow_v1/n10_wide_recovery/eval_test_mc16_raw/traces/test_gap_n10_0000.npz'),
    'n50_nominal_agent_collision': Path('diagnostics/gap_flow_v1/n50_wide_recovery/eval_test_mc16_raw/traces/test_gap_n50_0000.npz'),
    'n2_one_way_collision': ROOT/'controls/n2/one_way/traces/test_gap_n2_0000.npz',
    'n2_temporal_timeout': ROOT/'controls/n2/temporal_odd_first/traces/test_gap_n2_0000.npz',
}


def main():
    folder = ROOT/'videos'; folder.mkdir(exist_ok=True)
    records=[]
    for label,trace in SOURCES.items():
        meta,data=load_trace(trace)
        path=folder/f'{label}.mp4'
        frames=render_video(path,meta,data)
        records.append(dict(label=label,video=str(path),video_sha256=sha256(path),
            frames=frames,dt=meta['config']['dt'],trace=str(trace),
            trace_sha256=sha256(trace),termination=meta['termination'],
            collision=meta['collision'],checkpoint=meta['checkpoint'],seed=meta['seed']))
        print(f'{path}: {frames} frames',flush=True)
    (folder/'manifest.json').write_text(json.dumps({'complete':True,'videos':records},indent=2)+'\n')


if __name__=='__main__':main()
