"""Invoke the repository's required video entry and record any real gap."""
import json
import subprocess
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent


def main():
    cmd=[sys.executable,'-m','new_benchmark_common.batch_videos','--plan',str(HERE/'video_gap_plan.json'),
         '--output',str(HERE/'videos')]
    r=subprocess.run(cmd,cwd=HERE.parents[1],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (HERE/'video_gate.log').write_text(r.stdout)
    (HERE/'video_gate_exit_code.txt').write_text(str(r.returncode)+'\n')
    status=dict(batch_id='placement_strong_hypotheses_v1',complete=r.returncode==0,
        unified_entry_invoked=True,exit_code=r.returncode,
        selected_qualified_videos=0,
        missing_roles_by_scenario={scene:['deadlock_show','deadlock_resolution','safety_success']
            for scene in ('toy_giveway','double_bottleneck','four_way_intersection','ring_exchange')},
        scientific_gap='Mechanism probes did not select/export formally qualified full-video cases; Ring/Four-Way old evaluators lack a deadlock termination monitor. Timeout is not deadlock.',
        required_followup='Real complete baseline/resolution traces with documented deadlock evidence, plus independent collision-free safety successes for each required scenario.')
    (HERE/'video_gap_status.json').write_text(json.dumps(status,indent=2)+'\n')
    print(r.returncode,r.stdout[-500:])


if __name__=='__main__':main()
