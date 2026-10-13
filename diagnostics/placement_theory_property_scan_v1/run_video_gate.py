"""Execute required unified video entry and retain any validation gap."""
import json
import subprocess
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent

def main():
    cmd=[sys.executable,'-m','new_benchmark_common.batch_videos','--plan',str(HERE/'video_gap_plan.json'),
         '--output',str(HERE/'videos')]
    r=subprocess.run(cmd,cwd=HERE.parents[1],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (HERE/'video_gate.log').write_text(r.stdout)
    (HERE/'video_gate_exit_code.txt').write_text(str(r.returncode)+'\n')
    x=dict(batch_id='placement_theory_property_scan_v1',complete=r.returncode==0,
        unified_entry_invoked=True,exit_code=r.returncode,selected_qualified_videos=0,
        missing_roles_by_scenario={s:['deadlock_show','deadlock_resolution','safety_success']
            for s in ('toy_giveway','double_bottleneck','four_way_intersection','ring_exchange')},
        gap='This two-scenario diagnostic scan has no prequalified four-scenario full videos; timeout is not deadlock. Formal large-batch video delivery is incomplete.')
    (HERE/'video_gap_status.json').write_text(json.dumps(x,indent=2)+'\n')
    print(r.returncode,r.stdout[-350:])

if __name__=='__main__':main()
