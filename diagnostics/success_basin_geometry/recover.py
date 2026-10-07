"""Preserve completed chunks and schedule only missing jobs after solver aborts."""
import json
from datetime import datetime,timezone
from diagnostics.success_basin_geometry.setup import HERE,write,sha

def main():
    p=json.loads((HERE/'protocol.json').read_text());stats={}
    for sid in p['primary_states']:
        stage=f'main_{sid}';base=HERE/'raw'/stage
        assert not (base/'manifest.json').exists()
        r=json.loads((base/'partial.json').read_text())
        manifest={'stage':stage,'records':r,'new_rollouts':len(r),'physical_steps':sum(x['steps'] for x in r),
            'interrupted':True,'aborted_unsaved_batch_attempts_upper_bound':16,
            'device':['CPU'],'protocol_sha256':sha(HERE/'protocol.json')}
        (base/'manifest.json').write_text(json.dumps(manifest,indent=2))
        jobs=json.loads((HERE/f'main_{sid}.json').read_text());left=jobs[len(r):]
        write(f'resume_{sid}.json',left);stats[sid]=len(left)
    write('execution_failure_addendum.json',{'time_utc':datetime.now(timezone.utc).isoformat(),
        'frozen_solver_unchanged':True,'reason':'CBFSolverError aborts outside previously small parameter probes.',
        'handling':'Retain incomplete trajectories with outcome=null and execution_error. Never count them as any of the four outcomes. Lower bound success treats missing outcomes pessimistically; report identification intervals.',
        'resume_jobs':stats,'unsaved_attempts_upper_bound':48,
        'unsaved_physical_steps_upper_bound':16*sum(850-s['start_step'] for s in p['state_catalog'] if s['state_id'] in p['primary_states'])})
    print(stats)

if __name__=='__main__':main()
