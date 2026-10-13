"""Submit prospective Stage C confirmation of frozen Stage B passing arms."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from diagnostics.controller_loco_real_flow_v1.dispatch_conditional import HERE, PY, submit


def main():
    protocol=json.loads((HERE/'stage_c/protocol.json').read_text())
    b=json.loads((HERE/'stage_b/results.json').read_text())
    assert protocol['passing_arms']==b['primary']['passing_arms']==['R3','R3_R4_action']
    assert protocol['held_folds']==[1,4]
    assert json.loads((HERE/'stage_c/CACHE_INVENTORY.json').read_text())['missing']<=10000
    tasks=[[fold,arm] for fold in protocol['held_folds'] for arm in ['R0']+protocol['passing_arms']]
    (HERE/'stage_c/PREDICTION_TASKS.json').write_text(json.dumps(tasks,indent=2)+'\n')
    jobs=[]
    jobs.append(submit('locoCfeat',f'{PY} {HERE}/stage_c_features.py $SLURM_ARRAY_TASK_ID',array='0-6%5',gpu=True,hours=2))
    dep=jobs[-1]['job_id']
    jobs.append(submit('locoCraw',f'{PY} {HERE}/stage_c_run.py worker --index $SLURM_ARRAY_TASK_ID --workers 19',dependency=dep,array='0-18%19',hours=5))
    dep=jobs[-1]['job_id']
    jobs.append(submit('locoCmerge',f'bash {HERE}/finalize_stage_c.sh',dependency=dep,hours=2))
    dep=jobs[-1]['job_id']
    jobs.append(submit('locoCpredict',f'bash {HERE}/stage_c_predict_job.sh',dependency=dep,array=f'0-{len(tasks)-1}%5',gpu=True,hours=2))
    dep=jobs[-1]['job_id']
    jobs.append(submit('locoCanalyze',f'{PY} {HERE}/stage_c_analyze.py',dependency=dep,hours=2))
    dep=jobs[-1]['job_id']
    jobs.append(submit('locoReport',f'{PY} {HERE}/render_report.py',dependency=dep,hours=2))
    dep=jobs[-1]['job_id']
    jobs.append(submit('locoAudit',f'{PY} {HERE}/final_audit.py',dependency=dep,hours=2))
    record=dict(utc_time=datetime.now(timezone.utc).isoformat(),branch='STAGE C AFTER STAGE B',
                preregistration='PREREGISTRATION_STAGE_C_AFTER_B.md',tasks=tasks,jobs=jobs)
    p=HERE/'STAGE_C_AFTER_B_DISPATCH.json';assert not p.exists();p.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
