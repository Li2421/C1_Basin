"""Submit only the preregistered Stage B or C branch after Stage A analysis.

This program reads the completed discovery gate and independent-seed Stage A
selection result. It never reads individual outcome labels or changes the
frozen branch rule. Each stage is a Slurm dependency chain; failed jobs stop
their downstream jobs through afterok dependencies.
"""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PY = '/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'


def read(path):
    return json.loads(Path(path).read_text())


def submit(name, command, *, dependency=None, array=None, gpu=False, hours=1):
    args = ['sbatch', '--parsable', '-J', name, '-p', 'gpu' if gpu else 'cpu',
            '--cpus-per-task=1', '--mem=5G' if gpu else '--mem=4G',
            f'--time={hours:02d}:00:00',
            '-o', str(HERE / f'{name}_%A_%a.log' if array else HERE / f'{name}_%j.log')]
    if gpu:
        args.append('--gres=shard:1')
    if dependency:
        args.append(f'--dependency=afterok:{dependency}')
    if array:
        args.append(f'--array={array}')
    args.extend(['--wrap', command])
    result = subprocess.run(args, cwd=ROOT, check=True, capture_output=True, text=True)
    job_id = result.stdout.strip().split(';')[0]
    assert job_id.isdigit(), result.stdout
    return dict(name=name, job_id=job_id, command=command, dependency=dependency,
                array=array, gpu=gpu, hours=hours)


def main():
    gate = read(HERE / 'GATE_RESULTS.json')
    result = read(HERE / 'results.json')
    assert gate['status'] == 'COMPLETED DISCOVERY ONLY'
    assert result['status'] == 'COMPLETE' and result['primary']['necessary_folds'] == gate['necessary_folds']
    folds = gate['necessary_folds']
    passing = result['primary']['passing_arms']
    assert set(passing).issubset({'R3', 'R4_action', 'R4_openloop', 'R3_R4_action'})
    record = dict(utc_time=datetime.now(timezone.utc).isoformat(),
                  gate_necessary_folds=folds, stage_a_passing_arms=passing,
                  stage_a_primary_status=result['primary']['status'], jobs=[])
    if len(folds) < 2:
        record['branch'] = 'NO ADDITIONAL ROLLOUT: NECESSARY FOLDS < 2'
    elif not passing:
        record['branch'] = 'STAGE B: EXTRA SOURCE CONTROLLERS'
        jobs = record['jobs']
        jobs.append(submit('locoBprep', f'bash {HERE}/stage_b_prepare.sh'))
        dep = jobs[-1]['job_id']
        jobs.append(submit('locoBfeat', f'{PY} {HERE}/stage_b_features.py $SLURM_ARRAY_TASK_ID',
                           dependency=dep, array='0-6%5', gpu=True, hours=2))
        dep = jobs[-1]['job_id']
        jobs.append(submit('locoBraw', f'{PY} {HERE}/stage_b_run.py worker --index $SLURM_ARRAY_TASK_ID --workers 19',
                           dependency=dep, array='0-18%19', hours=8))
        dep = jobs[-1]['job_id']
        jobs.append(submit('locoBmerge', f'bash {HERE}/finalize_stage_b.sh', dependency=dep, hours=3))
        dep = jobs[-1]['job_id']
        jobs.append(submit('locoBfit', f'bash {HERE}/stage_b_train_job.sh',
                           dependency=dep, array=f'0-{5 * len(folds) - 1}%5', gpu=True, hours=2))
        dep = jobs[-1]['job_id']
        jobs.append(submit('locoBanalyze',
                           f'{PY} {HERE}/analyze.py --stage-b && {PY} {HERE}/stage_b_compare.py',
                           dependency=dep, hours=2))
    else:
        record['branch'] = 'STAGE C: INDEPENDENT CONFIRMATION'
        jobs = record['jobs']
        jobs.append(submit('locoCprep', f'bash {HERE}/stage_c_prepare.sh'))
        dep = jobs[-1]['job_id']
        jobs.append(submit('locoCfeat', f'{PY} {HERE}/stage_c_features.py $SLURM_ARRAY_TASK_ID',
                           dependency=dep, array='0-6%5', gpu=True, hours=2))
        dep = jobs[-1]['job_id']
        jobs.append(submit('locoCraw', f'{PY} {HERE}/stage_c_run.py worker --index $SLURM_ARRAY_TASK_ID --workers 19',
                           dependency=dep, array='0-18%19', hours=5))
        dep = jobs[-1]['job_id']
        jobs.append(submit('locoCmerge', f'bash {HERE}/finalize_stage_c.sh', dependency=dep, hours=2))
        dep = jobs[-1]['job_id']
        tasks = [[fold, arm] for fold in folds for arm in ['R0'] + passing]
        stage_c = HERE / 'stage_c'
        stage_c.mkdir(exist_ok=True)
        (stage_c / 'PREDICTION_TASKS.json').write_text(json.dumps(tasks, indent=2) + '\n')
        jobs.append(submit('locoCpredict', f'bash {HERE}/stage_c_predict_job.sh',
                           dependency=dep, array=f'0-{len(tasks) - 1}%5', gpu=True, hours=2))
        dep = jobs[-1]['job_id']
        jobs.append(submit('locoCanalyze', f'{PY} {HERE}/stage_c_analyze.py',
                           dependency=dep, hours=2))
    # Publish the branch decision before any report job can start. The report
    # and consistency audit run only after the selected branch succeeds.
    (HERE / 'CONDITIONAL_DISPATCH.json').write_text(json.dumps(record, indent=2) + '\n')
    dependency = record['jobs'][-1]['job_id'] if record['jobs'] else None
    record['jobs'].append(submit('locoReport', f'{PY} {HERE}/render_report.py',
                                 dependency=dependency, hours=2))
    record['jobs'].append(submit('locoAudit', f'{PY} {HERE}/final_audit.py',
                                 dependency=record['jobs'][-1]['job_id'], hours=2))
    (HERE / 'CONDITIONAL_DISPATCH.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
