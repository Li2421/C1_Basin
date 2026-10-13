"""Run the frozen confirmation under one12-CPU array, releasing idle CPUs."""
import json
import math
from collections import defaultdict
from datetime import datetime,timedelta
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PY='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'


def run(args):
    out=subprocess.run(args,cwd=ROOT,text=True,capture_output=True)
    assert out.returncode==0,str(args)+'\n'+out.stdout[-1500:]+'\n'+out.stderr[-1500:]
    return out.stdout


def wait(job):
    while run(['squeue','-h','-j',str(job)]).strip():time.sleep(15)


def slurm_module(module,args=(),minutes=20):
    assert all(' ' not in str(v) for v in args)
    cmd=f'cd {ROOT} && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu {PY} -u -m diagnostics.controller_q_diversity_v1.{module} '+' '.join(map(str,args))
    job=run(['sbatch','--parsable','--job-name=controller_q_analysis','--partition=cpu','--cpus-per-task=1','--mem=6G',
        f'--time=00:{minutes:02d}:00',f'--output={HERE}/logs/{module}_%j.log','--wrap',cmd]).strip().split(';')[0]
    print('Analysis job',module,job,flush=True);wait(job)
    # This cluster does not enable Slurm accounting storage. Query the live
    # controller immediately after completion instead of relying on sacct.
    state=run(['scontrol','show','job',job,'-o']).strip()
    assert 'JobState=COMPLETED' in state and 'ExitCode=0:0' in state,state


def allocate():
    dest=HERE/'confirmation_allocation.json'
    if dest.exists():return
    times=defaultdict(list)
    for path in (HERE/'batches/source_q1/raw').glob('*.json'):
        r=json.loads(path.read_text());c=r['case']
        if c['controller_name'].startswith('analytic'):times[c['scene'],c['controller_name']].append(r['elapsed_seconds'])
    means={str(k):sum(v)/len(v) for k,v in times.items()}
    costs={s:max(sum(v)/len(v) for (sc,c),v in times.items() if sc==s) for s in ('toy_give_way','ring_exchange')}
    # Use the slowest observed analytic-controller mean per scene and10%
    # margin. Safety is conservatively budgeted as an extra full FF rollout.
    seconds_per_state_both_scenes=1.10*(512+32)*sum(costs.values())/12
    now=datetime.now().astimezone()
    window_path=HERE/'execution_window_resume.json'
    window=json.loads(window_path.read_text()) if window_path.exists() else {}
    deadline_text=window.get('deadline_local','2026-10-09T09:00:00+08:00')
    deadline=datetime.fromisoformat(deadline_text) if deadline_text is not None else None
    seconds=(deadline-now-timedelta(minutes=20)).total_seconds() if deadline else None
    n=min(32,8*math.floor(seconds/seconds_per_state_both_scenes/8)) if seconds is not None else 32
    assert n>=16,('Resource window cannot fit minimum16 untouched states per scene',n,seconds)
    result=dict(states_per_scene=n,created_local=now.isoformat(),final_labels_opened=False,
        decision_inputs='Source runtimes and recorded execution window only; no held-controller outcomes',
        rule='All32 pre-frozen states per scene after explicit task resume without a renewed deadline; otherwise largest multiple of8 fitting resource window. No outcome-dependent allocation.',
        source_mean_seconds=means,cost_seconds_per_scene=costs,
        estimated_wall_seconds_per_state_across_both_scenes=seconds_per_state_both_scenes,
        estimated_rollout_wall_seconds=n*seconds_per_state_both_scenes,
        deadline_local=deadline.isoformat() if deadline else None,
        execution_window_sha256=__import__('hashlib').sha256(window_path.read_bytes()).hexdigest() if window_path.exists() else None,
        training_plan_sha256=__import__('hashlib').sha256((HERE/'training_plan.json').read_bytes()).hexdigest())
    dest.write_text(json.dumps(result,indent=2)+'\n');print('Frozen confirmation allocation',n,'states per scene',flush=True)


def main():
    assert (HERE/'source_review_accepted.json').exists(),'Main agent must review source-only diagnostics first'
    assert (HERE/'source_fits_ready.json').exists()
    if (HERE/'extra_controller_profiles.json').exists():assert (HERE/'coverage_fits_ready.json').exists()
    decision=json.loads((HERE/'source_review_accepted.json').read_text())
    variant=decision.get('selected_variant','source128')
    slurm_module('train',['freeze','--variant',variant])
    allocate();slurm_module('predict')
    assert (HERE/'confirmation_predictions_frozen.json').exists()
    print(run([PY,'-m','diagnostics.controller_q_diversity_v1.safety']),flush=True)
    for stage in ('confirmation_budget','safety_confirmation'):
        print(run([PY,'-m','diagnostics.controller_q_diversity_v1.experiment','preflight','--stage',stage]),flush=True)
    print(run([PY,'-m','diagnostics.controller_q_diversity_v1.join_confirmation']),flush=True)
    folder=HERE/'batches/confirmation_joint'
    print(run([PY,'-m','shared_rollout_db.plan','--manifest',str(folder/'planned_rollouts.json'),'--output',str(folder/'cache_preflight.json')])[-500:],flush=True)
    missing=json.loads((folder/'missing_cases.json').read_text());shards=max(12,math.ceil(len(missing)/160))
    if missing:
        job=run(['sbatch','--parsable',f'--array=0-{shards-1}%12','--time=01:00:00',
            str(HERE/'worker.sbatch'),'confirmation_joint',str(shards)]).strip().split(';')[0]
        (HERE/'confirmation_job.json').write_text(json.dumps(dict(job=job,shards=shards,CPU_cap=12,missing=len(missing)),indent=2)+'\n')
        print('Confirmation array',job,'shards',shards,flush=True);wait(job)
    print(run([PY,'-m','diagnostics.controller_q_diversity_v1.experiment','merge']),flush=True)
    for stage in ('confirmation_budget','safety_confirmation'):
        print(run([PY,'-m','diagnostics.controller_q_diversity_v1.experiment','preflight','--stage',stage]),flush=True)
        pre=json.loads((HERE/'batches'/stage/'exact_identity_preflight.json').read_text())
        assert pre['genuinely_missing']==0,('Unattempted confirmation: preserve and report incomplete',stage,pre['genuinely_missing'])
    slurm_module('dataset',['--stage','confirmation_budget'])
    slurm_module('evaluate')
    assert (HERE/'evaluation_complete.json').exists()
    print('CONFIRMATION_EVALUATION_READY; main agent must inspect failures and consolidate report',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as ex:
        (HERE/'final_coordinator_error.json').write_text(json.dumps(dict(time=datetime.now().astimezone().isoformat(),error=repr(ex)),indent=2)+'\n')
        raise
