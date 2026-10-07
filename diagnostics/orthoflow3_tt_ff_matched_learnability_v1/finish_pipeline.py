"""Run only after source collection completed. No job polling or TEST tuning."""
from __future__ import annotations
import json,re,subprocess,sys
from .design import ROOT,MAIN,read,write,sha
from . import cache,pipeline,dataset


def run_job(script,name,options=()):
    print(dict(event='submit',stage=name,options=list(options)),flush=True)
    with (ROOT/(name+'_submit.txt')).open('w') as f:
        result=subprocess.run(['sbatch','--wait',*options,str(ROOT/script)],cwd=MAIN,stdout=f,stderr=subprocess.STDOUT)
    assert result.returncode==0,(name,'Slurm job failed; inspect completed stderr before any next stage')
    print(dict(event='completed',stage=name),flush=True)


def training_resources():
    # One allocation-time read for coordination, never a completion poll.
    raw=subprocess.check_output(['squeue','-h','-u','zhihan','-t','RUNNING,COMPLETING','-o','%A|%b|%C|%j'],text=True)
    jobs=[]
    for line in raw.splitlines():
        jid,tres,cpu,name=line.split('|');m=re.search(r'shard:(\d+)',tres)
        jobs.append(dict(id=jid,gpu_shards=int(m[1]) if m else 0,cpus=int(cpu),name=name))
    own=min(6,6-sum(j['gpu_shards'] for j in jobs),16-sum(j['cpus'] for j in jobs))
    options=[]
    if own<1:
        assert jobs,'Resource accounting error'
        options=['--dependency=afterany:'+':'.join(sorted({j['id'] for j in jobs}))];own=6
    write(ROOT/'training_resource_admission.json',dict(other_running_jobs=jobs,own_training_concurrency=own,
        combined_gpu_cap=6,combined_cpu_cap=16,dependency=options,one_allocation_time_check=True))
    return [f'--array=0-23%{own}',*options]


def main():
    for b in read(ROOT/'preflight_source/batches.json'):
        assert read(ROOT/'batches'/b['batch']/'postflight_audit.json')['all_attempted_records_in_global_db']
    if not all((ROOT/'contexts'/f'worker{w}.json').exists() for w in range(5)):
        run_job('context.sbatch','context')
    if not (ROOT/'dataset_frozen.json').exists():dataset.materialize()
    if not (ROOT/'models_frozen.json').exists():
        run_job('train.sbatch','training',training_resources())
        run_job('predict.sbatch','freeze_predict')
    assert (ROOT/'test_predictions_frozen.json').exists()
    cache.preflight('test');cache.batches('test')
    print(dict(event='test_preflight_before_rollout',**read(ROOT/'preflight_test/exact_identity_audit.json')['summary']),flush=True)
    pipeline.collect('test')
    dataset.truth()
    subprocess.run([sys.executable,'-m','diagnostics.orthoflow3_tt_ff_matched_learnability_v1.evaluate'],cwd=MAIN,check=True)
    subprocess.run([sys.executable,'-m','diagnostics.orthoflow3_tt_ff_matched_learnability_v1.historical_audit'],cwd=MAIN,check=True)
    write(ROOT/'working_state.json',dict(phase='matched_confirmation_completed',classification=read(ROOT/'final_decision.json')['classification'],
        no_test_tuning=True,generator_unchanged=True,TT_control_law_frozen=True,protocol_sha256=sha(ROOT/'protocol.json')))
    print(dict(event='matched_confirmation_completed'),flush=True)


if __name__=='__main__':main()
