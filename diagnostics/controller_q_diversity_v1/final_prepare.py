"""Freeze the source-only choice, then enqueue durable confirmation jobs.

No process or Slurm slot remains reserved while dependencies are pending.
No confirmation outcome can influence selection, allocation or predictions.
"""
import json
import os
import math
import shlex
import subprocess
from datetime import datetime
from .experiment import HERE,MAIN,read,write,freeze,sha,preflight
from .train import KINDS,SEEDS,freeze_models

PY='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'


def command(args):
    return subprocess.check_output(args,cwd=MAIN,text=True).strip().split(';')[0]


def run():
    assert not (HERE/'data/final/manifest.json').exists()
    assert not (HERE/'models_frozen.json').exists(),'Resume from existing frozen stage instead of reselecting'
    variants=('source128','source_diverse64','source_augmented128');scores={};record_hashes={}
    for variant in variants:
        source_review=read(HERE/('source_review.json' if variant=='source128' else f'source_review_{variant}.json'))
        assert source_review['no_final_labels']
        data=HERE/'data'/variant/'dataset.npz';data_hash=sha(data)
        assert data_hash==source_review['dataset_sha256']
        rows=[]
        for kind in KINDS:
            for seed in SEEDS:
                path=HERE/'fits'/variant/kind/f'seed{seed}'/'complete.json';r=read(path)
                assert r['dataset_sha256']==data_hash and not r['final_labels_used']
                assert r['checkpoint_sha256']==sha(path.parent/'checkpoint.msgpack')
                rows.append(r);record_hashes[str(path.relative_to(HERE))]=sha(path)
        scores[variant]={k:sum(r['selected']['score'] for r in rows if r['kind']==k)/len(SEEDS)
                         for k in KINDS if k.startswith('state_') and k!='state_only'}
    candidates=[(value,variant,kind) for variant,values in scores.items() for kind,value in values.items()]
    score,variant,kind=min(candidates,key=lambda row:row[0])
    p=read(HERE/'protocol.json');state_sets={split:{s['state_uid'] for s in p['states'] if s['study_split']==split}
                                          for split in ('train','validation','final')}
    for a,b in [('train','validation'),('train','final'),('validation','final')]:
        assert not(state_sets[a]&state_sets[b])
    assert len({json.dumps(s['physical'],sort_keys=True) for s in p['states']})==len(p['states'])
    now=datetime.now().astimezone().isoformat()
    freeze(HERE/'coverage_fits_ready.json',dict(time=now,variants=list(variants),fits_per_variant=len(KINDS)*len(SEEDS),final_labels_opened=False))
    write(HERE/'source_fits_ready.json',dict(complete=len(KINDS)*len(SEEDS),time=now,final_labels_opened=False,coverage_expansion_pending=False))
    freeze(HERE/'source_review_accepted.json',dict(time=now,selected_variant=variant,primary=kind,score=score,
        source_scores=scores,fit_metadata_hashes=record_hashes,final_labels_seen=False,
        selection='Minimum mean3seed source-held-controller Q MSE +0.5controller contrast MSE; no target fitting or final threshold search',
        amendments='training_plan.json records source-only architecture and fixedFlow-weight changes. Cache-only scene-shift diagnostics are DEVELOPMENT, not final model-selection data.'))
    from .check_analysis import run as checks
    from .check_evaluation import run as evaluation_checks
    checks();evaluation_checks()
    freeze_models(variant=variant)
    from .final_coordinator import allocate
    allocate()
    from .predict import run as predict
    predict()
    from .safety import register
    register()
    for stage in ('confirmation_budget','safety_confirmation'):preflight(stage)
    command([PY,'-m','diagnostics.controller_q_diversity_v1.join_confirmation'])
    folder=HERE/'batches/confirmation_joint'
    command([PY,'-m','shared_rollout_db.plan','--manifest',str(folder/'planned_rollouts.json'),'--output',str(folder/'cache_preflight.json')])
    missing=read(folder/'missing_cases.json');shards=max(12,math.ceil(len(missing)/160))
    dependency=os.environ.get('SLURM_JOB_ID')
    args=['sbatch','--parsable',f'--array=0-{shards-1}%12','--time=01:00:00']
    if dependency:args+=['--dependency=afterok:'+dependency]
    assert missing,'Unexpected fully cached confirmation; evaluate frozen results directly'
    job=command(args+[str(HERE/'worker.sbatch'),'confirmation_joint',str(shards)])
    write(HERE/'confirmation_job.json',dict(job=job,shards=shards,CPU_cap=12,missing=len(missing),submitted=now))
    lines=['set -e','cd '+shlex.quote(str(MAIN)),
        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu']
    for module in ('experiment merge','experiment preflight --stage confirmation_budget',
                   'experiment preflight --stage safety_confirmation','dataset --stage confirmation_budget','evaluate','failure_audit'):
        lines.append(PY+' -u -m diagnostics.controller_q_diversity_v1.'+module)
    analysis=command(['sbatch','--parsable','--dependency=afterok:'+job,'--job-name=cq_confirmation_analysis',
        '--partition=cpu','--cpus-per-task=1','--mem=8G','--time=00:40:00',
        '--output='+str(HERE/'logs/confirmation_analysis_%j.log'),'--wrap','\n'.join(lines)])
    write(HERE/'confirmation_analysis_job.json',dict(job=analysis,dependency=job,requires_human_final_scientific_review=True))
    snapshot=HERE/'code_snapshots/final_frozen';snapshot.mkdir(exist_ok=True)
    code_hashes={}
    for path in list(HERE.glob('*.py'))+list(HERE.glob('*.sbatch')):
        target=snapshot/path.name;target.write_bytes(path.read_bytes());code_hashes[path.name]=sha(path)
    freeze(snapshot/'manifest.json',dict(time=now,code_hashes=code_hashes,training_plan_sha256=sha(HERE/'training_plan.json')))
    print('Frozen confirmation enqueued',dict(primary=kind,variant=variant,rollout_job=job,analysis_job=analysis),flush=True)


if __name__=='__main__':run()
