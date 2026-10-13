"""Wait for source expansion without CPU reservations, then fit fair controls."""
import json
import time
from datetime import datetime
from .final_coordinator import HERE,ROOT,PY,run,wait,slurm_module


def main():
    while not (HERE/'extra_rollout_job.json').exists():
        if (HERE/'coordinator_error.json').exists():raise RuntimeError('Source coordinator failed; inspect before proceeding')
        time.sleep(15)
    job=json.loads((HERE/'extra_rollout_job.json').read_text())['job'];wait(job)
    print(run([PY,'-m','diagnostics.controller_q_diversity_v1.extra_controllers','merge']),flush=True)
    print(run([PY,'-m','diagnostics.controller_q_diversity_v1.extra_controllers','preflight']),flush=True)
    pre=json.loads((HERE/'batches/source_extra/exact_identity_preflight.json').read_text())
    assert pre['genuinely_missing']==0,pre
    slurm_module('coverage_dataset')
    nfit=3*len(json.loads((HERE/'training_plan.json').read_text())['models'])
    for variant in ('source_diverse64','source_augmented128'):
        job=run(['sbatch','--parsable',f'--array=0-{nfit-1}%12',str(HERE/'fit.sbatch'),variant]).strip().split(';')[0]
        print('Coverage fit array',variant,job,flush=True);wait(job)
        done=list((HERE/'fits'/variant).glob('*/seed*/complete.json'))
        assert len(done)==nfit,(variant,len(done))
        slurm_module('source_review',['--variant',variant])
    (HERE/'coverage_fits_ready.json').write_text(json.dumps(dict(time=datetime.now().astimezone().isoformat(),
        variants=['source128','source_diverse64','source_augmented128'],fits_per_variant=nfit,
        final_labels_opened=False),indent=2)+'\n')
    print('COVERAGE_SOURCE_REVIEW_READY; select using source-only results before confirmation',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as ex:
        (HERE/'coverage_coordinator_error.json').write_text(json.dumps(dict(time=datetime.now().astimezone().isoformat(),error=repr(ex)),indent=2)+'\n')
        raise
