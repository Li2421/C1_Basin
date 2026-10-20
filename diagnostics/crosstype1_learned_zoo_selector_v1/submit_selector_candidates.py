"""Run the four recorded development comparisons; never open locked labels."""
from pathlib import Path
import json,hashlib,datetime,subprocess
from resource_guard import require

HERE=Path(__file__).resolve().parent
PYTHON='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'
STRATEGIES=('ridge','ridge_margin','nearest_type','shallow_tree')


def main():
    target=HERE/'INITIAL_SELECTOR_COMPARISONS_SUBMISSION.json';assert not target.exists()
    bundle=HERE/'DEVELOPMENT_BUNDLE.npz';assert bundle.exists()
    for name in ('DEVELOPMENT_BUNDLE.receipt.json','DEVELOPMENT_COMPARATOR_CHOICES.receipt.json','DEVELOPMENT_FEATURES.receipt.json'):
        for path,digest in json.loads((HERE/name).read_text())['files'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    for name in ('STATISTICS_UNIT_TESTS.json','SELECTOR_UNIT_TESTS.json','SELECTOR_CANDIDATE_TESTS.json'):
        assert json.loads((HERE/name).read_text())['passed']
    before=require(cpu=4)
    script=f'''#!/bin/bash
#SBATCH --job-name=crosstype1_selector_comparisons
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-3%4
#SBATCH --time=24:00:00
#SBATCH --output={HERE}/selector_comparisons_%A_%a.log
set -euo pipefail
cd {HERE.parents[1]}
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
strategies=(ridge ridge_margin nearest_type shallow_tree)
strategy="${{strategies[$SLURM_ARRAY_TASK_ID]}}"
{PYTHON} -B {HERE}/develop_complete.py {bundle} "${{strategy}}_v1" "$strategy"
'''
    path=HERE/'selector_comparisons.sbatch';path.write_text(script)
    record=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),queue_before=before,strategies=STRATEGIES,
        input_bundle_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),new_rollouts=0,locked_outcomes_read=False,
        computation='Small source-only selector fits and paired bootstrap on CPU; learned controller training already completed on GPU')
    (HERE/'INITIAL_SELECTOR_COMPARISONS_PREFLIGHT.json').write_text(json.dumps(record,indent=2)+'\n')
    record['job_id']=subprocess.check_output(['sbatch','--parsable',str(path)],text=True).strip()
    target.write_text(json.dumps(record,indent=2)+'\n');print(record['job_id'])


if __name__=='__main__':main()
