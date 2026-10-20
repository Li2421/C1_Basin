"""Finish the already-submitted immutable batch; never submit or repeat trials."""
from core import *
import subprocess
from audit_utils import source_inventory
PYTHON='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'
PLOTS='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1-plots/bin/python'
def call(script,*args,python=PYTHON):
    cmd=[python,'-B',str(HERE/script),*args];print('RUN',cmd,flush=True);subprocess.run(cmd,cwd=ROOT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','JAX_PLATFORMS':'cpu'},check=True)
def main():
    assert not (HERE/'DELIVERY.receipt.json').exists()
    assert all((HERE/'main'/f'worker_{i}.json').exists() for i in range(20))
    expected=read(HERE/'main_MISSING_CASES.json');assert len(expected)==896
    assert all((HERE/'main/raw'/f"{c['rollout_uid']}.json").exists() for c in expected)
    for path,h in read(HERE/'ANALYSIS_IMPLEMENTATION.receipt.json')['files'].items():assert sha(path)==h
    call('postflight.py','main');call('analysis.py');call('plots.py',python=PLOTS);call('validate.py');call('audit_extra.py');call('integrity.py','end');call('deliver.py');source_inventory('imports_finalize_existing.json');write(HERE/'FINAL_STATUS.json',dict(utc=utc(),status='COMPLETE',new_students=10,distillation_valid=read(HERE/'DISTILLATION_REPORT.json')['n_valid'],main_records=len(expected),rollout_retries=0,postprocessing='official merge/postflight, analysis, plots, validation, integrity audit and report completed',coordinator_interruption='AUTOMATION_INTERRUPTION.json; original STOPPED status preserved as history; no trial resubmission',delivery_sha256=sha(HERE/'DELIVERY.receipt.json')))
if __name__=='__main__':main()
