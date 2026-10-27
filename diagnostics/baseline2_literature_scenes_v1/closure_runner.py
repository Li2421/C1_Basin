"""Wait for untouched paper workers, then close v1 without new submissions."""
from experiment import *
import subprocess
import time
import traceback

PY='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'
PLOTS='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1-plots/bin/python'


def run(name,*args,python=PY,optional=False):
    print(utc(),name,*args,flush=True)
    with (HERE/'logs'/('closure_'+Path(name).stem+'.log')).open('a') as log:
        result=subprocess.run([python,'-B',str(HERE/name),*args],stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        if optional:
            write(HERE/'FIGURES_GAP.json',dict(utc=utc(),returncode=result.returncode,log=str(HERE/'logs'/('closure_'+Path(name).stem+'.log'))))
            return
        raise RuntimeError((name,args,result.returncode))


def once(receipt,name,*args):
    path=HERE/(receipt+'.receipt.json')
    if path.exists():
        for p,h in read(path)['files'].items():assert sha(p)==h,p
    else:run(name,*args)


def main():
    assert (HERE/'COORDINATOR_STOP').exists()
    last=0
    while True:
        rows=subprocess.check_output(['squeue','-u',os.environ['USER'],'-h','-r','-o','%i|%j']).decode().splitlines()
        ours=[line for line in rows if 'baseline2-' in line]
        assert all('baseline2-bvc-corrected' in line for line in ours),ours
        complete=all((HERE/'paper_bvc_v2/raw'/f"{c['rollout_uid']}.json").exists() for c in read(HERE/'paper_bvc_v2_MISSING_CASES.json'))
        if complete and not ours:break
        if not ours and not complete:raise RuntimeError('Paper BVC workers ended before all records completed; inspect logs; no automatic resubmission authorized')
        if time.monotonic()-last>600:run('closure_status.py');last=time.monotonic()
        time.sleep(20)
    run('db_merge.py')
    once('PAPER_REPRODUCTION','paper_results.py','--final')
    run('scene_documents.py')
    once('MIGRATION_RESULTS','aggregate_results.py')
    once('SAFETY_DIAGNOSTICS','diagnostic_documents.py')
    once('BACKGROUND_RESULTS','background_results.py')
    once('PILOT_RESULTS','pilot_results.py')
    once('REFERENCE_ENDPOINTS','reference_endpoints.py')
    once('SWITCHING_DIAGNOSTICS','switching_diagnostics.py')
    once('TRAINING_MENU_REFERENCE','training_menu_reference.py')
    once('DB_POSTFLIGHT','db_postflight.py')
    once('HANDOFF_FINALIZATION','finalize_handoff.py')
    once('INTEGRITY_END','audit_finish.py')
    once('FINAL_NUMERIC_RESULTS','validate_delivery.py','numeric')
    if not (HERE/'FIGURES.receipt.json').exists() and not (HERE/'FIGURES_GAP.json').exists():run('plot_results.py',python=PLOTS,optional=True)
    if not (HERE/'DELIVERY_SUMMARY.json').exists():run('report_closure.py')
    if not (HERE/'DELIVERY_VALIDATION.receipt.json').exists():run('closure_status.py')
    once('DELIVERY_VALIDATION','validate_delivery.py','delivery')
    # No subsequent write to v1 is authorized once v2 starts. All subprocesses
    # above have terminated; this final receipt is the read-only handoff.
    seal('V1_FREEZE.receipt.json',[HERE/n for n in (
        'DELIVERY_VALIDATION.receipt.json','FINAL_NUMERIC_RESULTS.receipt.json',
        'FINAL_CODE_MANIFEST.json','RAW_EVIDENCE_INDEX.jsonl','REPORT.md','GAPS.md',
        'HANDOFF.md','STATUS.md','FINAL_FLOW_SCOPE.json','DB_POSTFLIGHT.json',
        'INTEGRITY_END.json','INTEGRITY_COMPARISON.json','closure_runner.py')],
        v1_readonly_during_v2=True,no_active_v1_jobs=True,
        next_prompt='docs/prompt_baseline_literature_scenes_distributed_v2_20261020.md',
        all_original_four_scene_work_completed=False,user_directed_three_scene_closure=True)


if __name__=='__main__':
    try:main()
    except Exception:
        write(HERE/'CLOSURE_RUNNER_ERROR.json',dict(utc=utc(),traceback=traceback.format_exc()))
        raise
