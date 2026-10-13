"""Delivery check: no original mutation, no rollout, no target-label inputs."""
from util import *
import ast
def main():
    failures=[];checks={}
    def check(x,msg):
        if not x:failures.append(msg)
    for f in HERE.glob('PREREGISTRATION*.receipt.json'):
        r=read(f);check(sha(r['file'])==r['sha256'],'changed preregistration '+str(f))
    for r in read(HERE/'HISTORICAL_ASSET_INDEX.json')['files']:check(sha(r['path'])==r['sha256'],'changed historical asset '+r['path'])
    for r in read(HERE/'SUPPLEMENTAL_DEPENDENCY_INDEX.json')['files']:check(sha(r['path'])==r['sha256'],'changed supplemental dependency '+r['path'])
    eligibility=read(HERE/'STAGE2_ELIGIBILITY.json');check(eligibility['stage1_result_sha256']==sha(HERE/'blend_results.json'),'frozen stage-2 eligibility result changed')
    notes=[]
    db=read(HERE/'DB_READ_ONLY_BASELINE.json');current_db_sha=sha(db['path']);same=current_db_sha==db['sha256']
    if not same:
        investigation=read(HERE/'DB_CONCURRENT_ACTIVITY.json') if (HERE/'DB_CONCURRENT_ACTIVITY.json').exists() else {}
        external=investigation.get('status')=='CONCURRENT_EXTERNAL_ACTIVITY_IDENTIFIED' and investigation.get('current_sha256')==current_db_sha and investigation.get('stable_during_inspection') and investigation.get('study_experiment_rows')==0 and investigation.get('study_source_file_rows')==0
        check(external,'shared DB changed; investigate external writers before claiming no mutation')
        if external:notes.append('Shared DB global invariance NOT VERIFIED: concurrent external Drone3D imports recorded. Study uses immutable historical NPZ/checkpoints, all hash-verified; no study DB write path or provenance rows. See DB_CONCURRENT_ACTIVITY.json and preserved initial FAIL.')
    rep=read(HERE/'STAGE0_REPRODUCTION.json');check(rep['primary_gate_passed'] and rep['tasks']==38,'replay gate failed')
    stat=read(HERE/'STATISTICS_VERIFICATION.json');check(stat['status']=='PASS' and stat['statistics_code_sha256']==sha(HERE/'evaluate.py'),'statistics verification stale/failed')
    planned=read(HERE/'FIT_TASKS_BLEND.json')+read(HERE/'FIT_TASKS_REAL.json');models=[]
    for t in planned:
        folder=HERE/'fits'/task_name(t);meta=read(folder/'complete.json');check(meta['status']=='COMPLETE','missing completed fit')
        check(meta['test_labels_read'] is False and meta['source_only_normalization'] and meta['pca_source_only'],'training leakage metadata')
        check(set(t['sources']).isdisjoint(t['held']),'held controller in source fit')
        pred=np.load(folder/'predictions.npz')['held'];check(pred.shape==(3,len(t['held']),32 if t['bed']=='blend' else 64,16),'prediction shape mismatch');check(np.isfinite(pred).all(),'nonfinite predictions')
        for f,h in zip(meta['checkpoints'],meta['checkpoint_sha256']):
            check(sha(f)==h,'changed checkpoint');models.append(dict(path=f,sha256=h,task=task_name(t)))
        check(meta['jax_enable_x64'] is False,'x64 mismatch')
    for f in (HERE/'logistic').glob('*/complete.json'):
        meta=read(f);check(meta['test_labels_read'] is False and meta['target_labels_used'] is False,'logistic leakage')
    # No new code entry point invokes the simulator or SQL writes. imports are
    # read-only data/model helpers; this check complements the DB content hash.
    for f in HERE.glob('*.py'):
        tree=ast.parse(f.read_text())
        for node in ast.walk(tree):
            if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Attribute):continue
            check(not(node.func.attr=='step' and isinstance(node.func.value,ast.Name) and node.func.value.id=='env'),'simulator invocation '+str(f))
            if node.func.attr in ('execute','executemany') and node.args and isinstance(node.args[0],ast.Constant) and isinstance(node.args[0].value,str):
                check(not node.args[0].value.lstrip().upper().startswith(('INSERT ','UPDATE ','DELETE ','CREATE ','DROP ')),'SQL mutation '+str(f))
    figures=read(HERE/'FIGURE_INDEX.json');check(len(figures)==24,'expected E1–E3/D1–D3 PNG/PDF for two testbeds')
    for r in figures:check(sha(r['path'])==r['sha256'],'changed figure')
    result=read(HERE/'results.json');check(result['status']=='COMPLETE' and result['new_rollouts']==0 and result['shared_db_writes']==0,'final result incomplete')
    check(read(HERE/'blend_results.json')['evidence_level']=='DEVELOPMENT' and read(HERE/'real_results.json')['evidence_level']=='DEVELOPMENT','evidence overclaim')
    check(all((HERE/n).exists() for n in ['REPORT.md','HANDOFF.md','METRICS.csv']),'missing delivery')
    write(HERE/'CHECKPOINT_INDEX.json',models)
    index=[dict(path=str(f),sha256=sha(f)) for f in HERE.glob('*.py')]+[dict(path=str(HERE/n),sha256=sha(HERE/n)) for n in ['REPORT.md','HANDOFF.md','results.json','blend_results.json','real_results.json','METRICS.csv']]
    write(HERE/'ARTIFACT_INDEX.json',index)
    checks.update(historical_assets_verified=len(read(HERE/'HISTORICAL_ASSET_INDEX.json')['files']),shared_db_content_unchanged=same,replayed_tasks=38,new_fit_prediction_tasks=len(planned),new_unique_fit_seeds=len({m['path'] for m in models}),checkpoint_references=len(models),figure_files=len(figures),new_rollouts=0)
    verdict=dict(status=('PASS_WITH_NOTES' if notes else 'PASS') if not failures else 'FAIL',utc=utc(),checks=checks,failures=failures,notes=notes,evidence_level='DEVELOPMENT',code_sha256=sha(__file__))
    write(HERE/'FINAL_AUDIT.json',verdict);assert not failures,failures;print(json.dumps(verdict))
if __name__=='__main__':main()
