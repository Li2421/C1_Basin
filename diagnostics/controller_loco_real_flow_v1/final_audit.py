"""Final reproducibility and consistency audit, no outcome modification."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    prereg=[]
    for p in sorted(HERE.glob('PREREGISTRATION*.md')):
        receipt=p.with_suffix('.receipt.json')
        assert receipt.exists(),receipt
        x=read(receipt);assert x['sha256']==sha(p)
        prereg.append(dict(file=p.name,sha256=x['sha256'],utc_time=x['utc_time']))
    pre=read(HERE/'CACHE_PREFLIGHT.json')['summary'];inventory=read(HERE/'CACHE_INVENTORY.json')
    post=read(HERE/'POSTFLIGHT.json');labels=read(HERE/'LABEL_AUDIT.json')
    gate=read(HERE/'GATE_RESULTS.json');result=read(HERE/'STAGE_A_RESULTS_FROZEN.json')
    combined=read(HERE/'results.json')
    assert combined['primary']==result['primary'] and combined['rows']==result['rows']
    raw=list((HERE/'raw').glob('*.json'))
    assert pre['total_requested']==inventory['total']==post['total']==86016
    assert pre['genuinely_missing']==inventory['missing']+inventory['cached_unknown']
    assert post['missing']==post['quarantined']==0 and post['workers_complete']
    assert len(raw)==inventory['missing']==post['raw']==post['new_expected']
    assert post['valid']+post['unknown']==post['total']
    assert labels['success']+labels['failure']+labels['unknown']==post['total']
    assert len(gate['rows'])==7 and len(result['rows'])==35
    assert result['endpoint_seeds']==[4,5,6,7]
    features=[read(p) for p in (HERE/'static_features').glob('*_receipt.json')]
    assert len(features)==7 and all(not r['errors'] and r['physical_steps']==0 and not r['jax_enable_x64'] for r in features)
    fits=list((HERE/'fits').glob('fold*_*/complete.json'))
    checkpoints=list((HERE/'fits').glob('fold*_*/seed*_step*.msgpack'))
    predictions=list((HERE/'fits').glob('fold*_*/predictions.npz'))
    assert len(fits)==35 and len(checkpoints)==105 and len(predictions)==35
    assert all(not read(p)['jax_enable_x64'] for p in fits)
    keys=['AGENTS.md','PREREGISTRATION_STAGE_A.md','PREREGISTRATION_RESOURCE_SUPPLEMENT.md',
          'PREREGISTRATION_CONDITIONAL_BRANCHES.md','protocol.json','TRAIN_ETA_MASK.json',
          'STATE_DISJOINT_RECEIPT.json','CACHE_PREFLIGHT.json','COST_GATE.json','EXECUTION_IDENTITY_AUDIT.json',
          'POSTFLIGHT.json','LABEL_AUDIT.json','GATE_RESULTS.json','results.json','REPORT.md','HANDOFF.md']
    artifacts={}
    for k in keys:
        p=HERE/k if k!='AGENTS.md' else HERE.parents[1]/k
        if p.exists():artifacts[k]=dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
    for p in sorted(HERE.glob('*.py')):artifacts[p.name]=dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
    branch={};all_uids={p.stem for p in raw};artifact_folders=[]
    for name in ('stage_b','stage_c'):
        folder=HERE/name
        if (folder/'POSTFLIGHT.json').exists():
            q=read(folder/'POSTFLIGHT.json')
            assert q['missing']==q['quarantined']==0 and q['workers_complete']
            inventory=read(folder/'CACHE_INVENTORY.json');protocol=read(folder/'protocol.json')
            assert q['total']==q['valid']+q['unknown']==inventory['total']
            paths=list((folder/'raw').glob('*.json'));uids={p.stem for p in paths}
            assert len(paths)==len(uids)==q['raw']==inventory['missing']
            assert not all_uids.intersection(uids);all_uids.update(uids)
            authority=read(folder/'PILOT_OR_FULL_AUTHORITY.json')
            assert authority['protocol_sha256']==sha(folder/'protocol.json')
            branch_result=read(folder/'results.json');label=read(folder/'LABEL_AUDIT.json')
            assert label['success']+label['failure']+label['unknown']==q['total']
            receipts=[read(p) for p in (folder/'static_features').glob('*_receipt.json')]
            assert len(receipts)==7 and all(not r['errors'] and r['physical_steps']==0 and not r['jax_enable_x64'] for r in receipts)
            if name=='stage_b':
                fs=list((folder/'fits').glob('fold*_*/complete.json'));cks=list((folder/'fits').glob('fold*_*/seed*_step*.msgpack'))
                assert len(fs)==10 and len(cks)==30
                for p in fs:
                    m=read(p);assert len(m['source_controller_names'])==13 and m['held_controller'] not in m['source_controller_names']
                    assert m['test_labels_read'] is False and m['source_only_normalization'] is True and m['jax_enable_x64'] is False
                details=dict(fit_runs=len(fs),checkpoints=len(cks),primary=branch_result['primary'])
            else:
                preds=list((folder/'predictions').glob('*.json'));assert len(preds)==6
                for p in preds:
                    m=read(p);assert m['test_labels_read'] is False and m['physical_steps']==0 and m['jax_enable_x64'] is False
                assert branch_result['necessary_folds']==gate['necessary_folds']
                assert set(protocol['passing_arms'])==set(read(HERE/'stage_b/results.json')['primary']['passing_arms'])
                assert protocol['stage_b_results_sha256']==sha(HERE/'stage_b/results.json')
                receipt=read(HERE/'PREREGISTRATION_STAGE_C_AFTER_B.receipt.json')
                assert min(p.stat().st_mtime for p in paths)>datetime.fromisoformat(receipt['utc_time']).timestamp()
                details=dict(prediction_files=len(preds),confirmation=branch_result)
            for p in folder.glob('*.json'):artifacts[str(p.relative_to(HERE))]=dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
            branch[name]=dict(status='EXECUTED',postflight=q,**details)
        else:branch[name]=dict(status='NOT EXECUTED')
    assert combined['study_summary']['total_new_rollouts']==len(all_uids)
    training=read(HERE/'TRAINING_PROVENANCE_AUDIT.json');assert training['status']=='PASS' and training['fit_seed_checkpoints']==135
    for r in training['rows']:
        for filename,value in r['checkpoints_sha256'].items():assert sha(filename)==value
    for stage,count in [('AB',114688),('C',8192)]:
        index=HERE/f'RAW_CONTENT_INDEX_STAGE_{stage}.jsonl'
        receipt=read(index.with_suffix('.receipt.json'))
        assert receipt['episodes']==count and receipt['index_sha256']==sha(index)
        artifacts[index.name]=dict(path=str(index),sha256=sha(index),bytes=index.stat().st_size)
    reuse=read(HERE/'HISTORICAL_LABEL_REUSE_COUNTS.json')
    assert reuse['label_slots_reused']==21504 and reuse['valid_label_slots']==21232 and reuse['unknown_label_slots']==272
    prior=list((HERE/'controller_prior').glob('stage*_fold*.json'))
    assert len(prior)==45 and all(not read(p)['target_labels_read'] and not read(p)['test_labels_read'] for p in prior)
    assert read(HERE/'UNKNOWN_BOUND_CHECK.json')['status']=='PASS'
    assert 'interval' in read(HERE/'CONTROLLER_Q_DECOMPOSITION.json')['unknown_policy'] or '[0,1]' in read(HERE/'CONTROLLER_Q_DECOMPOSITION.json')['unknown_policy']
    for name in ['STAGE_A_RESULTS_FROZEN.json','TRAINING_PROVENANCE_AUDIT.json','HISTORICAL_LABEL_REUSE_COUNTS.json',
                 'SOURCE_REGIME_COVERAGE.json','SOURCE_SNAPSHOT_INDEX.json','PLOT_DEPENDENCIES.json',
                 'DECOMPOSITION_UNKNOWN_POLICY_AUDIT.md','UNKNOWN_BOUND_CHECK.json','EXPLORATORY_CONTROLLER_PRIOR_PROTOCOL.md',
                 'RUN_COMMANDS.md','controller_prior/results.json','figures/controller_selection_gains.png','figures/controller_selection_gains.pdf']:
        p=HERE/name;assert p.exists();artifacts[name]=dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
    audit=dict(status='PASS',preregistration=prereg,preflight=pre,postflight=post,label_audit=labels,
        necessary_folds=gate['necessary_folds'],primary=result['primary'],static_feature_receipts=len(features),
        fit_runs=len(fits),model_checkpoints=len(checkpoints),prediction_files=len(predictions),
        raw_files=len(raw),total_raw_files=len(all_uids),branches=branch,
        exploratory_frozen_prior_predictions=len(prior),overall=combined['study_summary']['classification'],
        audit_utc_time=datetime.now(timezone.utc).isoformat())
    (HERE/'FINAL_CONSISTENCY_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    for p in HERE.glob('*receipt.json'):artifacts[p.name]=dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
    (HERE/'ARTIFACT_INDEX.json').write_text(json.dumps(dict(artifacts=artifacts,raw_files=len(raw),total_raw_files=len(all_uids),
        raw_uids_sha256=hashlib.sha256('\n'.join(sorted(p.stem for p in raw)).encode()).hexdigest(),
        all_stage_raw_uids_sha256=hashlib.sha256('\n'.join(sorted(all_uids)).encode()).hexdigest()),indent=2)+'\n')
    print(json.dumps(dict(status='PASS',raw=len(raw),fits=len(fits),checkpoints=len(checkpoints),branch=branch),indent=2))

if __name__=='__main__':main()
