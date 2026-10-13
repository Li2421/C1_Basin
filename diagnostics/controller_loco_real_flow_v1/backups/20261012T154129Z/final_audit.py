"""Final reproducibility and consistency audit, no outcome modification."""
import hashlib
import json
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
    gate=read(HERE/'GATE_RESULTS.json');result=read(HERE/'results.json')
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
    branch={}
    for name in ('stage_b','stage_c'):
        folder=HERE/name
        if (folder/'POSTFLIGHT.json').exists():
            q=read(folder/'POSTFLIGHT.json')
            assert q['missing']==q['quarantined']==0
            branch[name]=dict(status='EXECUTED',postflight=q)
        else:branch[name]=dict(status='NOT EXECUTED')
    audit=dict(status='PASS',preregistration=prereg,preflight=pre,postflight=post,label_audit=labels,
        necessary_folds=gate['necessary_folds'],primary=result['primary'],static_feature_receipts=len(features),
        fit_runs=len(fits),model_checkpoints=len(checkpoints),prediction_files=len(predictions),
        raw_files=len(raw),branches=branch)
    (HERE/'FINAL_CONSISTENCY_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    (HERE/'ARTIFACT_INDEX.json').write_text(json.dumps(dict(artifacts=artifacts,raw_files=len(raw),
        raw_uids_sha256=hashlib.sha256('\n'.join(sorted(p.stem for p in raw)).encode()).hexdigest()),indent=2)+'\n')
    print(json.dumps(dict(status='PASS',raw=len(raw),fits=len(fits),checkpoints=len(checkpoints),branch=branch),indent=2))

if __name__=='__main__':main()
