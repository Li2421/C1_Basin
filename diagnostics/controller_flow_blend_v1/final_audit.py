"""Audit actual artifacts and exact identities; never mark planned work done."""
from common import *
import numpy as np
from shared_rollout_db.src.rollout_db import connect
from diagnostics.controller_q_diversity_v1.experiment import case

def main():
    failures=[];checks={};artifact=[]
    def verify(condition,message):
        if not condition:failures.append(message)
    for f in HERE.glob('PREREGISTRATION*.receipt.json'):
        r=read(f);verify(sha(r['file'])==r['sha256'],f'changed preregistration {f}');artifact.append(dict(path=str(f),sha256=sha(f)))
    verify(read(HERE/'ENDPOINT_EQUIVALENCE_AUDIT.json')['status']=='PASS','endpoint audit failed')
    verify(read(HERE/'STATISTICS_VERIFICATION.json')['status']=='PASS','statistics verification failed')
    verify(read(HERE/'STATISTICS_VERIFICATION.json')['statistics_code_sha256']==sha(HERE/'analyze.py'),'statistics code changed after verification')
    for r in read(HERE/'DATA_REUSE_INDEX.json')['files']+read(HERE/'DATA_REUSE_INDEX.json')['static_features']:
        verify(sha(r['path'])==r['sha256'],f'historical source changed {r["path"]}')
    for stage,cap in [('stage_1',5000),('stage_2',65000)]:
        d=HERE/stage
        if not (d/'POSTFLIGHT.json').exists():continue
        p=read(d/'protocol.json');post=read(d/'POSTFLIGHT.json');inventory=read(d/'CACHE_INVENTORY.json');verify(post['missing']==post['quarantined']==0,f'{stage} incomplete DB')
        verify(sha(HERE/'blend_runtime.py')==p['runtime_sha256'],f'{stage} runtime changed')
        verify(inventory['missing']+(832 if stage=='stage_2' else 0)<=cap,f'{stage} cap exceeded')
        states={s['state_uid']:s for s in p['states']};cases=read(d/'CASES.json');known=unknown=0
        with connect(True) as db:
            verify(db.execute('SELECT protocol_hash FROM experiment WHERE experiment_uid=?',(p['experiment_uid'],)).fetchone()[0]==sha(d/'protocol.json'),f'{stage} unregistered experiment')
            for r in cases:
                verify(case(p,states[r['state_uid']],r['controller_name'],r['eta_index'],r['seed'])==r,f'{stage} case mismatch')
                z=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(r['rollout_uid'],)).fetchone();verify(z is not None and not z['conflict_quarantined'],f'{stage} missing/quarantined UID')
                if z:
                    verify(all(z[k]==r[k] for k in ('state_uid','eta_uid','controller_uid','seed_key')),f'{stage} incompatible cache')
                    if z['numerical_failure']:unknown+=1
                    else:known+=1
        index=d/'RAW_CONTENT_INDEX.jsonl'
        with index.open('w') as out:
            for raw in sorted((d/'raw').glob('*.json')):
                q=read(raw);verify(q['protocol_sha256']==sha(d/'protocol.json'),f'{stage} raw protocol mismatch')
                verify(q['jax_enable_x64'] is False,f'{stage} x64 mismatch')
                verify(not(q['success'] and (q['numerical_failure'] or q['collision'])),f'{stage} contradictory success')
                out.write(json.dumps(dict(uid=q['case']['rollout_uid'],path=str(raw),sha256=sha(raw)))+'\n')
        verify(post['raw']==inventory['missing'],f'{stage} duplicated or missing new raw')
        checks[stage]=dict(requests=len(cases),new=inventory['missing'],valid=known,unknown=unknown,cap=cap,raw_index=str(index))
    if (HERE/'stage_2/results.json').exists():
        d=HERE/'stage_2';p=read(d/'protocol.json');tasks=read(d/'FIT_TASKS.json');models=[]
        verify(len(tasks)==60,'expected 60 prediction tasks')
        for task in tasks:
            f=d/'fits'/f"{task['split']}_fold{task['fold']}_{task['coverage']}_{task['arm']}";meta=read(f/'complete.json')
            if meta['status']!='COMPLETE':
                verify(task['arm']=='R5_behavior' and bool(read(d/'probes/POSTFLIGHT.json')['unknown_probe_uids']),'unexpected unexecuted fit');continue
            verify(meta['test_labels_read'] is False and meta['source_only_normalization'],'leakage metadata')
            verify(set(meta['source_controller_names']).isdisjoint(task['held']),'held controller in source')
            pred=np.load(f/'predictions.npz');verify(pred['held'].shape==(3,len(task['held']),32,16),'bad predictions shape')
            verify(np.isfinite(pred['held']).all(),'nonfinite prediction')
            for path,h in zip(meta['checkpoints'],meta['checkpoint_sha256']):verify(sha(path)==h,'checkpoint changed');models.append(dict(path=path,sha256=h))
        probe=read(d/'probes/POSTFLIGHT.json');verify(probe['requests']==832 and probe['missing']==0 and probe['terminal_label_rows_created']==0,'invalid unlabeled probes')
        checks['models']=dict(checkpoint_references=len(models),unique_checkpoint_paths=len({r['path'] for r in models}),prediction_tasks=60)
        write(d/'CHECKPOINT_INDEX.json',models)
    for name in ('REPORT.md','HANDOFF.md','results.json','STAGE_0_RESULTS.json','STAGE_1_GATE.json'):
        verify((HERE/name).exists(),f'missing delivery {name}')
        if (HERE/name).exists():artifact.append(dict(path=str(HERE/name),sha256=sha(HERE/name)))
    r=read(HERE/'results.json');verify(r['status']=='COMPLETE','report not final')
    if not read(HERE/'STAGE_1_GATE.json')['passed']:
        verify(not (HERE/'stage_2/DISPATCH.json').exists(),'executed Stage2 after failed gate')
    write(HERE/'ARTIFACT_INDEX.json',artifact)
    closed_loop=sum(v['new'] for k,v in checks.items() if k.startswith('stage_'))
    short_probes=read(HERE/'stage_2/probes/POSTFLIGHT.json')['requests'] if (HERE/'stage_2/probes/POSTFLIGHT.json').exists() else 0
    audit=dict(status='PASS' if not failures else 'FAIL',utc=utc(),checks=checks,failures=failures,audit_code_sha256=sha(__file__),new_rollouts=closed_loop+short_probes,new_closed_loop_rollouts=closed_loop,new_short_probes=short_probes,resource_policy=read(HERE/'RESOURCE_AUTHORIZATION.json'))
    write(HERE/'FINAL_CONSISTENCY_AUDIT.json',audit);assert not failures,failures;print(json.dumps(audit))
if __name__=='__main__':main()
