"""Missing-only original TT/FF rollouts; one immutable raw seed per case."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.design import FIELD, sha
from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import rollout
from shared_rollout_db.src.cache_writer import append_journal
from shared_rollout_db.src.rollout_db import eta_identity

EXPERIMENT='exp_formal_physical_state_critic_matched_20261008_v1'


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,separators=(',',':'),allow_nan=False)+'\n')
    tmp.replace(path)


def work_items():
    result=[]
    for scene in ('toy_give_way','ring_exchange'):
        for chain in ('TT','FF'):
            for split in ('train','validation'):
                panel=HERE/'matched'/scene/chain/split
                n=len(read(panel/'missing_cases.json'))
                for start in range(0,n,128):
                    result.append(dict(scene=scene,chain=chain,split=split,
                                       start=start,end=min(start+128,n)))
    return result


def freeze_items():
    items=work_items()
    path=HERE/'matched_source_work_items.json'
    if path.exists():assert read(path)==items
    else:write(path,items)
    print(json.dumps(dict(source_shards=len(items),missing_cases=sum(x['end']-x['start'] for x in items))))


def freeze_ring_ff_test():
    assert read(HERE/'ring_ff_models_frozen.json')['test_labels_used'] is False
    assert read(HERE/'ring_ff_test_prediction_audit.json')['test_labels_used'] is False
    panel=HERE/'matched'/'ring_exchange'/'FF'/'test'
    cases=read(panel/'cases.json')
    assert len(cases)==128*17*16
    path=panel/'missing_cases.json'
    if path.exists():assert read(path)==cases
    else:write(path,cases)
    items=[dict(scene='ring_exchange',chain='FF',split='test',
                start=i,end=min(i+128,len(cases))) for i in range(0,len(cases),128)]
    target=HERE/'matched_ring_ff_test_work_items.json'
    if target.exists():assert read(target)==items
    else:write(target,items)
    print(json.dumps(dict(test_shards=len(items),cases=len(cases))))


def freeze_remaining_test():
    assert read(HERE/'all_models_frozen.json')['test_labels_used'] is False
    items=[]
    for scene,chain,panel_name in (('toy_give_way','TT','within_toy_TT'),
                                   ('toy_give_way','FF','within_toy_FF'),
                                   ('ring_exchange','TT','within_ring_TT')):
        assert read(HERE/'fits'/panel_name/'prediction_audit.json')['test_labels_used'] is False
        panel=HERE/'matched'/scene/chain/'test'
        cases=read(panel/'cases.json')
        assert len(cases)==128*17*16
        missing=panel/'missing_cases.json'
        if missing.exists():assert read(missing)==cases
        else:write(missing,cases)
        for start in range(0,len(cases),128):
            items.append(dict(scene=scene,chain=chain,split='test',
                              start=start,end=min(start+128,len(cases))))
    target=HERE/'matched_remaining_test_work_items.json'
    if target.exists():assert read(target)==items
    else:write(target,items)
    print(json.dumps(dict(shards=len(items),cases=sum(x['end']-x['start'] for x in items))))


def run(index,mode='source'):
    assert mode in ('source','ring_ff_test','remaining_test')
    if mode=='ring_ff_test':
        assert read(HERE/'ring_ff_models_frozen.json')['test_labels_used'] is False
        assert read(HERE/'ring_ff_test_prediction_audit.json')['test_labels_used'] is False
    if mode=='remaining_test':
        assert read(HERE/'all_models_frozen.json')['test_labels_used'] is False
    items=read(HERE/('matched_source_work_items.json' if mode=='source' else
                     'matched_ring_ff_test_work_items.json' if mode=='ring_ff_test' else
                     'matched_remaining_test_work_items.json'))
    item=items[index]
    scene,chain,split=item['scene'],item['chain'],item['split']
    panel=HERE/'matched'/scene/chain/split
    cases=read(panel/'missing_cases.json')[item['start']:item['end']]
    protocol=read(HERE/'matched_protocol.json')
    profile=protocol['controllers'][scene][chain]
    states={s['state_uid']:s for s in protocol['states'] if s['scenario']==scene}
    from validation_stage2.run_rollouts import make_runtime
    import jax
    runtime=make_runtime(scene)
    assert sha(runtime.checkpoint)==profile['payload']['flow_checkpoint_sha256']
    assert bool(jax.config.jax_enable_x64)==profile['payload']['jax_enable_x64']
    assert runtime.steps==10
    for rel,digest in profile['payload']['action_code_hashes'].items():
        assert sha(FIELD/rel)==digest
    if chain=='FF':assert sha(profile['payload']['execution_callable'])==profile['payload']['execution_callable_sha256']
    completed=[];new=old=0
    for c in cases:
        assert c['scene']==scene and c['chain']==chain and c['split']==split
        assert c['controller_uid']==profile['controller_uid']
        assert eta_identity(c['eta'])[0]==c['eta_uid']
        s=states[c['state_uid']]
        path=panel/'raw'/(c['rollout_uid']+'.json')
        if path.exists():
            record=read(path)
            assert record['case']==c
            old+=1
        else:
            payload=dict(id=c['rollout_uid'],state=s,eta=c['eta'],
                         eta_index=c['eta_index'],seed=c['seed'],phase='formal_q_matched_v1')
            raw=rollout(runtime,payload,chain)
            assert raw['state_uid']==s['uid'] and raw['seed']==c['seed']
            if chain=='FF':assert raw['controller_uid']==c['controller_uid']
            numerical=int(raw['error'] is not None)
            success=int(raw['success'])
            collision=int(raw['collision'])
            assert not(success and (numerical or collision))
            record=dict(schema='formal_q_matched_seed_v1',experiment_uid=EXPERIMENT,
                        case=c,success=success,deadlock=int(raw['termination']=='deadlock'),
                        timeout=int(raw['termination']=='timeout'),collision=collision,
                        numerical_failure=numerical,episode_length=raw['steps'],
                        outcome=raw['termination'],raw=raw,
                        protocol_sha256=sha(HERE/'matched_protocol.json'))
            write(path,record);new+=1
        completed.append(record)
    if completed:
        append_journal(completed,EXPERIMENT,f'{mode}_{index:04d}')
    receipt=dict(shard=index,cases=len(cases),new=new,recovered=old,
                 complete=True,mode=mode,protocol_sha256=sha(HERE/'matched_protocol.json'))
    write(HERE/'matched'/'receipts'/f'{mode}_{index:04d}.json',receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('freeze_items','freeze_ring_ff_test','freeze_remaining_test','run'))
    parser.add_argument('--index',type=int)
    parser.add_argument('--mode',choices=('source','ring_ff_test','remaining_test'),default='source')
    args=parser.parse_args()
    if args.action=='freeze_items':freeze_items()
    elif args.action=='freeze_ring_ff_test':freeze_ring_ff_test()
    elif args.action=='freeze_remaining_test':freeze_remaining_test()
    else:run(args.index,args.mode)
