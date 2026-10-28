"""Explicit old-UID preflight and certified semantic aliases, never fake labels."""
from dispatch import *
from acceleration import accelerated_case

def legacy_case(scene,ctl,si,spec):
    c=case(scene,ctl,si,spec,purpose='v3_frozen_reference')
    old=read(HERE/'inputs/safety_only_planned.json')['exact_configs'][0]['controller_payload']
    c['controller_payload']['execution_sources']=old['execution_sources']
    if 'exact_acceleration'in old:c['controller_payload']['exact_acceleration']=old['exact_acceleration']
    c['controller_uid']=db.uid('ctl',dict(scenario=c['scenario_uid'],payload=c['controller_payload']))
    c['rollout_uid']=db.uid('roll',dict(state=c['state_uid'],eta=c['eta_uid'],controller=c['controller_uid'],seed=c['seed_key']))
    return c

def exact_old(c,stage):
    _,todo,done=register([c],stage)
    return done.get(c['rollout_uid'])

def alias(target,old,record,reason):
    path=HERE/'REUSE_MANIFEST.json';m=read(path)if path.exists()else dict(records={})
    m['records'][target['rollout_uid']]=dict(target_query_uid=target['rollout_uid'],source_rollout_uid=old['rollout_uid'],
        source_trace=record['trace'],reason=reason,certificate=str(HERE/('NO_PASSAGE_EQUIVALENCE.json'if reason=='no_passage_certified'else'EQUIVALENCE.json')))
    dump(path,m)
    return dict(record,logical_query_uid=target['rollout_uid'],reuse_reason=reason)

def run_requests(cases,stage,allow_no_passage=True):
    rows={};missing=[]
    scenes={e['scene']['name']:e['scene']for e in read(HERE/'SCENES.json')['entries']}
    passages=read(HERE/'PASSAGES.json')['by_scene']
    no_pass=(HERE/'NO_PASSAGE_EQUIVALENCE.json').exists()and read(HERE/'NO_PASSAGE_EQUIVALENCE.json')['passed']
    eq=(HERE/'EQUIVALENCE.json').exists()and read(HERE/'EQUIVALENCE.json')['passed']
    for i,c in enumerate(cases):
        sp=c['spec'];eligible=not sp.get('role_reselection')and (not sp.get('one_way')or(allow_no_passage and no_pass and not passages[c['scene']]['passages']))
        if eq and eligible:
            base={k:v for k,v in sp.items()if k!='one_way'}
            old=legacy_case(scenes[c['scene']],c['controller_payload']['nominal'],c['state_index'],base)
            # Register/preflight the exact old identity. Its existence never
            # implies the new identity is an official exact DB cache hit.
            r=exact_old(old,f'{stage}_old_{i:04d}')
            if r is not None:
                rows[c['rollout_uid']]=alias(c,old,r,'no_passage_certified'if sp.get('one_way')else'old_execution_certified');continue
        missing.append(c)
    if missing:
        for r in execute(missing,stage):rows[r['rollout_uid']]=r
    else:dump(HERE/f'{stage}_execution.json',dict(complete=True,new=0,certified_reused=len(rows)))
    # Keep the full planned new requests even if all executions were aliases.
    dump(HERE/f'{stage}_all_requests.json',dict(exact_configs=cases,certified_reused=len(cases)-len(missing)))
    return rows

def no_passage_certificate(make):
    from validation import compare
    path=HERE/'NO_PASSAGE_EQUIVALENCE.json'
    if path.exists():return read(path)['passed']
    replay=read(HERE/'REPLAY_INPUTS.json')['records'];ps=read(HERE/'PASSAGES.json')['by_scene']
    selected=[x for x in replay if not ps[x['case']['scene']]['passages']][:20]
    assert len(selected)==20 and len({x['case']['controller_payload']['nominal']['type_code']for x in selected})==2
    menu=read(HERE/'ONE_WAY_MENU.json')['default'];scenes={e['scene']['name']:e['scene']for e in read(HERE/'SCENES.json')['entries']}
    cases=[make(scenes[x['case']['scene']],x['case']['controller_payload']['nominal'],x['case']['state_index'],dict(x['case']['spec'],one_way=menu),purpose='no_passage_equivalence')for x in selected]
    rows={r['rollout_uid']:r for r in execute(cases,'no_passage20')};checks=[]
    for c,x in zip(cases,selected):
        if c['rollout_uid']in rows:checks.append(compare(x['reference'],rows[c['rollout_uid']]))
    passed=len(checks)==20 and all(x['equal']for x in checks)
    dump(path,dict(passed=passed,pairs=len(checks),checks=checks,policy='If this fails, no no-passage aliasing; all new requests run normally.'))
    return passed
