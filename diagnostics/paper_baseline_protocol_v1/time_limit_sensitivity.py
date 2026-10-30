"""Prespecified2x horizon check; saved mainline outcomes are never replaced."""
from core import *
from dispatch import execute,BudgetExhausted
import copy,math


def tail_progress(trace):
    if 'route_remaining'in trace and len(trace['route_remaining']):
        distance=np.mean(trace['route_remaining'],axis=1);metric='mean route remaining'
    else:
        distance=np.mean(np.linalg.norm(trace['positions']-trace['goals'][None,:,:],axis=-1),axis=1);metric='mean Euclidean distance to true goal'
    count=min(len(distance),max(3,int(math.ceil(.2*len(distance)))))
    y=-np.asarray(distance[-count:],dtype=np.float64);x=np.arange(count,dtype=np.float64)
    finite=bool(count>=3 and np.isfinite(y).all())
    covariance=float(np.dot(x-x.mean(),y-y.mean()))if finite else None
    return dict(eligible=bool(finite and covariance>0),tail_samples=count,samples=len(distance),
        progress_covariance=covariance,metric=metric,criterion='Strictly positive float64 least-squares progress slope over the final20%of saved samples; finite numeric estimate, not a deadlock certificate.')


def extended_request(parent):
    request=copy.deepcopy(parent);original_rng=request['rng_state_uid'];parent_uid=request['rollout_uid']
    request['scenario_payload']['config']['max_steps']*=2
    request['scenario_uid']=db.uid('scenario',request['scenario_payload'])
    physical={k:request['state'][k]for k in ('positions','velocities','goals')}
    request['state_uid']=db.uid('state',dict(scenario=request['scenario_uid'],physical=physical))
    # Changing a horizon otherwise changes the archived A RNG state identity.
    # Pin the original stream explicitly in the controller's semantic identity.
    request['controller_payload']['nonmain_rng']=dict(policy='Preserve parent stream for authorized doubled-horizon sensitivity',rng_state_uid=original_rng)
    request['controller_uid']=db.uid('ctl',dict(scenario=request['scenario_uid'],payload=request['controller_payload']))
    request['rollout_uid']=db.uid('roll',dict(state=request['state_uid'],eta=request['eta_uid'],controller=request['controller_uid'],seed=request['seed_key']))
    request.update(purpose='nonmain_timeout_positive_tail_2x',phase='test',nonmain_parent_rollout_uid=parent_uid)
    request['state_provenance'].update(nonmain_parent_rollout_uid=parent_uid,rng_state_uid=original_rng)
    assert request['rng_state_uid']==parent['rng_state_uid']and request['spec']==parent['spec']
    return request


def run():
    output=HERE/'TIME_LIMIT_SENSITIVITY.json'
    if output.exists()and read(output).get('finished'):return read(output)
    assert read(HERE/'MAINLINE_READY_FOR_OPTIONAL.json')['complete']
    from reporting import collect,write_csv
    observations=[];eligible={}
    for row in collect():
        if row['benchmark']!='A'or row['part']!='test'or row['variant']in('legacy_geometry','consistent_sensitivity'):continue
        result=row.get('result')
        if result is None or not result.get('timeout'):continue
        trace=result.get('trace');uid=result.get('rollout_uid')or(Path(trace).stem if trace else None)
        record={k:row[k]for k in ('benchmark','part','method','variant','scene','category','state','nominal')}
        record.update(parent_uid=uid,eligibility='unknown',reason='Historical trace unavailable; do not rerun to manufacture eligibility.')
        if trace and Path(trace).is_file():
            with np.load(trace)as saved:diagnostic=tail_progress(saved)
            record.update(eligibility='eligible'if diagnostic['eligible']else'ineligible',diagnostic=diagnostic,reason=None)
            if diagnostic['eligible']:eligible[uid]=None
        observations.append(record)
    # Reuse literal planned requests, preserving author-original adapters and
    # every nominal/checkpoint/eta. No parameter is reconstructed or reselected.
    sources={}
    for pattern in ('test_A_*_planned.json','frozen_ours_consistent_geometry_planned.json','conditional_old_success_time_backfill_planned.json'):
        for path in sorted((HERE/'plans').glob(pattern)):
            for parent in read(path)['exact_configs']:
                uid=parent['rollout_uid']
                if uid in eligible:
                    if uid in sources:
                        assert all(db.canonical(sources[uid][k])==db.canonical(parent[k])for k in ('state','scenario_payload','controller_payload','spec','seed_key','rng_state_uid')),('HARD_STOP: divergent parent execution semantics',uid)
                    sources[uid]=parent
    requests=[extended_request(parent)for uid,parent in sorted(sources.items())]
    parents={request['nonmain_parent_rollout_uid']:request['rollout_uid']for request in requests}
    budget_stop=False
    try:results=execute(requests,'nonmain_timeout_positive_tail_2x')if requests else[]
    except BudgetExhausted:
        from runner import register
        _,_,done=register(requests,'nonmain_timeout_positive_tail_2x');results=list(done.values());budget_stop=True
    found={r.get('target_query_uid',r['rollout_uid']):r for r in results}
    for row in observations:
        row['query_uid']=parents.get(row['parent_uid']);row['result']=found.get(row['query_uid'])
        if row['eligibility']=='eligible'and row['parent_uid']not in sources:row['reason']='Literal parent request unavailable; do not guess execution parameters.'
    groups={}
    for row in observations:groups.setdefault((row['method'],row['variant']),[]).append(row)
    tables=[]
    for(method,variant),rows in sorted(groups.items()):
        known=[r['result']for r in rows if r['result']is not None]
        tables.append(dict(method=method,variant=variant,original_timeout_rows=len(rows),
            eligibility_unknown=sum(r['eligibility']=='unknown'for r in rows),eligible=sum(r['eligibility']=='eligible'for r in rows),
            extended_observed=len(known),extended_successes=sum(r['success']for r in known),
            extended_success_conditional_seconds=float(np.mean([r['completion_time']for r in known if r['success']]))if any(r['success']for r in known)else None))
    result=dict(finished=True,budget_stop=budget_stop,rows=observations,tables=tables,
        unique_eligible_parent_uids=len(eligible),literal_parent_requests=len(sources),
        interpretation='Descriptive2x-horizon result only. Main outcomes, success criteria, parameters, nominal models and physics unchanged; parent RNG streams explicitly preserved. Missing historical eligibility is unknown, not failure.')
    dump(output,result);write_csv('TIME_LIMIT_SENSITIVITY_TABLE.csv',tables)
    with(HERE/'REPORT.md').open('a')as stream:stream.write('\n## 时限敏感性\n\n按预登记的末20%平均进度正斜率筛选超时轨迹，匹配原始请求并保持随机流，以2倍时限补跑一次，主表不改。结果见TIME_LIMIT_SENSITIVITY_TABLE.csv；缺旧轨迹的资格标未知，不补造轨迹或判死锁。\n')
    return result
