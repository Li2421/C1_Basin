"""Optional frozen-prefix Oracle@16; stop independently at first success."""
from core import *
from dispatch import execute,BudgetExhausted,status
from execution_policy import active_methods
from request_factory import case,paper
from scene_catalog import cohort
import datetime


def recorded_result(result):
    """Keep full raw evidence once; Oracle frontiers carry hash-bound references."""
    uid=result['rollout_uid'];source=HERE/'raw'/(uid+'.json')
    if not source.exists():
        source=HERE/'result_reference_copies'/(uid+'.json')
        if not source.exists():dump(source,result)
    keys=('success','collision','timeout','numerical_failure','outcome','episode_length','completion_time','truncated_time',
        'horizon_time','wall_seconds','setup_seconds','solver_failure_count','eta','spec','controller','seed','rollout_uid',
        'source_rollout_uid','target_query_uid','source','safety_layer','minimum_swept_clearance','trace','trace_sha256','dictionary',
        'paper_method','paper_options','paper_mechanism','paper_root_diagnostic_only','seed_key','scenario_uid','state_uid',
        'controller_uid','eta_uid','state_index','scene','benchmark','phase','purpose','future_index')
    compact={key:result[key]for key in keys if key in result}
    if'paper_original'in result:
        compact['paper_original']={key:result['paper_original'][key]for key in ('numerical_unknown','outcome','first_error')if key in result['paper_original']}
    compact['full_result_reference']=dict(path=str(source),sha256=sha(source))
    compact['cost_interpretation']='wall_seconds is the underlying saved execution cost, not fresh work for a reused UID. Incremental attempts/jobs are in the reservation ledger and stage queue/execution files.'
    return compact


def summarize(state,scenes):
    """Preserve partial frontiers without relabeling unsearched cases as failures."""
    groups=state['groups'];successful=[g['first_success']for g in groups if g['success']]
    resolved=lambda g:g['success']or len(g['queries'])>=16
    state['summary']=dict(cases=len(groups),successes=len(successful),
        not_found=sum(not g['success']and len(g['queries'])>=16 for g in groups),
        unfinished=sum(not resolved(g)for g in groups),
        first_success_mean=float(np.mean(successful))if successful else None,
        first_success_quartiles=np.quantile(successful,[.25,.5,.75]).tolist()if successful else None,
        all_case_logical_search_mean=float(np.mean([len(g['queries'])for g in groups])),
        numerical_unknown_queries=sum(bool(q['result'].get('numerical_failure'))for g in groups for q in g['queries']),
        native_unhandled_queries=sum(bool(q['result'].get('paper_original',{}).get('numerical_unknown',q['result'].get('numerical_failure',0)))for g in groups for q in g['queries']),
        nonoptimal_solver_returns=sum(q['result'].get('solver_failure_count',0)for g in groups for q in g['queries']),
        note='Finite-pool results only. Unfinished cases are separate from exhausted16. Tuned index17–32need not be in this prefix.')
    from statistic_rules import hierarchical_interval
    physical={};blocks={}
    for group in groups:physical.setdefault((group['scene'],group['state']),[]).append(group)
    for (name,si),values in physical.items():
        if not all(resolved(g)for g in values):continue
        blocks.setdefault(str(scenes[name]['type_id']),{}).setdefault(name,[]).append(sum(int(g['success'])for g in values)/len(values))
    incomplete=len(physical)-sum(len(v)for category in blocks.values()for v in category.values())
    resolved_interval=hierarchical_interval(blocks)
    state['summary']['physical_success_interval']=resolved_interval if incomplete==0 else None
    state['summary']['resolved_physical_blocks_interval']=resolved_interval
    state['summary']['incomplete_physical_blocks']=incomplete
    state['summary']['verified_coverage_lower_bound']=len(successful)/len(groups)
    state['summary']['coverage_upper_bound_if_unfinished_all_succeed']=(len(successful)+state['summary']['unfinished'])/len(groups)
    state['budget_curve']=[dict(k=k,verified_coverage=sum(g['success']and g['first_success']<=k for g in groups),
        complete=all(g['success']and g['first_success']<=k or len(g['queries'])>=k for g in groups))for k in range(1,17)]
    return state


def run_one(method):
    path=HERE/'oracles'/(method+'.json')
    if path.exists():state=read(path)
    else:
        groups=[]
        for scene in cohort('B','test'):
            for index in read(HERE/'SPLITS_FROZEN.json')['benchmark_B']['test_states']:
                for nominal in controllers()if method.startswith('ours_')else[None]:
                    groups.append(dict(scene=scene['name'],state=index,nominal=nominal,success=False,first_success=None,queries=[]))
        state=dict(method=method,complete=False,groups=groups,max_candidates=16,first_success_stop=True,finite_frozen_pool=True)
    if state['complete']:return state
    scenes={s['name']:s for s in cohort('B','test')};menus=read(HERE/'PAPER_MENUS_FROZEN.json')
    pool=(menus['ours'][method[5:]]if method.startswith('ours_')else menus['methods'][method])['initial']
    for index,item in enumerate(pool,1):
        requests=[];owners=[]
        for group in state['groups']:
            if group['success']or len(group['queries'])>=index:continue
            assert len(group['queries'])==index-1
            scene=scenes[group['scene']]
            if group['nominal']is None:
                request=paper('B',scene,group['state'],method,item['options'],phase='test',purpose='optional_frozen_prefix_oracle')
            else:
                request=case('B',scene,group['nominal'],group['state'],spec=item['spec'],phase='test',purpose='optional_frozen_prefix_oracle')
            requests.append(request);owners.append((group,request))
        if not requests:break
        stage=f'oracle16_{method}_k{index:02d}';budget_stop=False
        try:results=execute(requests,stage)
        except BudgetExhausted:
            # Dispatch already merged completed results before raising. Read
            # them back using the original preflighted plan, without a rollout.
            from runner import register
            _,_,done=register(requests,stage);results=list(done.values());budget_stop=True
            state['budget_exhausted']=True
        found={r.get('target_query_uid',r['rollout_uid']):r for r in results}
        for group,request in owners:
            result=found.get(request['rollout_uid'])
            if result is None:
                group['budget_unsearched'if budget_stop else'infrastructure_missing']=request['rollout_uid'];continue
            group.pop('infrastructure_missing',None)
            group.pop('budget_unsearched',None)
            group['queries'].append(dict(index=index,query_uid=request['rollout_uid'],result=recorded_result(result)))
            if result['success']:group.update(success=True,first_success=index)
        dump(path,state)
        if budget_stop or any(g.get('infrastructure_missing')for g in state['groups']):
            dump(path,summarize(state,scenes));return state
    state.update(complete=all(g['success']or len(g['queries'])==16 for g in state['groups']),time=datetime.datetime.now().astimezone().isoformat())
    dump(path,summarize(state,scenes));return state


def run():
    assert(HERE/'SUMMARY_TABLES.json').exists(),'Optional Oracle follows all fixed tests and the main report'
    assert read(HERE/'MAINLINE_READY_FOR_OPTIONAL.json')['complete'],'Optional Oracle cannot precede incomplete mainline fixed tests'
    methods=['ours_main']+active_methods()+['ours_four'];results=[]
    for method in methods:
        if read(HERE/'BUDGET.json')['actual_attempts']>=50000:break
        results.append(run_one(method));status('optional_finite_oracle',len(results),len(methods),'直接按冻结顺序搜索η/论文参数；找到成功即停，不使用生成器。')
    output=dict(optional=True,methods=results,not_run=[m for m in methods if m not in[r['method']for r in results]])
    dump(HERE/'ORACLE16_RESULTS.json',output)
    from oracle_subset_check import run as subset_check
    subset_check()
    return output
