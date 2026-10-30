"""Optional frozen-prefix Oracle@16; stop independently at first success."""
from core import *
from dispatch import execute,BudgetExhausted,status
from execution_policy import active_methods
from request_factory import case,paper
from scene_catalog import cohort
import datetime


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
        try:results=execute(requests,f'oracle16_{method}_k{index:02d}')
        except BudgetExhausted:
            state['budget_exhausted']=True;dump(path,state);return state
        found={r.get('target_query_uid',r['rollout_uid']):r for r in results}
        for group,request in owners:
            result=found.get(request['rollout_uid'])
            if result is None:group['infrastructure_missing']=request['rollout_uid'];continue
            group['queries'].append(dict(index=index,query_uid=request['rollout_uid'],result=result))
            if result['success']:group.update(success=True,first_success=index)
        dump(path,state)
        if any(g.get('infrastructure_missing')for g in state['groups']):return state
    state.update(complete=all(g['success']or len(g['queries'])==16 for g in state['groups']),time=datetime.datetime.now().astimezone().isoformat())
    successful=[g['first_success']for g in state['groups']if g['success']]
    state['summary']=dict(cases=len(state['groups']),successes=len(successful),not_found=sum(not g['success']for g in state['groups']),
        first_success_mean=float(np.mean(successful))if successful else None,
        first_success_quartiles=np.quantile(successful,[.25,.5,.75]).tolist()if successful else None,
        all_case_logical_search_mean=float(np.mean([len(g['queries'])for g in state['groups']])),
        numerical_unknown_queries=sum(bool(q['result'].get('numerical_failure'))for g in state['groups']for q in g['queries']),
        note='Failure to find a success is a finite-pool result; tuned index17–32need not be contained in this prefix.')
    from statistic_rules import hierarchical_interval
    physical={}
    for group in state['groups']:
        physical.setdefault((group['scene'],group['state']),[]).append(int(group['success']))
    blocks={}
    for (name,si),values in physical.items():
        blocks.setdefault(str(scenes[name]['type_id']),{}).setdefault(name,[]).append(sum(values)/len(values))
    state['summary']['physical_success_interval']=hierarchical_interval(blocks)
    state['budget_curve']=[dict(k=k,verified_coverage=sum(g['success']and g['first_success']<=k for g in state['groups']),
        complete=all(g['success']or len(g['queries'])>=k for g in state['groups']))for k in range(1,17)]
    dump(path,state);return state


def run():
    assert(HERE/'SUMMARY_TABLES.json').exists(),'Optional Oracle follows all fixed tests and the main report'
    assert read(HERE/'MAINLINE_READY_FOR_OPTIONAL.json')['complete'],'Optional Oracle cannot precede incomplete mainline fixed tests'
    methods=['ours_main']+active_methods()+['ours_four'];results=[]
    for method in methods:
        if read(HERE/'BUDGET.json')['actual_attempts']>=50000:break
        results.append(run_one(method));status('optional_finite_oracle',len(results),len(methods),'直接按冻结顺序搜索η/论文参数；找到成功即停，不使用生成器。')
    output=dict(optional=True,methods=results,not_run=[m for m in methods if m not in[r['method']for r in results]])
    dump(HERE/'ORACLE16_RESULTS.json',output)
    return output
