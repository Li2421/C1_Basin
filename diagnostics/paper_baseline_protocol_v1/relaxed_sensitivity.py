"""Descriptive old corridor/warehouse check, after all fixed tests and Oracle."""
from core import *
from dispatch import execute,BudgetExhausted
from execution_policy import active_methods
from request_factory import case,paper
from formal_tests import selection,options


def compatible_passages(scene,known):
    prior=known.get(scene['name'])
    if prior is None:return False
    config=lambda s:{k:v for k,v in s['config'].items()if k not in('max_steps','scene')}
    return all(db.canonical(a)==db.canonical(b)for a,b in (
        (config(scene),config(prior)),(scene['walls'],prior['walls']),
        (scene.get('bottleneck_config'),prior.get('bottleneck_config'))))


def run():
    output=HERE/'RELAXED_SENSITIVITY_RESULTS.json'
    if output.exists()and read(output).get('finished'):return read(output)
    assert read(HERE/'MAINLINE_READY_FOR_OPTIONAL.json')['complete']
    frozen=read(HERE/'RELAXED_SCENES_FROZEN.json')
    analysis=read(HERE/'ANALYSIS_PLAN_FROZEN.json')
    assert analysis['relaxed_scenes_sha256']==sha(HERE/'RELAXED_SCENES_FROZEN.json'),'HARD_STOP: historical sensitivity scene freeze changed'
    known={e['scene']['name']:e['scene']for e in read(HERE/'inputs/diagnostics/scene_types_oracle_coverage_v3/SCENES.json')['entries']}
    passage_names=set(read(HERE/'PASSAGES.json')['by_scene'])
    requests=[];metadata=[];methods=['ours_main','ours_four']+active_methods()
    for entry in frozen['entries']:
        scene=entry['scene']
        for method in methods:
            chosen=selection('B',method);index=chosen.get('global_index')
            available=chosen.get('available',False)
            reason=None
            if not available:reason='Training selection unavailable; no replacement parameter.'
            if method=='ours_main'and(scene['name']not in passage_names or not compatible_passages(scene,known)):
                available=False;reason='No frozen admission-passage definition for this historical geometry; do not invent or transfer incompatible passages.'
            for state in entry['states']:
                for nominal in controllers()if method.startswith('ours_')else[None]:
                    request=None
                    if available:
                        if nominal is None:request=paper('B',scene,state,method,options(method,index)['options'],phase='test',purpose='old_relaxed_geometry_sensitivity')
                        else:request=case('B',scene,nominal,state,spec=options(method,index)['spec'],phase='test',purpose='old_relaxed_geometry_sensitivity')
                        requests.append(request)
                    metadata.append(dict(benchmark='B',part='historical_sensitivity',variant='global_tuned',
                        source_version=entry['source_version'],scene=scene['name'],category=str(entry['type_id']),state=state,
                        method=method,nominal=nominal['type_code']if nominal else'paper',menu_index=index,
                        horizon_steps=scene['config']['max_steps'],dt=scene['config']['dt'],widen=scene['widen'],
                        query_uid=request['rollout_uid']if request else None,unavailable_reason=reason))
    budget_stop=False
    try:results=execute(requests,'nonmain_old_relaxed_geometry')
    except BudgetExhausted:
        from runner import register
        _,_,done=register(requests,'nonmain_old_relaxed_geometry');results=list(done.values());budget_stop=True
    found={r.get('target_query_uid',r['rollout_uid']):r for r in results}
    rows=[dict(row,result=found.get(row['query_uid']))for row in metadata]
    from reporting import summary,write_csv
    grouped={}
    for row in rows:grouped.setdefault((row['source_version'],row['method'],row['scene']),[]).append(row)
    tables=[dict(source_version=version,method=method,scene=name,**summary(group))for(version,method,name),group in sorted(grouped.items())]
    result=dict(finished=True,complete=all(row['result']is not None for row in rows),budget_stop=budget_stop,
        scenes_sha256=sha(HERE/'RELAXED_SCENES_FROZEN.json'),rows=rows,tables=tables,
        unavailable_logical_rows=sum(row['unavailable_reason']is not None for row in rows),
        interpretation='Historical descriptive sensitivity with the literal v1/v2horizons and states. No re-tuning; both sides use frozen BenchmarkBglobal settings. Missing admission geometry is unavailable, not a failed rollout. No ranking against the primary benchmark.')
    dump(output,result)
    write_csv('RELAXED_SENSITIVITY_TABLE.csv',[{k:json.dumps(v,ensure_ascii=False)if isinstance(v,(dict,list))else v for k,v in row.items()}for row in tables])
    with(HERE/'REPORT.md').open('a')as stream:
        stream.write('\n## 旧通道及仓库敏感性\n\n14个v1/v2历史通道、仓库及放宽档几何使用各自原时限与状态4–11，固定使用基准乙全局训练参数，不再选参。表见RELAXED_SENSITIVITY_TABLE.csv；缺少冻结通道定义的主族条目记不可用，不临时发明通道，也不计失败。原四列与论文方法的结果均保留。本附表不与6倍时限主表直接排名。\n')
    return result
