"""Frozen learned methods/internal controls on changed physical geometry."""
from core import *
from dispatch import execute,status
from learned_setup import rows as learned_rows
from scene_catalog import catalog,frozen_eta,legal_indices
from request_factory import case
import datetime

METHODS = ['学习选修正参数','只加安全投影','不加安全投影','经典目标右侧规则']


def engine(request,name):
    if name not in ('不加安全投影','经典目标右侧规则'):
        return request
    request['controller_payload'].update(engine='copied_nominal_only'if name=='不加安全投影'else'copied_goal_rhr',
                                         safety_layer='none'if name=='不加安全投影'else'distributed',dictionary=None)
    request['controller_uid']=db.uid('ctl',dict(scenario=request['scenario_uid'],payload=request['controller_payload']))
    request['rollout_uid']=db.uid('roll',dict(state=request['state_uid'],eta=request['eta_uid'],controller=request['controller_uid'],seed=request['seed_key']))
    return request


def run():
    assert(HERE/'ANALYSIS_PLAN_FROZEN.json').exists(),'Frozen-method environment tests follow all parameter/analysis freezes'
    path=HERE/'CONSISTENT_OURS_RESULTS.json'
    if path.exists()and read(path).get('complete'):return read(path)
    variants=read(HERE/'CONSISTENT_BOX_SCENES.json');split=read(HERE/'SPLITS_FROZEN.json')['benchmark_A']
    models=learned_rows();requests=[];owners=[]
    for name in split['validation']+split['test']:
        if name not in variants['scene_records']:continue
        scene=variants['scene_records'][name]
        # Both consistent and legacy geometries remain reportable. Even when
        # the90%rule retains legacy primary, consistent runs are sensitivity.
        for state in legal_indices(scene,range(8,24)):
            for model in models:
                for method in METHODS:
                    eta=frozen_eta(name)[1]if method=='学习选修正参数'else[0.,0.,0.]
                    request=case('A',scene,model,state,spec=dict(eta=eta,coefficients={}),
                                 phase='test',purpose='frozen_method_physical_consistency')
                    request=engine(request,method);requests.append(request)
                    owners.append(dict(benchmark='A',scene=name,category=scene['factor'],state=state,nominal=model['controller'],
                        method=method,variant='consistent_geometry',part='test'if name in split['test']else'validation',query_uid=request['rollout_uid']))
    results={r.get('target_query_uid',r['rollout_uid']):r for r in execute(requests,'frozen_ours_consistent_geometry')}
    records=[dict(o,result=results.get(o['query_uid']))for o in owners]
    output=dict(complete=all(r['result']is not None for r in records),planned=len(records),
                records=records,original_frozen_labels_remain_unchanged=True,
                justification='Changed physical execution semantics prevent label reuse. Internal controls also require the same changed environment for fair primary/mechanism comparisons; this is not optional historical timing backfill.',
                frozen_parameters_sha256=sha(HERE/'OURS_FIXED_FROZEN.json'),time=datetime.datetime.now().astimezone().isoformat())
    dump(path,output);status('frozen_learned_physical_consistency',sum(r['result']is not None for r in records),len(records),
                           '全部冻结学习规划器与内部参照使用同一一致几何；不重训，不依测试表现选择参数。')
    return output
