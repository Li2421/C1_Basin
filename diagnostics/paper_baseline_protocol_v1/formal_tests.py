"""Frozen fixed-policy tests; no test outcome can select a new parameter."""
from dispatch import execute,status
from core import *
from request_factory import case,paper
from scene_catalog import catalog,legal_indices
from execution_policy import active_methods
from method_names import NAMES
import datetime


def query_uid(row):return row.get('target_query_uid',row['rollout_uid'])


def options(method,index):
    frozen=read(HERE/'PAPER_MENUS_FROZEN.json')
    value=frozen['ours'][method[5:]]if method.startswith('ours_')else frozen['methods'][method]
    return next(r for r in value['initial']+value['extension']if r['index']==index)


def selection(benchmark,method):
    if method.startswith('ours_'):return read(HERE/'OURS_FIXED_FROZEN.json')['benchmark_B'][method]
    return read(HERE/'PARAMETERS_FROZEN.json')['benchmarks'][benchmark][method]


def altered_engine(request,engine,safety):
    request['controller_payload'].update(engine=engine,safety_layer=safety,dictionary=None)
    request['controller_uid']=db.uid('ctl',dict(scenario=request['scenario_uid'],payload=request['controller_payload']))
    request['rollout_uid']=db.uid('roll',dict(state=request['state_uid'],eta=request['eta_uid'],controller=request['controller_uid'],seed=request['seed_key']))
    return request


def plan(benchmark,variant):
    split=read(HERE/'SPLITS_FROZEN.json')['benchmark_'+benchmark]
    scenes=catalog(benchmark)
    names=(split['test']+split['validation'])if benchmark=='A'else split['test']
    states=split['test_states']if benchmark=='B'or variant=='global_tuned'else list(range(8,16))
    methods=active_methods()+(['ours_main','ours_four']if benchmark=='B'else[])
    if benchmark=='B'and variant=='internal':methods=['internal_safety','internal_nominal','internal_rhr_goal']
    requests=[];rows=[]
    for method in methods:
        if method.startswith('internal_'):chosen=None
        else:chosen=selection(benchmark,method)
        if method.startswith('ours_')and variant not in ('global_tuned','type_tuned','first_point'):continue
        if not method.startswith('ours_')and variant=='first_point':continue
        if chosen is not None and not chosen.get('available')and variant not in ('author_default','author_original','first_point'):continue
        for name in names:
            scene=scenes[name];category=scene['factor']if benchmark=='A'else str(scene['type_id'])
            if variant in ('author_default','author_original','first_point')or method.startswith('internal_'):index=1
            elif variant=='type_tuned':index=chosen['by_type'][category]['index']
            else:index=chosen['global_index']
            for state in legal_indices(scene,states):
                controls=controllers()if method.startswith(('ours_','internal_'))else[None]
                for controller in controls:
                    if controller is None:
                        request=paper(benchmark,scene,state,method,options(method,index)['options'],
                            mechanism=variant!='mechanism_disabled',phase='test',purpose='fixed_'+variant)
                        if variant=='author_original':
                            from original_author_reference import request_variant
                            request=request_variant(request)
                        nominal='paper'
                    else:
                        spec=(options(method,index)['spec']if method.startswith('ours_')else dict(eta=[0.,0.,0.],coefficients={}))
                        if method=='internal_rhr_goal':spec=dict(spec,reference='right_goal')
                        request=case(benchmark,scene,controller,state,spec=spec,phase='test',purpose='fixed_'+variant)
                        if method=='internal_nominal':request=altered_engine(request,'copied_nominal_only','none')
                        nominal=controller['type_code']
                    requests.append(request)
                    rows.append(dict(benchmark=benchmark,method=method,variant=variant,scene=name,
                        part='test'if name in split['test']else'validation',category=category,state=state,nominal=nominal,
                        menu_index=index,query_uid=request['rollout_uid']))
    return requests,rows


def fixed_stage(benchmark,variant):
    requests,metadata=plan(benchmark,variant)
    name=f'test_{benchmark}_{variant}'
    results={query_uid(r):r for r in execute(requests,name)}
    rows=[dict(row,result=results.get(row['query_uid']))for row in metadata]
    output=dict(benchmark=benchmark,variant=variant,planned_logical_rows=len(rows),
                observed_logical_rows=sum(r['result']is not None for r in rows),
                rows=rows,parameters_sha256=sha(HERE/'PARAMETERS_FROZEN.json'),
                analysis_plan_sha256=sha(HERE/'ANALYSIS_PLAN_FROZEN.json'))
    dump(HERE/'test_tables'/(name+'.json'),output)
    return output


def begin():
    analysis=read(HERE/'ANALYSIS_PLAN_FROZEN.json')
    assert analysis['parameters_sha256']==sha(HERE/'PARAMETERS_FROZEN.json')
    assert analysis['ours_fixed_sha256']==sha(HERE/'OURS_FIXED_FROZEN.json')
    assert analysis['execution_release_sha256']==sha(HERE/'EXECUTION_FROZEN.json')
    for name,expected in analysis['analysis_sources_sha256'].items():
        assert sha(HERE/name)==expected,('HARD_STOP: frozen analysis source changed',name)
    path=HERE/'FORMAL_TEST_STARTED.json'
    if path.exists():assert read(path)['analysis_sha256']==sha(HERE/'ANALYSIS_PLAN_FROZEN.json')
    else:dump(path,dict(time=datetime.datetime.now().astimezone().isoformat(),analysis_sha256=sha(HERE/'ANALYSIS_PLAN_FROZEN.json'),
                        parameter_sha256=sha(HERE/'PARAMETERS_FROZEN.json'),no_result_based_reruns=True))


def run_a():
    begin();outputs=[]
    for variant in ('global_tuned','author_default','author_original','type_tuned','mechanism_disabled'):
        outputs.append(fixed_stage('A',variant))
    status('benchmark_A_fixed_tests',1,1,'主检验固定政策测试已保存；验证集单列，默认/类型/关闭机制仅8个状态，不混入16状态主分母。')
    return outputs


def run_b():
    begin();outputs=[]
    for variant in ('global_tuned','author_default','author_original','type_tuned','mechanism_disabled','first_point','internal'):
        outputs.append(fixed_stage('B',variant))
    status('benchmark_B_fixed_tests',1,1,'十七类新初态固定政策与内部参照已保存；此阶段没有逐案例参数搜索。')
    return outputs


if __name__=='__main__':
    run_a();run_b()


def nonmain(module):
    """Implementation gaps in supplements never swallow a hard-stop condition."""
    import importlib,sqlite3,traceback
    try:return importlib.import_module(module).run()
    except Exception as exc:
        critical=('HARD_STOP','DB_AMBIGUOUS','DB_IDENTITY_CONFLICT','IMMUTABLE_STAGE_PLAN','QUARANTINED_RESULT','DB_AMBIGUOUS')
        if isinstance(exc,(PermissionError,sqlite3.Error))or any(word in str(exc)for word in critical):raise
        if(HERE/'PROTECTION_VIOLATION.jsonl').exists():raise
        dump(HERE/('NONMAIN_'+module+'_ERROR.json'),dict(error=repr(exc),traceback=traceback.format_exc()))
        with(HERE/'GAPS.md').open('a')as stream:stream.write('\n非主线'+module+'未完成：'+repr(exc)+'；保留已完成结果，不改变主表。\n')


def nonmain_plots():
    import subprocess
    interpreter=Path('/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1-plots/bin/python')
    if not interpreter.is_file():raise FileNotFoundError(str(interpreter))
    with(HERE/'figures.log').open('ab')as log:
        subprocess.run([str(interpreter),'-B',str(HERE/'plot_frozen_results.py')],stdout=log,stderr=log,check=True,timeout=180)


def nonmain_videos():
    import subprocess
    begin();status('nonmain_prespecified_videos',0,1,'只渲染已有真实轨迹，不补跑、不换更有利初态；完整状态逐帧保留。')
    interpreter=Path('/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1-plots/bin/python')
    with(HERE/'videos.log').open('ab')as log:
        subprocess.run([str(interpreter),'-B',str(HERE/'video_delivery.py')],stdout=log,stderr=log,check=True,timeout=7200)


def main():
    begin()
    from consistent_ours import run as changed_environment
    changed_environment()
    a=run_a();b=run_b()
    from reporting import run as report
    report()
    from time_backfill import run as times
    times()
    report()
    if read(HERE/'MAINLINE_READY_FOR_OPTIONAL.json')['complete']:
        from oracle import run as finite_oracle
        finite_oracle()
        report()
        nonmain('relaxed_sensitivity')
        nonmain('time_limit_sensitivity')
        nonmain('trace_time_diagnostic')
        try:nonmain_plots()
        except Exception as exc:
            if isinstance(exc,PermissionError)or(HERE/'PROTECTION_VIOLATION.jsonl').exists():raise
            dump(HERE/'NONMAIN_FIGURES_ERROR.json',dict(error=repr(exc)))
            with(HERE/'GAPS.md').open('a')as stream:stream.write('\n非主线绘图未完成：'+repr(exc)+'。\n')
        try:nonmain_videos()
        except Exception as exc:
            if isinstance(exc,PermissionError)or'HARD_STOP'in str(exc)or(HERE/'PROTECTION_VIOLATION.jsonl').exists():raise
            dump(HERE/'NONMAIN_VIDEOS_ERROR.json',dict(error=repr(exc)))
            with(HERE/'GAPS.md').open('a')as stream:stream.write('\n非主线完整轨迹视频未完成：'+repr(exc)+'。\n')
    from final_delivery import run as deliver
    deliver()
    return dict(benchmark_A=a,benchmark_B=b)
