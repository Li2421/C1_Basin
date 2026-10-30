"""Symmetric scene-equal two-round selection on frozen training cohorts only."""
from dispatch import execute,status
from core import *
from execution_policy import active_methods
from request_factory import case,paper
from scene_catalog import cohort,legal_indices
from statistic_rules import score,first_round,final_choice,saturation,expected_max_curve
from method_names import NAMES
import datetime


def query_uid(row):return row.get('target_query_uid',row['rollout_uid'])


def menu(method):
    frozen=read(HERE/'PAPER_MENUS_FROZEN.json')
    if method.startswith('ours_'):
        data=frozen['ours'][method[5:]]
    else:data=frozen['methods'][method]
    return {r['index']:r for r in data['initial']+data['extension']}


def requests_for(benchmark,method,indices,states):
    options=menu(method);requests=[];records=[]
    assert benchmark=='B' or not method.startswith('ours_')
    for index in indices:
        for scene in cohort(benchmark,'train'):
            for state in legal_indices(scene,states):
                controls=controllers()if method.startswith('ours_')else[None]
                for controller in controls:
                    if controller is None:
                        request=paper(benchmark,scene,state,method,options[index]['options'],
                                      phase='training',purpose='symmetric_two_round_tuning')
                        nominal='paper'
                    else:
                        request=case(benchmark,scene,controller,state,spec=options[index]['spec'],
                                     phase='training',purpose='symmetric_two_round_tuning')
                        nominal=controller['type_code']
                    requests.append(request)
                    records.append(dict(index=index,scene=scene['name'],state=state,nominal=nominal,
                                        category=scene['factor']if benchmark=='A'else str(scene['type_id']),
                                        query_uid=request['rollout_uid']))
    return requests,records


def observed_records(metadata,results):
    by_uid={query_uid(r):r for r in results}
    return [dict(row,result=by_uid.get(row['query_uid']))for row in metadata]


def scores_for(records,indices):
    result={}
    for index in indices:
        rows=[r for r in records if r['index']==index]
        expected={(r['scene'],r['state'],r['nominal'])for r in rows}
        result[index]=score(index,rows,expected)
    return result


def immutable(path,value):
    if path.exists():
        assert digest(read(path))==digest(value),('Frozen selection changed',path)
    else:dump(path,value)


def run_one(benchmark,method):
    path=HERE/'tuning'/f'{benchmark}_{method}.json'
    if path.exists()and read(path).get('complete'):
        prior=read(path)
        assert prior['execution_release_sha256']==sha(HERE/'EXECUTION_FROZEN.json'),'Training results belong to an older numerical release; preserve and redo affected method training.'
        assert prior['menu_sha256']==sha(HERE/'PAPER_MENUS_FROZEN.json')
        return prior
    # Release suffix prevents a repaired pre-test port from reusing an old plan.
    revision=read(HERE/'EXECUTION_FROZEN.json')['revision']
    prefix=f'tune_{benchmark}_{method}_v{revision:03d}'
    indices=list(range(1,17));requests,metadata=requests_for(benchmark,method,indices,(0,1))
    rows=observed_records(metadata,execute(requests,prefix+'_r1'))
    dump(HERE/'tuning'/(prefix+'_r1_records.json'),dict(records=rows))
    if any(r['result']is None for r in rows):
        result=dict(complete=False,available=False,benchmark=benchmark,method=method,
                    reason='Incomplete first round after bounded infrastructure retries; no selection by incomplete cohort.',records=rows)
        dump(path,result);return result
    first=scores_for(rows,indices)
    curve=saturation([first[i].success for i in indices])
    if curve['expand']:
        extension=list(range(17,33));requests,metadata=requests_for(benchmark,method,extension,(0,1))
        extra=observed_records(metadata,execute(requests,prefix+'_r1_extension'))
        rows+=extra;indices+=extension
        dump(HERE/'tuning'/(prefix+'_r1_extension_records.json'),dict(records=extra))
        if any(r['result']is None for r in extra):
            result=dict(complete=False,available=False,benchmark=benchmark,method=method,
                        reason='Required saturation expansion incomplete; no silent fallback to16.',saturation=curve,records=rows)
            dump(path,result);return result
        first=scores_for(rows,indices)
    shortlist=first_round(list(first.values()))
    selected=[s.index for s in shortlist]
    requests,metadata=requests_for(benchmark,method,selected,range(2,8))
    second=observed_records(metadata,execute(requests,prefix+'_r2'))
    dump(HERE/'tuning'/(prefix+'_r2_records.json'),dict(records=second))
    all_rows=[r for r in rows if r['index']in selected]+second
    if any(r['result']is None for r in second):
        result=dict(complete=False,available=False,benchmark=benchmark,method=method,
                    reason='Incomplete second round; final selection unavailable.',shortlist=selected,saturation=curve,records=all_rows)
        dump(path,result);return result
    final=scores_for(all_rows,selected)
    chosen=final_choice(shortlist,final)
    categories=sorted({r['category']for r in all_rows})
    wanted=categories if benchmark=='A'else[str(i)for i in range(1,18)]
    by_type={}
    for category in wanted:
        typed=[r for r in all_rows if r['category']==category]
        if not typed:
            by_type[category]=dict(index=chosen.index,fallback_global=True,
                reason='No same-type training geometry; use the frozen global training choice without looking at test outcomes.')
            continue
        typed_scores=scores_for(typed,selected)
        winner=min(typed_scores.values(),key=lambda s:s.key())
        by_type[category]=dict(index=winner.index,fallback_global=False,score=winner.json(),
                               ranking=[s.json()for s in sorted(typed_scores.values(),key=lambda s:s.key())])
    result=dict(complete=True,available=True,benchmark=benchmark,method=method,method_display=NAMES[method],
        first_round=[s.json()for s in sorted(first.values(),key=lambda s:s.key())],
        shortlist=selected,final_ranking=[s.json()for s in sorted(final.values(),key=lambda s:s.key())],
        global_index=chosen.index,by_type=by_type,saturation=curve,
        full_curve=expected_max_curve([s.success for s in first.values()]),
        tied_before_menu_index=[s.index for s in final.values()if s.key()[:-1]==chosen.key()[:-1]],
        execution_release_sha256=sha(HERE/'EXECUTION_FROZEN.json'),menu_sha256=sha(HERE/'PAPER_MENUS_FROZEN.json'),
        records_file=str((HERE/'tuning'/(prefix+'_r2_records.json')).relative_to(HERE)),
        limitation='Top4only: settings weak on states0–1 but strong on2–7 can be missed. Same rule for both sides.')
    dump(path,result)
    return result


def run():
    all_results={}
    for benchmark in ('A','B'):
        methods=active_methods()+(['ours_main','ours_four']if benchmark=='B'else[])
        for method in methods:
            all_results[(benchmark,method)]=run_one(benchmark,method)
            dump(HERE/'TUNING_PROGRESS.json',dict(time=datetime.datetime.now().astimezone().isoformat(),
                results={b+'::'+m:r for (b,m),r in all_results.items()}))
            status('training_parameter_selection',len(all_results),2*len(active_methods())+2,
                   '仅训练场景选参；完整第一轮及条件扩充之后，只有前4组进入第二轮。')
        if benchmark=='A':
            immutable(HERE/'PARAMETERS_A_FROZEN.json',dict(
                methods={m:all_results[(benchmark,m)]for m in active_methods()},
                execution_release_sha256=sha(HERE/'EXECUTION_FROZEN.json')))
    papers={b:{m:all_results[(b,m)]for m in active_methods()}for b in ('A','B')}
    ours={m:all_results[('B',m)]for m in ('ours_main','ours_four')}
    immutable(HERE/'PARAMETERS_FROZEN.json',dict(benchmarks=papers,menus_sha256=sha(HERE/'PAPER_MENUS_FROZEN.json'),
        withdrawn_methods=read(HERE/'METHOD_EXECUTION_POLICY.json')['withdrawn_methods'],
        execution_release_sha256=sha(HERE/'EXECUTION_FROZEN.json')))
    immutable(HERE/'OURS_FIXED_FROZEN.json',dict(benchmark_B=ours,benchmark_A='Original learned selections unchanged.',
        menus_sha256=sha(HERE/'PAPER_MENUS_FROZEN.json'),execution_release_sha256=sha(HERE/'EXECUTION_FROZEN.json')))
    dump(HERE/'TUNING_CURVES.json',dict(results={b+'::'+m:{k:r[k]for k in ('complete','saturation','full_curve')if k in r}
                                             for (b,m),r in all_results.items()},
        estimator='Exact expectation of the best scene-equal success in a uniformly random subset of k distinct settings.'))
    return all_results


if __name__=='__main__':run()


main=run
