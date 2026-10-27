"""Freeze descriptive points after all revised distributed fixed requests land."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
import collections


def main():
    os.sched_setaffinity(0, {read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
    prefix = 'REVISED_DISTRIBUTED_FIXED_POINT_SUMMARY_20261023'
    assert not (HERE/(prefix+'.receipt.json')).exists()
    verify_receipt('REVISED_DISTRIBUTED_FIXED_COMPLETE_20261021.receipt.json')
    verify_receipt('COMPLETED_SCENE_NUMBERS.receipt.json')
    live = read(HERE/'LIVE_REVISED_FIXED_SUCCESS.json')
    selected = [r for r in live['controllers'] if r['layer']=='distributed']
    assert len(selected)==3*(44+44+45+45) and all(r['complete'] for r in selected)
    assert sum(r['landed'] for r in selected)==68064
    snapshot = HERE/(prefix+'_COUNTS.json')
    write(snapshot, dict(utc=utc(), monitor_snapshot_utc=live['utc'], controllers=selected,
                         scope='Completed revised distributed fixed requests; deterministic seed aliases retain logical weight'))
    original = read(HERE/'COMPLETED_SCENE_RESULTS.json')
    rows=[]
    for scene in NAMES:
        baseline = next(r for r in original['results'][scene]['fixed_full']['statistics']
                        if (r['type'],r['layer'],r['method'])==('type_equal','distributed','safety'))
        for method in ('fixed_eta10','fixed_eta13','human_progress'):
            types=collections.defaultdict(list)
            for r in selected:
                if (r['scene'],r['method'])==(scene,method):types[r['type']].append(r)
            assert len(types)==8
            lo=sum(sum(r['success']/r['landed'] for r in g)/len(g) for g in types.values())/8
            hi=sum(sum((r['success']+r['unknown'])/r['landed'] for r in g)/len(g) for g in types.values())/8
            rows.append(dict(scene=scene,layer='distributed',method=method,
                             baseline_success=baseline['success_rate'],success_lower=lo,success_upper=hi,
                             delta_lower=lo-baseline['success_rate'],
                             logical_requests=sum(r['landed'] for g in types.values() for r in g),
                             unknown=sum(r['unknown'] for g in types.values() for r in g),
                             collision=sum(r['collision'] for g in types.values() for r in g)))
    write(HERE/(prefix+'.json'),dict(utc=utc(),rows=rows,
          evidence='Post-hoc revised development evidence; descriptive points only, confidence intervals pending main statistics',
          weighting='Equal type, equal qualified controller within type, equal logical state/seed within controller',
          unknown='Lower counts unknown as unsuccessful; upper counts unknown as successful. This interval is identification uncertainty, not a confidence interval.',
          new_rollouts=0,new_training=0))
    names=dict(lit_narrow_passage='窄通道',lit_crowded_random='二维拥挤随机',lit_circle_five='圆形围场五机',lit_doorway='门口')
    methods=dict(fixed_eta10='全程η10',fixed_eta13='全程η13',human_progress='人定进度切换')
    text='分布式修订固定修正：四场景全部68,064个逻辑请求已完成。采用合格实例、类型等权；这是事后修订的开发证据。表中区间只表示数值unknown带来的识别界，正式配对bootstrap置信区间由主结果统计生成。\n\n|场景|方法|只加安全成功率|加修正成功率识别界|数值unknown|碰撞|逻辑请求|\n|---|---|---:|---:|---:|---:|---:|\n'
    for r in rows:
        rate=f"{100*r['success_lower']:.3f}%" if r['success_lower']==r['success_upper'] else f"{100*r['success_lower']:.3f}–{100*r['success_upper']:.3f}%"
        text+=f"|{names[r['scene']]}|{methods[r['method']]}|{100*r['baseline_success']:.3f}%|{rate}|{r['unknown']}|{r['collision']}|{r['logical_requests']}|\n"
    text+='\n这些是全部主state上的固定方法结果；学习评估使用独立的指定子集，需在匹配子集上比较。controller自身停住实例的增益与安全层全局失败实例的增益将在主统计中分开。timeout不直接等于deadlock。\n'
    (HERE/(prefix+'.md')).write_text(text)
    seal(prefix+'.receipt.json',[Path(__file__),snapshot,HERE/(prefix+'.json'),HERE/(prefix+'.md'),
          HERE/'REVISED_DISTRIBUTED_FIXED_COMPLETE_20261021.receipt.json',HERE/'COMPLETED_SCENE_NUMBERS.receipt.json'],
          descriptive_only=True,new_rollouts=0,new_training=0,does_not_seal_mutating_live_monitor=True)
    print('Completed distributed fixed point summary',len(rows),flush=True)


if __name__=='__main__':main()
