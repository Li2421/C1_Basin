"""Report the completed goal-directed white-box reference as separate evidence."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
import numpy as np
import collections
import csv
import importlib.util


def main():
    if (HERE/'WHITEBOX_REFERENCE.receipt.json').exists():
        verify_receipt('WHITEBOX_REFERENCE.receipt.json');return
    spec=importlib.util.spec_from_file_location('sealed_baseline_statistics',HERE/'statistics.py')
    stats=importlib.util.module_from_spec(spec);spec.loader.exec_module(stats)
    results=[];sources=[]
    for scene in NAMES:
        stage='distributed_whitebox_'+scene;verify_receipt(stage+'_PREFLIGHT.receipt.json')
        cases=read(HERE/f'{stage}_CASES.json');counts=collections.Counter();states=collections.defaultdict(list)
        for c in cases:
            path=HERE/stage/'raw'/f"{c['rollout_uid']}.json";raw=read(path)
            assert all(raw[k]==c[k] for k in ('rollout_uid','state_uid','controller_uid','eta_uid','seed_key'))
            assert c['method']=='safety' and c['safety_layer']=='distributed' and not np.any(c['eta_a'])
            assert sha(raw['trace_path'])==raw['trace_sha256']
            counts.update({k:int(raw[k]) for k in ('success','collision','timeout','numerical_failure')})
            states[(c['state']['N'],c['state']['index'])].append(raw)
            sources.append(dict(rollout_uid=c['rollout_uid'],raw=str(path),raw_sha256=sha(path),trace_sha256=raw['trace_sha256']))
        assert all(len(rr)==2 for rr in states.values())
        matrix=np.asarray([[[np.mean([r['success'] for r in rr]),
            np.mean([min(1,r['success']+r['numerical_failure']) for r in rr])] for rr in states.values()]]).reshape(1,len(states),2)
        point,draws=stats.bootstrap(matrix,np.asarray(['goal_reference']))
        results.append(dict(scene=scene,controller='名义朝目标控制器（白盒参照）',
            layer='distributed',method='safety',completed=len(cases),state_clusters=len(states),
            **dict(counts),success_lower=float(point[0]),success_upper=float(point[1]),
            lower_ci95=np.quantile(draws[:,0],[.025,.975]).tolist(),
            upper_ci95=np.quantile(draws[:,1],[.025,.975]).tolist()))
    write(HERE/'WHITEBOX_REFERENCE.json',dict(utc=utc(),results=results,
        scope='Separate goal-directed reference; not pooled into the eight black-box controller types',
        bootstrap_iterations=5000,seed_cluster='Both seeds paired within each (N,state) cluster',
        success_definition='Original saved physical outcome; no inference of deadlock from timeout',new_rollouts=0))
    write(HERE/'WHITEBOX_REFERENCE_SOURCES.json',sources)
    with (HERE/'WHITEBOX_REFERENCE.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    lines=['名义朝目标控制器（白盒参照），分布式逐机器人安全投影，η=0。仅作背景，单独报告，不混入八类黑盒控制器的类型等权合并。每个 state 的两个 seed 同簇，沿用原统计函数做5000次 bootstrap；此处只有一个参照控制器，区间只反映该参照在采样 state 上的变异。',
        '| 场景 | 完成请求 | 成功数 | 成功率下界 [95% CI] | 成功率上界 | 碰撞 | 超时 | unknown |',
        '| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |']
    for r in results:
        ci=r['lower_ci95'];lines.append(f"| {NAMES[r['scene']]} | {r['completed']} | {r['success']} | {100*r['success_lower']:.2f}% [{100*ci[0]:.2f}, {100*ci[1]:.2f}] | {100*r['success_upper']:.2f}% | {r['collision']} | {r['timeout']} | {r['numerical_failure']} |")
    lines.append('原始逐局来源、trace SHA和上下界区间见 WHITEBOX_REFERENCE.json 与 WHITEBOX_REFERENCE_SOURCES.json。成功、碰撞、超时和 unknown 直接取已落盘结果；本表不把 timeout 当作死锁，也不新增 rollout。')
    (HERE/'WHITEBOX_REFERENCE.md').write_text('\n\n'.join(lines[:1])+'\n\n'+'\n'.join(lines[1:-1])+'\n\n'+lines[-1]+'\n')
    seal('WHITEBOX_REFERENCE.receipt.json',[HERE/n for n in ['WHITEBOX_REFERENCE.json','WHITEBOX_REFERENCE.csv','WHITEBOX_REFERENCE.md','WHITEBOX_REFERENCE_SOURCES.json','whitebox_reference_report.py','statistics.py']],completed_requests=512,new_rollouts=0,separate_background_reference=True)
    print(json.dumps(results,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
