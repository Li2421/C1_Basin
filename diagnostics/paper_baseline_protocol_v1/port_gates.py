"""Author-default solo applicability and training-only exception diagnostics."""
from dispatch import execute,status
from core import *
from execution_policy import active_methods
from freeze_menus import DEFAULTS
from request_factory import paper
from scene_catalog import catalog,cohort
from method_names import NAMES
import datetime


def unhandled(row):
    return bool(row.get('paper_original',{}).get('numerical_unknown',row.get('numerical_failure',0)))


def smoke():
    """Small subset of the required gates; later stages reuse exact UIDs."""
    requests=[]
    for method in active_methods():
        for benchmark in ('A','B'):
            training=cohort(benchmark,'train');selected={}
            for scene in sorted(training,key=lambda s:(s['config']['num_agents'],s['name'])):
                if 'valid_state_indices' in scene and 0 not in scene['valid_state_indices']:continue
                key=scene['factor']if benchmark=='A'else scene['type_id']
                if benchmark=='B'and key not in (1,8,16):continue
                selected.setdefault(key,scene)
            for scene in selected.values():
                for solo in (0,None):
                    requests.append(paper(benchmark,scene,0,method,DEFAULTS[method],solo=solo,
                        phase='applicability'if solo is not None else'pilot',purpose='initial_interface_smoke'))
    rows=execute(requests,'initial_paper_interface_smoke_v'+str(read(HERE/'EXECUTION_FROZEN.json')['revision']))
    failures=[dict(rollout_uid=r['rollout_uid'],method=r['controller'],error=r.get('setup_error'))
              for r in rows if r.get('setup_error')]
    output=dict(planned=len(requests),observed=len(rows),setup_failures=failures,
                numerical_unhandled=sum(unhandled(r)for r in rows),
                no_extra_candidate_or_state=True,subset_of_required_gates=True,
                note='Author-default cases only. Native numerical errors remain visible; high error rate is not a performance-based pruning rule.')
    dump(HERE/'PAPER_INTERFACE_SMOKE.json',output)
    # Resolve packaging/interface failures before hundreds of identical calls.
    structural=[f for f in failures if f['error']['type']in ('ModuleNotFoundError','ImportError','FileNotFoundError')]
    if structural:raise RuntimeError('Paper source/dependency interface needs review before gate expansion: '+repr(structural))
    return output


def run():
    output=HERE/'PORT_GATES.json'
    if output.exists() and read(output).get('complete'):
        return read(output)
    smoke()
    records=[];pilots=[];gaps=[]
    for method in active_methods():
        for benchmark in ('A','B'):
            solo=[];owners=[]
            for scene in catalog(benchmark).values():
                if 'valid_state_indices' in scene and 0 not in scene['valid_state_indices']:continue
                for robot in range(scene['states'][0]['N']):
                    request=paper(benchmark,scene,0,method,DEFAULTS[method],solo=robot,
                                  phase='applicability',purpose='author_default_single_robot_geometry_gate')
                    solo.append(request)
                    owners.append(dict(scene=scene['name'],robot=robot,rollout_uid=request['rollout_uid']))
            results={r['rollout_uid']:r for r in execute(solo,f'gate_v{read(HERE/"EXECUTION_FROZEN.json")["revision"]}_{benchmark}_{method}')}
            for name in catalog(benchmark):
                cases=[o for o in owners if o['scene']==name]
                observed=[results[o['rollout_uid']] for o in cases if o['rollout_uid']in results]
                scene=catalog(benchmark)[name]
                geometry_unavailable='valid_state_indices' in scene and 0 not in scene['valid_state_indices']
                complete=len(observed)==len(cases)

                # Integer comparison implements >=95% without rounding.
                successes=sum(int(r['success'])for r in observed)
                passed=(100*successes>=95*len(cases)) if complete and cases else None
                records.append(dict(benchmark=benchmark,method=method,method_display=NAMES[method],scene=name,
                    planned=len(cases),observed=len(observed),successes=successes,applicable=passed,geometry_state0_unavailable=geometry_unavailable,
                    unhandled=sum(unhandled(r)for r in observed),
                    cases=cases,missing=[o['rollout_uid'] for o in cases if o['rollout_uid']not in results]))
            requests=[paper(benchmark,scene,index,method,DEFAULTS[method],
                            phase='pilot',purpose='training_author_default_exception_gate')
                      for scene in cohort(benchmark,'train') for index in (0,1) if 'valid_state_indices' not in scene or index in scene['valid_state_indices']]
            results=execute(requests,f'pilot_v{read(HERE/"EXECUTION_FROZEN.json")["revision"]}_{benchmark}_{method}')
            bad=sum(unhandled(r)for r in results)
            complete=len(results)==len(requests)
            incomplete_port=10*bad>len(results) if results else None
            pilots.append(dict(benchmark=benchmark,method=method,method_display=NAMES[method],
                planned=len(requests),observed=len(results),complete=complete,unhandled=bad,
                port_problem_not_fully_resolved=incomplete_port,
                continues_regardless_of_exception_rate=True,
                elapsed_episode_seconds=[r['wall_seconds']for r in results],
                raw_uids=[r['rollout_uid']for r in results]))
            if not complete:gaps.append(f'{NAMES[method]}，基准{benchmark}：训练试跑存在基础设施缺失，不当作控制失败。')
            if incomplete_port:gaps.append(f'{NAMES[method]}，基准{benchmark}：默认训练试跑未处理异常率超过10%，按协议继续，标注端口问题未完全解决。')
            partial=dict(time=datetime.datetime.now().astimezone().isoformat(),complete=False,
                         applicability=records,pilots=pilots,gaps=gaps,
                         paused=read(HERE/'METHOD_EXECUTION_POLICY.json')['paused_methods'])
            dump(output,partial)
            status('port_gates',len(pilots),2*len(active_methods()),'单机门只判定适用性，不参与选参；未处理异常率高时保留标记并继续。')
    complete=all(not r['missing']for r in records)and all(p['complete']for p in pilots)
    partial.update(complete=complete)
    dump(output,partial)
    with(HERE/'GAPS.md').open('a')as stream:
        for gap in gaps:stream.write('\n- '+gap+'\n')
    return partial


if __name__=='__main__':run()
