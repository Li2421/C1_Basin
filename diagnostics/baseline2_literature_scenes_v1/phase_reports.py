from experiment import *
import collections
import csv
import numpy as np
import subprocess

METHODS={'impc_dr':'无限时域模型预测控制与死锁解除（IMPC-DR）','buffered_voronoi_right':'缓冲Voronoi单元与右手绕行点规则','circle_centralized':'集中式控制障碍函数控制器','circle_df':'去中心化全责控制障碍函数控制器（DF）','circle_dr':'去中心化互惠控制障碍函数控制器（DR）','doorway_off':'模型预测控制与控制障碍函数，活性关闭','doorway_on':'模型预测控制与控制障碍函数，活性开启'}

def records(stage):
    return [read(p) for p in (HERE/stage/'raw').glob('*.json')]

def paper():
    from paper_results import main
    return main()

def capability(stage):
    rows=records(stage);groups=collections.defaultdict(list)
    for x in rows:groups[(x['case']['state']['scene'],x['case']['controller_payload']['nominal']['controller'])].append(x)
    expected=collections.Counter((c['state']['scene'],c['controller_payload']['nominal']['controller']) for c in read(HERE/f'{stage}_CASES.json'));out=[]
    for (scene,ctl),planned in sorted(expected.items()):
        rr=groups[(scene,ctl)];success=sum(x['success'] for x in rr);unknown=sum(x['numerical_failure'] for x in rr);rate=success/planned
        out.append(dict(scene=scene,controller=ctl,planned=planned,completed=len(rr),success=success,unknown=unknown,rate_lower=rate,rate_upper=(success+unknown)/planned,qualified=len(rr)==planned and rate>=.95))
    write(HERE/f'{stage}_QUALIFICATION.json',out)
    return out

def status():
    info=dict(utc=utc(),stages=[])
    for p in HERE.glob('*_CASES.json'):
        if p.name.endswith('_MISSING_CASES.json'):continue
        stage=p.name[:-len('_CASES.json')];raw={q.stem for q in (HERE/stage/'raw').glob('*.json')};planned=len(read(p));missing=HERE/f'{stage}_MISSING_CASES.json'
        todo={c['rollout_uid'] for c in read(missing)} if missing.exists() else set();not_executed=planned-len(todo);completed=len(raw&todo);pf=HERE/f'{stage}_CACHE_PREFLIGHT.json';summary=read(pf)['summary'] if pf.exists() else {};reused=summary.get('exact_reusable',0)+summary.get('partial_reusable',0);preserved=not_executed-reused
        info['stages'].append(dict(stage=stage,planned=planned,completed_raw=len(raw),preflight_reused=reused,preserved_unknown_or_other_cached_identity=preserved,missing_requested=len(todo),missing_completed=completed,logical_completed=not_executed+completed,pending_missing=len(todo-raw),obsolete_implementation_branch=stage=='capability_lit_narrow_passage'))
    info['squeue']=subprocess.check_output(['squeue','-u',os.environ['USER'],'-o','%.18i %.30j %.8T %.10M %.6C %.20b']).decode()
    write(HERE/'STATUS.json',info)
    (HERE/'STATUS.md').write_text('# 当前阶段\n\n'+info['utc']+'。论文版与流程版按依赖推进。旧8条第0步实现错误分支已废弃，不计科学结果、不继续提交。已完成轨迹的序列化中断只恢复journal，不重跑。logical完成包括preflight精确复用；raw计数较小不代表复用案例缺失。数值unknown均保留。\n\n'+'\n'.join(f"- {r['stage']}：{'废弃实现分支；' if r['obsolete_implementation_branch'] else ''}逻辑完成{r['logical_completed']}/{r['planned']}，其中可用精确复用{r['preflight_reused']}、保留unknown/其他已有身份{r['preserved_unknown_or_other_cached_identity']}、本批落盘{r['missing_completed']}，待计算{r['pending_missing']}。" for r in info['stages'])+'\n\n```text\n'+info['squeue']+'```\n')
    print(canonical(info['stages']))

if __name__=='__main__':
    if sys.argv[1]=='paper':paper()
    elif sys.argv[1]=='capability':capability(sys.argv[2])
    else:status()
