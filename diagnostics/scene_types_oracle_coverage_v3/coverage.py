"""Descriptive finite-pool coverage, with pending cases kept separate from failures."""
from core import *
import csv

FAMILY_NAMES=[
 '原三列族（朝目标、横向放大、互斥分离）',
 '活跃约束切向投影族',
 '目标右侧旋转＋活跃约束切向投影族',
 '互斥分离＋目标右侧旋转＋活跃约束切向投影族',
 '朝目标＋互斥分离＋目标右侧旋转＋局部净空梯度族',
 '朝目标＋互斥分离＋目标右侧旋转＋局部净空梯度＋让行者后退族',
 '朝目标＋局部净空梯度族']

def scene_title(e):
    s=e['scene'];t=e['type_id'];n=s['config']['num_agents'];knob=s['knob']
    desc=f'{n} 台'
    if t in (2,8,10,11,12):desc=f'每队 {n//2} 台'
    if t==7:desc=f'{int(knob)} 块障碍、{n} 台'
    if t==14:desc=f'通道宽 {knob:g} 米、{n} 台'
    if t==16:desc=f'巷宽 0.4 米、{n} 台'
    if t==17:desc=f'{n} 台、目标间距 0.44 米'
    if t==6:desc=f'{n} 台、半径 {knob:g} 米'
    return e['title']+'，'+desc

def distribution(rows):
    vals=[r['first_success']for r in rows if r['success']]
    return dict(median=float(np.median(vals))if vals else None,
        quartiles=np.quantile(vals,[.25,.75]).tolist()if vals else None,
        found=len(vals),not_found_in64=sum(r['searched']==64 and not r['success']for r in rows),
        incomplete=sum(not r['success']and r['searched']<64 and not r.get('skipped')for r in rows),
        skipped=sum(bool(r.get('skipped'))for r in rows),
        average_queries_success=float(np.mean(vals))if vals else None,
        average_queries_completed=float(np.mean([r['searched']for r in rows if r['success']or r['searched']==64]))if any(r['success']or r['searched']==64 for r in rows)else None)

def write(groups,baseline):
    scenes=read(HERE/'SCENES.json')['entries'];families=[];scene_rows=[]
    for fi,name in enumerate(FAMILY_NAMES):
        rr=[g for g in groups if g['family']==fi]
        families.append(dict(index=fi,name=name,cases=len(rr),coverage={str(k):sum(g['success']and g['first_success']<=k for g in rr)for k in (16,32,64)},
            curve=[dict(K=k,covered=sum(g['success']and g['first_success']<=k for g in rr))for k in range(1,65)],
            distribution=distribution(rr),completion_times_success=[g['solution']['completion_time']for g in rr if g['success']],
            success_mean_completion_time=float(np.mean([g['solution']['completion_time']for g in rr if g['success']]))if any(g['success']for g in rr)else None,
            cases_detail=rr))
    for e in scenes:
        key=(e['type_id'],e['level']);rr=[g for g in groups if(g['type_id'],g['level'])==key]
        base=[b for b in baseline if(b['type_id'],b['level'])==key]
        values=[]
        for fi in range(7):
            rows=[g for g in rr if g['family']==fi]
            values.append(dict(family=fi,coverage={str(k):sum(g['success']and g['first_success']<=k for g in rows)for k in (16,32,64)},
                coverage_4x={str(k):sum(g['success']and g['first_success']<=k and g['solution']['episode_length']<=e['scene']['time_limits']['4']['max_steps']for g in rows)for k in (16,32,64)},
                distribution=distribution(rows),computed_cases=sum(g['success']or g['searched']==64 for g in rows)))
        identities={(g['controller'],g['state_index'])for g in rr if g['success']}
        complete=all(g['success']or g['searched']==64 or g.get('skipped')for g in rr)and len(rr)==112
        scene_rows.append(dict(type_id=e['type_id'],level=e['level'],name=scene_title(e),denominator=16,
            baseline_success=sum(b['success']for b in base),baseline_completed=len(base),families=values,
            any_family_success=len(identities),complete=complete,
            status='未见可解见证'if complete and not identities else'已见成功候选'if identities else'尚未完成',
            baseline_success_4x=sum(b['success']and b['episode_length']<=e['scene']['time_limits']['4']['max_steps']for b in base)))
    by_type=[]
    for t in range(1,18):
        items=[e for e in scene_rows if e['type_id']==t]
        by_type.append(dict(type_id=t,name=next(e['title']for e in scenes if e['type_id']==t),
            denominator=16*len(items),coverage64=[sum(e['families'][fi]['coverage']['64']for e in items)for fi in range(7)],
            curves={str(fi):[sum(g['success']and g['first_success']<=k for g in groups if g['type_id']==t and g['family']==fi)for k in range(1,65)]for fi in range(7)}))
    result=dict(scene_count=49,case_count=784,K=64,families=families,scenes=scene_rows,by_type=by_type,
        complete=all(e['complete']for e in scene_rows),
        scope='Finite frozen candidate pool; no generator, no family/dimension selection, no paper methods.',
        short_horizon_policy='4x reports only executed trajectories under the 6x-first-success stopping rule; unexecuted candidates remain unknown.')
    dump(HERE/'COVERAGE.json',result)
    header=['场景','仅安全投影 /16']+[f'{name} @16/@32/@64'for name in FAMILY_NAMES]+['原三列首次成功中位数','既有四列首次成功中位数','任一族成功 /16','状态']
    table=[]
    for e in scene_rows:
        cells=[]
        for v in e['families']:
            value='/'.join(str(v['coverage'][str(k)])for k in (16,32,64))
            if v['distribution']['incomplete']:value+=f"（待{v['distribution']['incomplete']}）"
            cells.append(value)
        table.append([e['name'],f"{e['baseline_success']}/16"]+cells+[str(e['families'][fi]['distribution']['median'])for fi in (0,4)]+[str(e['any_family_success']),e['status']])
    with(HERE/'coverage_scenes.csv').open('w',newline='')as f:
        writer=csv.writer(f);writer.writerow(header);writer.writerows(table)
    paragraphs=['# 十七类场景各修正族有限候选覆盖\n',
        '本轮'+('已完成'if result['complete']else'尚未完成')+'。49 个场景、784 个案例，各族当前 64 次内覆盖：'+
        '；'.join(f"{f['name']} {f['coverage']['64']}/784（从16到64增加{f['coverage']['64']-f['coverage']['16']}）"for f in families)+'。未运行完的案例不算作64次失败。\n',
        '64次内各族都未覆盖的完整场景：'+('；'.join(e['name']for e in scene_rows if e['complete']and e['any_family_success']==0)or'当前尚无已完成搜索且零覆盖的场景；未完成场景不作结论。')+'\n',
        '每个案例按冻结顺序搜索，第一次成功即停，最多64次；不训练或调用生成器。本轮不选族、不推荐维度、不运行论文方法。\n',
        '四倍时限结果来自同一次六倍时限搜索实际执行的轨迹，不追加候选搜索，不能把未执行的短时限候选当作失败。\n',
        '| '+' | '.join(header)+' |','| '+' | '.join(['---']*len(header))+' |']
    paragraphs.extend('| '+' | '.join(row)+' |'for row in table)
    paragraphs.append('\n首次成功序号的中位数、四分位、未找到数、搜索次数均值与成功条件平均完成时间保存在 COVERAGE.json。曲线同时保存总体和逐类型数据；未完成时只给当前已验证覆盖，不假定后续结果。\n')
    (HERE/'REPORT.md').write_text('\n\n'.join(paragraphs)+'\n')
    return result
