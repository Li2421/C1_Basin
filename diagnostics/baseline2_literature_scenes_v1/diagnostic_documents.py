from experiment import *
from closure_scope import FLOW_NAMES
import csv
import collections
import numpy as np

CATEGORIES={'arrived':'到达','collision':'碰撞','geometrically_unreachable':'几何不可达','filter_equilibrium_deadlock':'滤波器平衡点死锁（预注册有限窗口条件）','blocked_by_arrived':'被已到达者阻挡','zero_nominal':'名义输入为零','other_stationary':'其他静止未到达','oscillation':'振荡未到达','slow_progress':'缓慢前进未到达','unknown':'数值unknown','insufficient_window':'不足5秒诊断窗口'}

def main():
    allrows=[];summaries=[];census=[]
    for scene in FLOW_NAMES:
        p=HERE/f'SAFETY_BASELINE_{scene}.csv';assert p.exists(),scene
        rows=list(csv.DictReader(p.open()));allrows+=rows
        for population in ('all','qualified'):
            selected=rows if population=='all' else [r for r in rows if r['qualified']=='True']
            groups=collections.defaultdict(list)
            for r in selected:groups[r['type']].append(r)
            for typ,rr in sorted(groups.items()):summaries.append(dict(scene=FLOW_NAMES[scene],population=population,type=typ,robots=len(rr),counts=dict(collections.Counter(r['category'] for r in rr))))
        census+=read(HERE/f'GEOMETRY_CENSUS_{scene}.json')
    exported=[]
    for r in allrows:
        row=dict(r);row['controller']=f"{r['type']}（{r['controller']}）";row['category_name']=CATEGORIES.get(r['category'],r['category']);exported.append(row)
    with (HERE/'SAFETY_BASELINE.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=exported[0]);w.writeheader();w.writerows(exported)
    write(HERE/'FAILURE_DECOMPOSITION.json',summaries)
    text=['# 只加安全投影的失败拆分','','每行CSV代表一台机器人在一局中的互斥分类。顺序为到达、碰撞、几何不可达、静止未到达的四个子类、振荡或缓慢前进；数值unknown单列。几何无效状态已在rollout之前排除，不进入分母。静止按最后5秒相对该窗首帧的最大位移小于半径；4a还要求至少90%步的该机器人名义输入非零、投影改写，并且联合NNLS锥残差<.05。诊断只读取保存的模拟轨迹。','','|场景|统计人群|类型|机器人局数|分类计数|','|---|---|---|---|---|']
    for r in summaries:text.append('|'+ '|'.join([r['scene'],'全部实例' if r['population']=='all' else '通过单机能力门实例',r['type'],str(r['robots']),'；'.join(f"{CATEGORIES.get(k,k)} {v}" for k,v in sorted(r['counts'].items()))])+'|')
    text+=['','静止/timeout不是死锁证明。4a是题目指定的有限窗口操作判据，不意味着已证明无限时域平衡稳定性。被已到达者阻挡与名义输入为零不归入4a。原始每步联合残差、投影改写、起作用约束、名义与安全速度均在NPZ中。W=2.5/5/10秒与τ=.02/.05/.1的九组分类在各场景FAILURE_SENSITIVITY JSON中；敏感性只重算旧预测，没有新rollout。','','圆形围场五机随机场景的半径2米，而速度界乘5秒仅1.868255479米。故任意受速度界约束的5秒行进都满足最大位移小于半径；此静止窗口不能独立识别物理静止。按题设保留该定义和敏感性，不调整阈值。低联合锥残差也必须结合非零约束偏置与安全速度解读，不能据4a操作候选计数宣称精确平衡或无限时域死锁。见THEORY_CONDITIONS.md。']
    (HERE/'FAILURE_DECOMPOSITION.md').write_text('\n'.join(text)+'\n')
    groups=collections.defaultdict(list)
    for r in census:groups[(r['scene'],r['type'])].append(r)
    text=['# 滤波器平衡点事件的描述性几何普查','','只普查满足4a操作条件的事件。按联合投影的起作用约束记录机器人配对/墙组成、连通簇大小、名义输入与法向量夹角及局部坐标修正。以下仅为描述；没有据此宣称不同controller共享死锁几何。没有事件的类型仍列出。','','|场景|类型|事件局数|记录步数|约束组成|簇大小中位数|夹角中位数|离锥修正下确界中位数|','|---|---|---|---|---|---|---|---|']
    types={c['type_code']:c['type'] for c in read(HERE/'CONTROLLER_MANIFEST.json')};summ=[]
    for scene in FLOW_NAMES:
        for typ,name in sorted(types.items()):
            rr=groups[(scene,typ)];components=collections.Counter(k[0] for r in rr for k in r['active_constraints']);sizes=[x for r in rr for x in r['cluster_sizes']];angles=[x for r in rr for x in r['nominal_normal_angles_degrees']];distances=[r['escape']['infimum'] for r in rr if r['escape']['infimum'] is not None]
            row=dict(scene=scene,type=typ,events=len({r['rollout_uid'] for r in rr}),steps=len(rr),constraint_counts=dict(components),cluster_median=float(np.median(sizes)) if sizes else None,angle_median=float(np.median(angles)) if angles else None,escape_infimum_median=float(np.median(distances)) if distances else None);summ.append(row)
            text.append('|'+ '|'.join(map(str,[FLOW_NAMES[scene],name,row['events'],row['steps'],dict(components),row['cluster_median'],row['angle_median'],row['escape_infimum_median']]))+'|')
    text+=['','“离开闭锥的最小修正”一般不取到：这里报告到开补集的距离下确界。名义输入已经在锥外时，零修正达到要求、距离为0且方向无定义；名义输入在秩亏锥内时，法向正交方向可以任意小地离开其线性包，下确界为0，不存在唯一非零最小向量。平均约束法向量为0时第一轴未定义，局部方向明确留空；锥等于全空间时不存在离锥方向。完整方向、秩与原因在GEOMETRY_CENSUS.json中。该JSON还保留起作用约束的非零偏置、安全联合速度范数、最大单机安全速度和各机器人二维平均法向坐标；这些量用于区分有限窗口锥候选与精确零速平衡。步级分布存在同局时间相关性，未将记录步数当成独立事件数。']
    write(HERE/'GEOMETRY_CENSUS.json',census);write(HERE/'GEOMETRY_CENSUS_SUMMARY.json',summ);(HERE/'GEOMETRY_CENSUS.md').write_text('\n'.join(text)+'\n')
    seal('SAFETY_DIAGNOSTICS.receipt.json',[HERE/'SAFETY_BASELINE.csv',HERE/'FAILURE_DECOMPOSITION.json',HERE/'FAILURE_DECOMPOSITION.md',HERE/'GEOMETRY_CENSUS.json',HERE/'GEOMETRY_CENSUS_SUMMARY.json',HERE/'GEOMETRY_CENSUS.md',HERE/'diagnostic_documents.py',HERE/'failure_analysis.py'],predictions_only=True)

if __name__=='__main__':main()
