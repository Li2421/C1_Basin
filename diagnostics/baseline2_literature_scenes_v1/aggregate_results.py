from experiment import *
from closure_scope import FLOW_NAMES,circle_gap
from result_inventory import logical,qualification
from migration_statistics_complete import paired,METHOD_NAMES
from pooled_statistics import pooled_paired
from endpoint_metrics import metrics
from paper_evidence import LABELS
from copy import deepcopy
import csv

FIXED=['safety','fixed_eta10','fixed_eta13','human_progress']

def pooled(rows,scenes):
    out=[];names=set()
    for si,scene in enumerate(scenes):
        valid={q['controller'] for q in qualification(scene) if q['qualified']}
        for r in rows:
            c=r['case'];name=c['controller_payload']['nominal']['controller']
            if c['state']['scene']!=scene or name not in valid:continue
            x=deepcopy(r);s=x['case']['state'];s['source_scene']=scene;s['scene']='pooled_scenes';s['index']+=si*1000000;out.append(x);names.add(name)
    q=[dict(scene='pooled_scenes',controller=name,qualified=True) for name in sorted(names)]
    return out,q

def main():
    allrows=[];per_scene={};details={};sources=[]
    for scene in FLOW_NAMES:
        path=HERE/f'MIGRATION_RESULTS_{scene}.json';assert path.exists(),scene;per_scene[scene]=read(path)
        rows=[]
        for stage in ('safety_'+scene,'fixed_'+scene):rr,ss=logical(stage);rows+=rr;sources+=ss
        methods=FIXED.copy()
        if per_scene[scene]['learned_evaluation'] is not None:
            rr,ss=logical('learning_eval_'+scene);rows+=rr;sources+=ss;methods+=['learned_scene']
            keys={(r['case']['state']['N'],r['case']['state']['index']) for r in rr};subset=[r for r in rows if (r['case']['state']['N'],r['case']['state']['index']) in keys]
            complete=paired(subset,qualification(scene),scene,methods);original=deepcopy(complete)
            for r in original['statistics']:
                for comparison in r.get('delta_vs_fixed',{}).values():
                    comparison.pop('identified_delta_bounds',None);comparison.pop('unknown_sensitive_ci95',None)
            assert canonical(original)==canonical(per_scene[scene]['learned_evaluation']),('Existing sealed fields changed',scene)
            per_scene[scene]['learned_evaluation']=complete
            details[scene]=dict(fixed_full_inventory=metrics(rows,qualification(scene),scene,FIXED),learned_evaluation=metrics(subset,qualification(scene),scene,methods))
        else:details[scene]=dict(fixed_full_inventory=metrics(rows,qualification(scene),scene,FIXED),learned_evaluation=None)
        allrows+=rows
    pp,qq=pooled([r for r in allrows if r['case']['method'] in FIXED],list(FLOW_NAMES));combined=pooled_paired(pp,qq,'pooled_scenes',FIXED);combined_details=metrics(pp,qq,'pooled_scenes',FIXED)
    learned_scenes=[s for s in FLOW_NAMES if per_scene[s]['learned_evaluation'] is not None]
    supported=None
    if learned_scenes:
        learnrows=[]
        for scene in learned_scenes:
            keys={(r['case']['state']['N'],r['case']['state']['index']) for r in allrows if r['case']['state']['scene']==scene and r['case']['method']=='learned_scene'}
            learnrows+=[r for r in allrows if r['case']['state']['scene']==scene and (r['case']['state']['N'],r['case']['state']['index']) in keys]
        pp,qq=pooled(learnrows,learned_scenes);supported=pooled_paired(pp,qq,'pooled_scenes',FIXED+['learned_scene'])
    result=dict(evidence='DEVELOPMENT',per_scene=per_scene,endpoint_details=details,three_scene_fixed_combined=combined,three_scene_fixed_endpoint_details=combined_details,learned_supported_subset_combined=supported,learned_supported_scenes=learned_scenes,three_scene_learned_combined=None if len(learned_scenes)<3 else supported,aggregation='Pool geometry-valid state clusters (scene,N,index); resample type, controller, state, keeping seed twins together. Eligibility is applied separately in each scene. Scenes are not forced to equal weights.',bootstrap_iterations=5000)
    result['paper_evidence_labels']=LABELS
    result['closure_scope']=dict(included_scenes=list(FLOW_NAMES),excluded_flow_scenes={'lit_circle_five':circle_gap()},four_scene_combination_computed=False,learning_pilots='用户决定停止，转入 v2；未启动')
    write(HERE/'MIGRATION_RESULTS.json',result);write(HERE/'results.json',result);write(HERE/'MIGRATION_SOURCES.json',sources)
    flat=[]
    for scene,d in per_scene.items():
        for population in ('fixed_full_inventory','learned_evaluation'):
            if d[population]:flat += [dict(r,population=population) for r in d[population]['statistics']]
    for pop,d in [('three_scene_fixed_combined',combined),('learned_supported_subset_combined',supported)]:
        if d:flat += [dict(r,population=pop) for r in d['statistics']]
    for r in flat:r['paper_evidence_label']=LABELS[r['scene']]
    with (HERE/'MIGRATION_RESULTS.csv').open('w') as f:
        fields=list(dict.fromkeys(k for r in flat for k in r));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([{k:canonical(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in flat])
    make_table(per_scene,combined,supported,learned_scenes)
    seal('MIGRATION_RESULTS.receipt.json',[HERE/'MIGRATION_RESULTS.json',HERE/'results.json',HERE/'MIGRATION_RESULTS.csv',HERE/'MIGRATION_SOURCES.json',HERE/'AXIS_TABLE.md',HERE/'AXIS_TABLE.csv',HERE/'aggregate_results.py',HERE/'endpoint_metrics.py',HERE/'pooled_statistics.py',HERE/'POOLED_POINT_ESTIMATE_CHECK.receipt.json'],numeric_results_before_plots=True)

def make_table(per_scene,combined,supported,learned_scenes):
    types={r['type_code']:r['type'] for r in read(HERE/'CONTROLLER_MANIFEST.json')};types['type_equal']='类型等权合并';columns=[METHOD_NAMES[m] for m in FIXED]+['冻结学习双η与进度阈值·零样本','冻结学习双η与进度阈值·在线停滞触发','场景内重新学习双η与进度阈值']
    rows=[]
    def block(scene,label,data,population):
        lookup={(r['type'],r['method']):r for r in data['statistics']} if data else {}
        for t,name in types.items():
            row=dict(场景=label,类型=name,统计人群=population,论文证据=LABELS.get(scene,LABELS['pooled_scenes']))
            for m,col in zip(FIXED,columns[:4]):
                r=lookup.get((t,m));row[col]=cell(r)
            row[columns[4]]=row[columns[5]]='不适用：冻结四机圆环编码无定义'
            row[columns[6]]=cell(lookup.get((t,'learned_scene'))) if data and 'learned_scene' in data['methods'] else '缺口或按可用性仅做固定修正'
            rows.append(row)
    for scene,d in per_scene.items():
        use=d['learned_evaluation'] or d['fixed_full_inventory'];block(scene,FLOW_NAMES[scene],use,'独立32评估state；共同配对' if d['learned_evaluation'] else '全部有效state；共同配对')
    block('pooled','三个已完成场景有效状态池',combined,'固定修正；按场景分别执行能力门')
    if supported:block('supported','支持场景内学习的场景子集',supported,'独立32评估state/场景；不冒充完整三场景学习合并')
    for typ,name in types.items():
        row=dict(场景='圆形围场五机随机场景',类型=name,统计人群='不进入合并；部分已落盘证据只保留',论文证据=LABELS['lit_circle_five'])
        row.update({col:'缺口：用户决定停止，转入 v2' for col in columns})
        rows.append(row)
    with (HERE/'AXIS_TABLE.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    text=['# 三个已完成场景的流程迁移成功率','','按用户决定收尾v1：仅窄通道、二维拥挤随机、门口进入合并。圆形围场流程版停止、部分结果不作效果估计，不冒充四场景合并。单元格为成功率% [配对分簇bootstrap 95% CI]；unknown按下界，完整上界与差值CI见MIGRATION_RESULTS.json。仅纳入对应场景单机能力>=95%的实例；无合格实例保留缺口行。所有结果均为开发证据。','','|'+ '|'.join(rows[0])+'|','|'+'---|'*len(rows[0])]
    text+=['|'+ '|'.join(map(str,r.values()))+'|' for r in rows]
    text+=['','冻结流程的两种零样本方法均不适用：原编码固定四机圆环、100维环观测和33维上下文，SnapshotConditioning还原RingEnv；不修改冻结模型补定义。完整十二动作菜单的独立评估矩阵不存在，因此平均/最大完整菜单遗憾不识别；已有四臂菜单的交叉拟合/样本内参照、碰撞/超时/unknown率、逐局伤害及下降>=10个百分点的controller数见JSON。不同场景论文证据是否复现见PAPER_REPRODUCTION.md；本文不把流程版成功当作论文版成功。']
    (HERE/'AXIS_TABLE.md').write_text('\n'.join(text)+'\n')

def cell(r):return '缺口：无合格实例或该终点无数据' if r is None else f"{100*r['success_rate']:.2f} [{100*r['ci95'][0]:.2f}, {100*r['ci95'][1]:.2f}]"

if __name__=='__main__':main()
