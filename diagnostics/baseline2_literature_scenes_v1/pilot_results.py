from experiment import *
from closure_scope import FLOW_NAMES
from migration_statistics_complete import paired
from result_inventory import logical,qualification
from endpoint_metrics import metrics
import csv
from paper_evidence import LABELS

FIXED=['safety','fixed_eta10','fixed_eta13','human_progress']

def resolve(stage):
    rows=[];sources=[]
    for c in read(HERE/f'{stage}_CASES.json'):
        p=HERE/stage/'raw'/f"{c['rollout_uid']}.json"
        if not p.exists():
            paths=sorted(HERE.glob(f"*/raw/{c['rollout_uid']}.json"));assert paths,(stage,c['rollout_uid']);p=paths[0]
        r=read(p);assert all(r[k]==c[k] for k in ('rollout_uid','controller_uid','state_uid','eta_uid','seed_key'));r['case']=c;rows.append(r);sources.append(dict(request=c['rollout_uid'],source=str(p),sha256=sha(p)))
    return rows,sources

def main():
    result={};sources=[];flat=[]
    for scene in FLOW_NAMES:
        gap=HERE/f'PILOT_GAP_{scene}.json'
        if gap.exists():result[scene]=dict(gap=read(gap));continue
        q=read(HERE/f'pilot_capability_{scene}_QUALIFICATION.json');rows=[]
        for stage in ('pilot_safety_'+scene,'pilot_fixed_'+scene):rr,ss=resolve(stage);rows+=rr;sources+=ss
        statistics=paired(rows,q,scene,FIXED);details=metrics(rows,q,scene,FIXED)
        allq=[dict(x,qualified=True) for x in q]
        supplemental_all=metrics(rows,allq,scene,FIXED)
        for r in supplemental_all['controllers']:r['qualified']=next(x['qualified'] for x in q if x['controller']==r['controller'])
        result[scene]=dict(qualification=q,fixed_correction=statistics,endpoint_details=details,all_checkpoints_descriptive=supplemental_all['controllers'],ten_type_learned_supplement=None)
        if statistics:flat += [dict(r,population='BC qualified fixed correction') for r in statistics['statistics']]
        extended='learning_eval_'+scene+'_with_bc';learninggap=HERE/f'PILOT_LEARNING_GAP_{scene}.json'
        if (HERE/f'{extended}_CASES.json').exists():
            rr,ss=resolve(extended);sources+=ss;rows+=rr
            for stage in ('safety_'+scene,'fixed_'+scene):r,_=logical(stage);rows+=r
            keys={(x['case']['state']['N'],x['case']['state']['index']) for x in rr};subset=[x for x in rows if (x['case']['state']['N'],x['case']['state']['index']) in keys]
            combined=paired(subset,qualification(scene)+q,scene,FIXED+['learned_scene']);result[scene]['ten_type_learned_supplement']=combined;result[scene]['ten_type_endpoint_details']=metrics(subset,qualification(scene)+q,scene,FIXED+['learned_scene'])
            if combined:flat += [dict(r,population='Ten-type source-only learned supplement') for r in combined['statistics']]
        elif learninggap.exists():result[scene]['learning_gap']=read(learninggap)
        nominal,_=resolve('pilot_nominal_'+scene);result[scene]['nominal_background']=paired(rows+nominal,q,scene,['safety','nominal_only'])
    for scene,d in result.items():d['paper_evidence_label']=LABELS[scene]
    for r in flat:r['paper_evidence_label']=LABELS[r['scene']]
    write(HERE/'PILOT_RESULTS.json',result);write(HERE/'PILOT_RESULT_SOURCES.json',sources)
    if flat:
        with (HERE/'PILOT_RESULTS.csv').open('w') as f:
            fields=list(dict.fromkeys(k for r in flat for k in r));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([{k:canonical(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in flat])
    else:(HERE/'PILOT_RESULTS.csv').write_text('scene,type,method,population,success_rate\n')
    text=['# 学习型controller试点','','确定性行为克隆策略、混合密度行为克隆策略各两个训练seed。只复用封存训练状态中既有成功冲突搜索多机规划轨迹，不新生成专家rollout。第一个5000步global配方的结构、损失与训练规则保持；新几何/N输入和归一化说明见预注册补充四。所有结果为开发证据。','','|场景|controller|单机成功率下界|合格|只加安全投影成功率|全程η10|全程η13|人定双η进度切换|','|---|---|---|---|---|---|---|---|']
    for scene,d in result.items():
        if 'gap' in d:text.append(f"|{FLOW_NAMES[scene]}|缺口：{d['gap']['reason']}|—|—|—|—|—|—|");continue
        names={c['controller']:c['type'] for c in read(HERE/f'PILOT_CONTROLLER_MANIFEST_{scene}.json')};rr={r['controller']:r for r in d['all_checkpoints_descriptive']}
        for q in d['qualification']:
            rates=rr[q['controller']]['rates'];text.append('|'+ '|'.join([FLOW_NAMES[scene],f"{names[q['controller']]}（{q['controller']}）",f"{100*q['rate_lower']:.2f}%",'是' if q['qualified'] else '否',*[f"{100*rates[m]['success']:.2f}%" for m in FIXED]])+'|')
    text+=['','不合格checkpoint的完整数据如实保留，但不进入迁移主要比较。没有合格checkpoint时不训练无从解释的试点迁移critic。十类型补充只有在能力门与场景可用性满足时运行，保持原八类型结果/模型/选择独立不覆盖；新的源类型集合并不使旧执行标签失效。其配对CI、unknown上下界、碰撞/超时/伤害与相对固定修正差值见JSON/CSV。二维拥挤随机场景的8机、14机几何缺口不能用其他N代替；两类学习型试点按用户决定均未启动；没有试点迁移合并。']
    (HERE/'PILOT_RESULTS.md').write_text('\n'.join(text)+'\n')
    seal('PILOT_RESULTS.receipt.json',[HERE/'PILOT_RESULTS.json',HERE/'PILOT_RESULTS.csv',HERE/'PILOT_RESULTS.md',HERE/'PILOT_RESULT_SOURCES.json',HERE/'pilot_results.py'],qualified_checkpoints_only_for_migration=True,original_eight_type_branch_retained=True)

if __name__=='__main__':main()
