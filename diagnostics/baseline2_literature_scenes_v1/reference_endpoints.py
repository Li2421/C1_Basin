from experiment import *
from closure_scope import FLOW_NAMES
from result_inventory import logical,qualification
from pilot_results import resolve
from migration_statistics_complete import paired,METHOD_NAMES
import collections
import csv
from paper_evidence import LABELS

def main():
    grouped=collections.defaultdict(list);original=[];stat=[];sources=[]
    for scene in FLOW_NAMES:
        rows=[]
        stages=['safety_'+scene,'fixed_'+scene]
        if (HERE/f'learning_eval_{scene}_CASES.json').exists():stages.append('learning_eval_'+scene)
        for stage in stages:
            rr,ss=logical(stage);rows+=rr;sources+=ss
        q=qualification(scene);eligible={x['controller'] for x in q if x['qualified']}
        if scene in ('lit_narrow_passage','lit_doorway'):
            ss=[r for r in rows if r['case']['state']['index']==0]
            methods=['safety','fixed_eta10','fixed_eta13','human_progress']+(['learned_scene'] if any(r['case']['method']=='learned_scene' for r in ss) else [])
            value=paired(ss,q,scene,methods)
            if value:stat += [dict(r,population='eight nonlearned types, original state0') for r in value['statistics']]
        for stage in ('pilot_safety_'+scene,'pilot_fixed_'+scene):
            if (HERE/f'{stage}_CASES.json').exists():rr,ss=resolve(stage);rows+=rr;sources+=ss
        pq=HERE/f'pilot_capability_{scene}_QUALIFICATION.json'
        if pq.exists():eligible|={x['controller'] for x in read(pq) if x['qualified']}
        for r in rows:
            c=r['case'];ctl=c['controller_payload']['nominal'];isunknown=bool(r['numerical_failure']);position=bool(r.get('paper_position_convergence',False));velocity=bool(r.get('paper_velocity_convergence',False));reference=bool(position and (velocity if scene=='lit_circle_five' else True)) and not isunknown
            row=dict(scene=FLOW_NAMES[scene],scene_code=scene,N=c['state']['N'],state=c['state']['index'],controller=f"{ctl['type']}（{ctl['controller']}）",type=ctl['type'],type_code=ctl['type_code'],method=METHOD_NAMES[c['method']],method_code=c['method'],seed=c['seed'],qualified=ctl['controller'] in eligible,success=r['success'],collision=r['collision'],timeout=r['timeout'],unknown=int(isunknown),reference_arrival_lower=int(reference),reference_arrival_upper=int(reference or isunknown),reference_definition='all distance<.1 AND all speed<.1' if scene=='lit_circle_five' else 'all distance<.1; physical reference, not IMPC predicted cost_index' if scene in ('lit_narrow_passage','lit_crowded_random') else 'all distance<.1',rollout_uid=r['rollout_uid'])
            grouped[(scene,ctl['type'],c['method'],ctl['controller'] in eligible)].append(row)
            if scene in ('lit_narrow_passage','lit_doorway') and c['state']['index']==0:original.append(row)
    summaries=[]
    for (scene,typ,method,qualified),rows in sorted(grouped.items()):
        n=len(rows);summaries.append(dict(scene=FLOW_NAMES[scene],scene_code=scene,type=typ,method=METHOD_NAMES[method],qualified=qualified,episodes=n,task_success=sum(r['success'] for r in rows),reference_arrival_lower=sum(r['reference_arrival_lower'] for r in rows),reference_arrival_upper=sum(r['reference_arrival_upper'] for r in rows),unknown=sum(r['unknown'] for r in rows),definition=rows[0]['reference_definition']))
    for name,rows in [('REFERENCE_ENDPOINTS.csv',summaries),('ORIGINAL_INSTANCE_RESULTS.csv',original)]:
        with (HERE/name).open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    for r in stat:r['paper_evidence_label']=LABELS[r['scene']]
    write(HERE/'ORIGINAL_INSTANCE_RESULTS.json',stat);write(HERE/'REFERENCE_ENDPOINT_SOURCES.json',sources)
    types={r['type_code']:r['type'] for r in read(HERE/'CONTROLLER_MANIFEST.json')};types['type_equal']='类型等权合并'
    text=['# 文献原始实例state0的流程版结果','','窄通道场景state0是图8在1.8秒的声明reset假设，不是可验证真实t0；门口场景state0按代码原坐标。它们与扰动集合分开报告。下表仅为通过能力门的八类非训练controller；逐实例/seed与行为克隆试点原始实例均在CSV中，后者不混入八类型主比较。','','|场景|类型|做法|无碰撞成功率% [95% CI]|','|---|---|---|---|']
    for r in stat:text.append(f"|{FLOW_NAMES[r['scene']]}|{types[r['type']]}|{METHOD_NAMES[r['method']]}|{100*r['success_rate']:.2f} [{100*r['ci95'][0]:.2f}, {100*r['ci95'][1]:.2f}]|")
    text+=['','原论文判据参照见REFERENCE_ENDPOINTS.csv。圆形围场五机随机场景使用距离<.1且速度<.1，区别于流程版.08到达停止。门口场景列最终位置<.1。无限时域模型预测控制与死锁解除的预测cost_index无法定义于所有新controller，因此窄通道/拥挤随机仅提供明确标注的物理位置<.1参照，不冒充原预测终止判据。']
    (HERE/'ORIGINAL_INSTANCE_RESULTS.md').write_text('\n'.join(text)+'\n')
    seal('REFERENCE_ENDPOINTS.receipt.json',[HERE/'REFERENCE_ENDPOINTS.csv',HERE/'ORIGINAL_INSTANCE_RESULTS.csv',HERE/'ORIGINAL_INSTANCE_RESULTS.json',HERE/'ORIGINAL_INSTANCE_RESULTS.md',HERE/'REFERENCE_ENDPOINT_SOURCES.json',HERE/'reference_endpoints.py'],saved_predictions_only=True)

if __name__=='__main__':main()
