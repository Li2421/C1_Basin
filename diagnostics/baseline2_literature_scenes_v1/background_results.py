from experiment import *
from closure_scope import FLOW_NAMES
from result_inventory import logical,qualification
from migration_statistics import paired
import collections
import csv
from paper_evidence import LABELS

def main():
    result={};transitions=[];table=[]
    for scene,name in FLOW_NAMES.items():
        nominal,_=logical('nominal_'+scene);safety,_=logical('safety_'+scene);whitebox,_=logical('whitebox_'+scene)
        black=[r for r in nominal if r['case']['controller_payload']['nominal']['type_code']!='goal_reference'];q=qualification(scene)
        stats=paired(safety+black,q,scene,['safety','nominal_only']);valid={r['controller'] for r in q if r['qualified']}
        key=lambda r:(r['case']['controller_payload']['nominal']['controller'],r['case']['state']['N'],r['case']['state']['index'],r['case']['seed'])
        lookup={key(r):r for r in safety}
        for population in ('all','qualified'):
            groups=collections.defaultdict(list)
            for r in black:
                c=r['case'];ctl=c['controller_payload']['nominal']
                if population=='qualified' and ctl['controller'] not in valid:continue
                if r['collision']:groups[ctl['type']].append(lookup[key(r)])
            for typ,rr in sorted(groups.items()):
                counts=collections.Counter('success' if r['success'] else 'unknown' if r['numerical_failure'] else 'collision' if r['collision'] else 'timeout' if r['timeout'] else 'other' for r in rr)
                transitions.append(dict(scene=name,population=population,type=typ,nominal_collision_episodes=len(rr),safety_success=counts['success'],safety_timeout=counts['timeout'],safety_collision=counts['collision'],safety_unknown=counts['unknown'],safety_other=counts['other']))
        goal=[r for r in nominal if r['case']['controller_payload']['nominal']['type_code']=='goal_reference'];wstats=paired(goal+whitebox,[dict(scene=scene,controller='goal_reference',qualified=True)],scene,['safety','nominal_only','fixed_eta10','fixed_eta13','human_progress'])
        result[scene]=dict(black_box=stats,white_box_background=wstats)
        for r in stats['statistics']:table.append(dict(scene=name,type=r['type'],method='只执行名义控制器' if r['method']=='nominal_only' else '只加安全投影',rate_percent=100*r['success_rate'],ci95_low_percent=100*r['ci95'][0],ci95_high_percent=100*r['ci95'][1],success_upper_percent=100*r['success_upper']))
    for scene,d in result.items():d['paper_evidence_label']=LABELS[scene]
    write(HERE/'BACKGROUND_RESULTS.json',result);write(HERE/'NOMINAL_TO_SAFETY_TRANSITIONS.json',transitions)
    for filename,rr in [('BACKGROUND_TABLE.csv',table),('NOMINAL_TO_SAFETY_TRANSITIONS.csv',transitions)]:
        with (HERE/filename).open('w') as f:w=csv.DictWriter(f,fieldnames=rr[0]);w.writeheader();w.writerows(rr)
    text=['# 名义控制与安全投影背景参照','','名义参照不调用任何安全投影或修正，保留速度界、到达停止与原碰撞事件。八类黑盒按各场景同一能力门筛选；名义朝目标白盒单独报告，不进入迁移主结论。下表为5000次配对分簇bootstrap结果，完整上界、白盒结果及逐局转移在JSON/CSV中。','','|场景|类型|做法|成功率% [95% CI]|','|---|---|---|---|']
    types={r['type_code']:r['type'] for r in read(HERE/'CONTROLLER_MANIFEST.json')};types['type_equal']='类型等权合并'
    for r in table:text.append(f"|{r['scene']}|{types[r['type']]}|{r['method']}|{r['rate_percent']:.2f} [{r['ci95_low_percent']:.2f}, {r['ci95_high_percent']:.2f}]|")
    (HERE/'BACKGROUND_TABLE.md').write_text('\n'.join(text)+'\n')
    seal('BACKGROUND_RESULTS.receipt.json',[HERE/'BACKGROUND_RESULTS.json',HERE/'BACKGROUND_TABLE.csv',HERE/'BACKGROUND_TABLE.md',HERE/'NOMINAL_TO_SAFETY_TRANSITIONS.json',HERE/'NOMINAL_TO_SAFETY_TRANSITIONS.csv',HERE/'background_results.py'],background_only=True)

if __name__=='__main__':main()
