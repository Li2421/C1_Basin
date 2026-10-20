from common import *
from capability_statistics import summarize
from failure_diagnostics import diagnose
from audit_utils import source_inventory
import collections,csv
OLD=D/'axis4a_addendum_v1'
ARMS=['F10','S500'];NAMES=read(HERE/'HISTORICAL_INVENTORY.json')['controller_names']
def raw_for(c):
    path=HERE/'geometry_controls/raw'/f"{c['rollout_uid']}.json"
    if not path.exists():path=Path(c['reference_path'])
    r=read(path);assert sha(r['trace_path'])==r['trace_sha256'];return r,str(path)
def extract(cases,cohort,radius):
    lo=np.zeros((14,16,2,1));hi=lo.copy();seen=np.zeros_like(lo,bool)
    for c in cases:
        if c['cohort']!=cohort or c['r']!=radius or c['arm'] not in ARMS:continue
        r,path=raw_for(c);idx=(NAMES.index(c['base_controller_name']),c['state_index'],ARMS.index(c['arm']),0);assert not seen[idx];seen[idx]=True;lo[idx]=r['success'] if not r['numerical_failure'] else 0;hi[idx]=lo[idx]+r['numerical_failure']
    assert seen.all();return lo,hi

def main():
    prereg();assert (HERE/'geometry_controls_COMPLETE.receipt.json').exists();assert not (HERE/'A_COMPLETE.receipt.json').exists();cases=read(HERE/'geometry_controls_CASES.json');basecases=[]
    for q in read(OLD/'A2_BASELINE_REFERENCES.json'):
        c=dict(q['case'],cohort='baseline',r=.9,reference_path=q['raw_path']);s=next(i for i,v in enumerate(read(HERE/'A2_ORIGINAL_STATES.json')) if v['state_uid']==c['state_uid']);c['state_index']=s;basecases.append(c)
    baseline=extract(basecases,'baseline',.9);positive=extract(cases,'positive',.9);a1=summarize(*positive,202610163710);bounds=a1['median_max_success_pct'];a1['decision']='生成器有效，原几何能力门结论成立' if bounds[0]>=20 else '生成器无效，原几何能力门结论作废' if bounds[1]<20 else '不确定';a1['status']='GENERATOR_VALID' if bounds[0]>=20 else 'GENERATOR_INVALID' if bounds[1]<20 else 'INCONCLUSIVE';curves=[];a2={}
    for j,r in enumerate([.8,.85,.95,1.]):
        a=extract(cases,'same_state',r);q=summarize(*a,202610163711+j,base=baseline);a2[str(r)]=q
        for n,v in zip(NAMES,q['controller_max_success_bounds_pct']):curves.append(dict(controller=n,radius=r,lower_pct=v[0],upper_pct=v[1]))
    nominal={}
    for c in cases:
        if c['arm']=='nominal_only':r,path=raw_for(c);nominal[c['base_controller_name'],c['state_uid'],c['seed']]=(r,path)
    diagnostic_cases=[(c,*raw_for(c)) for c in cases]
    original_state_indices={s['state_uid']:s['study_index'] for s in read(R35/'pilot_protocol.json')['states']}
    for original_case in read(R35/'pilot_CASES.json'):
        c=dict(original_case,state_index=original_state_indices[original_case['state_uid']])
        path=R35/'pilot/raw'/f"{c['rollout_uid']}.json";diagnostic_cases.append((dict(c,cohort='old_geometry_gate'),read(path),str(path)))
    failed=[];groups={}
    for c,r,path in diagnostic_cases:
        if r['success'] and not r['numerical_failure']:continue
        d=diagnose(r);ref=nominal.get((c['base_controller_name'],c['state_uid'],c['seed']));d.update(controller=c['base_controller_name'],radius=c['r'],cohort=c['cohort'],arm=c['arm'],state_uid=c['state_uid'],state_index=c['state_index'],rollout_uid=c['rollout_uid'],source=path)
        if ref:d.update(nominal_termination='unknown' if ref[0]['numerical_failure'] else 'collision' if ref[0]['collision'] else ref[0]['outcome'],nominal_source=ref[1])
        else:d.update(nominal_termination='缺口',nominal_source=None)
        failed.append(d)
    for radius in [.8,.85,.9,.95,1.]:
        rr=[r for r in failed if r['radius']==radius];corr=[r for r in rr if r['arm'] in ['F10','S500']];cc=collections.Counter(r['category'] for r in corr);allc=collections.Counter(r['category'] for r in rr);groups[str(radius)]=dict(all_failed=len(rr),all_failure_classes=dict(allc),corrected_failed=len(corr),corrected_failure_classes=dict(cc),corrected_stagnation_fraction=cc['停滞候选']/len(corr) if corr else None,interpretation='修正未能解除的停滞，属于方法局限' if corr and cc['停滞候选']>len(corr)/2 else '失败以仍在移动的超时或碰撞及unknown为主；不判定正式死锁')
    corrected=[r for r in failed if r['arm'] in ['F10','S500']];pct=sum(r['category']=='停滞候选' for r in corrected)/len(corrected) if corrected else None;eligible=[float(r) for r,v in a2.items() if v['median_max_success_pct'][0]>=20] if a1['status']=='GENERATOR_INVALID' else [];out=dict(utc=utc(),status='COMPLETE',positive_control=a1,same_state_radii=a2,conditional_main=dict(triggered=bool(eligible),eligible_radii=eligible,reason='仅生成器无效且同state能力门通过才运行；不满足触发条件时不新增仿真'),failure_summary=groups,corrected_failure_stagnation_fraction=pct,failed_records=len(failed),missing_nominal_counterparts=sum(r['nominal_termination']=='缺口' for r in failed),source_evidence='Compatible historical correction labels reused; new nominal/safety baselines prospectively registered; classification descriptive',baseline_radius=.9)
    write(HERE/'A_RESULTS.json',out);write(HERE/'FAILURE_CLASSIFICATIONS.json',failed);write(HERE/'CONTROLLER_RADIUS_CURVES.json',curves)
    with (HERE/'FAILURE_CLASSIFICATIONS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(failed[0]));w.writeheader();w.writerows(failed)
    lines=['| 几何半径 | 两修正臂中较好成功率的控制器中位数界 | 95%CI | 相对半径0.90的配对差界 | 配对差95%CI |','|---|---|---|---|---|',f"| 0.90，生成器阳性对照 | {a1['median_max_success_pct']} | {a1['median_max_95CI_pct']} | 不适用 | 不适用 |"]
    for radius,q in a2.items():lines.append(f"| {radius}，同state换半径 | {q['median_max_success_pct']} | {q['median_max_95CI_pct']} | {q['paired_difference_of_medians_pp']} | {q['paired_difference_95CI_pp']} |")
    lines += ['',a1['decision']+'。','',f'修正失败中停滞候选总体比例为 {pct:.2%}；各半径分布见下表。','','| 半径 | 修正失败数 | 停滞候选比例 | 分类 | 解读 |','|---|---:|---:|---|---|']
    for radius,q in groups.items():lines.append(f"| {radius} | {q['corrected_failed']} | {q['corrected_stagnation_fraction']:.2%} | {q['corrected_failure_classes']} | {q['interpretation']} |")
    (HERE/'A_RESULTS.md').write_text('\n'.join(lines)+'\n');source_inventory('imports_analyze_A.json');seal('A_COMPLETE.receipt.json',[HERE/f for f in ['A_RESULTS.json','A_RESULTS.md','FAILURE_CLASSIFICATIONS.json','FAILURE_CLASSIFICATIONS.csv','CONTROLLER_RADIUS_CURVES.json','A_REPRODUCTION_GATE.json']],conditional_geometry_main_required=bool(eligible),new_unknown_not_rerun=True)
if __name__=='__main__':main()
