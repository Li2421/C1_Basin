"""Clarify management allocation and stage-two failure attribution;no rollout edits."""
from experiment import *
from registry import verify_receipt
from atomic_write import write
import ast,difflib

READY='FINAL_DELIVERY_CLARITY_20261022.receipt.json'
COPIES={
'optional_revision_diagnostics_supplement_20261022.py':'optional_revision_diagnostics_causeaware_20261022.py',
'optional_results_supplement_20261022.py':'optional_results_causeaware_20261022.py',
'final_reports_spatial_mask_20261022.py':'final_reports_delivery_clarity_20261022.py',
'optional_delivery_spatial_mask_20261022.py':'optional_delivery_final_clarity_20261022.py',
'correction_continuation_bootstrap_memo_20261022.py':'correction_continuation_final_clarity_20261022.py'}

HELPER='''def baseline_failure_modes(rows,failures):
 baseline=[r for r in rows if r['method']=='safety' and r['controller'] in failures]
 return dict(logical_episodes=len(baseline),collision=sum(bool(r['collision']) for r in baseline),unknown=sum(bool(r['unknown']) for r in baseline),constraint_violation_episodes=sum(r.get('constraint_violation_steps',0)>0 for r in baseline),sampled_speed_violation_episodes=sum(r.get('sampled_speed_violation_steps',0)>0 for r in baseline),acceleration_violation_episodes=sum(r.get('acceleration_bound_violation_steps',0)>0 for r in baseline),interpretation='Descriptive logical episode counts;seed aliases are not independent physical trials')

'''


def main():
    assert not (HERE/READY).exists();verify_receipt('MIGRATION_NUMBERS.receipt.json')
    verify_receipt('LOCAL_CPU_SHARED_OWN_FALLBACK_20261022.receipt.json')
    assert not (HERE/'OPTIONAL_NUMBERS.receipt.json').exists(),'Do not change sealed optional analysis'
    subs={Path(a).stem:Path(b).stem for a,b in COPIES.items()};rows=[]
    for old,new in COPIES.items():
        before=(HERE/old).read_text();after=before
        for a,b in sorted(subs.items(),key=lambda p:-len(p[0])):after=after.replace(a,b)
        if new.startswith('optional_revision_diagnostics_'):
            after=after.replace('def report(results,dynamics,sources):',HELPER+'def report(results,dynamics,sources):',1)
            after=after.replace(' discrimination=[];splits={};hovers={}',' discrimination=[];splits={};hovers={};failure_modes={}')
            needle="  split={}\n";assert after.count(needle)==1
            after=after.replace(needle,"  failure_modes[scene]=baseline_failure_modes(rows,failures)\n"+needle)
            needle="  splits[scene]=split;hovers[scene]=weighted_rates(rows,('near_goal_wandering',))"
            insert="  modes=failure_modes[scene]\n  if any(modes[k] for k in ('collision','unknown','constraint_violation_episodes','sampled_speed_violation_episodes','acceleration_violation_episodes')):\n   split['controller_failure']['gain_interpretation']='本层低成功率归类包含碰撞、数值失败或动力学违反；不能直接解释为controller自身停住被修正，也不计死锁解决'\n   split['controller_failure']['baseline_failure_modes']=modes\n"+needle
            assert after.count(needle)==1;after=after.replace(needle,insert)
            after=after.replace('instances=instances,discrimination=discrimination,','instances=instances,discrimination=discrimination,controller_failure_baseline_failure_modes=failure_modes,',1)
        if new.startswith('final_reports_'):
            oldtext='独立资源guard和watchdog持续校验，不能再次释放六核或恢复24核。'
            assert after.count(oldtext)==1
            after=after.replace(oldtext,oldtext+'本地管理优先绑定未分配核；若全机已分配满，仅与经用户/命令路径核验的本轮自有Slurm worker共享一核，绝不占其他用户核，有空核即迁回。当前guard以ACTIVE_ENTRYPOINT.local_cpu_guard为准，策略及核验见LOCAL_CPU_SHARED_OWN_FALLBACK_20261022与LOCAL_CPU_SHARED_OWN_FALLBACK_VERIFIED_20261022；18份额上限不变。')
            after=after.replace(" geometry();dynamics(optional)"," geometry();dynamics(optional)\n with (HERE/'DYNAMICS_STAGE.md').open('a') as stream:stream.write('\\n第二阶段低成功率类别若含碰撞、数值失败、约束/速度/加速度违反，不能直接称为controller自身停住，也不计死锁解决；具体逻辑局计数见INSTANCE_CLASSES_DYNAMICS_SUPPLEMENT_20261022.controller_failure_baseline_failure_modes及各类别gain_interpretation。分类与成功标准未改。\\n')",1)
        if new.startswith(('optional_delivery_','correction_continuation_','final_reports_')):
            indent='    ' if new.startswith('optional_delivery_') else ' '
            after=after.replace('def main():\n','def main():\n'+indent+f"verify_receipt('{READY}')\n",1)
        ast.parse(after);p=HERE/new
        if p.exists():assert p.read_text()==after,p
        else:p.write_text(after)
        diff=p.with_suffix('.py.diff');diff.write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=old,tofile=new)))
        rows.append(dict(original=old,new=new,original_sha256=sha(HERE/old),new_sha256=sha(p),diff=str(diff)))
    from optional_revision_diagnostics_causeaware_20261022 import baseline_failure_modes
    test=[dict(method='safety',controller='a',collision=1,unknown=0,constraint_violation_steps=3),dict(method='safety',controller='a',collision=0,unknown=1),dict(method='fixed_eta10',controller='a',collision=1,unknown=0),dict(method='safety',controller='b',collision=1,unknown=0)]
    counted=baseline_failure_modes(test,{'a'});assert counted['logical_episodes']==2 and counted['collision']==counted['unknown']==counted['constraint_violation_episodes']==1
    write(HERE/'FINAL_DELIVERY_CLARITY_MAP_20261022.json',dict(utc=utc(),copies=COPIES,static_copies=rows,classification_and_success_definitions_unchanged=True,conditional_cause_counts_test=counted,management_policy_reflects_actual_runtime=True,no_scientific_worker_changes=True))
    seal(READY,[Path(__file__),HERE/'FINAL_DELIVERY_CLARITY_MAP_20261022.json',HERE/'LOCAL_CPU_SHARED_OWN_FALLBACK_20261022.receipt.json',HERE/'MIGRATION_NUMBERS.receipt.json']+[p for r in rows for p in (HERE/r['new'],Path(r['diff']))],report_attribution_and_resource_prose_only=True,classification_unchanged=True,bootstrap_and_physics_unchanged=True)
    print('Sealed final-delivery attribution and resource-policy clarification',flush=True)


if __name__=='__main__':main()
