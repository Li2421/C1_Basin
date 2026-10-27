"""Static posthoc analysis revision; no execution/controller/physics edits."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
import ast,difflib

READY='ANALYSIS_SUPPLEMENT_READY_20261022.receipt.json'
INPUT='PARALLEL_NUMERIC_INPUTS_SUPPLEMENT_20261022.receipt.json'
COPIES={
 'statistics_header_reader_20261023.py':'statistics_training_selected_reader_20261022.py',
 'statistics_correction_header_20261023_quota18_now_20261022.py':'statistics_correction_supplement_20261022.py',
 'statistics_parallel_tasks_20261023_quota18_now_20261022.py':'statistics_tasks_supplement_20261022.py',
 'statistics_parallel_scoped_20261023_quota18_now_20261022.py':'statistics_supplement_20261022.py',
 'statistics_parallel_20261023_quota18_now_20261022.sbatch':'statistics_supplement_20261022.sbatch',
 'scene_learning_scheduler_pa_memo_20261022_quota18_now_20261022.py':'scene_learning_scheduler_supplement_20261022.py',
 'coordinator_pa_memo_20261022_quota18_now_20261022.py':'coordinator_supplement_20261022.py',
 'correction_after_distributed_fixed_pa_memo_20261022_quota18_now_20261022.py':'correction_continuation_supplement_20261022.py',
 'scheduler_watchdog_pa_memo_20261022_quota18_now_20261022.py':'scheduler_watchdog_supplement_20261022.py',
 'optional_delivery_pa_memo_20261022_quota18_now_20261022.py':'optional_delivery_supplement_20261022.py',
 'derived_metrics.py':'derived_metrics_original_supplement_20261022.py',
 'derived_metrics_correction_revised_20261021.py':'derived_metrics_supplement_20261022.py',
 'door_controller_gain_components_20261021.py':'door_controller_gain_components_supplement_20261022.py',
 'optional_revision_diagnostics_20261021.py':'optional_revision_diagnostics_supplement_20261022.py',
 'optional_results_correction_revised_20261021.py':'optional_results_supplement_20261022.py',
 'final_reports_pa_memo_20261022.py':'final_reports_supplement_20261022.py',
 'progress_correction_release_status_20261023_quota18_now_20261022.py':'progress_supplement_20261022.py',
}


def main():
    os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
    assert not (HERE/READY).exists()
    assert not any(v in COPIES.values() for v in read(HERE/'ACTIVE_ENTRYPOINT.json').values() if isinstance(v,str))
    verify_receipt('PA_QUERY_MEMO_PIPELINE_CODE_20261022.receipt.json')
    verify_receipt('INSTANCE_CLASSES_SUPPLEMENT_20261022.receipt.json')
    substitutions={Path(a).stem:Path(b).stem for a,b in COPIES.items()}
    substitutions.update({'INSTANCE_CLASSES_REVISED':'INSTANCE_CLASSES_SUPPLEMENT_20261022','CORRECTION_PARALLEL_NUMERICS_READY_20261023':READY.removesuffix('.receipt.json'),'PARALLEL_NUMERIC_INPUTS_20261023':INPUT.removesuffix('.receipt.json'),'parallel_numeric_fragments_20261023':'parallel_numeric_fragments_supplement_20261022'})
    rows=[]
    for old,new in COPIES.items():
        before=(HERE/old).read_text();after=before
        # Ordinary helper imports such as weighted_rates remain the frozen
        # originals; no mutable module replacement or monkeypatching.
        for src,dst in sorted(substitutions.items(),key=lambda p:-len(p[0])):
            if src in ('derived_metrics','statistics_header_reader_20261023'):continue
            after=after.replace(src,dst)
        after=after.replace("'derived_metrics.py'","'derived_metrics_original_supplement_20261022.py'")
        if new=='statistics_training_selected_reader_20261022.py':
            after=after.replace("FIXED=('safety','fixed_eta10','fixed_eta13','human_progress')","FIXED=('safety','train_selected_fixed')")
            after=after.replace("row['delta_vs_best_aggregate_fixed']","row['delta_vs_train_selected_fixed']").replace('bootstrap_reselects_best_fixed=True','bootstrap_reselects_best_fixed=False')
        if new=='statistics_correction_supplement_20261022.py':
            needle=" for stage in stages:\n";assert after.count(needle)==1
            after=after.replace(needle," stages += ['learning_eval_revised_20261021_trainfixed_'+s for s in NAMES]\n"+needle,1)
        if new=='statistics_tasks_supplement_20261022.py':
            after=after.replace('import signal,subprocess,re','import signal,subprocess,re\nfrom training_selected_primary_statistics_20261022 import primary_task,SCOPES')
            needle=" return {uid('numeric',t):t for t in tasks}"
            after=after.replace(needle," for scene in NAMES:\n  for scope in SCOPES:tasks.append(dict(kind='primary',scene=scene,scope=scope))\n tasks.append(dict(kind='primary',scenes=list(NAMES),scope='all_qualified'))\n"+needle)
            after=after.replace(" kind=task['kind'];arms=", " if task['kind']=='primary':return primary_task(task,new,classes)\n kind=task['kind'];arms=")
        if new=='statistics_supplement_20261022.py':
            after=after.replace("seal(INPUT_PROOF,[HERE/n for n in (", "seal(INPUT_PROOF,[HERE/f'TRAIN_SELECTED_FIXED_20261022_{s}.receipt.json' for s in NAMES]+[HERE/n for n in (")
            needle=" complete_original_landed_label_diagnoses()"
            after=after.replace(needle," for scene in NAMES:\n  stage='learning_eval_revised_20261021_trainfixed_'+scene\n  assert stage_done(stage),stage\n  verify_receipt('TRAIN_SELECTED_FIXED_20261022_'+scene+'.receipt.json')\n"+needle)
            needle="  results[scene]=entry"
            after=after.replace(needle,"  entry['revised']['primary_evaluation']=bytask[canonical(dict(kind='primary',scene=scene,scope='all_qualified'))]\n  entry['revised']['primary_instance_class_gain_splits']={scope:bytask[canonical(dict(kind='primary',scene=scene,scope=scope))] for scope in ('all_qualified','controller_failure','remaining_instances','safety_layer_global_failure')}\n"+needle)
            needle=" for sem,kind in [('original','old_hover')"
            after=after.replace(needle," results['four_scene_training_selected_primary']=bytask[canonical(dict(kind='primary',scenes=list(NAMES),scope='all_qualified'))]\n"+needle)
            needle=" fragments=[HERE/"
            after=after.replace(needle," names += ['training_selected_primary_statistics_20261022.py','statistics_training_selected_reader_20261022.py','analysis_standard_supplement_20261022.py']+[f'TRAIN_SELECTED_FIXED_20261022_{s}.receipt.json' for s in NAMES]\n"+needle)
            needle="    for scope in ('fixed_full','learned_evaluation'):flat.extend"
            # Primary rows are separate and keep the historical reference rows intact.
            marker=" fields=('scene','type','layer','method','success_rate','ci95','success_upper','upper_ci95','controllers','semantics')"
            after=after.replace(marker," for scene in NAMES:flat.extend(dict(scene=scene+'_primary_evaluation',semantics='revised',**r) for r in results[scene]['revised']['primary_evaluation'].get('statistics',[]))\n"+marker)
        if new=='scene_learning_scheduler_supplement_20261022.py':
            after=after.replace('    reviewed=set();diagnostics={};evaluations={}',"    from plan_training_selected_fixed_20261022 import plan,stage_name\n    fixed_evaluations=[]\n    reviewed=set();diagnostics={};evaluations={}")
            needle="        state=snapshot();mapping=jobs_by_stage(state)\n"
            pos=after.index(needle,after.index('    while True:'))
            after=after[:pos]+"        for scene in NAMES:\n            if not (HERE/f'TRAIN_SELECTED_FIXED_20261022_{scene}.receipt.json').exists() and not (HERE/f'LEARNING_BUNDLE_REVISED_20261021_{scene}.receipt.json').exists():continue\n            stage=stage_name(scene)\n            if stage not in fixed_evaluations:\n                if not (HERE/f'{stage}_PREFLIGHT.receipt.json').exists():action('train_selected_fixed_plan_'+scene,'plan_training_selected_fixed_20261022.py',scene)\n                if not stage_ready(stage):action('train_selected_fixed_queue_'+scene,'work_queue.py',stage)\n                fine_dispatch.prepare(stage);fixed_evaluations.append(stage)\n"+after[pos:]
            needle="        state=snapshot();mapping=jobs_by_stage(state)\n        for stage in rollout_stages:"
            after=after.replace(needle,"        rollout_stages += fixed_evaluations\n"+needle)
            after=after.replace("        finished=all(s in evaluations and stage_done(evaluations[s]) for s in active)","        finished=all(s in evaluations and stage_done(evaluations[s]) for s in active) and all(stage_done(s) for s in fixed_evaluations)")
        if new=='correction_continuation_supplement_20261022.py':
            after=after.replace("  evaluations=learning(['lit_narrow_passage','lit_crowded_random','lit_doorway'])", "  earlier=['learning_eval_revised_20261021_'+s for s in ('lit_narrow_passage','lit_crowded_random','lit_doorway')]\n  evaluations=earlier if all(stage_done(s) for s in earlier) else learning(['lit_narrow_passage','lit_crowded_random','lit_doorway'])")
            after=after.replace("  analyze(fixed+evaluations)","  evaluations += ['learning_eval_revised_20261021_trainfixed_'+s for s in NAMES]\n  analyze(fixed+evaluations)")
        if new=='derived_metrics_original_supplement_20261022.py':
            after=after.replace("census=read(HERE/'CONTROLLER_LEVEL_FAILURES.json')","census=read(HERE/'INSTANCE_CLASSES_SUPPLEMENT_20261022.json')['instances']").replace("if r['controller_failure_definite']","if r['remove_in_revised_exclusion']")
        if new=='optional_revision_diagnostics_supplement_20261022.py':
            after=after.replace('from reclassify_instances_20261021 import category,GRID,LABELS,small_diagnosis','from reclassify_instances_20261021 import GRID,LABELS,small_diagnosis\nfrom analysis_standard_supplement_20261022 import classify as category')
            after=after.replace('INSTANCE_CLASSES_DYNAMICS_REVISED_20261021','INSTANCE_CLASSES_DYNAMICS_SUPPLEMENT_20261022')
        if new in ('optional_results_supplement_20261022.py','final_reports_supplement_20261022.py'):
            after=after.replace('INSTANCE_CLASSES_DYNAMICS_REVISED_20261021','INSTANCE_CLASSES_DYNAMICS_SUPPLEMENT_20261022')
        if new=='final_reports_supplement_20261022.py':
            after=after.replace('import csv,collections','import csv,collections\nMETHODS=dict(METHODS,train_selected_fixed="训练集选择的最佳固定动作")\nfrom training_selected_report_tables_20261022 import primary_tables')
            after=after.replace('for method in METHODS:',"for method in ('safety','fixed_eta10','fixed_eta13','human_progress','learned_scene'):").replace('for m in METHODS]',"for m in ('safety','fixed_eta10','fixed_eta13','human_progress','learned_scene')]")
            after=after.replace('def axis(results,additional):','def axis_reference(results,additional):').replace("'AXIS_TABLE.md'","'APPENDIX_REFERENCE_AXIS.md'").replace("'AXIS_TABLE.csv'","'APPENDIX_REFERENCE_AXIS.csv'")
            after=after.replace(' axis(results,additional);decomposition(data,additional);class_gain_tables(results)',' axis_reference(results,additional);primary_tables(results,save,cell,delta,table);decomposition(data,additional);class_gain_tables(results)')
            after=after.replace("policy=read(HERE/'RESOURCE_SCHEDULE_CORRECTIONS_RELEASE_20261023.json')","policy=read(HERE/'RESOURCE_LIMIT_IMMEDIATE_18CPU_20261022.json')")
            start=after.index('资源最新用户授权：');end=after.index('\n\n主数值认',start)
            after=after[:start]+'资源最新用户授权：立即释放六核后固定18CPU总额度（17调度+1本地管理核），GPU保持2份额。RESOURCE_LIMIT_IMMEDIATE_18CPU_20261022.json为准；旧24→18日期安排已被立即释放决定替代。独立资源guard和watchdog持续校验，不能再次释放六核或恢复24核。'+after[end:]
            # Fixed legacy arms belong in the reference appendix. Preserve the
            # full source numbers while the report leads with the new primary.
            after=after.replace(" paragraphs.append(' '.join(gains)+' '.join(paired_gains)"," paragraphs.append('三种历史固定修正的成功率、增益和两层配对差见附表 APPENDIX_REFERENCE_AXIS.md；集中式仍以这三种固定修正为对照。' ")
            after=after.replace(" paragraphs.append('用于讨论死锁解决的实例范围：'+'；'.join(eligible_gains)+", " paragraphs.append('按新版实例类拆分的主固定与学习增益见 PRIMARY_CLASS_GAINS_20261022.md；历史三种固定修正拆分见附表。'+")
            marker=" save('GAPS.md','\\n\\n'.join(notes));save('REPORT.md','\\n\\n'.join(paragraphs)"
            after=after.replace(marker," from training_selected_report_tables_20261022 import report_lead\n paragraphs.insert(0,report_lead(results,cell,delta))\n"+marker)
            after=after.replace("outputs=['REPORT.md',", "outputs=['APPENDIX_REFERENCE_AXIS.md','APPENDIX_REFERENCE_AXIS.csv','MENU_TRAINING_ACTION_RANKING_20261022.md','PRIMARY_CLASS_GAINS_20261022.md','REPORT.md',")
            after=after.replace("outputs=['APPENDIX_REFERENCE_AXIS.md','APPENDIX_REFERENCE_AXIS.csv',", "outputs=['AXIS_TABLE.md','AXIS_TABLE.csv',")
            after=after.replace("'final_reports_supplement_20261022.py']],", "'final_reports_supplement_20261022.py','training_selected_report_tables_20261022.py']],")
            after=after.replace('4a′/4a″多数分类','滤波器平衡点死锁加已到达者阻挡的严格多数/墙前陷阱多数分类').replace('未到达机器人·局的4a′/4a″多数判断','未到达机器人·局的（4a′+已到达者阻挡）/4a″多数判断')
        if new=='progress_supplement_20261022.py':
            after=after.replace(" done=sum(r['completed'] for r in stages+labels+evaluation)"," selected_fixed=[count('learning_eval_revised_20261021_trainfixed_'+s,n) for s,n in zip(SCENES,EVAL)]\n done=sum(r['completed'] for r in stages+labels+evaluation+selected_fixed)")
            after=after.replace('for r in stages+labels+evaluation)','for r in stages+labels+evaluation+selected_fixed)')
            after=after.replace('177600','188352')
            after=after.replace('revised_learning_eval_stages=evaluation,','revised_learning_eval_stages=evaluation,training_selected_fixed_eval_stages=selected_fixed,previous_main_plan=177600,added_primary_comparator_requests=10752,')
            after=after.replace('10月23日09:00新加坡时间降至18CPU','已按用户最新指令立即释放六核，现为18CPU总额度')
        if new.endswith('.py'):ast.parse(after)
        p=HERE/new
        p.write_text(after)
        diff=p.with_suffix(p.suffix+'.diff');diff.write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=old,tofile=new)))
        rows.append(dict(original=old,new=new,original_sha256=sha(HERE/old),new_sha256=sha(p),diff=str(diff)))
    write(HERE/'ANALYSIS_SUPPLEMENT_SOURCE_MAP_20261022.json',dict(utc=utc(),copies=COPIES,static_copies=rows,changes='Posthoc category counts,training-only selected primary comparator,additional exact evaluation scope,report tables',execution_semantics_changed=False,total_cpu=18,gpu_shards=2))
    print('Staged',len(rows),'posthoc analysis variants; not sealed or activated',flush=True)


if __name__=='__main__':main()
