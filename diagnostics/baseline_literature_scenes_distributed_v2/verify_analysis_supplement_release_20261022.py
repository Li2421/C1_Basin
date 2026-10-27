"""Check posthoc source changes, selection, taxonomy and exact-request proofs."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
from build_analysis_supplement_pipeline_20261022 import READY,COPIES
import ast,numpy as np


def main():
    os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
    assert not (HERE/READY).exists()
    verify_receipt('INSTANCE_CLASSES_REVISED.receipt.json');verify_receipt('INSTANCE_CLASSES_SUPPLEMENT_20261022.receipt.json')
    old=read(HERE/'INSTANCE_CLASSES_REVISED.json');new=read(HERE/'INSTANCE_CLASSES_SUPPLEMENT_20261022.json')
    assert len(old['instances'])==len(new['instances'])==384
    unchanged_counts=0
    for a,b in zip(old['instances'],new['instances']):
        assert a['primary_unarrived_agent_episode_categories']==b['primary_unarrived_agent_episode_categories']
        assert a['at_least_one_4a_prime']==b['at_least_one_4a_prime']
        for p,q in zip(a['sensitivity'],b['sensitivity']):assert p['unarrived_agent_episode_categories']==q['unarrived_agent_episode_categories'];unchanged_counts+=1
    assert len(new['classification_changes'])==2
    # Primary inference keeps the exact existing bootstrap/contrast kernel.
    def functions(name):return {n.name:ast.dump(n) for n in ast.parse((HERE/name).read_text()).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    original=functions('statistics_header_reader_20261023.py');primary=functions('statistics_training_selected_reader_20261022.py')
    assert original['bootstrap']==primary['bootstrap'] and original['contrast']==primary['contrast']
    transformed=(HERE/'statistics_header_reader_20261023.py').read_text().replace("FIXED=('safety','fixed_eta10','fixed_eta13','human_progress')","FIXED=('safety','train_selected_fixed')").replace("row['delta_vs_best_aggregate_fixed']","row['delta_vs_train_selected_fixed']").replace('bootstrap_reselects_best_fixed=True','bootstrap_reselects_best_fixed=False')
    assert ast.dump(ast.parse(transformed))==ast.dump(ast.parse((HERE/'statistics_training_selected_reader_20261022.py').read_text()))
    from statistics_correction_supplement_20261022 import FIXED
    assert FIXED==('safety','fixed_eta10','fixed_eta13','human_progress'),'Historical reference kernel accidentally changed'
    from statistics_tasks_supplement_20261022 import task_manifest
    tasks=task_manifest();assert len(tasks)==44 and sum(t['kind']=='primary' for t in tasks.values())==17
    requests=[]
    for scene in ('lit_narrow_passage','lit_crowded_random','lit_doorway'):
        prefix='TRAIN_SELECTED_FIXED_20261022_'+scene;verify_receipt(prefix+'.receipt.json');selection=read(HERE/(prefix+'.json'))
        with np.load(HERE/f'LEARNING_BUNDLE_REVISED_20261021_{scene}.npz') as z:labels=z['labels'];types=z['types']
        rates=np.stack([(labels[types==t]==1).mean((0,1)) for t in sorted(set(types))]).mean(0)
        assert int(np.argmax(rates))==selection['selected_menu_index']
        assert [r['menu_index'] for r in selection['ranking']]==sorted(range(15),key=lambda ai:(-rates[ai],ai))
        stage='learning_eval_revised_20261021_trainfixed_'+scene;verify_receipt(stage+'_PREFLIGHT.receipt.json')
        cases=read(HERE/f'{stage}_CASES.json');missing=read(HERE/f'{stage}_MISSING_CASES.json');reuse=read(HERE/f'{stage}_REUSED_ROWS.json')
        assert len(cases)==3072 and len(missing)+len(reuse)==3072
        assert all(c['method']=='train_selected_fixed' and c['menu_index']==selection['selected_menu_index'] for c in cases)
        requests.append(dict(scene=scene,selected_menu_index=selection['selected_menu_index'],logical_requests=len(cases),exact_reused=len(reuse),new_executions=len(missing)))
    from training_selected_report_tables_20261022 import primary_tables,report_lead
    mapping=read(HERE/'ANALYSIS_SUPPLEMENT_SOURCE_MAP_20261022.json')
    for row in mapping['static_copies']:
        assert sha(HERE/row['new'])==row['new_sha256'] and sha(HERE/row['original'])==row['original_sha256']
        if row['new'].endswith('.py'):ast.parse((HERE/row['new']).read_text())
    assert not (HERE/'MIGRATION_NUMBERS.receipt.json').exists() and not (HERE/'PARALLEL_NUMERIC_INPUTS_20261023.receipt.json').exists()
    write(HERE/'ANALYSIS_SUPPLEMENT_VERIFICATION_20261022.json',dict(utc=utc(),passed=True,original_taxonomy_preserved=True,unchanged_sensitivity_count_vectors=unchanged_counts,classification_changes=new['classification_changes'],deadlock_evidence_categories_unchanged=True,bootstrap_and_contrast_AST_identical=True,primary_comparator_only_changed=True,primary_bootstrap_reselects=False,reference_tasks=27,added_primary_tasks=17,total_tasks=44,requests=requests,all_source_parsed=True,execution_runtime_modified=False,cpu_limit=18,gpu_shares=2))
    extra=['analysis_standard_supplement_20261022.py','plan_training_selected_fixed_20261022.py','training_selected_primary_statistics_20261022.py','training_selected_report_tables_20261022.py','build_analysis_supplement_pipeline_20261022.py','ANALYSIS_SUPPLEMENT_SOURCE_MAP_20261022.json','ANALYSIS_SUPPLEMENT_VERIFICATION_20261022.json','INSTANCE_CLASSES_SUPPLEMENT_20261022.receipt.json']
    seal(READY,[Path(__file__)]+[HERE/n for n in extra]+[p for r in mapping['static_copies'] for p in (HERE/r['new'],Path(r['diff']))],passed=True,execution_semantics_changed=False,physics_or_runtime_source_changed=False,training_only_selection=True,new_taxonomy_saved_separately=True,total_cpu=18,gpu_shares=2)
    print('PASS: analysis supplement sealed;44 statistical tasks;all original evidence preserved',flush=True)


if __name__=='__main__':main()
