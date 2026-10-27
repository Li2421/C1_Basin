"""Seal verified exact native geometry masks and latest static orchestration."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
import ast


def main():
    os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
    from build_pa_spatial_mask_pipeline_20261022 import EQ,CODE
    verify_receipt('PA_SPATIAL_MASK_CANDIDATE_20261022.receipt.json');verify_receipt('PA_SPATIAL_MASK_VALIDATION_PLAN_20261022.receipt.json')
    names=['PA_SPATIAL_MASK_FINITE_20261022.json','PA_SPATIAL_MASK_REAL_MATRIX_20261022.json','PA_SPATIAL_MASK_SOURCE_REVIEW_20261022.json']+[f'PA_SPATIAL_MASK_FULL_REPLAY_prioritized_astar{i}_20261022.json' for i in (0,1)]
    for name in names:assert read(HERE/name)['passed'],name
    assert read(HERE/names[0])['bitwise_path_checks']==658
    assert len(read(HERE/names[1])['saved_queries'])==20
    comparisons=[]
    for i in (0,1):
        r=read(HERE/f'PA_SPATIAL_MASK_FULL_REPLAY_prioritized_astar{i}_20261022.json');previous=read(HERE/f'PA_QUERY_MEMO_FULL_REPLAY_prioritized_astar{i}_20261022.json')
        assert r['steps']==2000 and r['all_non_timing_result_fields_equal'] and all(v['bitwise_equal'] for v in r['arrays'].values())
        assert r['cache_stats']==previous['cache_stats'],'Whole-query choices changed'
        assert r['stationary_edge_stats']['maximum_entries']<=65536
        assert sha(r['plan']['source_raw'])==r['plan']['raw_sha256']
        raw=read(r['plan']['source_raw']);assert sha(raw['trace_path'])==r['plan']['trace_sha256']
        comparisons.append(dict(controller=f'prioritized_astar{i}',previous_pa_memo_wall_seconds=previous['candidate_seconds'],candidate_wall_seconds=r['candidate_seconds'],previous_pa_memo_cpu_seconds=previous['candidate_cpu_seconds'],candidate_cpu_seconds=r['candidate_cpu_seconds'],case_specific_only=True))
    write(HERE/'PA_SPATIAL_MASK_TIMING_REVIEW_20261022.json',dict(utc=utc(),cases=comparisons,not_a_whole_pipeline_speedup_estimate=True))
    if not (HERE/EQ).exists():seal(EQ,[Path(__file__),HERE/'PA_SPATIAL_MASK_CANDIDATE_20261022.receipt.json',HERE/'PA_SPATIAL_MASK_VALIDATION_PLAN_20261022.receipt.json',HERE/'PA_SPATIAL_MASK_TIMING_REVIEW_20261022.json']+[HERE/n for n in names],passed=True,full_saved_steps=4000,finite_native_path_checks=658,saved_four_scene_queries=20,cache_capacity_fallback_exercised=True,scientific_execution_semantics_unchanged=True,new_research_rollouts=0,new_labels=0,new_journals=0)
    verify_receipt(EQ);assert not (HERE/CODE).exists()
    mapping=read(HERE/'PA_SPATIAL_MASK_PIPELINE_SOURCE_MAP_20261022.json')
    for row in mapping['static_copies']:
        assert sha(HERE/row['new'])==row['new_sha256'] and sha(HERE/row['original'])==row['original_sha256']
        if row['new'].endswith('.py'):ast.parse((HERE/row['new']).read_text())
    scene=(HERE/'scene_learning_scheduler_spatial_mask_20261022.py').read_text()
    assert "scene=='lit_circle_five' and not stage_done('learning_eval_revised_20261021_'+scene):continue" in scene
    assert 'merge_completed_circle_learned_before_fixed_preflight' in scene
    assert "originals += [s for s in fixed_evaluations if stage_done(s)]" in scene
    seal(CODE,[Path(__file__),HERE/'build_pa_spatial_mask_pipeline_20261022.py',HERE/'PA_SPATIAL_MASK_PIPELINE_SOURCE_MAP_20261022.json',HERE/EQ,HERE/'ANALYSIS_SUPPLEMENT_READY_20261022.receipt.json',HERE/'CIRCLE_EVALUATION_REUSE_ORDER_20261022.receipt.json']+[p for r in mapping['static_copies'] for p in (HERE/r['new'],Path(r['diff']))],passed=True,static_variants=16,total_cpu=18,gpu_shards=2,analysis_supplement_and_training_selection_unchanged=True,circle_reuse_order_unchanged=True,completed_saved_diagnosis_backfill_allowed=True)
    print('PASS:exact spatial-mask equivalence and16 latest pipeline variants sealed',flush=True)


if __name__=='__main__':main()
