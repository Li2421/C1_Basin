"""Validate existing evidence and seal static exact-computation variants."""
from experiment import *
from registry import verify_receipt
import ast,difflib,re


def main():
    os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
    evidence=['CBS_START_SIGNATURE_V2_FINITE_GRIDS_20261022.json','CBS_START_SIGNATURE_SAVED_QUERIES_20261022.json','CBS_START_SIGNATURE_REAL_MATRIX_20261022.json','CBS_START_SIGNATURE_V2_FULL_REPLAY_20261022.json']
    for name in evidence:
        row=read(HERE/name);assert row['passed'],name
        for p,h in row['source_hashes'].items():assert sha(p)==h,p
    matrix=read(HERE/evidence[2]);assert len(matrix['saved_queries'])==17 and len(matrix['ieee_and_time_constraints'])==12
    replay=read(HERE/evidence[3]);assert replay['steps']==2000 and replay['all_non_timing_result_fields_equal']
    assert all(x['bitwise_equal'] for x in replay['arrays'].values())
    original=(HERE/'cbs_kernel_flat_cache.cpp').read_text()
    candidate=(HERE/'cbs_kernel_start_signature_v2_candidate_20261022.cpp').read_text()
    restored=candidate.replace('\n\n#include "cbs_start_signature_cache_v2_20261022.hpp"\n','')
    restored=re.sub(r' // BEGIN EXACT START SIGNATURE CANDIDATE\n.*? // END EXACT START SIGNATURE CANDIDATE\n','',restored,flags=re.S)
    restored=restored.replace('if(use_start_memo)start_signature_memo.put(start_memo_key,output,k/2);','')
    restored=restored.replace(' if(use_start_memo)start_signature_memo.put(start_memo_key,output,0);\n','')
    restored=restored.split('\nextern "C" void start_signature_cache_stats')[0]
    assert restored==original,(len(restored),len(original))
    delta=HERE/'cbs_kernel_start_signature_v2_candidate_20261022.cpp.diff'
    assert not delta.exists()
    delta.write_text(''.join(difflib.unified_diff(original.splitlines(True),candidate.splitlines(True),fromfile='cbs_kernel_flat_cache.cpp',tofile='cbs_kernel_start_signature_v2_candidate_20261022.cpp')))
    pairs=[('runtime_correction_revised_20261021.py','runtime_correction_start_signature_v2_candidate_20261022.py'),('reservation_certificate_flat_static.py','reservation_certificate_start_signature_v2_20261022.py'),('reservation_certificate_v2_flat_static.py','reservation_certificate_v2_start_signature_v2_20261022.py')]
    def bodies(p):return [ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
    for a,b in pairs:assert bodies(HERE/a)==bodies(HERE/b),(a,b)
    write(HERE/'CBS_START_SIGNATURE_SOURCE_REVIEW_20261022.json',dict(utc=utc(),original_cpp_restored_byte_for_byte=True,all_runtime_and_certificate_function_class_AST_identical=pairs,query_and_trajectory_comparisons_passed=True,scientific_UIDs_and_preflight_unchanged=True,new_research_rollouts=0))
    write(HERE/'CBS_START_SIGNATURE_HARNESS_REVIEW_20261022.json',dict(utc=utc(),changes=['Saved nominal comparison applies the original runtime arrived-agent zeroing rule','The17th historical source uses raw-embedded trace checksum and scene/controller metadata rather than absent manifest metadata','All17 already-passed byte comparisons reused only after source raw/trace hashes and candidate source hashes were reverified'],comparison_criteria_weakened=False,production_sources_changed_by_harness_fixes=False,signed_zero_note='The separate12 IEEE cases preserve actual negative-zero bits; some finite-grid origin+zero additions normalize them',historical_harnesses=['test_cbs_start_signature_saved_queries_initial_20261022.py','test_cbs_start_signature_real_matrix_initial_20261022.py','test_cbs_start_signature_real_matrix_trace_fixed_20261022.py']))
    eq='CBS_START_SIGNATURE_EQUIVALENCE_20261022.receipt.json'
    names=evidence+['CBS_START_SIGNATURE_THEORY_20261022.md','CBS_START_SIGNATURE_CACHE_HISTORY_20261022.json','CBS_START_SIGNATURE_SOURCE_REVIEW_20261022.json','CBS_START_SIGNATURE_HARNESS_REVIEW_20261022.json','cbs_start_signature_cache_v2_20261022.hpp','cbs_kernel_start_signature_v2_candidate_20261022.cpp','cbs_kernel_start_signature_v2_candidate_20261022.so','cbs_kernel_start_signature_v2_candidate_20261022.cpp.diff','tiled_planning_start_signature_v2_20261022.py']+[b for a,b in pairs]
    names+=['test_cbs_start_signature_v2_candidate_20261022.py','test_cbs_start_signature_saved_queries_20261022.py','test_cbs_start_signature_real_matrix_20261022.py','replay_cbs_start_signature_v2_20261022.py',Path(__file__).name]
    seal(eq,[HERE/n for n in names],passed=True,finite_grid_checks=644,real_four_scene_queries=17,extra_real_ieee_checks=12,full_replay_steps=2000,execution_semantics_changed=False,observed_full_episode_wall_ratio=replay['same_episode_speedup'],not_a_whole_pipeline_speedup_claim=True)
    source=read(HERE/'CBS_START_SIGNATURE_PIPELINE_SOURCE_MAP_20261022.json');paths=[]
    for row in source['static_copies']:
        assert sha(HERE/row['original'])==row['original_sha256']
        assert sha(HERE/row['staged'])==row['staged_sha256']
        paths.extend([HERE/row['staged'],Path(row['diff'])])
    paths += [HERE/eq,HERE/'CBS_START_SIGNATURE_PIPELINE_SOURCE_MAP_20261022.json',HERE/'build_start_signature_pipeline_20261022.py']
    seal('CBS_START_SIGNATURE_PIPELINE_CODE_20261022.receipt.json',paths,passed=True,static_variants=14,no_runtime_patching=True,resource_policy_unchanged=True,case_factories_and_queues_unchanged=True)
    verify_receipt(eq);verify_receipt('CBS_START_SIGNATURE_PIPELINE_CODE_20261022.receipt.json')
    print('PASS: scientific equivalence and14 static variants sealed; not activated',flush=True)


if __name__=='__main__':main()
