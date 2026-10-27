"""Seal only completed exactness/performance gates; never activates a process."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
import difflib,ast

EQ='CBS_QUERY_DISK_EQUIVALENCE_20261022.receipt.json'
CODE='CBS_QUERY_DISK_PIPELINE_CODE_20261022.receipt.json'


def main():
    os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
    if (HERE/CODE).exists():verify_receipt(EQ);verify_receipt(CODE);print('Already sealed; no changes');return
    gates=['CBS_QUERY_DISK_STORAGE_TEST_20261022.json','CBS_QUERY_DISK_FINITE_GRIDS_cold_final_20261022.json','CBS_QUERY_DISK_FINITE_GRIDS_warm_final_20261022.json','CBS_QUERY_DISK_REAL_MATRIX_20261022.json','CBS_QUERY_DISK_SOURCE_REVIEW_20261022.json','CBS_QUERY_DISK_FULL_REPLAY_cold_20261022.json','CBS_QUERY_DISK_FULL_REPLAY_warm_20261022.json','CROSS_ETA_QUERY_REUSE_20261022.json']
    for name in gates:
        r=read(HERE/name);assert r['passed'],name
        for p,digest in r.get('source_hashes',{}).items():assert sha(p)==digest,(name,p)
    cold=read(HERE/gates[5]);warm=read(HERE/gates[6])
    for row in (cold,warm):
        assert row['steps']==2000 and row['all_non_timing_result_fields_equal']
        assert all(v['bitwise_equal'] for v in row['arrays'].values())
    assert cold['disk_namespace']==warm['disk_namespace']
    assert warm['disk_stats'][1]>0,'Warm run must exercise interprocess cache'
    assert warm['candidate_seconds']<cold['candidate_seconds']*.95,'No demonstrated full-episode warm benefit; review before deployment'
    for old in ('CBS_START_SIGNATURE_EQUIVALENCE_20261022.receipt.json','CBS_START_SIGNATURE_PIPELINE_CODE_20261022.receipt.json','CORRECTION_REVISION.receipt.json'):verify_receipt(old)
    pairs=[('cbs_kernel_start_signature_v2_candidate_20261022.cpp','cbs_kernel_query_disk_candidate_20261022.cpp'),('tiled_planning_start_signature_v2_20261022.py','tiled_planning_query_disk_candidate_20261022.py'),('reservation_certificate_start_signature_v2_20261022.py','reservation_certificate_query_disk_candidate_20261022.py'),('reservation_certificate_v2_start_signature_v2_20261022.py','reservation_certificate_v2_query_disk_candidate_20261022.py'),('runtime_correction_start_signature_v2_candidate_20261022.py','runtime_correction_query_disk_candidate_20261022.py'),('test_cbs_start_signature_v2_candidate_20261022.py','test_query_disk_finite_grids_candidate_20261022.py'),('test_cbs_start_signature_real_matrix_20261022.py','test_query_disk_real_matrix_candidate_20261022.py'),('replay_cbs_start_signature_v2_20261022.py','replay_query_disk_candidate_20261022.py')]
    paths=[]
    for origin,target in pairs:
        p=HERE/target;delta=p.with_suffix(p.suffix+'.diff')
        text=''.join(difflib.unified_diff((HERE/origin).read_text().splitlines(True),p.read_text().splitlines(True),fromfile=origin,tofile=target))
        if (HERE/EQ).exists():assert delta.read_text()==text
        else:delta.write_text(text)
        paths.extend([p,delta])
    core=['cbs_query_disk_memo_candidate_20261022.hpp','cbs_kernel_query_disk_candidate_20261022.so','cbs_start_signature_cache_v2_20261022.hpp','CBS_QUERY_DISK_CACHE_CONTRACT_20261022.md','test_query_disk_storage_candidate_20261022.cpp','test_query_disk_storage_candidate_20261022','build_query_disk_candidate_20261022.py',Path(__file__).name,'CBS_START_SIGNATURE_EQUIVALENCE_20261022.receipt.json','CBS_START_SIGNATURE_V2_FULL_REPLAY_20261022.json','measure_cross_eta_query_reuse_20261022.py','CROSS_ETA_QUERY_REUSE_PLAN_20261022.json']
    paths.extend(HERE/n for n in core+gates)
    raw=read(cold['plan']['source_raw'])
    paths.extend([Path(cold['plan']['source_raw']),Path(raw['trace_path']),HERE/'cbs_kernel_flat_cache.cpp',HERE/'cbs_kernel_flat_cache.so'])
    if not (HERE/EQ).exists():seal(EQ,paths,passed=True,original_search_cpp_restored=True,cold_warm_physical_array_bytes_equal=True,cold_seconds=cold['candidate_seconds'],warm_seconds=warm['candidate_seconds'],cold_cpu_seconds=cold['candidate_cpu_seconds'],warm_cpu_seconds=warm['candidate_cpu_seconds'],single_saved_case_only=True,no_whole_pipeline_factor_claim=True,scientific_semantics_changed=False,cache_slot_data_limit_bytes=65536*16384,no_runtime_patching=True)
    m=read(HERE/'CBS_QUERY_DISK_PIPELINE_SOURCE_MAP_20261022.json');assert m['current_label_batch_preserved']=='worker_correction_start_signature_20261022.sbatch'
    code=[]
    for row in m['static_copies']:
        assert sha(HERE/row['original'])==row['original_sha256'],row['original']
        assert sha(HERE/row['staged'])==row['staged_sha256'],row['staged']
        if row['staged'].endswith('.py'):ast.parse((HERE/row['staged']).read_text())
        code.extend([HERE/row['staged'],Path(row['diff'])])
    code.extend(HERE/n for n in (EQ,'CBS_QUERY_DISK_PIPELINE_SOURCE_MAP_20261022.json','build_query_disk_pipeline_candidate_20261022.py','worker_correction_start_signature_20261022.py','worker_correction_start_signature_20261022.sbatch','CBS_START_SIGNATURE_PIPELINE_CODE_20261022.receipt.json'))
    seal(CODE,code,passed=True,static_variants=14,current_labels_preserved_in_memory_backend=True,worker_signals_required=False,execution_UIDs_and_seeds_unchanged=True,resource_policy_unchanged='24now->18 Oct23 09:00 Singapore;2GPUshares')
    verify_receipt(EQ);verify_receipt(CODE)
    print('PASS: shared exact-query cache and14future-stage variants sealed; not activated',flush=True)


if __name__=='__main__':main()
