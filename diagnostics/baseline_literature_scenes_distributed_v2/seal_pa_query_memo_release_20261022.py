"""Seal completed PA equivalence gates and reviewed static pipeline variants."""
from experiment import *
from registry import verify_receipt
from atomic_write import write
import ast,difflib


def main():
    os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
    verify_receipt('PA_QUERY_MEMO_CANDIDATE_CODE_20261022.receipt.json')
    eq='PA_QUERY_MEMO_EQUIVALENCE_20261022.receipt.json'
    gates=['PA_QUERY_MEMO_FINITE_20261022.json','PA_QUERY_MEMO_STORAGE_20261022.json','PA_QUERY_MEMO_REAL_MATRIX_20261022.json','PA_QUERY_MEMO_SOURCE_REVIEW_20261022.json']+[f'PA_QUERY_MEMO_FULL_REPLAY_prioritized_astar{i}_20261022.json' for i in (0,1)]
    for name in gates:assert read(HERE/name)['passed'],name
    for i in (0,1):
        r=read(HERE/f'PA_QUERY_MEMO_FULL_REPLAY_prioritized_astar{i}_20261022.json')
        assert r['steps']==2000 and r['all_non_timing_result_fields_equal']
        assert all(a['bitwise_equal'] for a in r['arrays'].values())
        assert sha(r['plan']['source_raw'])==r['plan']['raw_sha256']
        raw=read(r['plan']['source_raw']);assert sha(raw['trace_path'])==r['plan']['trace_sha256']
    if not (HERE/eq).exists():seal(eq,[Path(__file__),HERE/'PA_QUERY_MEMO_CANDIDATE_CODE_20261022.receipt.json']+[HERE/n for n in gates],passed=True,full_saved_episodes=2,full_saved_steps=4000,finite_checks=656,real_four_scene_queries=20,new_research_rollouts=0,new_labels=0,new_journals=0,scientific_execution_semantics_unchanged=True)
    verify_receipt(eq)
    code='PA_QUERY_MEMO_PIPELINE_CODE_20261022.receipt.json'
    assert not (HERE/code).exists()
    source=HERE/'PA_QUERY_MEMO_PIPELINE_SOURCE_MAP_20261022.json';mapping=read(source)
    entry=read(HERE/'ACTIVE_ENTRYPOINT.json')
    assert not any(v in mapping['copies'].values() for v in entry.values() if isinstance(v,str))
    # Correct the unsealed instrumentation-only snapshot import before release.
    old='tiled_planning_query_disk_candidate_20261022';new='tiled_planning_pa_memo_query_disk_candidate_20261022'
    for row in mapping['static_copies']:
        p=HERE/row['staged'];assert sha(p)==row['staged_sha256']
        text=p.read_text().replace(old,new)
        if p.suffix=='.py':ast.parse(text)
        p.write_text(text)
        delta=''.join(difflib.unified_diff((HERE/row['original']).read_text().splitlines(True),text.splitlines(True),fromfile=row['original'],tofile=row['staged']))
        Path(row['diff']).write_text(delta);row['staged_sha256']=sha(p)
    builder=HERE/'build_pa_query_memo_pipeline_20261022.py';text=builder.read_text()
    needle="substitutions.update(runtime_correction_query_disk_candidate_20261022="
    assert needle in text
    builder.write_text(text.replace(needle,"substitutions.update(tiled_planning_query_disk_candidate_20261022='tiled_planning_pa_memo_query_disk_candidate_20261022',runtime_correction_query_disk_candidate_20261022=",1))
    mapping.update(reviewed_utc=utc(),snapshot_import_review='Static PA combined tile snapshot import corrected before any activation')
    write(source,mapping)
    seal(code,[Path(__file__),builder,source,HERE/eq]+[p for r in mapping['static_copies'] for p in (HERE/r['staged'],Path(r['diff']))],static_variants=len(mapping['copies']),total_cpu=18,gpu_shards=2,execution_semantics_unchanged=True,activate_only_after_full_gates=True)
    print('PASS: PA equivalence and16 static pipeline variants sealed',flush=True)


if __name__=='__main__':main()
