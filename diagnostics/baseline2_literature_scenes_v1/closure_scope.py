"""Explicit user-directed v1 closure; no new experiments may be submitted."""
from experiment import *

FLOW_NAMES={s:NAMES[s] for s in ('lit_narrow_passage','lit_crowded_random','lit_doorway')}
STOP_REASON='用户决定停止，转入 v2'


def circle_gap():
    return read(HERE/'CIRCLE_FLOW_USER_STOP.json')


def main():
    import subprocess
    assert (HERE/'COORDINATOR_STOP').exists()
    live=subprocess.check_output(['squeue','-u',os.environ['USER'],'-h','-r','-o','%i|%F|%j']).decode().splitlines()
    roots=set(read(HERE/'CIRCLE_USER_STOP_CANCELLATION.json')['cancelled_array_roots'])
    assert not any(line.split('|')[1] in roots for line in live),live
    stage='safety_lit_circle_five';planned=read(HERE/f'{stage}_CASES.json');raw=[]
    for p in sorted((HERE/stage/'raw').glob('*.json')):
        r=read(p);assert sha(r['trace_path'])==r['trace_sha256']
        raw.append(dict(path=str(p),sha256=sha(p),rollout_uid=r['rollout_uid'],
                        trace=r['trace_path'],trace_sha256=r['trace_sha256'],
                        numerical_failure=r['numerical_failure']))
    write(HERE/'CIRCLE_FLOW_RETAINED_RAW_INDEX.json',raw)
    write(HERE/'CIRCLE_FLOW_USER_STOP.json',dict(
        utc=utc(),scene='lit_circle_five',reason=STOP_REASON,planned=len(planned),
        completed_raw_retained=len(raw),unwritten_discarded=len(planned)-len(raw),
        already_recorded_numerical_unknowns_retained=sum(r['numerical_failure'] for r in raw),
        new_unknowns_from_cancellation=0,restore_unwritten=False,
        use_partial_results_for_effect_estimation=False,
        completed_capability_screening_retained=48000,
        no_future_circle_batches=True,no_learning_pilot_batches=True))
    for scene in NAMES:
        path=HERE/f'PILOT_GAP_{scene}.json'
        assert not path.exists()
        write(path,dict(reason=STOP_REASON,scope='Both BC families and every pilot extension were not started; no new batches authorized',user_directed=True))
    write(HERE/'FINAL_FLOW_SCOPE.json',dict(
        utc=utc(),included=FLOW_NAMES,excluded={'lit_circle_five':circle_gap()},
        combined_label='三个已完成流程版场景的有效状态池',
        four_scene_combination_permitted=False,paper_evidence_retained_for_all_four=True,
        paper_bvc_v2_workers_unchanged=True,v1_no_new_submissions=True))
    seal('USER_DIRECTED_CLOSURE.receipt.json',[HERE/n for n in (
        'USER_STOP_V1_TO_V2.json','CIRCLE_USER_STOP_CANCELLATION.json',
        'CIRCLE_FLOW_RETAINED_RAW_INDEX.json','CIRCLE_FLOW_USER_STOP.json',
        'FINAL_FLOW_SCOPE.json','RESOURCE_AUTHORIZATION_LATEST.json','closure_scope.py',
        *[f'PILOT_GAP_{s}.json' for s in NAMES])],
        authority='Explicit user instruction',no_synthetic_unknowns=True)
    print(canonical(circle_gap()))


if __name__=='__main__':main()
