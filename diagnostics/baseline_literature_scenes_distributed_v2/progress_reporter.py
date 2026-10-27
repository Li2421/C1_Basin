"""Count landed logical requests every minute, independently of queue batches."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
import fcntl
import time

REUSED={}


def count(stage,planned):
    proof=HERE/f'{stage}_REUSED_ROWS.json'
    if proof.exists() and stage not in REUSED:REUSED[stage]=len(read(proof))
    raw=sum(1 for _ in (HERE/stage/'raw').glob('*.json'))
    completed=raw+REUSED.get(stage,0)
    assert completed<=planned,(stage,completed,planned)
    return dict(stage=stage,completed=completed,planned=planned,remaining=planned-completed,
        new_records=raw,reused=REUSED.get(stage,0))


def snapshot_progress():
    states=dict(zip(NAMES,[64,112,16,64]));qualified=dict(zip(NAMES,[44,44,45,45]))
    baseline=[count(f'{layer}_baseline_{s}',48*n*2) for layer in ('distributed','central') for s,n in states.items()]
    baseline += [count('distributed_whitebox_'+s,n*2) for s,n in states.items()]
    fixed=[count(f'{layer}_fixed_{s}',qualified[s]*n*2*3) for layer in ('distributed','central') for s,n in states.items()]
    labels=[count('learning_labels_'+s,6144) for s in NAMES]
    evaluation=[count('learning_eval_'+s,1536 if s=='lit_circle_five' else 3072) for s in NAMES]
    central=[count('central_learning_eval_'+s,3072) for s in ('lit_narrow_passage','lit_doorway')]
    main=[*baseline,*fixed,*labels,*evaluation,*central]
    assert sum(r['planned'] for r in main)==227264
    ability=count('capability_lit_crowded_random',119808)
    scope=read(HERE/'OPTIONAL_SCOPE.json')
    optional=[count(stage,row['planned_requests']) for stage,row in scope['stages'].items()]
    assert sum(r['planned'] for r in optional)==127168
    completed=sum(r['completed'] for r in main);all_done=completed+ability['completed']+sum(r['completed'] for r in optional)
    status=read(HERE/'STATUS.json')
    out=dict(utc=utc(),phase=status['phase'],main_registered_or_preflight_verified_completed=completed,
        main_planned_logical_requests=227264,main_remaining=227264-completed,main_request_percent=100*completed/227264,
        fixed_completed=sum(r['completed'] for r in fixed),fixed_planned=136128,
        fixed_remaining=sum(r['remaining'] for r in fixed),fixed_percent=100*sum(r['completed'] for r in fixed)/136128,
        baseline_completed=sum(r['completed'] for r in baseline),fixed_stages=fixed,baseline_stages=baseline,
        eta_labels_completed=sum(r['completed'] for r in labels),eta_labels_planned=24576,eta_labels_by_scene=labels,
        distributed_learned_evaluation_completed=sum(r['completed'] for r in evaluation),distributed_learned_evaluation_planned=10752,
        distributed_learned_evaluation_by_scene=evaluation,central_learned_evaluation=central,
        central_learned_reuse_pending_preflight=sum(r['remaining'] for r in central),
        ability_gate_completed=ability['completed'],ability_gate_planned=119808,
        second_stage_completed=sum(r['completed'] for r in optional if r['stage'].startswith('optional_dynamics_')),
        second_stage_planned=90752,other_optional_planned=36416,optional_stages=optional,
        all_v2_completed=all_done,all_v2_planned_logical_requests=474240,all_v2_remaining=474240-all_done,
        all_v2_request_percent=100*all_done/474240,
        scope='Logical requests including verified exact reuse and landed unknowns. Completion is not success, new physical trajectories, elapsed-time fraction, or final source/DB audit.')
    write(HERE/'LIVE_PROGRESS_PERCENT.json',out)
    with (HERE/'PROGRESS_HISTORY.jsonl').open('a') as history:
        history.write(canonical({k:v for k,v in out.items() if k not in ('fixed_stages','baseline_stages','eta_labels_by_scene','distributed_learned_evaluation_by_scene','central_learned_evaluation','optional_stages')})+'\n')
    return out


def main():
    with (HERE/'PROGRESS_REPORTER.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify_receipt('PROGRESS_REPORTER.receipt.json')
        write(HERE/'PROGRESS_REPORTER_PID.json',dict(utc=utc(),pid=os.getpid()))
        while True:
            try:
                snapshot_progress()
                if (HERE/'V2_COMPLETE.json').exists():return
            except Exception as exc:write(HERE/'PROGRESS_REPORTER_ERROR.json',dict(utc=utc(),error=repr(exc)))
            time.sleep(60)


if __name__=='__main__':main()
