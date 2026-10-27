"""Use the already validated operational checkpoint to release queued work."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import signal

H = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    target = H / 'TAIL_CHECKPOINT_HANDOFF.receipt.json'
    assert not target.exists(), 'One operational handoff only'
    proof = read(H / 'CHECKPOINT_COMPATIBILITY.json')
    assert proof['passed']
    for path, expected in proof['implementation_sources'].items():
        assert sha(path) == expected, path
    guard = next(g for g in read(H / 'TAIL_REBALANCE.receipt.json')['original_guards']
                 if g['owner']['job'] == '29077_6')
    owner = guard['owner']
    cmdline = Path(f"/proc/{owner['pid']}/cmdline").read_bytes().replace(b'\0', b' ').decode()
    assert 'diagnostics/cross_scene_pipeline_v1/worker_resume.py main' in cmdline
    status = read(H / 'worker_status' / (owner['job'] + '.json'))
    case = status['task']
    assert case.get('cross_scene_new') and case['controller_payload']['nominal']['type_code'] == 'prioritized_astar'
    assert case['rollout_uid'] in guard['protected']
    saved = sorted((H / 'operational_checkpoints' / case['rollout_uid']).glob('*.receipt.json'),
                   key=lambda p: p.stat().st_mtime)
    assert saved
    snapshot = read(saved[-1])
    assert snapshot['scientific_label'] is None and sha(snapshot['path']) == snapshot['sha256']
    receipt = dict(utc=datetime.now(timezone.utc).isoformat(),
                   reason='Release one unstarted request behind a running PA request; both continue in parallel.',
                   owner=owner, current_rollout_uid=case['rollout_uid'],
                   current_case_sha256=hashlib.sha256(json.dumps(case,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest(),
                   latest_snapshot_before_signal=str(saved[-1]),
                   latest_snapshot_before_signal_sha256=sha(saved[-1]),
                   next_step_before_signal=snapshot['next_step'],
                   signal='SIGUSR1',
                   mechanism='Existing handler requests an operational snapshot and exits without a scientific label; helpers wait for old PID exit, then load the verified latest checkpoint.',
                   numerical_compatibility_proof=str(H / 'CHECKPOINT_COMPATIBILITY.json'),
                   numerical_compatibility_proof_sha256=sha(H / 'CHECKPOINT_COMPATIBILITY.json'),
                   worker_source_sha256=sha(H / 'worker_resume.py'),
                   helper_source_sha256=sha(H / 'worker_tail.py'),
                   script_sha256=sha(__file__),
                   no_running_request_reexecuted_from_start=True,
                   no_model_proposal_or_choice_changed=True,
                   signal_sent=False)
    target.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    os.kill(owner['pid'],signal.SIGUSR1)
    receipt['signal_sent']=True
    target.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(receipt,ensure_ascii=False))


if __name__ == '__main__':
    main()
