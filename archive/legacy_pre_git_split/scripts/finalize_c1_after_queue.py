"""Wait for a specific live queue job, then gate and assess its frozen test."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--queue',type=Path,required=True)
    p.add_argument('--index',type=int,required=True)
    p.add_argument('--training',type=Path,required=True)
    p.add_argument('--sets',type=Path,required=True)
    p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--family-size',type=int,required=True)
    args=p.parse_args()
    fingerprint=digest(args.queue)
    spec=json.loads(args.queue.read_text())
    if not 0<=args.index<len(spec['jobs']):raise ValueError('Queue index out of range')
    folder=args.queue.parent/args.queue.stem
    marker=folder/f'{args.index:02d}.complete.json'
    failed=folder/f'{args.index:02d}.failed.json'
    process=json.loads(args.queue.with_suffix('.process.json').read_text())
    pid=process['pid']
    proc=Path(f'/proc/{pid}')
    def identity():
        try:
            fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
            command=(proc/'cmdline').read_bytes().split(b'\0')
            return fields[19] if str(args.queue).encode() in command else None
        except FileNotFoundError:return None
    started=identity()
    print(dict(waiting_for=str(marker),queue_pid=pid),flush=True)
    while not marker.exists():
        if failed.exists():raise RuntimeError('Upstream queue job failed; inspect '+str(failed))
        if started is None or identity()!=started:
            raise RuntimeError('Upstream queue is no longer live; do not infer completion or restart it')
        time.sleep(30)
    record=json.loads(marker.read_text())
    if record['spec_sha256']!=fingerprint or record['argv']!=spec['jobs'][args.index]:
        raise ValueError('Completed queue source mismatch')
    complete=json.loads((args.training/'complete.json').read_text())
    config=json.loads((args.training/'config.json').read_text())
    if complete['updates']!=64 or config['updates']!=64 or config['objective']!='constrained':
        raise ValueError('Expected complete frozen constrained training')
    expected_family=48 if config['version']=='c1_scene_deadlock_union_v1' else 6
    if args.family_size!=expected_family:raise ValueError('Wrong multiplicity family')
    checkpoint=args.training/'best_feasible.pkl'
    if not checkpoint.exists():
        with args.out.open('x') as stream:
            json.dump(dict(status='no_validation_feasible_checkpoint',test_opened=False,goal_met=False,
                training=str(args.training),config_sha256=digest(args.training/'config.json')),stream,indent=2)
        print('Training finished without feasibility; test remains unopened',flush=True)
        return
    modules={'c1_deadlock_union_v1':'single_integrator.c1.evaluate_deadlock_union',
             'c1_scene_deadlock_union_v1':'single_integrator.c1.evaluate_scene_deadlock_union'}
    if not (args.evaluation/'complete.json').exists():
        subprocess.run([sys.executable,'-u','-m',modules[config['version']],
            '--checkpoint',str(checkpoint),'--sets',str(args.sets),'--out',str(args.evaluation),'--split','test'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'scripts/audit_c1_final_evidence.py','--checkpoint',str(checkpoint),
        '--sets',str(args.sets),'--evaluation',str(args.evaluation),'--family-size',str(args.family_size),
        '--out',str(args.out)],cwd=ROOT,check=True)


if __name__=='__main__':main()
