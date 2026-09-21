"""One frozen scene/baseline/seed: train, gate feasibility, test, and assess."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',required=True)
    p.add_argument('--baseline-seed',type=int,choices=(0,1),required=True)
    p.add_argument('--seed',type=int,choices=(0,1,2),required=True)
    args=p.parse_args()
    root=ROOT/'results/c1_scene_deadlock_union'
    experiment=json.loads((root/'experiment.json').read_text())
    matches=[b for b in experiment['baselines'] if b['scene']==args.scene and b['baseline_seed']==args.baseline_seed]
    if len(matches)!=1:raise ValueError('Comparison not in the frozen experiment')
    entry=matches[0];manifest=Path(entry['path'])
    if digest(manifest)!=entry['manifest_sha256']:raise ValueError('Baseline manifest changed')
    for path,sha in experiment['source_hashes'].items():
        if digest(ROOT/path)!=sha:raise ValueError('Experiment source changed: '+path)
    sets=root/args.scene/'sets.json'
    if digest(sets)!=json.loads((root/'sets_manifest.json').read_text())[args.scene]:raise ValueError('Scene set changed')
    name=f'baseline{args.baseline_seed}_seed{args.seed}'
    train=root/args.scene/name; test=root/args.scene/(name+'_test')
    report=root/args.scene/(name+'_acceptance.json')
    locks=root/'locks';locks.mkdir(exist_ok=True)
    with (locks/(args.scene+'_'+name+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not (train/'complete.json').exists():
            # Partial jobs require explicit recovery inspection, rather than
            # possibly duplicating a trainer launched outside this wrapper.
            if train.exists():raise RuntimeError('Existing incomplete training needs process/checkpoint inspection')
            command=[sys.executable,'-u','-m','single_integrator.c1.train_scene_deadlock_union',
                '--frozen-baseline',str(manifest),'--sets',str(sets),'--out',str(train),
                '--baseline-seed',str(args.baseline_seed),'--seed',str(args.seed),
                '--updates','64','--batch-size','8','--lr','.0001','--dual-lr','.1','--validation-every','8']
            subprocess.run(command,cwd=ROOT,check=True)
        config=json.loads((train/'config.json').read_text())
        complete=json.loads((train/'complete.json').read_text())
        if (config['source_hashes']!=experiment['source_hashes'] or config['sets_sha256']!=digest(sets)
                or config['baseline_sha256']!=entry['checkpoint_sha256'] or config['seed']!=args.seed
                or config['baseline_seed']!=args.baseline_seed or complete['updates']!=64
                or config['objective']!='constrained'):
            raise ValueError('Completed training does not match frozen comparison')
        checkpoint=train/'best_feasible.pkl'
        status_path=root/args.scene/(name+'_pipeline_result.json')
        if not checkpoint.exists():
            result=dict(status='training_completed_without_validation_feasibility',test_opened=False,
                goal_met=False,seed=args.seed,baseline_seed=args.baseline_seed,scene=args.scene)
        else:
            saved=pickle.loads(checkpoint.read_bytes())
            if saved['config']!=config or saved['selection']['J_live']>config['epsilon']:
                raise ValueError('Selected checkpoint is not validation-feasible')
            if not (test/'complete.json').exists():
                subprocess.run([sys.executable,'-u','-m','single_integrator.c1.evaluate_scene_deadlock_union',
                    '--checkpoint',str(checkpoint),'--sets',str(sets),'--out',str(test),'--split','test'],cwd=ROOT,check=True)
            if not report.exists():
                subprocess.run([sys.executable,'scripts/audit_c1_final_evidence.py','--checkpoint',str(checkpoint),
                    '--sets',str(sets),'--evaluation',str(test),'--family-size','48','--out',str(report)],cwd=ROOT,check=True)
            assessment=json.loads(report.read_text())
            if assessment['checkpoint_sha256']!=digest(checkpoint) or assessment['sets_sha256']!=digest(sets):
                raise ValueError('Acceptance report changed inputs')
            result=dict(status='independent_test_assessed',test_opened=True,
                endpoint_gates_passed=assessment['all_endpoint_gates_passed'],
                gates=assessment['gates'],seed=args.seed,baseline_seed=args.baseline_seed,scene=args.scene,
                note='One comparison only; entire research goal requires all planned evidence')
        if status_path.exists():
            if json.loads(status_path.read_text())!=result:raise ValueError('Pipeline result changed')
        else:
            with status_path.open('x') as stream:json.dump(result,stream,indent=2)
        print(result,flush=True)


if __name__=='__main__':main()
