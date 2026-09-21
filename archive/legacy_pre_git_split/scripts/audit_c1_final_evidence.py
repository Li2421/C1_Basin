"""Verify a complete frozen test before assessing the declared efficacy gates."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
import math

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from single_integrator.c1.acceptance import assess


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--sets',type=Path,required=True)
    p.add_argument('--family-size',type=int,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    summary=json.loads((args.evaluation/'summary.json').read_text())
    complete=json.loads((args.evaluation/'complete.json').read_text())
    saved=pickle.loads(args.checkpoint.read_bytes()); config=saved['config']
    if summary['split']!='test' or summary['objective']!='constrained' or not summary['selected_feasible']:
        raise ValueError('Requires an independent test of a selected C1 constrained model')
    if summary['checkpoint_sha256']!=digest(args.checkpoint):raise ValueError('Checkpoint mismatch')
    if summary['sets_sha256']!=digest(args.sets) or config['sets_sha256']!=digest(args.sets):raise ValueError('Test set mismatch')
    if config['objective']!='constrained' or saved['selection']['J_live']>saved['selection']['epsilon']:
        raise ValueError('Checkpoint lacks validation feasibility')
    for name,sha in config['source_hashes'].items():
        if digest(ROOT/name)!=sha:raise ValueError('Frozen source changed: '+name)
    sets=json.loads(args.sets.read_text())
    if summary['execution_noise_seeds']!=sets['test_noise_seeds']:raise ValueError('Test noise changed')
    paths=sorted(args.evaluation.glob('Safety_*.json'))+sorted(args.evaluation.glob('C1_*.json'))
    rows=[json.loads(f.read_text()) for f in paths]
    if len(rows)!=complete['episodes']:raise ValueError('Incomplete evaluation')
    for path in paths:
        if not path.with_suffix('.npz').is_file():raise ValueError('Missing recorded trace')
    report=assess(rows,[r['rid'] for r in sets['pools']['test']],sets['test_noise_seeds'],family_size=args.family_size)
    # Check the first goal independently of the efficacy gates: the recorded
    # trajectory risk must still upper-bound the original deadlock indicator.
    for row in rows:
        risk=row['risk']['J_live']
        dead=row['six_class_outcome'] in ('safe_deadlock','stalled_deadlock')
        if not math.isfinite(risk) or risk<0 or (dead and risk<1-1e-8):
            raise ValueError('Invalid risk or deadlock certificate violation in held-out execution')
    correspondence={}
    for method in ('Safety','C1'):
        group=[r for r in rows if r['method']==method]
        labels=sorted({r['six_class_outcome'] for r in group})
        correspondence[method]=dict(
            mean_risk=sum(r['risk']['J_live'] for r in group)/len(group),
            deadlock_rate=sum(r['six_class_outcome'] in ('safe_deadlock','stalled_deadlock') for r in group)/len(group),
            risk_at_least_one_without_deadlock=sum(r['risk']['J_live']>=1 and r['six_class_outcome'] not in ('safe_deadlock','stalled_deadlock') for r in group),
            ordinary_timeouts_with_zero_risk=sum(r['six_class_outcome']=='other_timeout' and r['risk']['J_live']==0 for r in group),
            by_outcome={label:dict(n=sum(r['six_class_outcome']==label for r in group),
                min_risk=min(r['risk']['J_live'] for r in group if r['six_class_outcome']==label),
                mean_risk=sum(r['risk']['J_live'] for r in group if r['six_class_outcome']==label)/sum(r['six_class_outcome']==label for r in group)) for label in labels})
    report['risk_correspondence']=correspondence
    report['risk_interpretation']='Retrospective full-trajectory event upper bound; not calibrated probability or 20-second prediction; ordinary timeout is not bounded by this certificate'
    report.update(evaluation=str(args.evaluation),checkpoint_sha256=digest(args.checkpoint),sets_sha256=digest(args.sets),
        assessment_sha256=digest(ROOT/'single_integrator/c1/acceptance.py'),auditor_sha256=digest(Path(__file__)),
        record_sha256={f.name:digest(f) for f in paths})
    with args.out.open('x') as stream:json.dump(report,stream,indent=2)
    print(json.dumps(report['gates'],indent=2))


if __name__=='__main__':main()
