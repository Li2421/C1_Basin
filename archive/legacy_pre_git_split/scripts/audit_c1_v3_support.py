"""Read-only analysis of saved executions; outputs a new V3 audit artifact."""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def audit(root):
    reports = {}
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        rows = [(f, json.loads(f.read_text())) for f in sorted(folder.glob('4*.json'))]
        if not rows:
            continue
        counts = np.zeros(3, int)
        onsets = []
        for f, r in rows:
            if not r['deadlock']:
                continue
            with np.load(f.with_suffix('.npz')) as z:
                flags = z['candidate_deadlock'].astype(bool)
                counts += [flags[:300].sum(), flags[300:500].sum(), flags[500:].sum()]
                onsets.append(float((np.flatnonzero(z['deadlock'])[0]+1)*.05))
        reports[folder.name] = dict(episodes=len(rows), unresolved_deadlocks=len(onsets),
            first_event_after15=sum(t>15 for t in onsets), first_event_after25=sum(t>25 for t in onsets),
            stagnation_seconds_0_15_25_42_5=(counts*.05).tolist())
    return dict(source=str(root), scope='Descriptive audit of already-seen data; not V3 validation.',
        finding='Legacy P+g scores only [5,15]s; the last 27.5s have no direct P/g cost.',
        caveat='Early geometry may predict late outcomes; timing alone does not prove causation.', runs=reports)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = audit(ROOT/'results/c1_four_objectives_multiseed/evaluation')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
