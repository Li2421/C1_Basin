"""Summarize a completed paired evaluation without reading policy parameters."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from single_integrator.c1.paired_statistics import summarize
from single_integrator.c1.training.persistence import atomic_save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--family-size', type=int, default=6)
    p.add_argument('--endpoint', choices=('strict','strict_or_stalled'), default='strict')
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    complete = json.loads((args.evaluation/'complete.json').read_text())
    rows = [json.loads(f.read_text()) for name in ('Safety', 'C1')
            for f in sorted(args.evaluation.glob(name+'_*.json'))]
    if len(rows) != complete['episodes']:
        raise ValueError('evaluation is incomplete or changed')
    result = summarize(rows, family_size=args.family_size, endpoint=args.endpoint)
    result['source'] = str(args.evaluation)
    result['evaluation_protocol'] = json.loads((args.evaluation/'summary.json').read_text())
    atomic_save(args.out, result)
    print(json.dumps({k:v for k,v in result.items() if k!='evaluation_protocol'}, indent=2))


if __name__ == '__main__':
    main()
