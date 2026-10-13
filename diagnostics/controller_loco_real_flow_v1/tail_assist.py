"""Two-hour, five-CPU outcome-blind assistance on disjoint frozen tail UIDs.

The original 19 round-robin workers begin at the head of MISSING_CASES and
advance in order. These five workers start from the final 12,288 UIDs, so they
cannot contend with the original workers during the two-hour allocation.
They use the exact Stage A execute implementation and preserve raw/journals.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from diagnostics.controller_loco_real_flow_v1 import run

HERE = Path(__file__).resolve().parent
TAIL_SIZE = 12_288
WORKERS = 5


def main(index):
    assert 0 <= index < WORKERS
    rows = json.loads((HERE / 'MISSING_CASES.json').read_text())
    tail = rows[-TAIL_SIZE:]
    assert len(tail) == TAIL_SIZE
    # Equal, nonoverlapping exact-UID subsets; no outcome-dependent selection.
    subset = tail[index::WORKERS]
    rec = run.execute(subset, f'tail_assist_{index}')
    dest = HERE / f'TAIL_ASSIST_{index}.json'
    dest.write_text(json.dumps(rec, indent=2) + '\n')
    print(json.dumps(rec), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('index', type=int)
    main(parser.parse_args().index)
