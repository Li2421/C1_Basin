"""Produce per-control trajectory diagnostics from immutable simulator traces."""
import json
from pathlib import Path

from .audit_existing import OUTPUT, SPECS, audit
from .run_controls import MODES


def main():
    records=[]
    for n,_ in SPECS:
        for mode in MODES:
            folder=OUTPUT/'controls'/f'n{n}'/mode
            expected=json.loads((folder/'summary.json').read_text())['rollouts']
            rows=[]
            for item in expected:
                row=audit(folder/'traces'/f"{item['rollout_id']}.npz")
                row.update(N=n,mode=mode)
                rows.append(row)
            records.extend(rows)
    (OUTPUT/'all_control_diagnostics.json').write_text(json.dumps(records,indent=2)+'\n')
    print(f'Wrote {len(records)} control diagnostics')


if __name__=='__main__': main()
