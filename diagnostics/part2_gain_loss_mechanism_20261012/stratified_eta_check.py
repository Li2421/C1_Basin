"""Check whether frozen indicators add state information beyond eta identity."""
from __future__ import annotations
import csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent

def main():
    with (HERE/'heldout_prediction.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    out={};rng=np.random.default_rng(20261012)
    for scene in ('toy_give_way','ring_exchange'):
        for split in ('validation','heldout_test'):
            z=[r for r in rows if r['scene']==scene and r['split']==split]
            by=defaultdict(list)
            for r in z:by[int(r['eta_index'])].append(r)
            for cond in ('condition0','condition1'):
                observed=0.
                for a in by.values():
                    y=np.asarray([int(r['target']) for r in a]);p=np.asarray([int(r[cond]) for r in a])
                    observed+=float(np.dot(y,p-p.mean()))
                null=[]
                for _ in range(5000):
                    v=0.
                    for a in by.values():
                        y=rng.permutation([int(r['target']) for r in a]);p=np.asarray([int(r[cond]) for r in a])
                        v+=float(np.dot(y,p-p.mean()))
                    null.append(v)
                out[f'{scene}:{split}:{cond}']=dict(statistic=observed,
                    one_sided_permutation_p=(1+sum(v>=observed-1e-12 for v in null))/(1+len(null)),
                    permutations=len(null),eta_strata=len(by),
                    note='Outcome labels permuted among states within each fixed eta index; no threshold refit')
    (HERE/'stratified_eta_summary.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:round(v['one_sided_permutation_p'],4) for k,v in out.items()}))

if __name__=='__main__':main()
