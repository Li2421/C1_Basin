"""Nonphysical label-trained eta-only comparator; never a mechanism condition."""
from __future__ import annotations
import csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from score_frozen import average_precision

HERE=Path(__file__).resolve().parent

def main():
    with (HERE/'heldout_prediction.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    out={}
    for scene in ('toy_give_way','ring_exchange'):
        a=[r for r in rows if r['scene']==scene];dis=[r for r in a if r['split']=='discovery']
        prior=float(np.mean([int(r['target']) for r in dis]));rates={j:float(np.mean([int(r['target']) for r in dis if int(r['eta_index'])==j])) for j in range(16)}
        for split in ('validation','heldout_test'):
            test=[r for r in a if r['split']==split];y=np.asarray([int(r['target']) for r in test],bool)
            score=np.asarray([rates[int(r['eta_index'])] for r in test]);p=score>prior
            tp=int(np.sum(p&y));fp=int(np.sum(p&~y));fn=int(np.sum(~p&y));tn=int(np.sum(~p&~y))
            out[f'{scene}:{split}']=dict(tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),
                balanced_accuracy=.5*(tp/max(tp+fn,1)+tn/max(tn+fp,1)),average_precision=average_precision(y,score),
                discovery_prevalence=prior,discovery_eta_rates=rates,
                warning='uses discovery final B15 labels; benchmark only, prohibited as a mechanism condition')
    (HERE/'eta_prior_baseline.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print({k:(v['balanced_accuracy'],v['average_precision']) for k,v in out.items()})

if __name__=='__main__':main()
