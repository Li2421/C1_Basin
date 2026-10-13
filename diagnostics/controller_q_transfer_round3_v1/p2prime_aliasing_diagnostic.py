"""Post-label diagnostic for controllers aliased by static R3.

The split-half product removes the positive sampling bias of a squared Q4/Q8
difference. Unknowns are kept as unknown; any all-unknown half-cell is omitted
from this auxiliary diagnostic and counted, never treated as a failed episode.
"""
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent


def rate(x):
    ok=x>=0
    n=ok.sum(-1)
    return np.divide((x==1).sum(-1),n,out=np.full(n.shape,np.nan),where=n>0)


def pair(name,a,b,block,first,second,seed):
    qa0=rate(block[a][...,first]);qb0=rate(block[b][...,first])
    qa1=rate(block[a][...,second]);qb1=rate(block[b][...,second])
    d0=qa0-qb0;d1=qa1-qb1;product=d0*d1
    state=np.nanmean(product,axis=1)
    draws=np.random.default_rng(seed).integers(len(state),size=(5000,len(state)))
    ci=np.quantile(np.nanmean(state[draws],axis=1),[.025,.975]).tolist()
    return dict(pair=name,states=len(state),eta=16,
                R3_identical_by_static_audit=True,
                split_half_true_squared_Q_difference=float(np.nanmean(state)),
                split_half_state_cluster_ci95=ci,
                identical_input_minimum_pair_mean_MSE=float(np.nanmean(state)/4),
                incomplete_state_eta_halfcells=int((~np.isfinite(product)).sum()),
                unknown_episode_count=int((block[a]==-1).sum()+(block[b]==-1).sum()))


def main():
    assert json.loads((HERE/'P2PRIME_LABEL_AUDIT.json').read_text())['status']=='COMPLETE EXACT UID LABEL TENSOR'
    names=json.loads((HERE/'P2PRIME_PROTOCOL.json').read_text())['source_controllers']+json.loads((HERE/'P2PRIME_PROTOCOL.json').read_text())['held_controllers']
    x=np.load(HERE/'P2PRIME_LABELS.npz')['outcomes']
    rows=[pair('FF_vs_TT',names.index('FF'),names.index('TT'),x[:,256:288,:,:4],slice(0,2),slice(2,4),2026101211),
          pair('TF_vs_FT',names.index('TF'),names.index('FT'),x[:,288:,:,:],slice(0,4),slice(4,8),2026101212)]
    result=dict(status='EXPLORATORY POST-LABEL REPRESENTATION DIAGNOSTIC',rows=rows,
                interpretation='A positive split-half squared contrast proves R3 cannot represent both controllers Q exactly at identical input; does not establish selection transfer or causality. Negative finite-sample estimates are possible.')
    (HERE/'P2PRIME_R3_ALIASING_DIAGNOSTIC.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
