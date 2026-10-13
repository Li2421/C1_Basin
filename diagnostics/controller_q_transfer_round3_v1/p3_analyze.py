"""Exploratory Q-level/η-contrast and no-step probe failure localization."""
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

HERE=Path(__file__).resolve().parent


def read(name):return json.loads((HERE/name).read_text())
def write(name,v):(HERE/name).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def center(x,axes):
    for ax in axes:x=x-x.mean(axis=ax,keepdims=True)
    return x


def main():
    assert read('P3_TRAINING.json')['status']=='COMPLETE'
    p2=read('P2PRIME_RESULTS.json');lab=np.load(HERE/'P2PRIME_LABELS.npz')['outcomes'][5:,288:]
    Y=(lab[:,:,:,4:]==1).mean(-1);U=(lab[:,:,:,4:]==-1).mean(-1)
    draws=np.random.default_rng(2026101209).integers(64,size=(5000,64));rows=[]
    for count in (3,5):
        for arm in ('R0','R4_action','R4_t0','R4_openloop'):
            z=np.load(HERE/'p2prime_fits'/f'C{count}_{arm}'/'predictions.npz')
            logits=z['held'];wrong=z['mismatched'];P=expit(logits).mean(0)
            level_error=float(np.mean((P.mean((1,2))-Y.mean((1,2)))**2))
            eta_error=float(np.mean((P.mean(1)-P.mean((1,2))[:,None]-(Y.mean(1)-Y.mean((1,2))[:,None]))**2))
            three_error=float(np.mean((center(P,(0,1,2))-center(Y,(0,1,2)))**2))
            for c,name in enumerate(('TF','FT')):
                pick=logits[:,c].argmax(-1);mispick=wrong[:,c].argmax(-1)
                a=np.take_along_axis(Y[c][None],pick[:,:,None],axis=2)[:,:,0].mean(0)
                b=np.take_along_axis(Y[c][None],mispick[:,:,None],axis=2)[:,:,0].mean(0)
                pooled=Y[c,:,p2['pooled_best_fixed_eta']]
                diff=a-pooled;match=a-b
                rows.append(dict(source_controllers=count,arm=arm,held_controller=name,success=float(a.mean()),
                                 gain_vs_source_pooled=float(diff.mean()),gain_ci95=np.quantile(diff[draws].mean(1),[.025,.975]).tolist(),
                                 matched_minus_mismatched=float(match.mean()),mismatch_ci95=np.quantile(match[draws].mean(1),[.025,.975]).tolist(),
                                 q_level_mse=level_error,controller_eta_pattern_mse=eta_error,threeway_residual_mse=three_error,
                                 q_total_mse=float(np.nanmean((P[c]-np.divide(Y[c],1-U[c],out=np.full_like(Y[c],np.nan),where=U[c]<1))**2))))
    result=dict(status='COMPLETED EXPLORATORY DIAGNOSTIC',new_rollouts=0,rows=rows,
                interpretation='P3 was activated after primary selection result; no new outcome was used to construct static features, but held test already influenced activation. Not independent confirmation.',
                source_controller_count_contrast='C3 vs C5 uses same held state and protocol; C5=all available 5 source controllers',
                physical_steps_for_probe=0,probe='one analytic step p+dt*a0,v=a0; no env.step')
    write('P3_RESULTS.json',result)
    print(json.dumps([{k:r[k] for k in ('source_controllers','arm','held_controller','success','gain_vs_source_pooled','matched_minus_mismatched')} for r in rows],indent=2))


if __name__=='__main__':main()
