"""Paired C5 minus C3 held-controller selection from frozen predictions."""
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent


def main():
    labels=np.load(HERE/'P2PRIME_LABELS.npz')['outcomes'][5:,288:,:,:,]
    Y=(labels[:,:,:,4:]==1).mean(-1)
    U=(labels[:,:,:,4:]==-1).mean(-1)
    draws=np.random.default_rng(2026101217).integers(64,size=(5000,64))
    rows=[]
    for arm in ('R0','R1','R3','R4_action','R3_R4'):
        z3=np.load(HERE/'p2prime_fits'/f'C3_{arm}'/'predictions.npz')['held']
        z5=np.load(HERE/'p2prime_fits'/f'C5_{arm}'/'predictions.npz')['held']
        for c,name in enumerate(('TF','FT')):
            pick3=z3[:,c].argmax(-1);pick5=z5[:,c].argmax(-1)
            v3=np.take_along_axis(Y[c][None],pick3[:,:,None],axis=2)[:,:,0]
            v5=np.take_along_axis(Y[c][None],pick5[:,:,None],axis=2)[:,:,0]
            u3=np.take_along_axis(U[c][None],pick3[:,:,None],axis=2)[:,:,0]
            u5=np.take_along_axis(U[c][None],pick5[:,:,None],axis=2)[:,:,0]
            diff=(v5-v3).mean(0)
            robust=(v5-v3-u3*(pick5!=pick3)).mean(0)
            upper=(v5+u5*(pick5!=pick3)-v3).mean(0)
            rows.append(dict(arm=arm,held_controller=name,source_count_3_success=float(v3.mean()),
                             source_count_5_success=float(v5.mean()),c5_minus_c3=float(diff.mean()),
                             paired_ci95=np.quantile(diff[draws].mean(1),[.025,.975]).tolist(),
                             unknown_robust_lower=float(robust.mean()),unknown_robust_upper=float(upper.mean()),
                             unknown_robust_ci95=np.quantile(robust[draws].mean(1),[.025,.975]).tolist(),
                             unknown_robust_simultaneous_ci95=np.quantile(robust[draws].mean(1),[.0025,.9975]).tolist(),
                             changed_eta_fraction=float(np.mean(pick5!=pick3))))
    result=dict(status='COMPLETE PAIRED SOURCE CONTROLLER COUNT ABLATION',
                source_count_3=['FF','TT','isotropic'],source_count_5=['FF','TT','rotate_negative','isotropic','rotate_positive'],
                bootstrap='5000 state-cluster resamples; same held physical states and outcome seeds; Bonferroni over 10 arm-controller count contrasts',rows=rows)
    (HERE/'P2PRIME_SOURCE_COUNT_ANALYSIS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for r in rows:print(r['arm'],r['held_controller'],round(100*r['c5_minus_c3'],2),
                        [round(100*x,2) for x in r['unknown_robust_simultaneous_ci95']])


if __name__=='__main__':main()
