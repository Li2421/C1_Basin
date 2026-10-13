"""Exploratory, threshold-free horizon diagnostics on frozen cached trajectories."""
import csv
import json
from pathlib import Path
import numpy as np
from score_frozen import average_precision

HERE = Path(__file__).resolve().parent

def read(name):
    with (HERE / name).open(newline='') as f:
        return list(csv.DictReader(f))

def ap_ci(rows, y, score, seed):
    states = sorted(set(r['uid'] for r in rows))
    groups = {s: [i for i, r in enumerate(rows) if r['uid'] == s] for s in states}
    rng = np.random.default_rng(seed)
    a = []
    for _ in range(1000):
        idx = [i for s in rng.choice(states, len(states), replace=True) for i in groups[s]]
        v = average_precision(y[idx], score[idx])
        if v is not None: a.append(v)
    return np.quantile(a, [.025, .975]).tolist() if a else [None, None]

def main():
    out=[]
    for scene, name, positive in [('toy_give_way','toy_loss_analysis.csv','loss'),
                                  ('ring_exchange','ring_gain_analysis.csv','gain')]:
        for split in ('discovery','validation','heldout_test'):
            rows=[r for r in read(name) if r['split']==split and r['b15_outcome']!='unknown']
            y=np.array([r['b15_outcome']==positive for r in rows],bool)
            for H in (10,20,80):
                measures=[('negative_worst_progress',np.array([-float(r[f'H{H}_worst_progress_delta'])
                          if r[f'H{H}_worst_progress_delta'] else np.nan for r in rows]))]
                if scene=='ring_exchange':
                    measures.append(('absolute_angular_travel_advantage',np.array([
                        float(r[f'TF_H{H}_angular_sum_abs'])-float(r[f'TT_H{H}_angular_sum_abs'])
                        if r[f'TF_H{H}_angular_sum_abs'] and r[f'TT_H{H}_angular_sum_abs'] else np.nan for r in rows])))
                for metric,score in measures:
                    finite=np.isfinite(score)
                    if not finite.all():
                        rr=[r for r,z in zip(rows,finite) if z]
                        yy=y[finite];ss=score[finite]
                    else: rr=rows;yy=y;ss=score
                    out.append(dict(scene=scene,split=split,horizon=H,metric=metric,n=len(rr),
                        positives=int(yy.sum()),prevalence=float(yy.mean()),ap=average_precision(yy,ss),
                        ap_state_bootstrap_95=ap_ci(rr,yy,ss,20261012+H) if split=='heldout_test' else None,
                        median_score_positive=float(np.median(ss[yy])) if yy.any() else None,
                        median_score_negative=float(np.median(ss[~yy])) if (~yy).any() else None))
    with (HERE/'horizon_ablation.csv').open('w',newline='') as f:
        fields=['scene','split','horizon','metric','n','positives','prevalence','ap','ap_state_bootstrap_95',
                'median_score_positive','median_score_negative']
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(out)
    (HERE/'horizon_ablation.json').write_text(json.dumps(out,indent=2)+'\n')
    for r in out:
        if r['split']=='heldout_test':print(r['scene'],r['horizon'],r['metric'],'AP',round(r['ap'],3),'base',round(r['prevalence'],3))

if __name__=='__main__':main()
