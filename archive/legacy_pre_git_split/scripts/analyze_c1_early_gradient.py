"""Paired start-cluster analysis of a fixed local early-intervention probe."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    args=parser.parse_args();p=args.folder
    protocol=json.loads((p/'protocol.json').read_text())
    complete=json.loads((p/'complete.json').read_text())
    records=json.loads((p/'records.json').read_text())
    gradients=json.loads((p/'gradients.json').read_text())
    assert complete['replays']==len(records)==400
    indexed={(r['rid'],r['seed'],r['arm']):r for r in records}
    assert len(indexed)==400
    seeds=protocol['evaluation_seeds'];arms=protocol['arms']
    matrices={a:np.array([[[float(indexed[i,s,a][key]) for key in ('either_deadlock','success')]
                           for s in seeds] for i in range(20)]) for a in arms}
    baseline=matrices['reference']
    rng=np.random.default_rng(2026091806);bootstrap=rng.integers(0,20,(5000,20))
    def paired_difference(a,b,column):
        per_start=(matrices[a][:,:,column]-matrices[b][:,:,column]).mean(axis=1)
        return dict(mean=float(per_start.mean()),ci95=np.quantile(per_start[bootstrap].mean(axis=1),[.025,.975]).tolist())
    summaries={}
    for arm in arms:
        rows=[r for r in records if r['arm']==arm]
        dead_to_success=dead_to_timeout=new_dead=0
        for row in rows:
            base=indexed[row['rid'],row['seed'],'reference']
            dead_to_success+=bool(base['either_deadlock'] and row['success'])
            dead_to_timeout+=bool(base['either_deadlock'] and row['outcome']=='other_timeout')
            new_dead+=bool(not base['either_deadlock'] and row['either_deadlock'])
        summaries[arm]=dict(n=len(rows),deadlocks=sum(r['either_deadlock'] for r in rows),
            successes=sum(r['success'] for r in rows),ordinary_timeouts=sum(r['outcome']=='other_timeout' for r in rows),
            baseline_deadlock_to_success=dead_to_success,baseline_deadlock_to_timeout=dead_to_timeout,new_deadlocks=new_dead,
            mean_actual_early_action_rms=float(np.mean([r['actual_early_action_rms'] for r in rows])),
            deadlock_reduction_vs_reference=paired_difference('reference',arm,0),
            success_increase_vs_reference=paired_difference(arm,'reference',1))
    safety={key:sum(r['safety'][key] for r in records) for key in
            ('agent_collision_steps','wall_collision_steps','outside_endpoints','cbf_violations','speed_violations')}
    assert not any(safety.values())
    result=dict(scope='local per-start diagnostic only; no trained shared-policy claim',
        records_sha256=hashlib.sha256((p/'records.json').read_bytes()).hexdigest(),
        difficulty_passed=bool(baseline[:,:,0].mean()>=.1 and np.sum(baseline[:,:,0].sum(axis=1)>0)>=5),
        safety=safety,arms=summaries,
        negative_early_deadlock_advantage_vs_positive=paired_difference('positive_early','negative_early',0),
        negative_early_deadlock_advantage_vs_random=paired_difference('random_early','negative_early',0),
        nonzero_gradient_starts=sum(g['gradient_norm']>0 for g in gradients),
        prediction_risk_lowered_starts=sum(g['negative_prediction_risk']<g['mean_prediction_risk'] for g in gradients),
        limitations=['20 independent starts, four outcome seeds; uncertainty may be large',
                     'Per-start fitted directions; not shared-policy training',
                     'Directions normalized to linearized residual RMS, not equal realized applied-action RMS',
                     'Random direction is one fixed draw per start',
                     'No automatic full-training launch or confirmation claim'])
    (p/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
