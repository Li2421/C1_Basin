"""Matched optimizer development report; never opens independent test data."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path)
    root=parser.parse_args().folder
    runs={}
    for name in ('adam','sgd'):
        folder=root/name
        runs[name]={k:json.loads((folder/(k+'.json')).read_text()) for k in
                    ('config','history','validation','complete','difficulty')}
        assert runs[name]['complete']['completed_updates']==16
        assert not runs[name]['complete']['test_opened']
    a,b=runs['adam'],runs['sgd']
    for key in a['config']:
        if key!='optimizer':assert a['config'][key]==b['config'][key],key
    for x,y in zip(a['history'],b['history']):
        assert x['rids']==y['rids'] and x['noise_seed']==y['noise_seed']
    for key in ('J_live','J_def','either_deadlock','success','timeout'):
        np.testing.assert_allclose(a['validation'][0][key],b['validation'][0][key],rtol=1e-6,atol=1e-10)
    summary={}
    for name,run in runs.items():
        validation=[{k:v for k,v in row.items() if k!='episodes'} for row in run['validation']]
        groups={}
        for label,zero in [('zero_risk',True),('positive_risk',False)]:
            rows=[r for r in run['history'] if (r['pre']['J_live']==0)==zero]
            ratios=[r['post']['J_def']/r['pre']['J_def'] for r in rows if r['pre']['J_def']>1e-12]
            groups[label]=dict(n=len(rows),restarts=sum(r['decision']['proposal']=='restart' for r in rows),
                median_deviation_ratio=float(np.median(ratios)) if ratios else None,
                median_gradient_norm=float(np.median([r['gradient_norm'] for r in rows])) if rows else None)
        summary[name]=dict(validation=validation,validation_feasible=run['complete']['validation_feasible'],
            batch_groups=groups,accepted_updates=sum(r['decision']['status']=='accepted' for r in run['history']),
            safety_baseline_deadlock_rate=run['difficulty']['either_deadlock'])
    result=dict(scope='single-seed matched development audit, not efficacy confirmation',
        configurations_matched_except_optimizer=True,training_batches_and_noise_matched=True,
        warm_validation_matched=True,arms=summary,
        limitations=['Equal numeric learning rates do not imply equal effective step sizes',
                     'One training seed and reused development starts, not cross-scene generalization',
                     'Preserving initial performance without lowering risk is not a solution',
                     'No automatic independent evaluation or full-training continuation'])
    (root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
