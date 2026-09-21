"""Frozen two-gate analysis; no hyperparameter selection or training launch."""
import argparse
import hashlib
import json
from pathlib import Path
import warnings
import numpy as np
from scipy.stats import rankdata, spearmanr, ConstantInputWarning


def analyze(folder):
    read=lambda name:json.loads((folder/name).read_text())
    protocol=read('protocol.json');complete=read('complete.json')
    rows=read('records.json');diagnostics=read('diagnostics.json')
    n=len(protocol['starts']);seeds=protocol['evaluation_seeds'];arms=protocol['arms']
    assert n==25 and len(seeds)==2 and len(rows)==complete['replays']==200
    assert not complete['independent_test_opened']
    indexed={(r['rid'],r['seed'],r['arm']):r for r in rows}
    assert len(indexed)==200
    assert {r['rid'] for r in diagnostics}==set(range(n)) and len(diagnostics)==n
    diagnostics=sorted(diagnostics,key=lambda r:r['rid'])
    scores=np.array([r['prediction_risk'] for r in diagnostics])
    assert np.isfinite(scores).all()
    mat={a:np.array([[[float(indexed[i,s,a][k]) for k in ('either_deadlock','success')]
                         for s in seeds] for i in range(n)]) for a in arms}
    base=mat['reference'];frequency=base[:,:,0].mean(axis=1)
    bootstrap=np.random.default_rng(2026091869).integers(0,n,(10000,n))
    def auc(s,d):
        labels=d.reshape(-1);rank=rankdata(np.repeat(s,d.shape[1]))
        positives=labels.sum();negative=len(labels)-positives
        if positives==0 or negative==0:return np.nan
        return float((rank@labels-positives*(positives+1)/2)/(positives*negative))
    def correlation(s,f):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',ConstantInputWarning)
            return float(spearmanr(s,f).statistic)
    def estimate(value,draws):
        draws=np.asarray(draws);valid=np.isfinite(draws)
        return dict(value=float(value) if np.isfinite(value) else None,
            ci95=np.quantile(draws[valid],[.025,.975]).tolist() if valid.any() else None,
            valid_bootstrap_fraction=float(valid.mean()))
    prediction=dict(spearman=estimate(correlation(scores,frequency),
        [correlation(scores[b],frequency[b]) for b in bootstrap]),
        auc=estimate(auc(scores,base[:,:,0]),[auc(scores[b],base[b,:,0]) for b in bootstrap]))
    prediction['zero_risk_starts']=int(np.sum(scores==0))
    prediction['deadlocks_at_zero_risk_starts']=int(base[scores==0,:,0].sum())
    prediction['by_start']=[dict(rid=i,prediction_risk=float(scores[i]),future_deadlock_frequency=float(frequency[i])) for i in range(n)]
    # Do not split tied risks across display bins: intervals are descriptive only.
    edges=np.unique(np.quantile(scores,[0,.25,.5,.75,1]))
    prediction['score_groups']=[]
    for i in range(max(1,len(edges)-1)):
        mask=(scores>=edges[i]) & ((scores<=edges[-1]) if i==len(edges)-2 or len(edges)==1 else scores<edges[i+1])
        prediction['score_groups'].append(dict(starts=int(mask.sum()),
            mean_score=float(scores[mask].mean()) if mask.any() else None,
            deadlock_frequency=float(frequency[mask].mean()) if mask.any() else None))
    def difference(a,b,column,alpha=.05):
        per_start=(mat[a][:,:,column]-mat[b][:,:,column]).mean(axis=1)
        return dict(mean=float(per_start.mean()),confidence=1-alpha,
            ci=np.quantile(per_start[bootstrap].mean(axis=1),[alpha/2,1-alpha/2]).tolist())
    summaries={}
    for arm in arms:
        selected=[r for r in rows if r['arm']==arm]
        summaries[arm]=dict(deadlocks=int(mat[arm][:,:,0].sum()),successes=int(mat[arm][:,:,1].sum()),
            ordinary_timeouts=sum(r['outcome']=='other_timeout' for r in selected),
            original_deadlock_to_success=sum(bool(indexed[r['rid'],r['seed'],'reference']['either_deadlock'] and r['success']) for r in selected),
            original_deadlock_to_timeout=sum(bool(indexed[r['rid'],r['seed'],'reference']['either_deadlock'] and r['outcome']=='other_timeout') for r in selected),
            new_deadlocks=sum(bool(not indexed[r['rid'],r['seed'],'reference']['either_deadlock'] and r['either_deadlock']) for r in selected))
    advantages={a:difference(a,'negative',0,alpha=.05/3) for a in ('reference','positive','random')}
    success=difference('negative','reference',1)
    nonzero={d['rid'] for d in diagnostics if d['gradient_norm']>0}
    calibration_passed=bool(nonzero) and all(d['calibration'][a]['matched'] for d in diagnostics if d['rid'] in nonzero for a in ('negative','positive','random'))
    amplitudes={}
    for arm in ('negative','positive','random'):
        selected=[r for r in rows if r['arm']==arm and r['rid'] in nonzero]
        total=sum(r['direct_action_components'] for r in selected)
        amplitudes[arm]=float(np.sqrt(sum(r['direct_action_sum_squares'] for r in selected)/total)) if total else 0.
    mean_amplitude=np.mean(list(amplitudes.values()))
    amplitude_passed=bool(mean_amplitude>0 and all(abs(a/mean_amplitude-1)<=.1 for a in amplitudes.values()))
    safety={k:sum(r['safety'][k] for r in rows) for k in ('agent_collision_steps','wall_collision_steps','outside_endpoints','cbf_violations','speed_violations')}
    difficulty=bool(frequency.mean()>=.1 and np.sum(frequency>0)>=5)
    prediction_passed=all(prediction[k]['value'] is not None and prediction[k]['valid_bootstrap_fraction']>=.95
        and prediction[k]['ci95'][0]>threshold for k,threshold in [('spearman',0.),('auc',.5)])
    resolved=summaries['negative']['original_deadlock_to_success']/max(summaries['reference']['deadlocks'],1)
    gradient_passed=bool(all(a['ci'][0]>0 for a in advantages.values()) and success['ci'][0]>0 and resolved>=.5)
    gates=dict(difficulty=difficulty,early_prediction=prediction_passed,calibration=calibration_passed,
        actual_amplitude_match=amplitude_passed,gradient_usefulness=gradient_passed,safety=not any(safety.values()))
    return dict(scope='zero-initialized G_phi, local action diagnostic, not trained-policy efficacy',
        records_sha256=hashlib.sha256((folder/'records.json').read_bytes()).hexdigest(),
        gates=gates,both_tests_and_controls_passed=all(gates.values()),prediction=prediction,
        arms=summaries,negative_deadlock_advantages=advantages,negative_success_increase=success,
        original_deadlock_to_success_fraction=resolved,nonzero_gradient_starts=len(nonzero),
        actual_direct_action_rms=amplitudes,safety=safety,
        limitations=['25 independent starts; two outcome noises per start are clustered',
            'One random direction per start and one fixed action amplitude',
            'Only zero-initialized G_phi tested; no parameter-gradient or generalization proof',
            'Bootstrap intervals are approximate; failure does not prove impossibility',
            'No automated training or test-set access'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path)
    folder=parser.parse_args().folder
    result=analyze(folder)
    (folder/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='prediction'},indent=2))


if __name__=='__main__':main()
