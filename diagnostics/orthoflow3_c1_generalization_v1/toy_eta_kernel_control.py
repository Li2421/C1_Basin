"""Non-neural continuous eta-only comparator using identical Toy W1 pairs."""

import csv
import json

import numpy as np
from scipy.spatial.distance import cdist

from offline_baselines import OUT, dump_json
from train_toy_eta_only import OLD, SWEEP, prepare


def predict(ztrain,alpha,prior,bw,z):
    k=np.exp(-cdist(z,ztrain,'sqeuclidean')/(2*bw*bw))
    return np.clip(prior+k@alpha,1e-5,1-1e-5)


def main():
    data=prepare();x,q=data['train'];xval,yval=data['val']
    ztrain,inv=np.unique(x,axis=0,return_inverse=True)
    count=np.bincount(inv)
    value=np.bincount(inv,weights=q)/count
    prior=float(np.mean(q));trials=[]
    d=cdist(ztrain,ztrain,'sqeuclidean')
    for bw in (0.03,0.05,0.08,0.12,0.18,0.27,0.4,0.6,0.9):
        k=np.exp(-d/(2*bw*bw))
        for ridge in (0.001,0.01,0.1,1.0):
            alpha=np.linalg.solve(k+np.diag(ridge/np.sqrt(count)),value-prior)
            p=predict(ztrain,alpha,prior,bw,xval)
            nll=float(np.mean(-yval*np.log(p)-(1-yval)*np.log(1-p)))
            trials.append((nll,bw,ridge))
    nll,bw,ridge=min(trials)
    alpha=np.linalg.solve(np.exp(-d/(2*bw*bw))+np.diag(ridge/np.sqrt(count)),value-prior)
    x_test,y_test=data['test']
    p_test=predict(ztrain,alpha,prior,bw,x_test)
    test_nll=float(np.mean(-y_test*np.log(p_test)-(1-y_test)*np.log(1-p_test)))
    test_mae=float(np.mean(abs(y_test-p_test)))
    oldp=json.loads((OLD/'frozen_proposals.json').read_text())['states']
    newp=json.loads((SWEEP/'frozen_proposals.json').read_text())['states']
    oldq=list(csv.DictReader((OLD/'per_state_results.csv').open()))
    raw={}
    for path in sorted((SWEEP/'raw').glob('shard*.jsonl')):
        for line in path.read_text().splitlines():
            if line.strip():
                r=json.loads(line);raw[(r['episode_index'],r['kind'],r['future_index'])]=r
    detail=[]
    for i in range(200):
        xyz=np.asarray([oldp[i]['eta'][f'sample_{j}'] for j in range(4)]+[
                        newp[i]['eta'][f'sample_{j}'] for j in range(4,16)])
        z=(xyz-data['center'])/data['scale']
        score=predict(ztrain,alpha,prior,bw,z)
        q16=np.asarray([float(oldq[i][f'Q16_sample_{j}']) for j in range(4)]+[
             sum(r['success'] for r in (raw[(i,f'sample_{j}',seed)] for seed in range(16)) if r['scientific_outcome_valid'])/16
             for j in range(4,16)])
        ix=int(np.argmax(score));old=int(np.argmax(newp[i]['all_critic_scores']))
        detail.append({'episode_index':i,'eta_only_choice':ix,'eta_only_q16':float(q16[ix]),
                       'eta_only_b15':bool(q16[ix]>=15/16),'old_critic_b15':bool(q16[old]>=15/16),
                       'oracle_b15':bool(np.any(q16>=15/16))})
    summary={'train_pairs':len(q),'train_unique_eta':len(ztrain),'val_pairs':len(yval),
             'selected_bandwidth':bw,'selected_ridge':ridge,'val_nll':nll,
             'frozen_simultaneous_holdout_nll':test_nll,
             'frozen_simultaneous_holdout_mae':test_mae,
             'hard200_eta_only_b15':sum(x['eta_only_b15'] for x in detail),
             'hard200_old_critic_b15':sum(x['old_critic_b15'] for x in detail),
             'hard200_oracle_b15':sum(x['oracle_b15'] for x in detail),
             'rescue_vs_old':sum(x['eta_only_b15'] and not x['old_critic_b15'] for x in detail),
             'break_vs_old':sum(x['old_critic_b15'] and not x['eta_only_b15'] for x in detail),
             'status':'old hard200 diagnostic; no target outcome used in fit or bandwidth selection',
             'new_rollout':0}
    dump_json('toy_hard200_eta_kernel_summary.json',summary)
    np.savez(OUT/'toy_eta_kernel_model.npz',ztrain=ztrain,alpha=alpha,prior=prior,bw=bw,
             center=data['center'],scale=data['scale'])
    with (OUT/'toy_hard200_eta_kernel.csv').open('w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=list(detail[0]));w.writeheader();w.writerows(detail)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
