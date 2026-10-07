#!/usr/bin/env python3
"""One frozen final evaluation of uncertainty selection on Toy K16 hard cohort."""
import csv,json,statistics
from collections import Counter
from pathlib import Path
import numpy as np
import common

ROOT=common.ROOT;HERE=Path(__file__).resolve().parent
OLD=ROOT/'diagnostics/orthoflow3_mode_free_generator_critic_hard_cohort_v1'
KSW=ROOT/'diagnostics/orthoflow3_mode_free_k_sweep_latency_v1'
LAMBDAS=(0.,.25,.5,1.,2.)

def save_csv(name,rows):
    with (HERE/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def load_q():
    old=list(csv.DictReader((OLD/'per_state_results.csv').open()))
    keyed={}
    for p in sorted((KSW/'raw').glob('shard*.jsonl')):
        for line in p.open():
            r=json.loads(line)
            keyed[(int(r['episode_index']),r['kind'],int(r['future_index']))]=r
    assert len(old)==200
    q=np.empty((200,16),np.float64);upper=np.empty_like(q)
    for i in range(200):
        for j in range(4):q[i,j]=float(old[i][f'Q16_sample_{j}']);upper[i,j]=q[i,j]
        for j in range(4,16):
            rec=[keyed[(i,f'sample_{j}',seed)] for seed in range(16)]
            good=sum(bool(r['success']) for r in rec if r['scientific_outcome_valid'])
            invalid=sum(not r['scientific_outcome_valid'] for r in rec)
            q[i,j]=good/16;upper[i,j]=(good+invalid)/16
    assert np.array_equal(q>=15/16,upper>=15/16)
    return q,upper

def main():
    freeze=json.loads((HERE/'frozen_selection.json').read_text())
    assert freeze['selected_before_hard_test_evaluation'] and freeze['lambda'] in LAMBDAS
    old=json.loads((OLD/'frozen_proposals.json').read_text())['states']
    extended=json.loads((KSW/'frozen_proposals.json').read_text())['states']
    assert len(old)==len(extended)==200
    h=np.load(KSW/'cohort_features.npz')['h_raw']
    eta=np.asarray([[old[i]['eta'][f'sample_{j}'] if j<4 else extended[i]['eta'][f'sample_{j}'] for j in range(16)] for i in range(200)],np.float32)
    assert eta.shape==(200,16,3)
    q,upper=load_q()
    member=common.predict_members(np.repeat(h,16,axis=0),eta.reshape(-1,3)).reshape(5,200,16)
    mu=member.mean(0);sigma=member.std(0)
    old_score=np.asarray([s['all_critic_scores'] for s in extended])
    original=np.argmax(old_score,axis=1)
    robust=q>=15/16
    oracle=robust.any(axis=1)
    assert int(oracle.sum())==194 and int(robust[np.arange(200),original].sum())==181
    local_h,local_eta=common.virtual_neighborhood(np.repeat(h,16,axis=0),eta.reshape(-1,3),.05)
    local_member=common.predict_members(local_h,local_eta).reshape(5,200,16,6)
    local_score=local_member.mean((0,3))
    summary={'oracle_B15':int(oracle.sum()),'original_critic_B15':181,'frozen_lambda':freeze['lambda'],
             'lambda_selected_on_VAL_only':True,'new_rollout':0,'methods':{},'uncertainty':{}}
    current=robust[np.arange(200),original]
    choices={}
    for name,score in [('original_mean_logits',old_score),('M5_mean_probability',mu),
                       *[(f'M5_LCB_lambda_{lam:g}',mu-lam*sigma) for lam in LAMBDAS],
                       ('virtual_local_mean_r0p05',local_score)]:
        ix=np.argmax(score,axis=1);choices[name]=ix
        selected=robust[np.arange(200),ix]
        summary['methods'][name]={'B15_states':int(selected.sum()),'oracle_gap':int(oracle.sum()-selected.sum()),
          'mean_Q16':float(q[np.arange(200),ix].mean()),
          'rescue_vs_original':int((selected&~current).sum()),
          'break_vs_original':int((~selected&current).sum()),
          'rescued_of_original_13_failures':int((selected&oracle&~current).sum()),
          'broken_of_original_181_successes':int((~selected&current).sum()),
          'false_high_Q_lt_0p5_p_gt_0p9':int(((q[np.arange(200),ix]<.5)&(mu[np.arange(200),ix]>.9)).sum())}
    failed=np.where(oracle&~current)[0]
    correct=np.where(current)[0]
    failed_sigma=sigma[failed,original[failed]]
    correct_sigma=sigma[correct,original[correct]]
    b15_sigma=sigma[robust]
    all_sigma=sigma.reshape(-1)
    summary['uncertainty']={'original_false_top1_count':len(failed),
      'false_top1_sigma_mean':float(failed_sigma.mean()),'false_top1_sigma_median':float(np.median(failed_sigma)),
      'correct_top1_sigma_mean':float(correct_sigma.mean()),'correct_top1_sigma_median':float(np.median(correct_sigma)),
      'B15_proposal_sigma_mean':float(b15_sigma.mean()),'B15_proposal_sigma_median':float(np.median(b15_sigma)),
      'all_proposal_sigma_mean':float(all_sigma.mean()),'all_proposal_sigma_median':float(np.median(all_sigma)),
      'false_top1_gt_correct_median_sigma':int((failed_sigma>np.median(correct_sigma)).sum()),
      'false_top1_all_five_predict_gt_0p9':int((member[:,failed,original[failed]]>.9).all(axis=0).sum()),
      'false_top1_mu_gt_0p9_and_sigma_le_correct_median':int(((mu[failed,original[failed]]>.9)&(failed_sigma<=np.median(correct_sigma))).sum())}
    rows=[]
    for i in range(200):
        for j in range(16):
            rows.append({'episode_index':i,'proposal_index':j,'eta1':float(eta[i,j,0]),'eta2':float(eta[i,j,1]),'eta3':float(eta[i,j,2]),
              'Q16_lower':float(q[i,j]),'Q16_upper':float(upper[i,j]),'B15':int(robust[i,j]),
              'original_critic_logit':float(old_score[i,j]),'ensemble_mean_p':float(mu[i,j]),
              'ensemble_std_p':float(sigma[i,j]),'member_probabilities_json':json.dumps(member[:,i,j].tolist()),
              'virtual_local_mean_r0p05':float(local_score[i,j]),
              'original_selected':int(j==original[i]),'oracle_coverable':int(oracle[i]),
              'mean_selected':int(j==choices['M5_mean_probability'][i]),
              'val_selected_LCB':int(j==choices[f'M5_LCB_lambda_{freeze["lambda"]:g}'][i]),
              'local_selected':int(j==choices['virtual_local_mean_r0p05'][i])})
    save_csv('frozen_k16_proposal_predictions.csv',rows)
    failure_rows=[r for r in rows if r['episode_index'] in failed and r['original_selected']]
    save_csv('original_13_false_top1_ensemble.csv',failure_rows)
    state_rows=[]
    for i in range(200):
        state_rows.append({'episode_index':i,'oracle_B15':int(oracle[i]),'original_B15':int(current[i]),
          **{name+'_choice':int(ix[i]) for name,ix in choices.items()},
          **{name+'_B15':int(robust[i,ix[i]]) for name,ix in choices.items()}})
    save_csv('selection_per_state.csv',state_rows)
    (HERE/'selection_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
