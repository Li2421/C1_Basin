#!/usr/bin/env python3
"""Offline 13-state K16 critic mis-selection audit; no rollout or model changes."""
import csv, json, math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

ROOT=Path('/home/zhihan/research/Basin_C1')
OUT=Path(__file__).resolve().parent
GEN=ROOT/'diagnostics/orthoflow3_mode_free_generator_critic_hard_cohort_v1'
KSW=ROOT/'diagnostics/orthoflow3_mode_free_k_sweep_latency_v1'
STRUCT=ROOT/'diagnostics/orthoflow3_structured_continuous_q_data_v1'
R=np.array([.625,.5,.375])
def rows(p): return list(csv.DictReader(p.open()))
def save(name,rr):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rr for k in r)))
        w.writeheader();w.writerows(rr)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    old=json.loads((GEN/'frozen_proposals.json').read_text())['states']
    new=json.loads((KSW/'frozen_proposals.json').read_text())['states']
    oldq={int(r['episode_index']):r for r in rows(GEN/'per_state_results.csv')}
    ks={int(r['episode_index']):r for r in rows(KSW/'per_state_k.csv') if int(r['K'])==16}
    failed=sorted(i for i,r in ks.items() if int(r['covered_B15']) and not int(r['critic_B15']))
    assert len(failed)==13,failed
    structured=pq.read_table(STRUCT/'structured_pair_table.parquet').to_pylist()
    sparse=pq.read_table(STRUCT/'sparse_matched_control.parquet').to_pylist()
    train=[r['eta'] for r in structured if r['matrix_partition']=='TRAIN_TRAIN']
    train += [[r['eta1'],r['eta2'],r['eta3']] for r in sparse]
    train=np.asarray(train,float)/R
    anchor=np.asarray([[float(r[x]) for x in ('eta1','eta2','eta3')] for r in rows(ROOT/'diagnostics/orthoflow3_shared_eta_codebook_v1/codebook_eta.csv')])/R
    n_q=defaultdict(lambda:[0,0])
    for p in (KSW/'raw').glob('shard*.jsonl'):
        for line in p.open():
            r=json.loads(line);kind=r.get('kind','')
            if kind.startswith('sample_') and r.get('scientific_outcome_valid'):
                key=(int(r['episode_index']),int(kind.split('_')[1]))
                n_q[key][0]+=int(bool(r['success']));n_q[key][1]+=1
    table=[];summ=[];types=Counter()
    for i in failed:
        eta=np.asarray([old[i]['eta'][f'sample_{j}'] if j<4 else new[i]['eta'][f'sample_{j}'] for j in range(16)],float)/R
        score=np.asarray(new[i]['all_critic_scores'],float)
        q=[];n=[]
        for j in range(16):
            if j<4:q.append(float(oldq[i][f'Q16_sample_{j}']));n.append(16)
            else:
                k,trials=n_q[(i,j)];n.append(trials);q.append(k/16 if trials==16 else float('nan'))
        q=np.asarray(q);n=np.asarray(n)
        cr=np.argsort(np.argsort(-score))+1
        true_rank=np.argsort(np.argsort(-np.nan_to_num(q,nan=-1)))+1
        pick=int(np.argmax(score));b15=(q>=15/16)&(n==16)
        top2=int(b15[np.argsort(-score)[:2]].any())
        top3=int(b15[np.argsort(-score)[:3]].any())
        neartrain=cKDTree(train).query(eta)[0]
        nearanchor=cKDTree(anchor).query(eta)[0]
        distmean=np.linalg.norm(eta-np.asarray(old[i]['eta']['generator_mean'],float)/R,axis=1)
        distfixed=np.linalg.norm(eta-np.asarray(old[i]['eta']['fixed_common'],float)/R,axis=1)
        distselector=np.linalg.norm(eta-np.asarray(old[i]['eta']['old_selector_anchor'],float)/R,axis=1)
        p_pred=1/(1+np.exp(-np.clip(score,-30,30)))
        # Boundary-locality is a descriptive proxy, not a certified Basin boundary.
        opp=np.array([np.min(np.linalg.norm(eta[j]-eta[b15],axis=1)) if b15.any() and not b15[j] else
                      np.min(np.linalg.norm(eta[j]-eta[~b15],axis=1)) if (~b15).any() and b15[j] else float('nan')
                      for j in range(16)])
        for j in range(16):
            table.append({'episode_index':i,'proposal_index':j,'eta1':eta[j,0]*R[0],
              'eta2':eta[j,1]*R[1],'eta3':eta[j,2]*R[2],
              'successes16':int(round(q[j]*16)) if math.isfinite(q[j]) else '',
              'valid_trials':int(n[j]),'Q16':float(q[j]) if math.isfinite(q[j]) else '',
              'B15':int(b15[j]) if n[j]==16 else '',
              'critic_logit':float(score[j]),'critic_p':float(p_pred[j]),'critic_rank':int(cr[j]),
              'oracle_rank':int(true_rank[j]),'nearest_critic_train_eta_norm':float(neartrain[j]),
              'distance_to_generator_mean_norm':float(distmean[j]),'nearest_old_anchor_norm':float(nearanchor[j]),
              'distance_to_fixed_common_norm':float(distfixed[j]),
              'distance_to_old_selector_anchor_norm':float(distselector[j]),
              'nearest_opposite_label_proposal_norm':float(opp[j]),
              'near_boundary_proxy':int(opp[j]<=.1 or (math.isfinite(q[j]) and .5<q[j]<15/16))})
        tags=['PAIRWISE_RANKING_ERROR']
        if p_pred[pick]-q[pick]>=.2:tags.append('CALIBRATION_OVERPREDICTION_GE_0P2')
        if p_pred[pick]>.9 and q[pick]<=.5:tags.append('CALIBRATION_OVERCONFIDENCE')
        if .5<q[pick]<15/16 or opp[pick]<=.1:tags.append('BOUNDARY_OVERESTIMATION_PROXY')
        if neartrain[pick]>.2:tags.append('PROPOSAL_DISTRIBUTION_SHIFT_PROXY')
        if np.isnan(q).any():tags.append('INCOMPLETE_Q16_PROPOSAL')
        for t in tags:types[t]+=1
        summ.append({'episode_index':i,'critic_choice':pick,'critic_Q16':float(q[pick]),
             'critic_predicted_Q':float(p_pred[pick]),'oracle_B15_count':int(b15.sum()),
             'top2_contains_B15':top2,'top3_contains_B15':top3,
             'critic_chosen_nearest_train_eta_norm':float(neartrain[pick]),
             'critic_chosen_distance_to_mean_norm':float(distmean[pick]),
             'critic_chosen_distance_to_fixed_common_norm':float(distfixed[pick]),
             'critic_chosen_distance_to_old_selector_anchor_norm':float(distselector[pick]),
             'critic_chosen_nearest_old_anchor_norm':float(nearanchor[pick]),'error_tags':';'.join(tags)})
    save('toy_critic_13_proposals.csv',table)
    save('toy_critic_13_summary.csv',summ)
    (OUT/'toy_critic_13_error_types.json').write_text(json.dumps({'n_states':13,'type_counts':types,
      'top2_success_states':sum(r['top2_contains_B15'] for r in summ),
      'top3_success_states':sum(r['top3_contains_B15'] for r in summ),
      'definitions':'Boundary/OOD flags are proxies, not causal diagnoses; missing Q16 is left unresolved.'},indent=2)+'\n')
    print(json.dumps({'states':13,'type_counts':types,'top2':sum(r['top2_contains_B15'] for r in summ),'top3':sum(r['top3_contains_B15'] for r in summ)}))
if __name__=='__main__':main()
