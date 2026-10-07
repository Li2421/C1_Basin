#!/usr/bin/env python3
"""Read-only audit of the frozen Toy K16 critic failures; no rollouts."""
from __future__ import annotations

import csv
import importlib.util
import json
import sqlite3
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow.parquet as pq
from flax import serialization
from scipy.special import expit
from scipy.stats import beta, binom, mannwhitneyu

ROOT = Path('/home/zhihan/research/Basin_C1')
D = ROOT / 'diagnostics'
OUT = Path(__file__).resolve().parent
SRC = D / 'orthoflow3_structured_continuous_q_data_v1'
W = D / 'orthoflow3_nll_weighting_ablation_v1'
R = D / 'orthoflow3_ranking_aware_critic_v1'
OLD = D / 'orthoflow3_continuous_basin_critic_v1'
KDIR = D / 'orthoflow3_mode_free_k_sweep_latency_v1'
KOLD = D / 'orthoflow3_mode_free_generator_critic_hard_cohort_v1'
U = D / 'orthoflow3_toy_critic_uncertainty_local_v1'
DB = ROOT / 'shared_rollout_db/rollout.sqlite'
SEEDS = (17, 23, 41)

def csvwrite(name, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def dump(name, x):
    (OUT / name).write_text(json.dumps(x, indent=2, sort_keys=True) + '\n')

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def params_and_model():
    rank = load_module(R / 'run_experiment.py', 'audit_ranklib')
    model = rank.SingleCritic()
    pp = []
    for seed in SEEDS:
        template = model.init(jax.random.PRNGKey(seed), jnp.zeros((1, 214)), jnp.zeros((1, 3)))
        p = W / 'models' / 'secondary_combined' / 'W1' / f'seed{seed}' / 'checkpoint.msgpack'
        pp.append(serialization.from_bytes(template, p.read_bytes()))
    return model, pp

def batch_predict(model, params, h, e):
    h = np.asarray(h, np.float32)
    e = np.asarray(e, np.float32)
    member = []
    for p in params:
        parts = []
        for start in range(0, len(h), 512):
            v = model.apply(p, jnp.asarray(h[start:start+512]), jnp.asarray(e[start:start+512]))
            parts.append(np.asarray(v))
        member.append(np.concatenate(parts))
    return expit(np.mean(member, axis=0)), np.stack(member)

def bins_rows(label, p, q, robust):
    edges = [0, .5, .7, .8, .9, .95, 1.00000001]
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        ix = (p >= a) & ((p < b) if b < 1. else (p <= 1.))
        rows.append({'population':label, 'bin':f'[{a:.2f},{b:.2f})', 'n':int(ix.sum()),
                     'predicted_mean':float(p[ix].mean()) if ix.any() else None,
                     'true_q_mean':float(q[ix].mean()) if ix.any() else None,
                     'b15_precision':float(robust[ix].mean()) if ix.any() else None,
                     'non_b15_fraction':float((~robust[ix]).mean()) if ix.any() else None})
    return rows

def main():
    OUT.mkdir(exist_ok=True)
    structured = pq.read_table(SRC/'structured_pair_table.parquet').to_pylist()
    tr = [dict(r, data_source='structured') for r in structured if r['matrix_partition']=='TRAIN_TRAIN']
    va = [dict(r, data_source='structured') for r in structured if r['matrix_partition']=='VAL_VAL']
    te = [dict(r, data_source='structured') for r in structured if r['matrix_partition']=='TESTSTATE_TESTETA']
    wide0 = pq.read_table(SRC/'sparse_matched_control.parquet').to_pylist()
    keys = {(r['state_uid'],r['eta_uid']) for r in tr}
    wide = [dict(r, eta=[r['eta1'],r['eta2'],r['eta3']], data_source='wide') for r in wide0 if (r['state_uid'],r['eta_uid']) not in keys]
    train = tr + wide
    assert len(train)==len({(r['state_uid'],r['eta_uid']) for r in train})
    assert not ({r['state_uid'] for r in train} & {r['state_uid'] for r in te})
    assert not ({r['eta_uid'] for r in train} & {r['eta_uid'] for r in te})
    man = json.loads((OLD/'dataset_manifest.json').read_text())
    hm = np.asarray(man['state_normalization']['Toy']['mean'],np.float32)
    hs = np.asarray(man['state_normalization']['Toy']['std'],np.float32)
    ec = np.asarray(man['eta_normalization']['center'],np.float32)
    es = np.asarray(man['eta_normalization']['scale'],np.float32)
    nh = lambda rows: (np.asarray([r['h_raw'] for r in rows],np.float32)-hm)/hs
    ne = lambda rows: (np.asarray([r['eta'] for r in rows],np.float32)-ec)/es
    th, vh, eh = nh(train), nh(va), nh(te)
    tz, vz, ez = ne(train), ne(va), ne(te)
    tq = np.asarray([r['empirical_q'] for r in train],np.float32)
    vq = np.asarray([r['empirical_q'] for r in va],np.float32)
    eq = np.asarray([r['empirical_q'] for r in te],np.float32)
    model, params = params_and_model()
    train_p,_ = batch_predict(model,params,th,tz)
    ordinary_p,_ = batch_predict(model,params,eh,ez)
    predictions = list(csv.DictReader((U/'frozen_k16_proposal_predictions.csv').open()))
    predictions.sort(key=lambda r:(int(r['episode_index']),int(r['proposal_index'])))
    assert len(predictions)==3200
    q = np.asarray([float(r['Q16_lower']) for r in predictions]).reshape(200,16)
    upper = np.asarray([float(r['Q16_upper']) for r in predictions]).reshape(200,16)
    assert np.array_equal(q>=15/16,upper>=15/16)
    frozen = json.loads((KDIR/'frozen_proposals.json').read_text())
    logits = np.asarray([r['all_critic_scores'] for r in frozen['states']],float)
    p = expit(logits)
    assert np.max(np.abs(logits.reshape(-1)-np.asarray([float(r['original_critic_logit']) for r in predictions])))<1e-5
    hardh = np.load(KDIR/'cohort_features.npz')['h_raw']
    hardz = (np.asarray([[[float(predictions[i*16+j][f'eta{k}']) for k in (1,2,3)]
                         for j in range(16)] for i in range(200)])-ec)/es
    hardhn = (hardh-hm)/hs
    replay_p,replay_member = batch_predict(model,params,np.repeat(hardhn,16,axis=0),hardz.reshape(-1,3))
    replay_logit=np.mean(replay_member,axis=0).reshape(200,16)
    replay_delta=float(np.max(abs(replay_logit-logits)))
    assert replay_delta<1e-5, replay_delta
    chosen = np.argmax(logits,axis=1)
    covered = np.max(q,axis=1)>=15/16
    bad = np.flatnonzero(covered & (q[np.arange(200),chosen]<15/16))
    assert len(bad)==13 and int((q[np.arange(200),chosen]>=15/16).sum())==181
    dump('dataset_and_identity.json',{'train_pairs':len(train),'structured_pairs':len(tr),'wide_pairs':len(wide),
         'train_states':len({r['state_uid'] for r in train}),'train_eta':len({r['eta_uid'] for r in train}),
         'val_pairs':len(va),'test_pairs':len(te),'state_overlap_train_test':0,'eta_uid_overlap_train_test':0,
         'hard_states':200,'hard_proposals':3200,'selection_failures':bad.tolist(),
         'train_q_mean':float(tq.mean()),'hard_proposal_q_mean':float(q.mean()),
         'train_h_normalized_finite':bool(np.isfinite(th).all()),'hard_h_normalized_finite':bool(np.isfinite(hardhn).all()),
         'eta_center':ec.tolist(),'eta_scale':es.tolist(),
         'train_h_abs_gt5_fraction':float((abs(th)>5).mean()),'hard_h_abs_gt5_fraction':float((abs(hardhn)>5).mean())})
    dump('inference_replay.json',{'all_3200_logits_replayed':True,'max_abs_logit_difference':replay_delta,
      'model_decoding':'sigmoid(mean logits of seed17/23/41)',
      'max_abs_probability_difference':float(np.max(abs(replay_p-expit(logits.reshape(-1))))),
      'stored_probability_decoding_matches':bool(np.max(abs(replay_p-expit(logits.reshape(-1))))<5e-6)})

    calibration=[]
    calibration+=bins_rows('train_combined',train_p,tq,tq>=15/16)
    calibration+=bins_rows('ordinary_unseen_state_unseen_eta',ordinary_p,eq,eq>=15/16)
    calibration+=bins_rows('all_generator_proposals',p.reshape(-1),q.reshape(-1),(q>=15/16).reshape(-1))
    calibration+=bins_rows('critic_top1',p[np.arange(200),chosen],q[np.arange(200),chosen],q[np.arange(200),chosen]>=15/16)
    csvwrite('upper_tail_calibration.csv',calibration)
    dump('fit_generalization.json',{'train_pair_MAE':float(np.mean(abs(train_p-tq))),
      'ordinary_test_pair_MAE':float(np.mean(abs(ordinary_p-eq))),
      'hard_generator_proposal_MAE':float(np.mean(abs(p-q))),
      'train_pair_NLL':float(np.mean(-tq*np.log(np.clip(train_p,1e-7,1-1e-7))-(1-tq)*np.log(np.clip(1-train_p,1e-7,1-1e-7)))),
      'ordinary_test_pair_NLL':float(np.mean(-eq*np.log(np.clip(ordinary_p,1e-7,1-1e-7))-(1-eq)*np.log(np.clip(1-ordinary_p,1e-7,1-1e-7)))),
      'hard_generator_proposal_NLL':float(np.mean(-q*np.log(np.clip(p,1e-7,1-1e-7))-(1-q)*np.log(np.clip(1-p,1e-7,1-1e-7))))})
    sweeps=[]
    for k in (1,2,4,8,16):
        ch=np.argmax(logits[:,:k],axis=1)
        sp=p[np.arange(200),ch]
        sq=q[np.arange(200),ch]
        oq=q[:,:k].max(axis=1)
        sweeps.append({'K':k,'oracle_B15':int((oq>=15/16).sum()),'selected_B15':int((sq>=15/16).sum()),
                       'selected_pred_p_mean':float(sp.mean()),'selected_true_q_mean':float(sq.mean()),
                       'mean_overestimation':float((sp-sq).mean()),'median_overestimation':float(np.median(sp-sq)),
                       'false_high_confidence_p95_nonB15':int(((sp>=.95)&(sq<15/16)).sum()),
                       'oracle_true_q_mean':float(oq.mean()),'mean_regret':float((oq-sq).mean()),
                       'selection_failures_when_covered':int(((oq>=15/16)&(sq<15/16)).sum())})
    csvwrite('max_selection_bias.csv',sweeps)

    # Cache-only exact evidence and binomial intervals for the 13 bad/good comparisons.
    lookup={}
    for path in (KOLD/'candidate_cache_keys.csv', KDIR/'candidate_cache_keys.csv'):
        lookup.update({(int(r['episode_index']),r['kind']):r for r in csv.DictReader(path.open())})
    con=sqlite3.connect(DB)
    label_rows=[]
    for i in bad:
        good=int(np.argmax(q[i]))
        for role,j in [('critic_bad',int(chosen[i])),('oracle_good',good)]:
            key=lookup[(int(i),f'sample_{j}')]
            rr=con.execute('''SELECT success,seed_key FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=?
               AND conflict_quarantined=0 AND numerical_failure=0 AND compatibility_quality='EXACT_REUSE' ''',
               (key['state_uid'],key['eta_uid'],key['controller_uid'])).fetchall()
            n=len(rr);s=sum(x[0] for x in rr)
            lo=float(beta.ppf(.025,s,n-s+1)) if s else 0.
            hi=float(beta.ppf(.975,s+1,n-s)) if s<n else 1.
            label_rows.append({'episode_index':int(i),'role':role,'proposal_index':j,'Q16':float(q[i,j]),
                'critic_p':float(p[i,j]),'compatible_exact_n':n,'compatible_exact_success':s,
                'q_95_lower':lo,'q_95_upper':hi,'extra_exact_beyond_16':max(n-16,0),
                'P_at_most_k_if_true_Q09':float(binom.cdf(s,n,.9))})
    con.close()
    csvwrite('binomial_uncertainty.csv',label_rows)

    # Joint-input support audit. h distance is RMS of normalized 214-D coordinates;
    # eta distance is normalized physical Euclidean. Neighbors are chosen with
    # d_joint=sqrt(d_h^2+d_eta^2), with separate distances retained.
    train_state_ids=sorted({r['state_uid'] for r in train})
    state_index={s:j for j,s in enumerate(train_state_ids)}
    train_state_h=np.asarray([th[next(i for i,r in enumerate(train) if r['state_uid']==s)] for s in train_state_ids])
    train_state_col=np.asarray([state_index[r['state_uid']] for r in train])
    hdist=np.sqrt(np.mean((hardhn[:,None,:]-train_state_h[None,:,:])**2,axis=2))
    val_state_ids=sorted({r['state_uid'] for r in va})
    vh_state=np.asarray([vh[next(i for i,r in enumerate(va) if r['state_uid']==s)] for s in val_state_ids])
    val_map=np.asarray([val_state_ids.index(r['state_uid']) for r in va])
    val_hdist=np.sqrt(np.mean((vh_state[:,None,:]-train_state_h[None,:,:])**2,axis=2))[val_map][:,train_state_col]
    val_eta=np.sqrt(((vz[:,None,:]-tz[None,:,:])**2).sum(axis=2))
    def local_eval(hd,ed,alpha,k,kind):
        dist=np.sqrt(hd*hd+(alpha*ed)**2)
        ix=np.argpartition(dist,k-1,axis=1)[:,:k]
        dd=np.take_along_axis(dist,ix,axis=1)
        yy=tq[ix]
        if kind=='uniform':return yy.mean(axis=1)
        if kind=='weighted':w=1/np.maximum(dd,.01)**2
        else:w=np.exp(-.5*(dd/.5)**2)
        return (w*yy).sum(axis=1)/np.maximum(w.sum(axis=1),1e-12)
    configs=[]
    for alpha in (.25,.5,1.,2.,4.):
        for k in (1,5,10,20):
            for kind in ('uniform','weighted','kernel'):
                pv=local_eval(val_hdist,val_eta,alpha,k,kind)
                configs.append({'alpha':alpha,'k':k,'kind':kind,'val_mae':float(np.mean(abs(pv-vq))),
                               'val_nll':float(np.mean(-vq*np.log(np.clip(pv,1e-5,1-1e-5))-(1-vq)*np.log(np.clip(1-pv,1e-5,1-1e-5))))})
    csvwrite('local_baseline_val_selection.csv',configs)
    selected=min(configs,key=lambda r:(r['val_mae'],r['val_nll']))
    hz=hardz.reshape(-1,3)
    hed=np.sqrt(((hz[:,None,:]-tz[None,:,:])**2).sum(axis=2))
    hhd=np.repeat(hdist[:,train_state_col],16,axis=0)
    local_predictions=[]
    for cfg in [dict(alpha=1.,k=5,kind='weighted'),selected]:
        pred=local_eval(hhd,hed,cfg['alpha'],cfg['k'],cfg['kind']).reshape(200,16)
        picked=np.argmax(pred,axis=1)
        local_predictions.append({'config':cfg,'all_proposal_mae':float(np.mean(abs(pred-q))),
            'bad13_predicted_q_mean':float(pred[bad,chosen[bad]].mean()),
            'bad13_predicted_q_median':float(np.median(pred[bad,chosen[bad]])),
            'correct181_predicted_q_median':float(np.median(pred[np.flatnonzero(q[np.arange(200),chosen]>=15/16),chosen[q[np.arange(200),chosen]>=15/16]])),
            'bad13_predicted_above09':int((pred[bad,chosen[bad]]>.9).sum()),
            'local_selected_B15':int((q[np.arange(200),picked]>=15/16).sum()),
            'bad13_local_selected_B15':int((q[bad,picked[bad]]>=15/16).sum())})
    dump('local_baseline_summary.json',{'val_selected':selected,'results':local_predictions})
    neighbors=[]; neighbor_summary=[]
    for i in bad:
        j=int(chosen[i]);ed=hed[i*16+j];hd=hhd[i*16+j];dist=np.sqrt(hd**2+ed**2)
        ix=np.argsort(dist)[:20]
        qv=tq[ix]
        for rank,t in enumerate(ix,1):
            neighbors.append({'episode_index':int(i),'bad_proposal_index':j,'neighbor_rank':rank,
                 'joint_distance':float(dist[t]),'state_distance':float(hd[t]),'eta_distance':float(ed[t]),
                 'train_true_q':float(tq[t]),'train_trials':int(train[t]['n_trials']),
                 'train_critic_p':float(train_p[t]),'source':train[t]['data_source'],
                 'train_state_uid':train[t]['state_uid'],'train_eta_uid':train[t]['eta_uid']})
        row={'episode_index':int(i),'bad_proposal_index':j,'true_Q16':float(q[i,j]),'predicted_p':float(p[i,j]),
             'nearest_state_dist':float(hd.min()),'nearest_eta_dist':float(ed.min()),
             'nearest_joint_dist':float(dist.min()),'nearest20_min_q':float(qv.min()),
             'nearest20_max_q':float(qv.max()),'near_low_q_h025_eta005_count':int(((hd<=.25)&(ed<=.05)&(tq<=.5)).sum())}
        for k in (1,5,10,20):
            row[f'k{k}_mean_q']=float(tq[ix[:k]].mean())
            row[f'k{k}_high_q_fraction']=float((tq[ix[:k]]>=.9).mean())
            row[f'k{k}_low_q_fraction']=float((tq[ix[:k]]<=.5).mean())
            row[f'k{k}_mean_model_p']=float(train_p[ix[:k]].mean())
        neighbor_summary.append(row)
    csvwrite('nearest_train_neighbors.csv',neighbors)
    csvwrite('failure_support_summary.csv',neighbor_summary)
    dump('distance_summary.json',{'bad13_nearest_joint_median':float(np.median([r['nearest_joint_dist'] for r in neighbor_summary])),
      'bad13_nearest_state_median':float(np.median([r['nearest_state_dist'] for r in neighbor_summary])),
      'bad13_nearest_eta_median':float(np.median([r['nearest_eta_dist'] for r in neighbor_summary])),
      'hard_state_nearest_train_h_median':float(np.median(hdist.min(axis=1))),
      'hard_state_nearest_train_h_p95':float(np.quantile(hdist.min(axis=1),.95)),
      'bad_near_low_q_count':int(sum(r['near_low_q_h025_eta005_count']>0 for r in neighbor_summary))})
    schema=json.loads((D/'gphi_training_dataset_startup_complete_v1/feature_schema.json').read_text())['segments']
    near_state=hdist.argmin(axis=1)
    hard_h_distance=hdist.min(axis=1)
    val_state_near=np.sqrt(np.mean((vh_state[:,None,:]-train_state_h[None,:,:])**2,axis=2)).min(axis=1)
    selected_eta_distance=hed[np.arange(200)*16+chosen].min(axis=1)
    selected_correct=(q[np.arange(200),chosen]>=15/16)
    selected_hd=hhd[np.arange(200)*16+chosen]
    selected_ed=hed[np.arange(200)*16+chosen]
    joint=np.sqrt(selected_hd**2+selected_ed**2).min(axis=1)
    supported=((selected_hd<=.25)&(selected_ed<=.05)).any(axis=1)
    sem=[]
    for seg in schema:
        a,b=seg['offset'],seg['offset']+seg['length']
        diff=np.sqrt(np.mean((hardhn[:,a:b]-train_state_h[near_state,a:b])**2,axis=1))
        sem.append({'segment':seg['name'],'length':seg['length'],
          'bad13_nearest_state_segment_distance_median':float(np.median(diff[bad])),
          'correct181_nearest_state_segment_distance_median':float(np.median(diff[selected_correct])),
          'bad13_abs_normalized_feature_median':float(np.median(abs(hardhn[bad,a:b]))),
          'correct181_abs_normalized_feature_median':float(np.median(abs(hardhn[selected_correct,a:b])))})
    csvwrite('semantic_feature_groups.csv',sem)
    normalized_eta=hardz[np.arange(200),chosen]
    all_eta_min=tz.min(axis=0);all_eta_max=tz.max(axis=0)
    within_train_box=np.all((normalized_eta>=all_eta_min)&(normalized_eta<=all_eta_max),axis=1)
    dump('distribution_shift.json',{'hard_vs_train_source_groups_disjoint':True,
      'train_t0_flow_root':2026092806,'hard_t0_flow_root':42,
      'train_future_root':2026092811,'hard_future_root_rule':'42+future_index',
      'same_physical_environment_flow_checkpoint_basis_safety':True,
      'strict_cache_controller_uid_equal':False,
      'val_state_nearest_train_h_median':float(np.median(val_state_near)),
      'hard_state_nearest_train_h_median':float(np.median(hard_h_distance)),
      'bad13_nearest_train_h_median':float(np.median(hard_h_distance[bad])),
      'correct181_nearest_train_h_median':float(np.median(hard_h_distance[selected_correct])),
      'bad13_nearest_train_eta_median':float(np.median(selected_eta_distance[bad])),
      'correct181_nearest_train_eta_median':float(np.median(selected_eta_distance[selected_correct])),
      'bad13_nearest_train_joint_median':float(np.median(joint[bad])),
      'correct181_nearest_train_joint_median':float(np.median(joint[selected_correct])),
      'joint_distance_bad_vs_correct_mannwhitney_p':float(mannwhitneyu(joint[bad],joint[selected_correct]).pvalue),
      'strict_joint_support_bad13':int(supported[bad].sum()),
      'strict_joint_support_correct181':int(supported[selected_correct].sum()),
      'selected_eta_outside_train_coordinate_box_bad13':int((~within_train_box[bad]).sum()),
      'selected_eta_outside_train_coordinate_box_correct181':int((~within_train_box[selected_correct]).sum()),
      'fraction_all_proposals_eta_distance_gt01':float((hed.min(axis=1)>.1).mean()),
      'fraction_all_proposals_eta_distance_gt02':float((hed.min(axis=1)>.2).mean())})
    # A same-exact-eta TRAIN-only check for feature-space aliasing. Q8 is noisy,
    # so this is a warning signal, not proof of irreducible representation loss.
    groups={}
    for ix,r in enumerate(tr):
        if r['n_trials']>=8:groups.setdefault(r['eta_uid'],[]).append(ix)
    conflicting=[]
    counts={'0.10':0,'0.20':0,'0.30':0}
    for uid,ix in groups.items():
        for aa in range(len(ix)):
            for bb in range(aa+1,len(ix)):
                a,b=ix[aa],ix[bb]
                if abs(float(tq[a])-float(tq[b]))<.75:continue
                d=float(np.sqrt(np.mean((th[a]-th[b])**2)))
                for threshold in (.10,.20,.30):
                    if d<=threshold:counts[f'{threshold:.2f}']+=1
                if d<=.20:conflicting.append({'eta_uid':uid,'state_a':tr[a]['state_uid'],
                    'state_b':tr[b]['state_uid'],'state_distance':d,
                    'Q_a':float(tq[a]),'Q_b':float(tq[b]),'trials_a':tr[a]['n_trials'],'trials_b':tr[b]['n_trials']})
    csvwrite('train_same_eta_feature_conflicts.csv',conflicting)
    dump('train_aliasing_screen.json',{'same_exact_eta_q_gap_at_least075_count_by_h_radius':counts,
      'interpretation':'Q8-or-higher screen only; a near-conflict is not proof that h omits necessary information.'})
    print(json.dumps({'bad':bad.tolist(),'local':local_predictions,'sweep':sweeps,'distance':json.loads((OUT/'distance_summary.json').read_text())},indent=2))

if __name__=='__main__':main()
