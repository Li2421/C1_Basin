"""Existing-record alignment, information variation, and matched-seed audit."""
from __future__ import annotations
import json
import numpy as np
from scipy.stats import binomtest
from .experiment import OUT,SRC,WIDE,read,write,csvwrite

def main():
    from shared_rollout_db.src.rollout_db import connect
    d=np.load(SRC/'dataset.npz');x=np.load(SRC/'entities.npz');states=read(SRC/'states.json');pairs=read(SRC/'pairs.json')
    profiles=read(SRC/'protocol.json')['profiles'];records={};mis=[];valid=numeric=0
    with connect(True) as db:
      for c,profile in enumerate(profiles):
       for i,pair in enumerate(pairs):
        rows=db.execute('SELECT seed_key,success,numerical_failure,conflict_quarantined,compatibility_quality FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=?',(pair['state_uid'],pair['eta_uid'],profile['controller_uid'])).fetchall()
        seeds={}
        for row in rows:
            k=json.loads(row['seed_key']).get('future_index',-1)
            if not 0<=k<16:continue
            assert k not in seeds and not row['conflict_quarantined'] and row['compatibility_quality']=='EXACT_REUSE'
            seeds[k]=None if row['numerical_failure'] else int(row['success'])
        assert len(seeds)==16
        ss=sum(v for v in seeds.values() if v is not None);nn=sum(v is not None for v in seeds.values())
        if ss!=d['success'][c,i] or nn-ss!=d['failure'][c,i]:mis.append([c,i])
        valid+=nn;numeric+=16-nn;records[(c,i)]=seeds
    assert not mis
    rows=[]
    for c,profile in enumerate(profiles):
      for state in range(len(states)):
        a,b=2*state,2*state+1
        both=[k for k in range(16) if records[c,a][k] is not None and records[c,b][k] is not None]
        wins=sum(records[c,a][k]>records[c,b][k] for k in both)
        losses=sum(records[c,a][k]<records[c,b][k] for k in both)
        p=binomtest(wins,wins+losses,.5).pvalue if wins+losses else 1.
        rows.append(dict(controller=profile['name'],state_index=state,split=states[state]['split'],
            matched_valid_seeds=len(both),eta10_only_success=wins,eta15_only_success=losses,
            paired_Q_difference=(wins-losses)/len(both),two_sided_matched_seed_pvalue=p,
            eta10_success=int(d['success'][c,a]),eta15_success=int(d['success'][c,b]),
            eta10_B15=bool(d['success'][c,a]>=15),eta15_B15=bool(d['success'][c,b]>=15)))
    csvwrite(OUT/'matched_seed_comparisons.csv',rows)
    summaries=[]
    for split in ('train','validation'):
      for c in ('alt','second'):
        rr=[r for r in rows if r['split']==split and r['controller']==c]
        summaries.append(dict(split=split,controller=c,states=len(rr),
            nominal_significant_eta10=sum(r['two_sided_matched_seed_pvalue']<.05 and r['paired_Q_difference']>0 for r in rr),
            nominal_significant_eta15=sum(r['two_sided_matched_seed_pvalue']<.05 and r['paired_Q_difference']<0 for r in rr),
            eta10_only_B15=sum(r['eta10_B15'] and not r['eta15_B15'] for r in rr),
            eta15_only_B15=sum(r['eta15_B15'] and not r['eta10_B15'] for r in rr)))
    csvwrite(OUT/'matched_seed_summary.csv',summaries)
    tr=np.flatnonzero(d['split']=='train');ctx=d['context'][:,tr].astype(float).reshape(2,46,2,24)
    totalvar=ctx.reshape(-1,24).var(0);within=ctx.var(1).mean((0,1))
    names=['nominal_progress_integral','nominal_progress_late','nominal_goal_delta','nominal_min_pair_distance','nominal_pair_closing_max','nominal_safety_mean','nominal_safety_max','nominal_safety_active','nominal_final_speed','nominal_flow_change','eta_progress_integral','eta_progress_late','eta_goal_delta','eta_min_pair_distance','eta_pair_closing_max','eta_safety_mean','eta_safety_max','eta_safety_active','eta_correction_mean','eta_correction_max','eta_correction_active','eta_nominal_progress_difference','eta_nominal_final_position_distance','eta_nominal_min_pair_distance_difference']
    feature_rows=[dict(feature=name,std_total=float(np.sqrt(totalvar[j])),std_within_controller_eta=float(np.sqrt(within[j])),
        within_fraction=float(within[j]/totalvar[j]) if totalvar[j]>1e-12 else None,
        old_normalized_std=float(np.sqrt(totalvar[j])/max(np.sqrt(totalvar[j]),.05)),
        constant=bool(totalvar[j]<1e-12)) for j,name in enumerate(names)]
    csvwrite(OUT/'context_feature_variation.csv',feature_rows)
    result=dict(dataset_DB_counts_match=True,valid_records=valid,numerical_separate=numeric,
        mismatches=mis,source_family_count=46,independent_source_validation_count=16,
        narrow_training_controller_pairs=184,wide_restored_controller_pairs=1472,
        historical_source_pairs_previously_omitted=1288,
        context_features=len(names),context_constant_channels=sum(r['constant'] for r in feature_rows),
        significantly_downscaled_varying_channels=sum(not r['constant'] and r['old_normalized_std']<.2 for r in feature_rows),
        context_safety_note='Nominal safety response is identically zero in the observed H20 source probes; this context mostly precedes safety interventions.',
        normalizer_note='Physical state had fixed physical-unit scaling only; context used an absolute std floor .05 across mixed units. These are conditioning concerns, not proven semantic corruption.',
        Q_B15_note='For independent identically distributed Bernoulli trials with true probability p, P(B15)=p^15*(16-15*p), strictly increasing in p. Aggregate mean-Q vs B15 discrepancies do not by themselves prove a loss conflict.',
        source_probe_selection_caveat='Eta10/15 was selected from these source outcomes; source crossfit is exploratory. The previously opened 16-family panel is not reused for model choice.',
        hypothesis='Controller-level changes are learnable; state-specific residual signal requires separate held-family evidence.',new_rollout=0)
    write(OUT/'data_path_audit.json',result)
    print(json.dumps(dict(audit=result,matched_seed_summary=summaries),indent=2,default=lambda v:v.item()))

if __name__=='__main__':main()
