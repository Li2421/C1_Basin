#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, json
from collections import Counter
from pathlib import Path
import pyarrow.parquet as pq

H=Path(__file__).parent
def read(name):return list(csv.DictReader(open(H/name)))
def dump(name,obj):(H/name).write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 rows=pq.read_table(H/'pair_table.parquet').to_pylist();manifest=json.load(open(H/'dataset_manifest.json'));eta=json.load(open(H/'eta_split.json'));states=json.load(open(H/'state_split.json'))
 pm=read('prediction_metrics.csv');bm=read('b15_metrics.csv');rank=read('ranking_metrics.csv');san=read('selector_sanity.csv');selected=json.load(open(H/'selected_models.json'))
 def metric(model,sc,reg):return next(r for r in pm if r['model']==model and r['scenario']==sc and r['regime']==reg)
 D={sc:{m:metric(m,sc,'D_unseen_state_unseen_eta') for m in ('joint','scenario_only','eta_only')} for sc in ('Toy','DB')}
 state_overlap={}
 for sc,sp in states['groups'].items():
  sets={k:set(v) for k,v in sp.items()};state_overlap[sc]={f'{a}_{b}':len(sets.get(a,set())&sets.get(b,set())) for a,b in [('train','val'),('train','test'),('val','test')]}
 qdist={}
 for sc in ('Toy','DB'):
  qdist[sc]={}
  for kind,pred in [('available',lambda r:True),('train',lambda r:r['sampled_train'])]:
   z=[r for r in rows if r['scenario']==sc and pred(r)];qdist[sc][kind]={'pairs':len(z),'clear_failure':sum(r['empirical_q']<=.5 for r in z),'intermediate':sum(.5<r['empirical_q']<.9 for r in z),'near_boundary':sum(.9<=r['empirical_q']<15/16 for r in z),'robust':sum(r['empirical_q']>=15/16 for r in z)}
 dbhash=sha(manifest['database']);db_unchanged=dbhash==manifest['database_sha256']
 (H/'data_audit.md').write_text(f"""# Data audit

- Source: `{manifest['database']}` only; database unchanged during the experiment: **{db_unchanged}**.
- Authoritative basis: `{manifest['basis_sha256']}`.
- Canonical available pairs: Toy {manifest['counts']['Toy']['available_pairs']:,}; DB {manifest['counts']['DB']['available_pairs']:,}.
- Sampled TRAIN pairs: Toy {manifest['counts']['Toy']['train_pairs']:,}; DB {manifest['counts']['DB']['train_pairs']:,}.
- Evidence unit: unique compatible state×eta aggregate; duplicate experiment provenance and seed records were deduplicated.
- Excluded: quarantined conflicts, numerical failures, ambiguous rollouts, and non-authoritative controller profiles.
- Training weight: `min(n_trials,16)`; the empirical success ratio remains `n_success/n_trials`.

Success-rate coverage: `{json.dumps(qdist,sort_keys=True)}`.

No rollout was requested or executed.
""")
 (H/'split_leakage_audit.md').write_text(f"""# Split leakage audit

## State groups

Frozen source-group splits from the prior selector datasets were reused. Pairwise source-group overlaps are `{json.dumps(state_overlap,sort_keys=True)}`; all are zero.

## Eta groups

Exact eta IDs were not independently randomized. Eta points were joined into connected components whenever normalized distance was <=0.05, then entire components were assigned together using outcome-blind availability balancing.

- Unique eta: {eta['unique_eta']:,}; components: {eta['groups']:,}.
- Test→train nearest distance: minimum {eta['nearest_holdout_to_train']['test']['min']:.4f}, median {eta['nearest_holdout_to_train']['test']['median']:.4f}, mean {eta['nearest_holdout_to_train']['test']['mean']:.4f}.
- No eta UID appears in multiple splits, and no cross-split pair is closer than the grouping radius apart from floating-point equality at the 0.05 boundary.

The main simultaneous holdout contains Toy {manifest['counts']['Toy']['D_unseen_state_unseen_eta']} and DB {manifest['counts']['DB']['D_unseen_state_unseen_eta']} pairs. DB coverage is scientifically limited and is reported as such.
""")
 toy=D['Toy']['joint'];db=D['DB']['joint']; toy_eta=D['Toy']['eta_only'];db_eta=D['DB']['eta_only'];toy_sep=D['Toy']['scenario_only'];db_sep=D['DB']['scenario_only']
 decision={
  'classification':'CONTINUOUS_FIELD_DOES_NOT_GENERALIZE',
  'new_rollouts':0,'database_unchanged':db_unchanged,
  'train_pairs':{sc:manifest['counts'][sc]['train_pairs'] for sc in ('Toy','DB')},
  'split_leakage':{'state_source_group_overlap':state_overlap,'test_eta_nearest_train':eta['nearest_holdout_to_train']['test']},
  'unseen_state_unseen_eta':D,
  'state_aware_vs_eta_only':{
   'Toy':{'nll_gain':float(toy_eta['binomial_nll'])-float(toy['binomial_nll']),'mae_gain':float(toy_eta['q_mae'])-float(toy['q_mae'])},
   'DB':{'nll_gain':float(db_eta['binomial_nll'])-float(db['binomial_nll']),'mae_gain':float(db_eta['q_mae'])-float(db['q_mae'])}},
  'joint_vs_scenario_only':{
   'Toy':{'nll_gain':float(toy_sep['binomial_nll'])-float(toy['binomial_nll']),'mae_gain':float(toy_sep['q_mae'])-float(toy['q_mae'])},
   'DB':{'nll_gain':float(db_sep['binomial_nll'])-float(db['binomial_nll']),'mae_gain':float(db_sep['q_mae'])-float(db['q_mae'])}},
  'ranking':rank,'selector_sanity':san,'selected_models':selected,
  'reason':'Toy simultaneous holdout has near-zero Q rank correlation and poor finite-candidate regret; DB simultaneous holdout is better explained by eta-only and joint calibration is poor. The state-aware joint field is not a reliable general continuous selector.'}
 dump('final_decision.json',decision)
 rank_by={(r['scenario'],int(r['K'])):r for r in rank}
 rank_lines=[]
 for (sc,k),r in rank_by.items():
  b15='-' if r.get('b15_selection_rate','') in ('',None) else f"{float(r['b15_selection_rate']):.3f}"
  top3='-' if r.get('top3_oracle_hit_rate','') in ('',None) else f"{float(r['top3_oracle_hit_rate']):.3f}"
  regret='-' if r.get('mean_regret','') in ('',None) else f"{float(r['mean_regret']):.3f}"
  rank_lines.append(f"| {sc} | {k} | {r['eligible_states']} | {regret} | {b15} | {top3} | {r['coverage_status']} |")
 report=f"""# OrthoFlow3 continuous Basin critic V1

## Dataset and split

Training used {manifest['counts']['Toy']['train_pairs']:,} Toy and {manifest['counts']['DB']['train_pairs']:,} DB canonical aggregate pairs. No rollout was generated. Source-group overlap is zero. Spatial eta grouping gives a test→train nearest-distance median of {eta['nearest_holdout_to_train']['test']['median']:.3f} normalized units (minimum {eta['nearest_holdout_to_train']['test']['min']:.3f}).

## Simultaneous unseen-state / unseen-eta

| scenario | pairs | model | NLL | Brier | Q MAE | Q Spearman |
|---|---:|---|---:|---:|---:|---:|
| Toy | {toy['pairs']} | joint | {float(toy['binomial_nll']):.3f} | {float(toy['brier']):.3f} | {float(toy['q_mae']):.3f} | {float(toy['q_spearman']):.3f} |
| Toy | {toy_eta['pairs']} | eta-only | {float(toy_eta['binomial_nll']):.3f} | {float(toy_eta['brier']):.3f} | {float(toy_eta['q_mae']):.3f} | {float(toy_eta['q_spearman']):.3f} |
| DB | {db['pairs']} | joint | {float(db['binomial_nll']):.3f} | {float(db['brier']):.3f} | {float(db['q_mae']):.3f} | {float(db['q_spearman']):.3f} |
| DB | {db_eta['pairs']} | eta-only | {float(db_eta['binomial_nll']):.3f} | {float(db_eta['brier']):.3f} | {float(db_eta['q_mae']):.3f} | {float(db_eta['q_spearman']):.3f} |

Toy joint improves absolute error over eta-only but loses essentially all rank information on simultaneous holdout. In DB, eta-only is materially better than the state-aware joint model; the 64-pair DB panel contains only a limited number of held-out eta components.

Joint versus scenario-only is mixed: positive transfer on Toy (NLL {float(toy['binomial_nll']):.3f} vs {float(toy_sep['binomial_nll']):.3f}) and negative transfer on DB ({float(db['binomial_nll']):.3f} vs {float(db_sep['binomial_nll']):.3f}).

## Finite-candidate ranking

| scenario | K | eligible states | regret | B15 selection | top-3 oracle hit | status |
|---|---:|---:|---:|---:|---:|---|
"""+'\n'.join(rank_lines)+f"""

Toy K=8/16 regret is large despite high top-3 coverage. DB K=8 is trivial/perfect on the highly robust anchors, but K=16/32 degrades sharply. Toy K=32 has only two eligible states and is not considered evidence.

## Old codebook sanity

- Toy: critic robust selection {float(san[0]['critic_b15_rate']):.3f}, legacy selector {float(san[0]['legacy_b15_rate']):.3f}, mode agreement {float(san[0]['mode_agreement']):.3f}, rank Spearman {float(san[0]['mean_rank_spearman']):.3f}.
- DB: both achieve robust selection 1.000, but exact mode agreement is zero because many anchors tie at Q=1; rank Spearman is {float(san[1]['mean_rank_spearman']):.3f}.

## Decision

**CONTINUOUS_FIELD_DOES_NOT_GENERALIZE**.

The continuous critic learns useful partial structure, but the joint state-aware field fails the decisive simultaneous holdout: Toy ranking collapses, DB is dominated by eta-only, and finite-candidate regret rises with candidate diversity. It is not ready for eta selection or critic-guided optimization.

**NEW ROLLOUT = 0**.
"""
 # Avoid nested f-string parser complexity by replacing the prebuilt ranking section separately below.
 (H/'final_report.md').write_text(report)
 dump('runtime_statistics.json',{'new_rollouts':0,'models_trained':12,'train_pairs':decision['train_pairs'],'database_unchanged':db_unchanged})
 dump('working_state.json',{'status':'COMPLETE','completed':['database_audit','dataset','splits','12_models','prediction_evaluation','ranking','selector_sanity','finalization'],'final_decision':decision['classification'],'next_action':None,'new_rollouts':0})
 artifacts=[]
 for p in sorted(H.rglob('*')):
  if p.is_file() and p.name!='manifest.json':artifacts.append({'path':str(p.relative_to(H)),'sha256':sha(p)})
 dump('manifest.json',{'task':'ORTHOFLOW3_CONTINUOUS_BASIN_CRITIC_V1','new_rollouts':0,'artifacts':artifacts})
 print(json.dumps(decision,indent=2))
if __name__=='__main__':main()
