"""Write immutable handoff summaries from completed, deduplicated audit data."""
import csv, hashlib, json, os
from pathlib import Path
import numpy as np

H=Path('/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_local_basin_continuity_v1')
def rows(n): return list(csv.DictReader((H/n).open()))
def f(x): return float(x)
def stat(a): return {'mean':float(np.mean(a)),'median':float(np.median(a)),'min':float(np.min(a)),'max':float(np.max(a))}
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as q:
  for x in iter(lambda:q.read(1<<20),b''):h.update(x)
 return h.hexdigest()
def main():
 q=rows('q_field_similarity.csv'); o=rows('basin_overlap.csv'); x=rows('cross_transfer.csv'); d=rows('state_pair_distances.csv'); z=rows('zero_boundary_analysis.csv')
 summary={}
 for label,fn in [('all',lambda r:True),('abs1',lambda r:abs(int(r['offset_steps']))==1),('abs4',lambda r:abs(int(r['offset_steps']))==4)]:
  a=[r for r in q if fn(r)]; b=[r for r in o if fn(r) and r['screening_set']=='S8']; c=[r for r in x if fn(r)]
  summary[label]={'pairs':len(a),'q_abs_delta':stat([f(r['mean_abs_delta_Q8']) for r in a]),
    'spearman_available':stat([f(r['spearman_q_field']) for r in a if r['spearman_q_field']!='nan']),
    'pearson_available':stat([f(r['pearson_q_field']) for r in a if r['pearson_q_field']!='nan']),
    'S8_jaccard':stat([f(r['jaccard']) for r in b if r['jaccard']!='nan']),
    'S8_anchor_to_neighbor_retention':stat([f(r['retention_anchor_to_neighbor']) for r in b if r['retention_anchor_to_neighbor']!='nan']),
    'directional_cross_pairs':len(c),'directional_B63_retained':sum(r['B63_retained']=='True' for r in c),
    'directional_Q64':stat([f(r['Q_64']) for r in c]) if c else None}
 transition=[r for r in z if abs(f(r['anchor_zero_Q8'])-f(r['neighbor_zero_Q8']))>0]
 raw=[]
 for directory in (H/'raw').iterdir():
  if directory.is_dir():
   for p in directory.glob('*.jsonl'): raw += [json.loads(s) for s in p.read_text().splitlines() if s]
 steps=sum(int(r.get('continuation_steps',0)) for r in raw)
 runtime={'new_continuation_records_executed':len(raw),'unique_screening_tuples':11520,'duplicate_screening_records_excluded':1120,
  'cross_transfer_new_continuations':2288,'physical_steps_executed':steps,'gpu_shards_peak':6,'cpu_threads_peak':6,
  'gpu_memory_peak_observed_mib':3602,'max_process_rss_observed_mib':718,'note':'Six idle-server shards were explicitly user-authorized. Duplicate screening work resulted from a documented cancellation race and was excluded from analysis; the final executed continuation count remains below 15000.'}
 (H/'runtime_statistics.json').write_text(json.dumps(runtime,indent=2)+'\n')
 conclusion={
  'classification':'BASIN_LOCALLY_STABLE_CANONICAL_DISCONTINUOUS',
  'learning_implication':'BASIN_AWARE_DIRECT_ETA',
  'true_basin_switch_cases_supported':0,
  'reason':'The common eta-success field is almost invariant across valid t±1/t±4 replay states and 38/39 selected parent-B63 eta transfers remain B63. The sole promoted loss is eta=0 at t-4, 58/64 (timeout), not a bidirectionally established basin relocation. Existing canonical-target non-smoothness therefore is better explained by representative selection inside a locally stable basin than by demonstrated local basin switching.',
  'limitations':['Neighbor canonical eta values were deliberately not re-optimized; directional robust transfer is anchor-to-neighbor only.','Common-cloud basin sets are 8-seed screening sets, not topological proofs.','Six long-horizon t±4 transfer pairs were removed solely to remain inside the hard continuation budget after the scheduler cancellation race.']}
 (H/'audit_conclusion.json').write_text(json.dumps({'summary':summary,'zero_transition_pairs':transition,'conclusion':conclusion},indent=2)+'\n')
 report=f'''# OrthoFlow3 local basin continuity audit

## Result

**Classification: `{conclusion['classification']}`.** Across 12 uniformly selected provenance-valid anchors and 48 exact same-trajectory neighbors (`t±1`, `t±4`), the 24-point common eta Q-fields were almost unchanged. The conclusion is empirical: it supports local stability of the sampled success field, not mathematical basin connectedness.

| separation | pairs | mean / median `|ΔQ8|` | mean / median S8 Jaccard | mean / median Spearman (available pairs) | directional B63 retained |
|---|---:|---:|---:|---:|---:|
| `±1` | {summary['abs1']['pairs']} | {summary['abs1']['q_abs_delta']['mean']:.6f} / {summary['abs1']['q_abs_delta']['median']:.6f} | {summary['abs1']['S8_jaccard']['mean']:.6f} / {summary['abs1']['S8_jaccard']['median']:.6f} | {summary['abs1']['spearman_available']['mean']:.6f} / {summary['abs1']['spearman_available']['median']:.6f} | {summary['abs1']['directional_B63_retained']}/{summary['abs1']['directional_cross_pairs']} |
| `±4` | {summary['abs4']['pairs']} | {summary['abs4']['q_abs_delta']['mean']:.6f} / {summary['abs4']['q_abs_delta']['median']:.6f} | {summary['abs4']['S8_jaccard']['mean']:.6f} / {summary['abs4']['S8_jaccard']['median']:.6f} | {summary['abs4']['spearman_available']['mean']:.6f} / {summary['abs4']['spearman_available']['median']:.6f} | {summary['abs4']['directional_B63_retained']}/{summary['abs4']['directional_cross_pairs']} |
| all | {summary['all']['pairs']} | {summary['all']['q_abs_delta']['mean']:.6f} / {summary['all']['q_abs_delta']['median']:.6f} | {summary['all']['S8_jaccard']['mean']:.6f} / {summary['all']['S8_jaccard']['median']:.6f} | {summary['all']['spearman_available']['mean']:.6f} / {summary['all']['spearman_available']['median']:.6f} | {summary['all']['directional_B63_retained']}/{summary['all']['directional_cross_pairs']} |

The selected directional robust transfers have mean `Q64={summary['all']['directional_Q64']['mean']:.6f}`. All `±1` transfer tests retained B63; `14/15` selected `±4` tests did. The one non-retention was an anchor zero eta at `t-4`, `58/64` success (six timeouts); its common-cloud mean `|ΔQ8|` was only `0.005208`. It is not bidirectional evidence of a genuine basin switch.

## Valid nearby-state construction

Neighbors are exact replay states on the same frozen trajectory, preserving physical state, Flow continuation identity, history, monitor, timers, latches and absolute timestep. No feature perturbation or reconstructed physical-only state was used. Across all pairs aggregate position displacement had mean/median `{np.mean([f(r['aggregate_position_displacement']) for r in d]):.6f}/{np.median([f(r['aggregate_position_displacement']) for r in d]):.6f}`, relative-geometry change `{np.mean([f(r['relative_geometry_change']) for r in d]):.6f}/{np.median([f(r['relative_geometry_change']) for r in d]):.6f}`, and normalized 214-D feature distance `{np.mean([f(r['normalized_feature_distance']) for r in d]):.6f}/{np.median([f(r['normalized_feature_distance']) for r in d]):.6f}`. No sampled pair crossed a discrete monitor transition or changed its stuck timer.

## Zero boundary and canonical selection

`eta=0` changed its 8-seed status on only `{len(transition)}/48` pairs. The promoted zero transition above is a local feasibility boundary candidate, but not a dominant pattern and not a confirmed far-away active-basin relocation. Canonical neighbor targets were not searched in this strictly bounded continuity audit; consequently direct per-pair canonical-jump vs set-jump and bidirectional neighbor-canonical transfer remain unavailable. The prior documented non-smooth canonical representative together with the present stable local Q-fields supports the interpretation that canonical selection is the more likely source of the apparent non-smoothness, while this audit does not prove it at every pair.

## Set distance and J_def

The screening `S8` / `S7` overlaps are reported, but robust sampled Hausdorff/centroid distances are marked `UNDERRESOLVED`: neighbor robust sets were not re-searched and 8/8 is not B63. J_def is secondary; directional canonical-transfer J_def is stored, but a changing neighbor minimum-J_def point cannot be inferred without neighbor canonical optimization.

## Safety and integrity

There were zero agent/wall collisions, invalid actions, numerical errors, or projection failures in all retained screening and cross-transfer rollouts. Of 12,640 raw screening records, 1,120 deterministic cancellation-race duplicates were verified identical on success/outcome/terminal step and excluded, leaving exactly 11,520 required unique screening tuples. Full details are in `integrity_checks.json` and `resource_amendment.json`.

## Learning implication

The evidence supports `{conclusion['learning_implication']}` over treating canonical-point MSE as a sufficient representation of the local feasible set. It does **not** justify a generative/multimodal model or a claim of true local basin switching. The smallest justified next experiment is a bounded, source-uniform neighbor-oracle confirmation on a small predeclared subset of these pairs, producing both neighbor canonical representatives and two-way B63 cross-transfer; no model training should start automatically.
'''
 (H/'basin_continuity_report.md').write_text(report)
 files=[]
 for p in sorted(H.iterdir()):
  if p.is_file() and p.name not in {'manifest.json'}: files.append({'name':p.name,'sha256':sha(p),'bytes':p.stat().st_size})
 (H/'manifest.json').write_text(json.dumps({'schema':'orthoflow3_local_basin_continuity_manifest_v1','basis_sha256':'51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38','classification':conclusion['classification'],'files':files},indent=2)+'\n')
 print(json.dumps({'classification':conclusion['classification'],'runtime':runtime,'summary':summary},indent=2))
if __name__=='__main__':main()
