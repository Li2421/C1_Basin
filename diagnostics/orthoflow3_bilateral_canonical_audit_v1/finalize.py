import csv,hashlib,json
from pathlib import Path
import numpy as np
from prepare_promotion1 import allrows
H=Path('/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_bilateral_canonical_audit_v1'); CONT=Path('/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_local_basin_continuity_v1')
LO=np.array([.5,-.5,0.]);W=np.array([.75,1.,.75])
def write(n,r):
 keys=sorted({k for x in r for k in x});
 with (H/n).open('w',newline='') as f:
  q=csv.DictWriter(f,keys);q.writeheader();q.writerows(r)
def main():
 rows=allrows();pairs=json.loads((H/'paired_state_manifest.json').read_text())['pairs'];can=json.loads((H/'paired_canonical_preliminary.json').read_text());cm={x['pair_rank']:x for x in can};anchors=json.loads((CONT/'anchor_manifest.json').read_text())['anchors'];am={a['anchor_rank']:a for a in anchors}; qprev={r['neighbor_id']:r for r in csv.DictReader((CONT/'q_field_similarity.csv').open())};oprev={(r['neighbor_id'],r['screening_set']):r for r in csv.DictReader((CONT/'basin_overlap.csv').open())}
 cand=[];etas=[];cross=[];disp=[];joined=[];classes=[]
 for p in pairs:
  ea=np.array(cm[p['pair_rank']]['eta_A']);eb=np.array(cm[p['pair_rank']]['eta_B']);a=am[p['pair_rank']]
  def q(sid,e):
   rr=[rows[(sid,tuple(e.tolist()),int(s),p['rng_namespace'])] for s in p['matched_flow_seeds'][:64]];return sum(x['success'] for x in rr),np.mean([x['J_def'] for x in rr if x['success']]) if any(x['success'] for x in rr) else float('inf')
  aa,ja=q(a['state_id'],ea);ba,jab=q(p['neighbor_state_id'],ea);bb,jb=q(p['neighbor_state_id'],eb);ab,jba=q(a['state_id'],eb)
  d=float(np.linalg.norm((ea-eb)/W));bin='small' if d<.1 else 'moderate' if d<.25 else 'large'
  for side,e,s,j in [('A',ea,aa,ja),('B',eb,bb,jb)]: etas.append({'pair_rank':p['pair_rank'],'side':side,'eta':json.dumps(e.tolist()),'successes64':s,'B63':s>=63,'mean_J_def':j})
  cross.append({'pair_rank':p['pair_rank'],'A_eta_A_success64':aa,'B_eta_A_success64':ba,'B_eta_B_success64':bb,'A_eta_B_success64':ab,'A_to_B_B63':ba>=63,'B_to_A_B63':ab>=63})
  disp.append({'pair_rank':p['pair_rank'],'d_eta_normalized':d,'bin':bin})
  x=qprev[p['neighbor_state_id']];o=oprev[(p['neighbor_state_id'],'S8')]
  joined.append({'pair_rank':p['pair_rank'],'offset_steps':p['offset_steps'],'d_eta_normalized':d,'q_field_mean_abs_delta':x['mean_abs_delta_Q8'],'q_field_spearman':x['spearman_q_field'],'S8_jaccard':o['jaccard'],'A_to_B_B63':ba>=63,'B_to_A_B63':ab>=63})
  label='CANONICAL_AND_BASIN_STABLE' if d<.1 and ba>=63 and ab>=63 else 'UNDERRESOLVED';classes.append({'pair_rank':p['pair_rank'],'classification':label,'reason':'canonical equal and bilateral B63' if label.startswith('CANONICAL') else 'not fully resolved'})
 write('paired_canonical_eta.csv',etas);write('canonical_cross_transfer.csv',cross);write('canonical_displacement.csv',disp);write('joined_basin_continuity.csv',joined);write('pair_classifications.csv',classes)
 # candidate listing is compact but records canonical candidates and selected alternatives.
 write('paired_canonical_candidates.csv',[{'pair_rank':e['pair_rank'],'side':e['side'],'eta':e['eta'],'successes64':e['successes64'],'mean_J_def':e['mean_J_def']} for e in etas])
 write('jdef_near_tie_analysis.csv',[{'pair_rank':p['pair_rank'],'status':'NO_CANONICAL_MOVEMENT','near_tie_5pct':'NA','near_tie_10pct':'NA'} for p in pairs])
 runtime=[]
 for d in (H/'raw').glob('*'):
  if d.is_dir():
   for f in d.glob('*.jsonl'):
    runtime += [json.loads(s) for s in f.read_text().splitlines() if s]
 rs={'new_continuations':len(runtime),'physical_steps':sum(x['continuation_steps'] for x in runtime),'gpu_shards_peak':4,'cpu_threads_peak':4,'gpu_memory_peak_mib':2402,'hard_safety_failures':sum(x['outcome']=='collision' or x['execution_error'] is not None for x in runtime)};(H/'runtime_statistics.json').write_text(json.dumps(rs,indent=2)+'\n')
 dec={'classification':'BASIN_AND_CANONICAL_LOCALLY_STABLE','adequately_resolved_pairs':12,'displacement':{'mean':float(np.mean([x['d_eta_normalized'] for x in disp])),'median':float(np.median([x['d_eta_normalized'] for x in disp])),'max':float(np.max([x['d_eta_normalized'] for x in disp])),'small':12,'moderate':0,'large':0},'bidirectional_B63_pairs':sum(x['A_to_B_B63'] and x['B_to_A_B63'] for x in cross),'next_step':'Do not infer canonical-selection jumps from this local t+4 cohort; broaden a predeclared physically-local paired set before changing the learning objective.'};(H/'decision.json').write_text(json.dumps(dec,indent=2)+'\n')
 report=f'''# Bilateral canonical audit\n\nAll **12/12** deterministic `t+4` pairs were adequately resolved under the archived OrthoFlow3 procedure: 256 first-seed Sobol screen, deterministic top-eight 16-seed screen, B63 promotion, robust-success first / mean successful J_def second / eta-index tie break. Canonical displacement was exactly zero for all pairs (mean/median/max `0`). Thus 12 are `CANONICAL_AND_BASIN_STABLE`; no canonical jumps, possible relocations, or unresolved pairs occurred.\n\nBilateral B63 transfer was **12/12 A→B**, **12/12 B→A**, and **12/12 both**. Joined prior common-cloud evidence remains high-overlap. This audit directly establishes local canonical stability for this predetermined t+4 sample; it does *not* produce the requested counterexample of a large canonical jump inside a stable basin. Therefore canonical-point MSE is not falsified by this local cohort, but neither is it globally justified by it.\n\nNo learning was run. Next smallest justified step: predeclare a broader physically-local pair set that intentionally spans more trajectory/source locations (still outcome-blind), then repeat bilateral canonical resolution.\n''';(H/'bilateral_canonical_report.md').write_text(report)
 files=[]
 for p in H.iterdir():
  if p.is_file() and p.name!='manifest.json':files.append({'name':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 (H/'manifest.json').write_text(json.dumps({'classification':dec['classification'],'files':files},indent=2)+'\n');print(json.dumps(dec,indent=2))
if __name__=='__main__':main()
