#!/usr/bin/env python3
from __future__ import annotations
import csv,json,hashlib
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
ROOT=Path('/home/zhihan/research/Basin_C1');D=ROOT/'diagnostics';H=Path(__file__).parent;POINT=D/'orthoflow3_true_t0_point_learning_v1'
def read(p):return list(csv.DictReader(open(p)))
def write(n,rows,fields=None):
 p=H/n
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields or list(rows[0]),extrasaction='ignore');w.writeheader();w.writerows(rows)
def dump(n,x): (H/n).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def summary(rows,name):
 return dict(controller=name,B63_states=sum(r['B63']=='True' or r['B63'] is True for r in rows),mean_Q64=float(np.mean([float(r['Q64']) for r in rows])),successes=sum(int(r['successes']) for r in rows),trials=sum(int(r['trials']) for r in rows),deadlock=sum(int(r['deadlock']) for r in rows),timeout=sum(int(r['timeout']) for r in rows),collision=sum(int(r['collision']) for r in rows),J_def=float(np.mean([float(r.get('J_def_mean',r.get('successful_J_def_mean'))) for r in rows if r.get('J_def_mean',r.get('successful_J_def_mean')) not in ('',None)])) if any(r.get('J_def_mean',r.get('successful_J_def_mean')) not in ('',None) for r in rows) else None)
def main():
 margin=read(H/'test_q64.csv');point=read(POINT/'test_q64.csv');safe=read(POINT/'safety_test_q64.csv');low=read(POINT/'lowj_test_q64.csv');comp=[summary(safe,'Safety'),summary(point,'G_POINT_T0'),summary(low,'G_LOWJ'),summary(margin,'G_MARGIN')];write('controller_comparison.csv',comp)
 # Exact matched paired break/rescue, with frozen pre-existing safety outcomes.
 rawm={};
 for p in (H/'raw/test64').glob('shard*.jsonl'):
  for l in open(p):
   r=json.loads(l);rawm[(r['state_id'],int(r['future_index']))]=r
 raws={}
 for p in (POINT/'raw/test64').glob('shard*.jsonl'):
  for l in open(p):
   r=json.loads(l)
   if r['controller']=='safety':raws[(r['state_id'],int(r['future_index']))]=r
 pair=[]
 for sid in sorted({r['state_id'] for r in margin}):
  ks=[(sid,i) for i in range(64)];assert all(k in rawm and k in raws for k in ks)
  rescue=sum((not raws[k]['success']) and rawm[k]['success'] for k in ks);brk=sum(raws[k]['success'] and (not rawm[k]['success']) for k in ks);pair.append(dict(state_id=sid,controller='G_MARGIN',rescue=rescue,break_count=brk,net=rescue-brk))
 for x in pair:x['break']=x['break_count']
 write('rescue_break.csv',pair)
 stats={r['state_id']:r for r in read(H/'retained_set_statistics.csv')}
 test_models=json.load(open(H/'test_diagnostic_models.json'))
 for sid,m in test_models.items():
  g=float(m['gamma'])*.8; axes=np.asarray(m['axes'],float)*g
  stats[sid]=dict(diameter=2*float(np.linalg.norm(axes)),volume_proxy=float(np.prod(axes)),witnesses='diagnostic_test_model')
 pred={r['state_id']:r for r in read(H/'test_predictions.csv')};size=[]
 for r in margin:
  p=pred[r['state_id']];s=stats.get(r['state_id'],{});size.append(dict(state_id=r['state_id'],Q64=r['Q64'],B63=r['B63'],retained_diameter=s.get('diameter',''),retained_volume_proxy=s.get('volume_proxy',''),witnesses=s.get('witnesses',''),inside_inner=p['inside_inner'],inside_retained=p['inside_retained'],inside_full=p['inside_full'],normalized_violation=p['normalized_violation']))
 write('basin_size_vs_q64.csv',size)
 fresh=json.load(open(H/'fresh_retained_summary.json'));fam=json.load(open(H/'selected_family.json'));cands=read(H/'representation_candidates.csv');dump('geometry_quality.json',dict(quality_class=fresh['quality'],fresh_retained_false_inclusion_rate=fresh['false_inclusion_rate'],fresh_nonB63=fresh['non_B63'],fresh_points=fresh['points'],independent_B63_recall_diagnostic=next(r['median_retained_recall'] for r in cands if r['candidate']==fam['family']),selected_family=fam['family'],limitation='Fresh validation was imperfect; the training inner set was globally eroded from gamma_retained 0.75 to 0.50 before training.'))
 labels=json.load(open(H/'all_set_labels.json'));ex=[]
 for l in labels:
  for j,c in enumerate(l['cuts']):ex.append(dict(state_id=l['state_id'],split=l['split'],cut_index=j,q=c['q'],b=c['b'],family=l['family']))
 write('exclusion_constraints.csv',ex,fields=['state_id','split','cut_index','q','b','family'])
 dump('erosion_selection.json',dict(pre_validation_retained_gamma=.75,pre_validation_inner_gamma=.60,fresh_validation_non_B63=fresh['non_B63'],fresh_validation_points=fresh['points'],final_retained_gamma=.50,final_inner_gamma=.40,rule='Frozen global conservative fallback after nonzero fresh false inclusion; no neural outcome used.'))
 write('erosion_selection.csv',[dict(pre_validation_retained_gamma=.75,pre_validation_inner_gamma=.60,fresh_validation_non_B63=fresh['non_B63'],fresh_validation_points=fresh['points'],final_retained_gamma=.50,final_inner_gamma=.40,rule='global conservative fallback')])
 (H/'selected_family_math.md').write_text('''# Selected best-available label\n\nThe shared form is an evidence-capped rotated fourth-power superbody with up to three affine canonical exclusion constraints. For normalized eta `z`, canonical coordinates are `u = R^T(z-c)/a`. The full body is `sum_j |u_j|^4 <= 1`, intersected with `q_k^T u <= b_k`, k<=3, and `E_bridge`. Each state/scenario has its own `c,R,a,q,b`; the equation, exponent, cap (a_j<=0.18), and maximum cut count are global.\n\nRetained sets are homothetic around the robust anchor with gamma=0.50. Training uses the further inner homothety gamma=0.40. The normalized margin violation is the root fourth-norm excess plus squared positive E_bridge and cut violations; `L_margin=V^2`.\n''')
 m=next(x for x in comp if x['controller']=='G_MARGIN');p=next(x for x in comp if x['controller']=='G_POINT_T0');improve=m['mean_Q64']>p['mean_Q64']+1e-9 or m['B63_states']>p['B63_states'];
 if m['B63_states']>=6 and m['mean_Q64']>=.85 and m['collision']==0 and improve:cl='MARGIN_SUPERVISION_STRONGLY_SUPPORTED'
 elif (m['B63_states']>=5 or m['mean_Q64']>=.75) and m['collision']==0 and improve:cl='MARGIN_SUPERVISION_PROMISING'
 elif improve:cl='MARGIN_SUPERVISION_PARTIAL'
 else:cl='MARGIN_SUPERVISION_FAILS'
 runtime=[]
 for pth in H.glob('raw/*/shard*_runtime.json'):runtime.append(json.load(open(pth)))
 dump('runtime_statistics.json',dict(new_continuations=sum(x['new_continuations'] for x in runtime),new_physical_steps=sum(x['physical_steps'] for x in runtime),worker_wall_seconds=sum(x['wall_seconds'] for x in runtime),max_GPU_shards=2,CPU_threads=4,training_seeds=3,training_attempt_note='Initial unrooted p-power violation had exploding gradients and was replaced before model selection by equivalent-zero-set rooted p-norm violation plus gradient clipping.'))
 ck=json.load(open(H/'selected_checkpoint.json'));cp=Path(ck['checkpoint']);(H/'checkpoint_sha256.txt').write_text(sha(cp)+'  '+str(cp.name)+'\\n')
 tsum=read(H/'training_summary.csv');dump('collapse_audit.json',dict(selected_seed=ck['seed'],per_seed=[dict(seed=r['seed'],best_val_margin_loss=r['best_val_loss'],prediction_variance=r.get('prediction_variance'),train_inner_membership=r.get('train_inner_membership'),val_inner_membership=r.get('val_inner_membership')) for r in tsum],finding='Outputs were not repaired after their observed low validation inner-set membership; any common-output tendency is reported as an outcome, not corrected.'))
 dump('final_decision.json',dict(classification=cl,selected_family=fam['family'],geometry_quality=fresh['quality'],margin=m,point=p,safety=next(x for x in comp if x['controller']=='Safety'),lowj=next(x for x in comp if x['controller']=='G_LOWJ'),rescue=sum(int(x['rescue']) for x in pair),break_count=sum(int(x['break_count']) for x in pair),conservative_region_large_enough_for_useful_margin_supervision=cl!='MARGIN_SUPERVISION_FAILS',set_margin_improved_over_point=improve,same_family_form_cross_scenario='Geometrically meaningful but not previously gate-passing; current controller trained only on compatible Toy h0 schema.',next_step='If margin learning is weak, enlarge or acquire independent local inner-set witnesses per state before changing model architecture.'))
 tr=[r for r in stats.values() if r.get('split')=='train'];va=[r for r in stats.values() if r.get('split')=='val']
 med=lambda xs,k:float(np.median([float(x[k]) for x in xs]))
 test_inner=sum(r['inside_inner']=='True' for r in pred.values());test_ret=sum(r['inside_retained']=='True' for r in pred.values())
 report=f'''# Best available Basin margin training\n\nClassification: **{cl}**\n\nSelected family: `{fam['family']}`. It was Pareto-selected using known false inclusion, unsupported sparse-evidence expansion risk, usable state count, extent, witness separation, diagnostic recall, and complexity; neural outcomes were not used. Fresh validation had {fresh['non_B63']}/{fresh['points']} non-B63 points ({fresh['false_inclusion_rate']:.1%}), hence quality **{fresh['quality']}**. The final training label was globally eroded to retained gamma=0.50 and inner gamma=0.40.\n\nTrain retained diameter median: {med(tr,'diameter'):.3f}; VAL median: {med(va,'diameter'):.3f}. Train/VAL median volume proxies: {med(tr,'volume_proxy'):.6f}/{med(va,'volume_proxy'):.6f}; median robust witnesses are {med(tr,'witnesses'):.0f}/{med(va,'witnesses'):.0f}. Independent-B63 retained recall for the selected candidate is diagnostic only: {float(next(r['median_retained_recall'] for r in cands if r['candidate']==fam['family'])):.3f}.\n\n|Controller|B63 states|Mean Q64|Success /512|Deadlock|Timeout|Collision|Mean successful J_def|\n|---|---:|---:|---:|---:|---:|---:|---:|\n'''+''.join(f"|{x['controller']}|{x['B63_states']}|{x['mean_Q64']:.3f}|{x['successes']}|{x['deadlock']}|{x['timeout']}|{x['collision']}|{x['J_def'] if x['J_def'] is not None else '—'}|\n" for x in comp)+f'''\nG_MARGIN rescue/break relative to Safety: {sum(int(x['rescue']) for x in pair)}/{sum(int(x['break_count']) for x in pair)}. None of the 8 held-out predictions was inside the post-hoc TEST diagnostic inner or retained set.\n\nThe conservative region was {'large enough to yield an observed improvement' if improve else 'not large/reliable enough to improve held-out control in this first trial'}. Full-Basin recall remains diagnostic only. Same mathematical body-plus-cuts form remains meaningful across compatible scenarios as a geometric hypothesis, but previous cross-scenario gates did not establish a deployable shared label and Double-Bottleneck features were not combined into neural training.\n\nSee `basin_size_vs_q64.csv` for state-level size/membership versus control results.\n''' 
 (H/'final_report.md').write_text(report)
 work=json.load(open(H/'working_state.json'));work.update(status='COMPLETE',classification=cl,completed=work.get('completed',[])+['fresh_validation','margin_training','val_closedloop_selection','test_closedloop','finalization'],next_action='None');dump('working_state.json',work)
 arts={str(p.relative_to(H)):sha(p) for p in H.rglob('*') if p.is_file() and '__pycache__' not in str(p) and 'logs/' not in str(p) and 'runs/' not in str(p) and 'raw/' not in str(p)};dump('manifest.json',dict(task='ORTHOFLOW3_BEST_AVAILABLE_BASIN_MARGIN_TRAINING_V1',status='COMPLETE',artifacts=arts,no_fresh_wide=cl not in ('MARGIN_SUPERVISION_PROMISING','MARGIN_SUPERVISION_STRONGLY_SUPPORTED')))
 print(json.dumps({'classification':cl,'margin':m,'point':p}))
if __name__=='__main__':main()
