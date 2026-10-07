#!/usr/bin/env python3
import csv,json,hashlib,time
from pathlib import Path
import numpy as np
H=Path(__file__).parent
def read(n):return list(csv.DictReader(open(H/n)))
def num(x):
 try:return float(x)
 except:return None
def dump(n,x):(H/n).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')

toy=read('toy_prediction_metrics.csv');rank=read('toy_ranking_metrics.csv');abl=read('matrix_density_ablation.csv');db=read('db_transfer_results.csv')
def metric(model,reg='D_unseen_state_unseen_eta'):return next(r for r in toy if r['model']==model and r['regime']==reg)
st=metric('toy_structured');sm=metric('toy_structured_matched');sp=metric('toy_sparse_control');sem=metric('toy_structured_eta_matched');se=metric('toy_sparse_eta_control')
old={'nll':.558,'mae':.369,'spearman':.015}
improve_old={k:old[k]-num(st[k]) if k!='spearman' else num(st[k])-old[k] for k in old}
improve_sparse={'nll':num(sp['nll'])-num(sm['nll']),'mae':num(sp['mae'])-num(sm['mae']),'spearman':num(sm['spearman'])-num(sp['spearman'])}
improve_sparse_eta={'nll':num(se['nll'])-num(sem['nll']),'mae':num(se['mae'])-num(sem['mae']),'spearman':num(sem['spearman'])-num(se['spearman'])}
r32=next(r for r in rank if int(r['K'])==32)
a=sorted(abl,key=lambda r:int(r['density_percent']));monotonic=(num(a[2]['nll'])<=num(a[1]['nll'])<=num(a[0]['nll']) and num(a[2]['mae'])<=num(a[1]['mae'])<=num(a[0]['mae']))
strong=(improve_old['nll']>.1 and improve_old['mae']>.1 and num(st['spearman'])>.3 and improve_sparse['nll']>.05 and improve_sparse['mae']>.05 and num(r32['mean_regret'])<.1 and num(r32['b15_selection_rate'])>=.8 and monotonic)
partial=(improve_old['nll']>0 and improve_old['mae']>0 and num(st['spearman'])>.1 and num(r32['mean_regret'])<.2)
toy_class='STRUCTURED_CONTINUOUS_Q_STRONGLY_SUPPORTED' if strong else ('STRUCTURED_CONTINUOUS_Q_PARTIALLY_SUPPORTED' if partial else 'CONTINUOUS_ETA_GENERALIZATION_STILL_FAILS')
dm={r['model']:r for r in db};base=dm['db_only'];pre=dm['toy_pretrained_db_adapter'];joint=dm['joint_finetune']
pre_gain={'nll':num(base['nll'])-num(pre['nll']),'mae':num(base['mae'])-num(pre['mae']),'spearman':num(pre['spearman'])-num(base['spearman'])}
joint_gain={'nll':num(base['nll'])-num(joint['nll']),'mae':num(base['mae'])-num(joint['mae']),'spearman':num(joint['spearman'])-num(base['spearman'])}
if ((pre_gain['nll']>.02 and pre_gain['mae']>.02 and pre_gain['spearman']>=0) or
    (joint_gain['nll']>.02 and joint_gain['mae']>.02 and joint_gain['spearman']>=0)):dbclass='POSITIVE'
elif (sum(v>0 for v in pre_gain.values())>=2 or sum(v>0 for v in joint_gain.values())>=2):dbclass='WEAK'
else:dbclass='NOT_DEMONSTRATED'
pre=json.load(open(H/'cache_preflight.json'))['summary'];post=json.load(open(H/'cache_postflight.json'))['summary'];runt=[]
for p in sorted((H/'raw').glob('shard*_runtime.json')):runt.append(json.load(open(p)))
new=pre['genuinely_missing']
design=json.load(open(H/'toy_eta_split.json'));states=json.load(open(H/'toy_state_split.json'))
decision={'task':'ORTHOFLOW3_STRUCTURED_CONTINUOUS_Q_DATA_V1','toy_classification':toy_class,'db_transfer_diagnostic':dbclass,'new_db_rollout':0,'new_toy_continuations':new,
 'global_db_postflight_exact':post['exact_reusable']==post['total_requested'] and post['genuinely_missing']==0,'preflight':pre,'postflight':post,
 'toy_D_structured':st,'toy_D_structured_matched_basis':sm,'toy_D_sparse_pair_matched':sp,'toy_D_structured_eta_matched_basis':sem,'toy_D_sparse_eta_matched':se,'old_continuous_critic':old,'improvement_vs_old':improve_old,'improvement_vs_sparse':improve_sparse,'improvement_vs_sparse_eta_matched':improve_sparse_eta,
 'ranking':rank,'density_ablation':abl,'density_monotonic':monotonic,'db_results':db,'db_gain_vs_db_only':{'toy_pretrained':pre_gain,'joint_finetune':joint_gain},
 'interpretation':'Toy structured cross-matrix isolates h×eta interaction geometry. DB result is diagnostic only because its frozen training matrix remains sparse and lacks robust-transition coverage.'}
dump('final_decision.json',decision)
report=f'''# ORTHOFLOW3_STRUCTURED_CONTINUOUS_Q_DATA_V1\n\n## Decision\n\n- Toy: **{toy_class}**\n- DB transfer diagnostic: **{dbclass}**\n- New Toy continuations: **{new:,}**; new DB continuations: **0**.\n- Global DB postflight exact reuse: **{decision['global_db_postflight_exact']}**.\n\n## Frozen design\n\nStates: {states['counts']['train']}/{states['counts']['val']}/{states['counts']['test']} TRAIN/VAL/TEST with zero source-group overlap. Eta: {design['counts']['train']['total']}/{design['counts']['val']['total']}/{design['counts']['test']['total']}; TRAIN types global/transition/boundary = {design['counts']['train']['global']}/{design['counts']['train']['transition']}/{design['counts']['train']['boundary']}. TEST-to-TRAIN normalized eta distance min/median = {design['test_to_train_distance']['min']:.4f}/{design['test_to_train_distance']['median']:.4f}.\n\nPreflight requested {pre['total_requested']:,}: exact {pre['exact_reusable']:,}, partial {pre['partial_reusable']:,}, aggregate {pre['aggregate_reusable']:,}, truly missing {pre['genuinely_missing']:,}.\n\n## Toy simultaneous unseen state + unseen eta\n\n| Model | NLL | MAE | Spearman | B15 AUROC | B15 accuracy |\n|---|---:|---:|---:|---:|---:|\n| Structured full | {num(st['nll']):.4f} | {num(st['mae']):.4f} | {num(st['spearman']):.4f} | {num(st['b15_auroc']):.4f} | {num(st['b15_accuracy']):.4f} |\n| Structured matched basis | {num(sm['nll']):.4f} | {num(sm['mae']):.4f} | {num(sm['spearman']):.4f} | {num(sm['b15_auroc']):.4f} | {num(sm['b15_accuracy']):.4f} |\n| Pair/Q-bin/state-matched sparse | {num(sp['nll']):.4f} | {num(sp['mae']):.4f} | {num(sp['spearman']):.4f} | {num(sp['b15_auroc']):.4f} | {num(sp['b15_accuracy']):.4f} |\n| Eta-count-matched sparse | {num(se['nll']):.4f} | {num(se['mae']):.4f} | {num(se['spearman']):.4f} | {num(se['b15_auroc']):.4f} | {num(se['b15_accuracy']):.4f} |\n| Old critic | 0.5580 | 0.3690 | 0.0150 | — | — |\n\nFull structured improvement over old: NLL {improve_old['nll']:+.4f}, MAE {improve_old['mae']:+.4f}, Spearman {improve_old['spearman']:+.4f}. Fair matched-basis structured improvement over sparse: NLL {improve_sparse['nll']:+.4f}, MAE {improve_sparse['mae']:+.4f}, Spearman {improve_sparse['spearman']:+.4f}.\n\n## Finite-candidate ranking\n\n| K | Regret | B15 selection | Top-3 B15 |\n|---:|---:|---:|---:|\n'''
report=report.replace(
 '| Eta-count-matched sparse |',
 f"| Structured eta-count matched basis | {num(sem['nll']):.4f} | {num(sem['mae']):.4f} | {num(sem['spearman']):.4f} | {num(sem['b15_auroc']):.4f} | {num(sem['b15_accuracy']):.4f} |\n| Eta-count/Q-bin/pair-matched sparse |"
)
report=report.replace(
 'Fair matched-basis structured improvement over sparse:',
 f"Pair/Q-bin/state matched structured improvement over sparse:"
).replace(
 '\n\n## Finite-candidate ranking',
 f" Eta-count/pair/Q-bin/state matched structured improvement over sparse: NLL {improve_sparse_eta['nll']:+.4f}, MAE {improve_sparse_eta['mae']:+.4f}, Spearman {improve_sparse_eta['spearman']:+.4f}.\n\n## Finite-candidate ranking"
)
for r in rank:report+=f"| {r['K']} | {num(r['mean_regret']):.4f} | {num(r['b15_selection_rate']):.4f} | {num(r['top3_b15_hit_rate']):.4f} |\n"
report+='\n## Density ablation\n\n| Density | Train pairs | NLL | MAE | Spearman |\n|---:|---:|---:|---:|---:|\n'
for r in a:report+=f"| {r['density_percent']}% | {r['train_pairs']} | {num(r['nll']):.4f} | {num(r['mae']):.4f} | {num(r['spearman']):.4f} |\n"
report+=f"\nMonotonic NLL/MAE improvement: **{monotonic}**.\n\n## Frozen DB diagnostic\n\n| Model | NLL | MAE | Spearman | B15 accuracy |\n|---|---:|---:|---:|---:|\n"
for r in db:report+=f"| {r['model']} | {num(r['nll']):.4f} | {num(r['mae']):.4f} | {num(r['spearman']):.4f} | {num(r['b15_accuracy']):.4f} |\n"
report+=f'''\nToy-pretrained gain versus DB-only: NLL {pre_gain['nll']:+.4f}, MAE {pre_gain['mae']:+.4f}, Spearman {pre_gain['spearman']:+.4f}. Joint-finetune gain: NLL {joint_gain['nll']:+.4f}, MAE {joint_gain['mae']:+.4f}, Spearman {joint_gain['spearman']:+.4f}. A null transfer result cannot falsify shared continuous structure because no DB rollout or DB cross-matrix enrichment was permitted.\n'''
(H/'final_report.md').write_text(report)
dump('working_state.json',{'status':'complete','completed':['design','preflight','structured_matrix','journal_merge','postflight','matched_controls','toy_training','density_ablation','ranking','db_frozen_transfer','finalize'],'new_toy_continuations':new,'new_db_continuations':0,'next_action':'none'})
dump('runtime_statistics.json',{'rollout_shards':6,'new_toy_continuations':new,'rollout_wall_seconds_max':max(x['wall_seconds'] for x in runt),'finalized_unix':time.time()})
print(json.dumps({'toy':toy_class,'db':dbclass,'structured_D':st,'sparse_D':sp,'ranking':rank,'ablation':abl,'db_results':db},indent=2))
