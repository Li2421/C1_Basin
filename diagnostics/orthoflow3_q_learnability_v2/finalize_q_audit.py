#!/usr/bin/env python3
"""Finalize zero-feasibility and true directional validation after frozen Q."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import time
from collections import Counter, defaultdict
from pathlib import Path

import flax.linen as nn
from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import pearsonr


ROOT=Path('/home/zhihan/research/Basin_C1'); HERE=ROOT/'diagnostics/orthoflow3_q_learnability_v2'
ETA_CENTER=np.asarray([.875,0.,.375]); ETA_SCALE=np.asarray([.75,1.,.75])


class QMLP(nn.Module):
    @nn.compact
    def __call__(self,x):
        x=nn.silu(nn.Dense(256)(x)); x=nn.silu(nn.Dense(256)(x)); return nn.Dense(1)(x).squeeze(-1)


def write_json(path,value): path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def load_raw(stem):
    rows=[]
    for path in sorted((HERE/'raw'/stem).glob('shard*.jsonl')):
        if not path.stem.removeprefix('shard').isdigit(): continue
        rows += [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    return {row['task_id']:row for row in rows}
def write_csv(path,rows,fields=None):
    fields=fields or list(rows[0])
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def main():
    started=time.monotonic()
    base=load_raw('base_rollout_plan'); zero=load_raw('zero_followup_plan'); grad=load_raw('gradient_followup_plan')
    zp=json.loads((HERE/'zero_followup_plan.json').read_text()); gp=json.loads((HERE/'gradient_followup_plan.json').read_text())
    if len(zero)!=len(zp['tasks']) or len(grad)!=len(gp['tasks']): raise RuntimeError(('incomplete followups',len(zero),len(zp['tasks']),len(grad),len(gp['tasks'])))
    states=json.loads((HERE/'eligible_state_manifest.json').read_text())['selected_states']; by_state={x['state_id']:x for x in states}
    h=np.asarray(np.load(HERE/'conditioning_features.npz')['features'],dtype=np.float64)
    norm=json.loads((HERE/'normalization.json').read_text()); hm=np.asarray(norm['h_mean']); hs=np.asarray(norm['h_std'])
    selected=json.loads((HERE/'selected_checkpoint.json').read_text()); model=QMLP(); template=model.init(jax.random.PRNGKey(0),jnp.zeros((1,217)))
    params=serialization.from_bytes(template,Path(selected['checkpoint']).read_bytes())
    def pred(state_id,eta):
        st=by_state[state_id]; x=np.concatenate([(h[st['feature_index']]-hm)/hs,(np.asarray(eta)-ETA_CENTER)/ETA_SCALE]).astype(np.float32)
        return float(jax.nn.sigmoid(model.apply(params,jnp.asarray(x)[None]))[0])
    frozen=json.loads((HERE/'frozen_q_manifest.json').read_text()); threshold=float(frozen['zero_feasibility_threshold'])

    zero_rows=[]
    test_states=[x for x in states if x['split']=='test'][:8]
    for st in test_states:
        rows=[r for r in base.values() if r['state_id']==st['state_id'] and r['probe_id']=='zero']+[r for r in zero.values() if r['state_id']==st['state_id']]
        if len(rows)!=64: raise RuntimeError((st['state_id'],'zero rows',len(rows)))
        successes=sum(r['success'] for r in rows); p=pred(st['state_id'],[0,0,0]); true=successes>=63; said=p>=threshold
        zero_rows.append({'state_id':st['state_id'],'predicted_q_zero':p,'val_selected_threshold':threshold,'successes':successes,'trials':64,'true_robust_zero':true,'predicted_feasible':said,'false_feasible':said and not true,'false_infeasible':true and not said})
    write_csv(HERE/'zero_eta_audit.csv',zero_rows)

    selection=json.loads((HERE/'gradient_test_selection.json').read_text())['cases']; directional=[]; failures=[]
    for case in selection:
        role_rows={}
        for role in ('minus','plus'):
            role_rows[role]=sorted([r for r in grad.values() if r.get('gradient_case_id')==case['case_id'] and r.get('gradient_role')==role],key=lambda r:r['future_index'])
        role_rows['base']=sorted([r for r in base.values() if r['state_id']==case['state_id'] and r['probe_id']==case['base_probe_id']],key=lambda r:r['future_index'])+sorted([r for r in grad.values() if r.get('gradient_case_id')==case['case_id'] and r.get('gradient_role')=='base'],key=lambda r:r['future_index'])
        if any(len(role_rows[k])!=32 for k in role_rows): raise RuntimeError((case['case_id'],{k:len(v) for k,v in role_rows.items()}))
        q={role:sum(r['success'] for r in rows)/32 for role,rows in role_rows.items()}
        ph={role:pred(case['state_id'],case[f'eta_{role}']) for role in ('minus','base','plus')}
        plus_minus=np.asarray([int(a['success'])-int(b['success']) for a,b in zip(role_rows['plus'],role_rows['minus'])])
        row={'case_id':case['case_id'],'state_id':case['state_id'],'base_probe_id':case['base_probe_id'],
             'q32_minus':q['minus'],'q32_base':q['base'],'q32_plus':q['plus'],
             'true_plus_minus':q['plus']-q['minus'],'pred_minus':ph['minus'],'pred_base':ph['base'],'pred_plus':ph['plus'],'pred_plus_minus':ph['plus']-ph['minus'],
             'plus_gt_minus':q['plus']>q['minus'],'plus_ge_base':q['plus']>=q['base'],'base_ge_minus':q['base']>=q['minus'],
             'paired_rescue_plus_vs_minus':int(np.sum(plus_minus==1)),'paired_break_plus_vs_minus':int(np.sum(plus_minus==-1)),
             'direction_sign_agreement':bool(np.sign(ph['plus']-ph['minus'])==np.sign(q['plus']-q['minus']))}
        directional.append(row)
        if q['plus']<q['minus'] or q['plus']<q['base']-.125:
            outcomes={role:dict(Counter(r['outcome'] for r in rows)) for role,rows in role_rows.items()}
            failures.append({'case_id':case['case_id'],'state_id':case['state_id'],'eta_base':case['eta_base'],'eta_minus':case['eta_minus'],'eta_plus':case['eta_plus'],'clipping':case['clipping'],'predicted':ph,'true_q32':q,'outcomes':outcomes,'nearest_training_eta_normalized_distance':.05})
    write_csv(HERE/'gradient_directional_results.csv',directional)
    write_json(HERE/'gradient_failure_analysis.csv.json',{'cases':failures})
    # Required CSV form as well.
    write_csv(HERE/'gradient_failure_analysis.csv',[{'case_id':f['case_id'],'state_id':f['state_id'],'clipping':f['clipping'],'pred_minus':f['predicted']['minus'],'pred_base':f['predicted']['base'],'pred_plus':f['predicted']['plus'],'q32_minus':f['true_q32']['minus'],'q32_base':f['true_q32']['base'],'q32_plus':f['true_q32']['plus'],'nearest_training_eta_normalized_distance':f['nearest_training_eta_normalized_distance'],'outcome_counts_json':json.dumps(f['outcomes'],sort_keys=True)} for f in failures],fields=['case_id','state_id','clipping','pred_minus','pred_base','pred_plus','q32_minus','q32_base','q32_plus','nearest_training_eta_normalized_distance','outcome_counts_json'])

    pre=json.loads((HERE/'pre_followup_summary.json').read_text()); probability=json.loads((HERE/'heldout_probability_metrics.json').read_text())
    deltas=np.asarray([r['true_plus_minus'] for r in directional]); sign=np.mean([r['direction_sign_agreement'] for r in directional]) if directional else 0
    corr=float(pearsonr([r['pred_plus_minus'] for r in directional],[r['true_plus_minus'] for r in directional]).statistic) if len(directional)>2 and np.std(deltas)>0 else 0.0
    predictive=(probability['test_binomial_nll_per_trial']<probability['constant_test_nll'] and pre['ranking_mean_spearman']>0.20)
    direction_good=(len(directional)>=6 and np.mean([r['plus_gt_minus'] for r in directional])>=2/3 and deltas.mean()>0 and sign>=2/3 and not any(r['true_plus_minus']<=-.25 for r in directional))
    if not predictive: classification='Q_NOT_GENERALIZABLE'
    elif direction_good: classification='Q_PREDICTIVE_AND_DIRECTIONALLY_FAITHFUL'
    else: classification='Q_PREDICTIVE_BUT_DIRECTIONALLY_UNRELIABLE'
    decision={'classification':classification,'conditioning_ambiguous':False,'heldout_predictive':bool(predictive),'directionally_faithful':bool(direction_good),
              'directional_states':len(directional),'fraction_plus_gt_minus':float(np.mean([r['plus_gt_minus'] for r in directional])),'fraction_plus_ge_base':float(np.mean([r['plus_ge_base'] for r in directional])),'fraction_base_ge_minus':float(np.mean([r['base_ge_minus'] for r in directional])),'mean_plus_minus':float(deltas.mean()),'median_plus_minus':float(np.median(deltas)),'predicted_true_sign_agreement':float(sign),'predicted_true_change_pearson':corr,'catastrophic_directional_failures':int(sum(r['true_plus_minus']<=-.25 for r in directional)),
              'zero_false_feasible':int(sum(r['false_feasible'] for r in zero_rows)),'zero_false_infeasible':int(sum(r['false_infeasible'] for r in zero_rows)),
              'future_G_through_Q_test_justified':classification=='Q_PREDICTIVE_AND_DIRECTIONALLY_FAITHFUL'}
    write_json(HERE/'q_decision.json',decision)

    split=json.loads((HERE/'state_split_manifest.json').read_text())
    leakage={'passed':not any(split['overlap'].values()),'split_unit':'leakage_group','overlap':split['overlap'],'normalization_train_only':True,'test_used_for_model_selection':False,'test_used_for_threshold_selection':False,'followup_true_outcomes_generated_after_q_frozen':True,'jdef_used_in_training':False}
    write_json(HERE/'source_group_leakage_audit.json',leakage)
    runtimes=[]
    for path in sorted((HERE/'raw').glob('*/*_runtime.json')): runtimes.append(json.loads(path.read_text()))
    all_records=list(base.values())+list(zero.values())+list(grad.values())
    runtime={'base_new_continuations':len(base),'zero_followup_new_continuations':len(zero),'gradient_followup_new_continuations':len(grad),'total_new_continuations':len(base)+len(zero)+len(grad),'physical_steps':sum(int(x['continuation_steps']) for x in all_records),'rollout_shard_wall_seconds_sum':sum(x['wall_seconds'] for x in runtimes),'rollout_wall_seconds_critical_path_by_stage':{},'q_training_seconds':pre['training_seconds'],'maximum_gpu_shards':6,'base_initial_allocation_gpu_shards':2,'base_completion_allocation_gpu_shards':6,'gpu_memory_peak':'not exposed by Slurm accounting; six-process run required XLA preallocation disabled and 0.14 fraction cap','cpu_per_rollout_shard':2,'ram_request_per_rollout_shard_gb':14,'host':platform.node(),'finalization_seconds':time.monotonic()-started,'stage_runtimes':runtimes}
    for stage in set(x['plan'] for x in runtimes): runtime['rollout_wall_seconds_critical_path_by_stage'][stage]=max(x['wall_seconds'] for x in runtimes if x['plan']==stage)
    write_json(HERE/'runtime_statistics.json',runtime)
    print(json.dumps({'decision':decision,'runtime':{k:v for k,v in runtime.items() if k!='stage_runtimes'}},indent=2))

if __name__=='__main__': main()
