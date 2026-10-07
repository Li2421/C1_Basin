"""One-time evaluation after VAL-selected checkpoints and TEST DB postflight."""
from __future__ import annotations
import csv
import hashlib
import json
import os
from datetime import datetime,timezone
from pathlib import Path
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('CUDA_VISIBLE_DEVICES','')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from scipy.special import expit

import train_val as train
from shared_rollout_db.src.rollout_db import connect
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]

def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def csvwrite(p,rs):
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def metrics(logits,rows):return train.metrics(logits,rows)
def predict_checkpoint(cp,seed,data):
    template=train.model.init(jax.random.PRNGKey(seed),jnp.zeros((1,214)),jnp.zeros((1,3)))
    params=serialization.from_bytes(template,Path(cp).read_bytes())
    return train.predict(params,data)
def knn_family(rows_train,rows_test):
    train_d=train.from_rows(rows_train);test_d=train.from_rows(rows_test)
    values=[]
    for j,r in enumerate(rows_test):
        cand=[i for i,z in enumerate(rows_train) if z['parent_episode_index']==r['parent_episode_index'] and z['kind']==r['kind']]
        assert len(cand)==3
        ix=min(cand,key=lambda i:float(np.sqrt(np.mean((train_d['h'][i]-test_d['h'][j])**2))))
        values.append({'test_parent':r['parent_episode_index'],'test_offset_m':r['offset_m'],'kind':r['kind'],
            'nearest_train_offset_m':rows_train[ix]['offset_m'],
            'nearest_h_rms':float(np.sqrt(np.mean((train_d['h'][ix]-test_d['h'][j])**2))),
            'predicted_Q16':rows_train[ix]['empirical_q'],'true_Q16':r['empirical_q']})
    return values
def main():
    if (OUT/'final_decision.json').exists():raise RuntimeError('Final TEST evaluation already completed')
    cpinfo=read(OUT/'selected_checkpoints.json')
    post=read(OUT/'test/cache_postflight.json')['summary']
    assert post['exact_reusable']==320 and post['genuinely_missing']==0
    import_time=(OUT/'selected_checkpoints.json').stat().st_mtime
    raw=list((OUT/'test/raw').glob('shard*.jsonl'))
    assert len(raw)==6 and all(import_time < p.stat().st_mtime for p in raw)
    frozen=read(OUT/'test/frozen_proposals.json')
    testrows=train.db_rows(frozen['states'],train.csvrows(OUT/'test/candidate_cache_keys.csv'))
    valrows=train.val_rows();localrows=train.local_train_rows()
    assert len(testrows)==20 and len(valrows)==20 and len(localrows)==30
    assert not ({(r['state_uid'],r['eta_uid']) for r in testrows} &
                {(r['state_uid'],r['eta_uid']) for r in valrows+localrows})
    test=train.from_rows(testrows);val=train.from_rows(valrows)
    modelruns=[];predictions=[]
    frozenpaths=[ROOT/'diagnostics/orthoflow3_nll_weighting_ablation_v1/models/secondary_combined/W1'/f'seed{s}'/'checkpoint.msgpack' for s in train.SEEDS]
    variants={'frozen_pretrained':[{'seed':s,'checkpoint':str(cp)} for s,cp in zip(train.SEEDS,frozenpaths)]}
    for v in ('control','local'):
        variants[v]=[r for r in cpinfo['selected'] if r['variant']==v]
    by_variant={}
    for variant,items in variants.items():
        test_logits=[];val_logits=[]
        for item in items:
            cp=Path(item['checkpoint'])
            if 'checkpoint_sha256' in item:
                assert hashlib.sha256(cp.read_bytes()).hexdigest()==item['checkpoint_sha256']
            tl=predict_checkpoint(cp,item['seed'],test);vl=predict_checkpoint(cp,item['seed'],val)
            test_logits.append(tl);val_logits.append(vl)
            modelruns.append({'variant':variant,'seed':item['seed'],'checkpoint':str(cp),
               **{f'TEST_{k}':v for k,v in metrics(tl,testrows).items()},
               **{f'VAL_{k}':v for k,v in metrics(vl,valrows).items()}})
        test_avg=np.mean(np.stack(test_logits),axis=0)
        val_avg=np.mean(np.stack(val_logits),axis=0)
        by_variant[variant]=test_avg
        modelruns.append({'variant':variant,'seed':'ensemble_logit_mean','checkpoint':'frozen_three_seed_mean',
           **{f'TEST_{k}':v for k,v in metrics(test_avg,testrows).items()},
           **{f'VAL_{k}':v for k,v in metrics(val_avg,valrows).items()}})
        for i,r in enumerate(testrows):
            predictions.append({'variant':variant,'parent_episode_index':r['parent_episode_index'],
                'offset_m':r['offset_m'],'kind':r['kind'],'state_uid':r['state_uid'],'eta_uid':r['eta_uid'],
                'success_count':r['success_count'],'Q16':r['empirical_q'],
                'p_hat':float(expit(test_avg[i])),'logit':float(test_avg[i])})
    csvwrite(OUT/'model_metrics.csv',modelruns)
    csvwrite(OUT/'test_pair_predictions.csv',predictions)
    local_knn=knn_family(localrows,testrows)
    csvwrite(OUT/'family_local_1nn.csv',local_knn)
    knn_q=np.asarray([r['predicted_Q16'] for r in local_knn]);actual=np.asarray([r['empirical_q'] for r in testrows])
    knn_selected=sum(actual[i+int(np.argmax(knn_q[i:i+2]))]>=15/16 for i in range(0,20,2))
    knn_ordered=sum(int(np.argmax(knn_q[i:i+2])==np.argmax(actual[i:i+2])) for i in range(0,20,2))
    metrics_table={v:metrics(logits,testrows) for v,logits in by_variant.items()}
    for m in metrics_table.values():assert m['B15_oracle']==10
    bad_idx=[i for i,r in enumerate(testrows) if r['kind']=='frozen_bad']
    bad_over={v:{'bad_eta_predicted_mean':float(np.mean(expit(z[bad_idx]))),
                 'bad_eta_true_mean':float(np.mean(actual[bad_idx])),
                 'bad_eta_predicted_at_least_0p9':int(np.sum(expit(z[bad_idx])>=.9))}
              for v,z in by_variant.items()}
    total_new=640
    decision={'experiment':'ORTHOFLOW3_TOY_CRITIC_LOCAL_SUPERVISION_V1','status':'COMPLETE',
      'goal':'local-data intervention to adjudicate support versus model fitting',
      'TEST_protocol':'five previously selected diagnostic source families; held-out +/-1.5cm perturbations; two frozen eta per state; 16 matched seeds',
      'TRAIN_local':'center and +/-1cm, 30 cached state-eta pairs; original Toy structured+wide data retained',
      'VAL':'held-out +/-0.5cm, 20 pairs; checkpoint selected on VAL NLL only',
      'TEST':'held-out +/-1.5cm, 20 pairs; started after checkpoint freeze',
      'new_continuations':total_new,'new_eta':0,'no_generator_change':True,
      'model_architecture_unchanged':True,'loss':'pure pair-equal soft-label NLL',
      'val_postflight':read(OUT/'val/cache_postflight.json')['summary'],
      'test_postflight':post,
      'TEST_ensemble_metrics':metrics_table,
      'TEST_bad_eta_bias':bad_over,
      'family_aware_local_1NN':{'B15_selected':int(knn_selected),'correct_pairwise_orderings':int(knn_ordered),
         'MAE':float(np.mean(abs(knn_q-actual))),
         'note':'diagnostic uses known source family identity and local labels; not deployable to unseen families'},
      'interpretation_rule':('If local-data model and family-aware local baseline both recover held-out ordering, '
                             'the original errors were learnable from nearby h/eta observations. If the local baseline '
                             'recovers but the unchanged neural model does not, fitting/inductive bias is implicated. '
                             'Neither outcome identifies missing physical variables or supports population generalization.'),
      'scope_limit':'New perturbations share all five TRAIN source families. This adjudicates local interpolation/extrapolation, not novel-family generalization or the other eight K16 error states.',
      'original_hard_cohort_result':'181/200 frozen critic vs 194/200 K16 oracle; no adapted model is evaluated as a new population system on those reused states'}
    local=metrics_table['local'];frozenm=metrics_table['frozen_pretrained'];control=metrics_table['control']
    if local['B15_selected']>max(frozenm['B15_selected'],control['B15_selected']) and local_knn and knn_selected>=local['B15_selected']:
        decision['root_cause_update']='LOCAL_DATA_SUPPORT_GAP_SUPPORTED; SYSTEMATIC_EXTRAPOLATION_ERROR_CORRECTABLE_IN_FIVE_FAMILIES'
    elif knn_selected>local['B15_selected']:
        decision['root_cause_update']='LOCAL_DATA_CONTAINS_SIGNAL_BUT_NEURAL_ADAPTATION_WEAK'
    else:
        decision['root_cause_update']='UNDERRESOLVED_LOCAL_INTERVENTION'
    write(OUT/'final_decision.json',decision)
    with connect() as con:
        con.execute('UPDATE experiment SET end_time=?,reused_rollout_count=?,new_rollout_count=? WHERE experiment_uid=?',
             (datetime.now(timezone.utc).isoformat(),480,total_new,read(OUT/'frozen_protocol.json')['experiment_uid']))
    write(OUT/'working_state.json',{'stage':'COMPLETE','new_rollout':640,'val_postflight_exact':320,'test_postflight_exact':320,
           'selected_checkpoints':'selected_checkpoints.json','result':'final_decision.json'})
    print(json.dumps({'TEST_ensemble_metrics':metrics_table,'family_local_1NN':decision['family_aware_local_1NN'],
                      'root_cause_update':decision['root_cause_update']},indent=2))
if __name__=='__main__':main()
