"""Strong continuous eta-only Toy control on W1 structured+wide TRAIN pairs.

VAL selects checkpoint/seed. Frozen Toy-200 K16 outcomes are read only after
training. The model sees empirical Q targets but never state h.
"""

import argparse
import csv
import json

import flax.linen as nn
import flax.serialization as serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pyarrow.parquet as pq

from offline_baselines import ROOT, OUT, dump_json


STRUCT=ROOT/'diagnostics/orthoflow3_structured_continuous_q_data_v1'
SWEEP=ROOT/'diagnostics/orthoflow3_mode_free_k_sweep_latency_v1'
OLD=ROOT/'diagnostics/orthoflow3_mode_free_generator_critic_hard_cohort_v1'
MANIFEST=ROOT/'diagnostics/orthoflow3_continuous_basin_critic_v1/dataset_manifest.json'


class EtaOnly(nn.Module):
    @nn.compact
    def __call__(self,eta):
        x=nn.silu(nn.Dense(64)(eta))
        x=nn.silu(nn.Dense(64)(x))
        return nn.Dense(1)(x)[...,0]


def prepare():
    sr=pq.read_table(STRUCT/'structured_pair_table.parquet').to_pylist()
    train=[r for r in sr if r['matrix_partition']=='TRAIN_TRAIN']
    val=[r for r in sr if r['matrix_partition']=='VAL_VAL']
    test=[r for r in sr if r['matrix_partition']=='TESTSTATE_TESTETA']
    wide=pq.read_table(STRUCT/'sparse_matched_control.parquet').to_pylist()
    seen={(r['state_uid'],r['eta_uid']) for r in train}
    wide=[r for r in wide if (r['state_uid'],r['eta_uid']) not in seen]
    norm=json.loads(MANIFEST.read_text())['eta_normalization']
    center=np.asarray(norm['center'],np.float32);scale=np.asarray(norm['scale'],np.float32)
    def convert(rows,is_wide=False):
        eta=np.asarray([[r['eta1'],r['eta2'],r['eta3']] if is_wide else r['eta'] for r in rows],np.float32)
        q=np.asarray([r['empirical_q'] for r in rows],np.float32)
        return (eta-center)/scale,q
    xtr,ytr=convert(train)
    xwide,ywide=convert(wide,True)
    xv,yv=convert(val);xt,yt=convert(test)
    xtr=np.concatenate([xtr,xwide]);ytr=np.concatenate([ytr,ywide])
    assert len(xtr)==2876 and len(xv)>0 and len(xt)==512
    return {'train':(xtr,ytr),'val':(xv,yv),'test':(xt,yt),'center':center,'scale':scale,
            'train_states':len({r['state_uid'] for r in train+wide}),
            'train_eta':len({r['eta_uid'] for r in train+wide})}


def metrics(model,params,part):
    x,q=part
    z=np.asarray(model.apply(params,jnp.asarray(x)))
    p=np.clip(1/(1+np.exp(-z)),1e-6,1-1e-6)
    return {'pairs':len(q),'nll':float(np.mean(-q*np.log(p)-(1-q)*np.log(1-p))),
            'mae':float(np.mean(abs(q-p))),'brier':float(np.mean((q-p)**2))}


def fit(seed):
    data=prepare();model=EtaOnly();params=model.init(jax.random.PRNGKey(seed),jnp.zeros((1,3)))
    opt=optax.adamw(1e-3,weight_decay=1e-4);state=opt.init(params);rng=np.random.default_rng(seed+28981)
    @jax.jit
    def step(p,st,x,q):
        def loss(pp):return jnp.mean(optax.sigmoid_binary_cross_entropy(model.apply(pp,x),q))
        value,g=jax.value_and_grad(loss)(p)
        updates,st=opt.update(g,st,p)
        return optax.apply_updates(p,updates),st,value
    history=[];best=(float('inf'),None,0);stale=0
    for iteration in range(1,4001):
        ix=rng.integers(0,len(data['train'][1]),size=256)
        params,state,loss=step(params,state,jnp.asarray(data['train'][0][ix]),jnp.asarray(data['train'][1][ix]))
        if iteration%50:continue
        val=metrics(model,params,data['val'])
        history.append({'step':iteration,'train_loss':float(loss),**val})
        if val['nll']<best[0]-1e-5:
            best=(val['nll'],serialization.to_bytes(params),iteration);stale=0
        else:stale+=1
        if stale>=12 and iteration>=800:break
    params=serialization.from_bytes(params,best[1])
    folder=OUT/'toy_eta_only_mlp'/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    (folder/'checkpoint.msgpack').write_bytes(best[1])
    (folder/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    summary={'seed':seed,'train_pairs':len(data['train'][1]),'train_states':data['train_states'],
             'train_eta':data['train_eta'],'best_step':best[2],
             'val_metrics':metrics(model,params,data['val']),
             'frozen_simultaneous_holdout':metrics(model,params,data['test']),
             'checkpoint':str(folder/'checkpoint.msgpack'),'new_rollout':0}
    (folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def evaluate_hard200():
    data=prepare()
    fitted=[json.loads((OUT/'toy_eta_only_mlp'/f'seed{s}'/'summary.json').read_text()) for s in (17,23,41)]
    selected=min(fitted,key=lambda r:(r['val_metrics']['nll'],r['seed']))['seed']
    model=EtaOnly()
    init=model.init(jax.random.PRNGKey(0),jnp.zeros((1,3)))
    params={s:serialization.from_bytes(init,(OUT/'toy_eta_only_mlp'/f'seed{s}'/'checkpoint.msgpack').read_bytes()) for s in (17,23,41)}
    old_prop=json.loads((OLD/'frozen_proposals.json').read_text())['states']
    new_prop=json.loads((SWEEP/'frozen_proposals.json').read_text())['states']
    old_result=list(csv.DictReader((OLD/'per_state_results.csv').open()))
    raw={}
    for path in sorted((SWEEP/'raw').glob('shard*.jsonl')):
        for line in path.read_text().splitlines():
            if line.strip():
                r=json.loads(line);raw[(r['episode_index'],r['kind'],r['future_index'])]=r
    assert len(old_prop)==len(new_prop)==len(old_result)==200 and len(raw)==38400
    rows=[]
    for i in range(200):
        assert old_prop[i]['episode_index']==new_prop[i]['episode_index']==i
        xyz=np.asarray([old_prop[i]['eta'][f'sample_{j}'] for j in range(4)]+[
                        new_prop[i]['eta'][f'sample_{j}'] for j in range(4,16)])
        z=(xyz-data['center'])/data['scale']
        q=np.empty(16)
        for j in range(4):q[j]=float(old_result[i][f'Q16_sample_{j}'])
        for j in range(4,16):
            rr=[raw[(i,f'sample_{j}',f)] for f in range(16)]
            q[j]=sum(r['success'] for r in rr if r['scientific_outcome_valid'])/16
        original_idx=int(np.argmax(new_prop[i]['all_critic_scores']))
        logit_by_seed={seed:np.asarray(model.apply(params[seed],jnp.asarray(z))) for seed in (17,23,41)}
        logit_by_seed['ensemble']=np.mean(np.stack(list(logit_by_seed.values())),axis=0)
        for seed,logit in logit_by_seed.items():
            score=1/(1+np.exp(-logit));idx=int(np.argmax(score))
            rows.append({'seed':seed,'episode_index':i,'state_source_group':new_prop[i]['source_group'],
                         'selected_index':idx,'selected_score':float(score[idx]),
                         'selected_q16':float(q[idx]),'selected_b15':bool(q[idx]>=15/16),
                         'original_index':original_idx,'original_q16':float(q[original_idx]),
                         'original_b15':bool(q[original_idx]>=15/16),
                         'oracle_b15':bool(np.any(q>=15/16)),
                         'eta_nearest_train_distance':float(np.min(np.linalg.norm(data['train'][0]-z[idx],axis=1)))})
    with (OUT/'toy_hard200_eta_only.csv').open('w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={}
    for seed in (17,23,41,'ensemble'):
        rr=[r for r in rows if r['seed']==seed]
        summary[str(seed)]={'selected_b15':sum(r['selected_b15'] for r in rr),
                            'mean_q16':float(np.mean([r['selected_q16'] for r in rr])),
                            'oracle_b15':sum(r['oracle_b15'] for r in rr),
                            'original_b15':sum(r['original_b15'] for r in rr),
                            'rescue_vs_original':sum(r['selected_b15'] and not r['original_b15'] for r in rr),
                            'break_vs_original':sum(r['original_b15'] and not r['selected_b15'] for r in rr),
                            'median_selected_eta_nearest_train_distance':float(np.median([r['eta_nearest_train_distance'] for r in rr]))}
    dump_json('toy_hard200_eta_only_summary.json',{'seeds':summary,'val_selected_seed':selected,
              'val_selected_result':summary[str(selected)],'original_critic_historical':181,
              'oracle_historical':194,'status':'previously examined hard200 diagnostic, not new independent confirmation',
              'new_rollout':0})
    return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,choices=(17,23,41));parser.add_argument('--evaluate',action='store_true');args=parser.parse_args()
    if args.evaluate:
        print(json.dumps(evaluate_hard200(),indent=2))
    else:
        assert args.seed is not None
        print(json.dumps(fit(args.seed),indent=2))


if __name__=='__main__':main()
