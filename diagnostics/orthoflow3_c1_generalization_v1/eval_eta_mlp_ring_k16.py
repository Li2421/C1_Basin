"""All three pre-frozen spatial eta-only models on the identical Ring K16 pool."""

import csv
import json

import flax.serialization as serialization
import jax.numpy as jnp
import numpy as np

from offline_baselines import K16, OUT, dump_json
from train_eta_only_mlp import EtaOnly
from train_interaction import SPLIT


def main():
    proposals=json.loads((K16/'frozen_proposals.json').read_text())['states']
    results={r['state_uid']:r for r in json.loads((K16/'per_state_results.json').read_text())}
    model=EtaOnly()
    init=model.init(jnp.asarray([17,0],dtype=jnp.uint32),jnp.zeros((1,3)))
    center=np.asarray(SPLIT['eta_coord_center'])
    radius=np.asarray(SPLIT['eta_coord_radius'])
    detail=[];summary={}
    for source in ('ring_exchange','four_way_intersection'):
      for seed in (17,23,41):
        path=OUT/'eta_only_mlp'/f'seed{seed}'/source/'checkpoint.msgpack'
        params=serialization.from_bytes(init,path.read_bytes())
        scores=[]
        for p in proposals:
            sid=p['state_uid'];xyz=np.asarray([p['mean']]+p['samples']);z=(xyz-center)/radius
            logit=np.asarray(model.apply(params,jnp.asarray(z,dtype=jnp.float32)))
            score=1/(1+np.exp(-logit));idx=int(np.argmax(score))
            q=np.asarray([results[sid]['candidate_evidence']['mean']['Q16']]+[
                results[sid]['candidate_evidence'][f'sample_{i}']['Q16'] for i in range(1,17)])
            old=int(p['critic_index'])
            detail.append({'source_scenario':source,'seed':seed,'state_uid':sid,'eta_only_index':idx,'eta_only_score':float(score[idx]),
                           'eta_only_q16':float(q[idx]),'eta_only_b15':bool(q[idx]>=15/16),
                           'old_critic_index':old,'old_critic_q16':float(q[old]),'old_critic_b15':bool(q[old]>=15/16),
                           'oracle_b15':bool(np.any(q>=15/16))})
        rr=[x for x in detail if x['seed']==seed and x['source_scenario']==source]
        summary[f'{source}_seed{seed}']={'states':len(rr),'oracle_b15':sum(x['oracle_b15'] for x in rr),
                            'eta_only_b15':sum(x['eta_only_b15'] for x in rr),
                            'eta_only_mean_q16':float(np.mean([x['eta_only_q16'] for x in rr])),
                            'old_critic_b15':sum(x['old_critic_b15'] for x in rr),
                            'rescue_vs_old':sum(x['eta_only_b15'] and not x['old_critic_b15'] for x in rr),
                            'break_vs_old':sum(x['old_critic_b15'] and not x['eta_only_b15'] for x in rr)}
    with (OUT/'ring_k16_eta_only_mlp.csv').open('w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=list(detail[0]));w.writeheader();w.writerows(detail)
    selected=min((json.load(open(OUT/'eta_only_mlp'/f'seed{s}'/'ring_exchange'/'summary.json'))['dev_nll'],s)
                 for s in (17,23,41))[1]
    dump_json('ring_k16_eta_only_mlp_summary.json',{'seeds':summary,'dev_selected_seed':selected,
               'dev_selected_result':summary[f'ring_exchange_seed{selected}'],
               'training': '48 audited TRAIN source families and 161 spatial TRAIN eta groups; corrected Ring safety',
               'test': 'previously examined 60-state K16 diagnostic, not untouched final confirmation',
               'new_rollout':0})
    print(json.dumps({'seeds':summary,'dev_selected_seed':selected},indent=2))


if __name__=='__main__':main()
