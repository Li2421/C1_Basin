"""Matched-seed evidence summary; no selection based on rollout outcomes."""
import json
import numpy as np
import jax
from . import build,train
from diagnostics.orthoflow3_closed_loop_contract_v1 import repair
from diagnostics.orthoflow3_k16_critic_canonicalization_v1 import phase_a as a,evaluate_b as ev

def run():
    oldgen={r['seed']:r for r in a.load(repair.OUT/'generator/training.json')['results']}
    rr=[]
    for seed in train.SEEDS:
        newgen=a.load(build.OUT/f'generator/seed{seed}/training.json')
        oldc=a.load(repair.OUT/f'critic/seed{seed}/training.json');newc=a.load(build.OUT/f'critic/seed{seed}/training.json')
        om=a.load(repair.OUT/f'critic/seed{seed}/metrics.json')['summary'];nm=a.load(build.OUT/f'critic/seed{seed}/metrics.json')['summary']
        rr.append({'seed':seed,'old_generator':oldgen[seed],'unified_generator':newgen,
            'old_critic':oldc,'unified_critic':newc,'same_scientific_labels':True,
            'old_fixed_proposal_metrics':om,'unified_fixed_proposal_metrics':nm})
    gm,gp,cm,cp=train.load_models();repair.install();og,ogp,oc,ocp=ev.load_models()
    count=lambda p:sum(x.size for x in jax.tree_util.tree_leaves(p))
    value={'matched_seeds':rr,'parameter_counts':{'old_generator':count(ogp),'unified_generator':count(gp),
             'old_critic':count(ocp),'unified_critic':count(cp)},
        'data_equality':'same v2 robust labels; same Phase-A aligned full-Q16 evidence; same initial expansion states',
        'training_budgets':'same max updates, optimizer, loss, scenario/state sampling and validation stopping rules',
        'capacity_note':'Entity encoder changes representation and parameterization; not a parameter-count-matched causal isolation of capacity.',
        'closed_loop':'selected frozen checkpoints evaluated separately; not a two-seed confidence interval on rollout success'}
    build.dump('old_vs_unified_ablation.json',value)
    p=build.OUT/'development/proposals.json'
    if p.exists():
        metrics=[]
        for row in a.load(p)['states']:
            for mode,props in [('old',row['old']),('unified',row)]:
                e=np.asarray(props['etas'])[1:];z=(e-a.learn.CENTER)/a.learn.RADIUS
                dist=np.linalg.norm(z[:,None]-z[None,:],axis=-1)
                metrics.append({'state_uid':row['state_uid'],'scenario':row['scenario'],'representation':mode,
                    'unique_stochastic_proposals':len(np.unique(e,axis=0)),
                    'mean_pairwise_distance':float(dist[np.triu_indices(16,1)].mean()),
                    'normalized_covariance':np.cov(z.T).tolist(),'boundary_fraction':float(np.mean(np.abs(z)>.99))})
        build.dump('proposal_diversity.json',metrics)
    return value['parameter_counts']

if __name__=='__main__':print(json.dumps(run(),indent=2))
