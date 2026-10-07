"""Frozen interaction models on the unchanged Ring true-t0 K16 proposals.

No rollout or eta generation. The original frozen critic is re-scored to
verify exact conditioning/preprocessing reproduction before new comparisons.
"""

import csv
import json
from collections import defaultdict

import flax.serialization as serialization
import jax
import jax.numpy as jnp
import numpy as np

from offline_baselines import ROOT, OUT, K16, dump_json
from train_interaction import InteractionCritic, prepare, SPLIT
from diagnostics.orthoflow3_generator_critic_frozen_test_v1.run_frozen_test import FrozenNewRuntime
from diagnostics.orthoflow3_generator_critic_v1 import train_evaluate as original
from diagnostics.orthoflow3_ring_revision_v1 import run_fresh as revision


def main():
    manifest=json.loads((ROOT/'diagnostics/orthoflow3_ring_revision_v1/fresh_test_manifests.json').read_text())['states']
    frozen=json.loads((K16/'frozen_proposals.json').read_text())['states']
    outcome={r['state_uid']:r for r in json.loads((K16/'per_state_results.json').read_text())}
    fstates={r['state_uid']:r for r in manifest if r['scenario']=='ring_exchange'}
    assert len(fstates)==len(frozen)==60
    # Runtime construction registers only already-known experiment identities;
    # no rollout is invoked. Conditioning uses the exact frozen Flow key.
    runtime=FrozenNewRuntime('ring_exchange')
    norm=json.loads((ROOT/'diagnostics/orthoflow3_generator_critic_v1/normalization.json').read_text())
    gm,gp,oldmodel,oldparams=revision.load_models(norm)
    n=norm['scenarios']['ring_exchange']
    data=prepare()
    hmean,hstd=data['ring_exchange']['h_mean'],data['ring_exchange']['h_std']
    model=InteractionCritic()
    dims={s:len(data[s]['h_mean']) for s in data}
    init=model.init(jax.random.PRNGKey(0),jnp.zeros((1,dims['four_way_intersection'])),jnp.zeros((1,3)),
                    jnp.zeros((1,dims['ring_exchange'])),jnp.zeros((1,3)))
    variants=('ring_full','joint_full','ring_25_scratch','ring_25_transfer_fixed','ring_25_finetune')
    params={}
    for seed in (17,23,41):
        for run in variants:
            path=OUT/'interaction_models'/f'seed{seed}'/run/'checkpoint.msgpack'
            params[(seed,run)]=serialization.from_bytes(init,path.read_bytes())
    rows=[];max_old_score_diff=0.0
    for p in frozen:
        sid=p['state_uid'];s=fstates[sid]
        state={'uid':sid,'alias':s['state_id'],'physical':s['physical'],'index':int(s['state_id'].split('_')[-1])}
        flat,env=runtime.conditioning(state)
        h=np.asarray(flat,np.float32)
        assert h.shape==(100,)
        h_old=(h-np.asarray(n['h_mean'],np.float32))/np.asarray(n['h_std'],np.float32)
        c=revision.context_vector(env,norm,'ring_exchange')
        xyz=np.asarray([p['mean']]+p['samples'],np.float32)
        assert xyz.shape==(17,3)
        old_logits=np.asarray(oldmodel.apply(oldparams,jnp.asarray(np.repeat(h_old[None],17,axis=0)),
                                             jnp.asarray(np.repeat(c[None],17,axis=0)),
                                             jnp.asarray((xyz-original.CENTER)/original.RADIUS),method=oldmodel.ring))
        old_scores=1/(1+np.exp(-old_logits))
        err=float(np.max(np.abs(old_scores-np.asarray(p['critic_scores']))))
        max_old_score_diff=max(max_old_score_diff,err)
        assert err<1e-4,(sid,err)
        h_new=(h-hmean)/hstd
        eta=(xyz-np.asarray(SPLIT['eta_coord_center']))/np.asarray(SPLIT['eta_coord_radius'])
        true=np.asarray([outcome[sid]['candidate_evidence']['mean']['Q16']]+[
            outcome[sid]['candidate_evidence'][f'sample_{i}']['Q16'] for i in range(1,17)])
        old_idx=int(p['critic_index'])
        for seed in (17,23,41):
            for run in variants:
                logits=np.asarray(model.apply(params[(seed,run)],jnp.asarray(np.repeat(h_new[None],17,axis=0)),
                                              jnp.asarray(eta),method=model.ring))
                score=1/(1+np.exp(-logits));idx=int(np.argmax(score))
                rows.append({'seed':seed,'run':run,'state_uid':sid,'selected_index':idx,
                             'selected_score':float(score[idx]),'selected_q16':float(true[idx]),
                             'selected_b15':bool(true[idx]>=15/16),'old_b15':bool(true[old_idx]>=15/16),
                             'oracle_b15':bool(np.any(true>=15/16))})
    with (OUT/'ring_k16_interaction_models.csv').open('w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={'original_critic_score_reproduction_max_abs_diff':max_old_score_diff,
             'training': 'v2-audited source-family and spatial-eta train only; corrected Ring safety',
             'test': 'same previously examined 60-state true-t0 K16 proposal pool; diagnostic, not independent final confirmation',
             'new_rollout':0,'models':{}}
    for seed in (17,23,41):
        for run in variants:
            rr=[r for r in rows if r['seed']==seed and r['run']==run]
            summary['models'][f'{run}_seed{seed}']={'selected_b15':sum(r['selected_b15'] for r in rr),
                'mean_q16':float(np.mean([r['selected_q16'] for r in rr])),
                'rescue_vs_original':sum(r['selected_b15'] and not r['old_b15'] for r in rr),
                'break_vs_original':sum(r['old_b15'] and not r['selected_b15'] for r in rr)}
    dump_json('ring_k16_interaction_summary.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
