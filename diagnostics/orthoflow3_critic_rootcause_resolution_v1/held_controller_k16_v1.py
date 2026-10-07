"""Prospective held-controller + held-family + continuous-eta benchmark.

Public candidate construction uses source TRAIN evidence/geometry, no models.
Full K16 and predeclared seen/unseen eta subsets are all reported.
"""
import argparse, collections, copy, hashlib, json, os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
from pathlib import Path
import numpy as np
from .db_transfer_data import ROOT, BASE, OUT as DATA, read, write, sha
OUT=ROOT/'held_controller_k16_v1'
SEEDS=(88138,88139)
FAMILY_ROOTS=(940151000,940161000)


def prepare_rule():
    from scipy.spatial import ConvexHull, distance
    from scipy.stats import qmc
    from shared_rollout_db.src.rollout_db import eta_identity
    assert not (OUT/'protocol.json').exists()
    rows=[r for r in read(DATA/'pairs.json') if r['split']=='train']
    byeta=collections.defaultdict(list)
    for r in rows:
        if r['scenario']=='ring_exchange':byeta[r['eta_uid']].append(r)
    old=read(ROOT/'controller_function_support/pairs.json')[:2]
    chosen=[dict(eta_uid=r['eta_uid'],eta=r['eta'],source='previous_fixed_public_two',seen=True) for r in old]
    eligible=[]
    for uid,a in byeta.items():
        # Observed standard-seed classifications, never fabricated complete Q16.
        known=[r for r in a if r['B15'] or r['nonB15']]
        states=len({r['state_uid'] for r in known})
        if states<24:continue
        prevalence=sum(r['B15'] for r in known)/len(known)
        eligible.append(dict(eta_uid=uid,eta=a[0]['eta'],states=states,known_pairs=len(known),B15_prevalence=prevalence))
    robust=max(eligible,key=lambda r:(r['B15_prevalence'],r['states'],r['eta_uid']))
    if robust['eta_uid'] not in {x['eta_uid'] for x in chosen}:
        chosen.append(dict(eta_uid=robust['eta_uid'],eta=robust['eta'],source='source_TRAIN_strong_common_eta',seen=True))
    support=np.array([a[0]['eta'] for a in byeta.values()]);lo=support.min(0);scale=np.maximum(np.ptp(support,axis=0),.1)
    transition=[r for r in eligible if .15<=r['B15_prevalence']<=.85]
    while len(chosen)<8:
        pool=[r for r in transition if r['eta_uid'] not in {x['eta_uid'] for x in chosen}]
        assert pool,'Insufficient source transition support; do not silently relax rule'
        oldx=np.array([r['eta'] for r in chosen])
        best=max(pool,key=lambda r:(float(np.min(np.linalg.norm((oldx-r['eta'])/scale,axis=1))),r['eta_uid']))
        chosen.append(dict(eta_uid=best['eta_uid'],eta=best['eta'],source='source_TRAIN_state_variable_space_filling',seen=True))
    known=np.array([r['eta'] for r in chosen]);hull=ConvexHull((known-lo)/scale)
    candidates=known.min(0)+qmc.Sobol(3,scramble=True,seed=2026100561).random_base2(13)*np.ptp(known,axis=0)
    z=(candidates-lo)/scale
    interior=np.max(z@hull.equations[:,:3].T+hull.equations[:,3],axis=1)<-1e-5
    candidates=candidates[interior]
    alltrain=np.unique(np.array([r['eta'] for r in rows]),axis=0)
    dist=distance.cdist((candidates-lo)/scale,(alltrain-lo)/scale).min(1)
    candidates=candidates[dist>=.05];assert len(candidates)>=8
    for _ in range(8):
        used=np.array([r['eta'] for r in chosen]);dd=distance.cdist(candidates/scale,used/scale).min(1)
        j=int(np.argmax(dd));eta=candidates[j].tolist();uid=eta_identity(eta)[0]
        chosen.append(dict(eta_uid=uid,eta=eta,source='source_convex_interior_Sobol_unseen_eta',seen=False));candidates=np.delete(candidates,j,axis=0)
    assert len({r['eta_uid'] for r in chosen})==16
    for j,r in enumerate(chosen):
        r['index']=j;r['nearest_TRAIN_eta_distance']=float(np.linalg.norm((alltrain-r['eta'])/scale,axis=1).min())
    write(OUT/'candidate_panel.json',chosen)
    write(OUT/'candidate_audit.json',dict(source_TRAIN_rows_sha256=sha(DATA/'pairs.json'),normalization_origin=lo.tolist(),normalization_scale=scale.tolist(),eligible_source_statistics=eligible,source_common_eta=robust['eta_uid'],unseen_eta_distances=[r['nearest_TRAIN_eta_distance'] for r in chosen if not r['seen']],source_targets_used=False,model_predictions_used=False))
    write(OUT/'protocol.json',dict(controller_seeds=SEEDS,family_roots=FAMILY_ROOTS,families_per_controller=32,first_stage_families_per_controller=16,K=16,standard_seeds=16,new_rollout_upper_bound=16384,batch_upper_bound=4096,flow_recipe='Unchanged native Ring expert-only base_u_v10_local_dataset,100000steps,batch256,DEVcheckpoint; both predetermined seeds retained regardless of outcomes',source_models='source_controller_cv_v2/final_models, plus frozen db_transfer_v1 controls and original Ring-only frozen references where compatible',model_selection='Source-only controller-held-out CV; source-family selected models secondary, all target labels unopened',candidate_construction='2 old public eta + source TRAIN strongest common eta +5 spatially separated source TRAIN state-variable eta +8 Sobol convex-interior eta separated >=.05 from every source TRAIN eta',candidates_sha256=sha(OUT/'candidate_panel.json'),generation_rule='ring_exchange.environment.sample_instance(root+j,test,nativeConfig);true-t0;nooutcomefiltering',scope='Unseen native future Flow function and source family,16publiccandidatesincluding8unseeneta; NOT cross-scene or generator coverage',partial_stage_rule='Freeze all64families before outcomes. First16families/controller. Extend remaining16/controller only if pooled fixed-panel best-global-eta leaves >=4 oracle-B15 states unresolved AND at least8 states have mixed B15/nonB15 candidates. Rule never sees model scores; all stage outcomes retained.',selection_eligibility='At leastoneB15candidate;coveragefailuresandall-B15underdiscriminativecasesreportednotremoved',numeric_policy='Unknown unresolved,not failure/success;no retry',generator_changed=False,success_safety_horizon_changed=False,feature_design_target_used=False,code_sha256=sha(__file__)))
    print(dict(K=16,seen_eta=8,unseen_eta=8,minimum_unseen_separation=min(r['nearest_TRAIN_eta_distance'] for r in chosen if not r['seen']),rollout_upper_bound=16384,rollouts_launched=0),flush=True)


def flow(index):
    import jax
    from new_benchmark_common.training import train_stage1
    assert jax.default_backend()=='gpu'
    rule=read(OUT/'protocol.json');seed=SEEDS[index];dest=OUT/f'flow_seed{seed}'
    assert rule['code_sha256']==sha(__file__) and not (dest/'best.pkl').exists()
    result=train_stage1(BASE/'diagnostics/ring_exchange_stage1/base_u_v10_local_dataset',dest,seed=seed,steps=100000,batch_size=256,log_interval=250,validation_batches=8,early_transition_fraction=.5,early_steps=15,early_nominal_only=False,source_balanced_sampling=False)
    write(dest/'frozen_controller.json',dict(seed=seed,checkpoint_sha256=sha(dest/'best.pkl'),training_summary=result,protocol_sha256=sha(OUT/'protocol.json'),critic_target_labels_used=False,critic_model_selection_used=False,controller_tournament=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare_rule','flow'));p.add_argument('--index',type=int,default=0);a=p.parse_args()
    flow(a.index) if a.action=='flow' else prepare_rule()
