"""All-fold falsification, reusing audited analysis engines and cached labels."""
import argparse,importlib.util,json,types
import numpy as np
from .study import HERE,ROOT,DIV,MAT,FOLDS,ARMS,SEEDS,read,write,freeze,sha

ALIASES={'eta_only_same_all_labels':'eta_only','field_same_all_labels':'union_current_entity',
         'encounter_same_all_labels':'union_native_slots','encounter_flow_eta_prior':'union_slots_state_erased'}


def prepare():
    freeze(HERE/'evaluation_protocol.json',dict(
        method_protocol_sha256=sha(HERE/'protocol.json'),
        panels=['Original four-scene factorial natural40states/heldscene, K4/Q2-Q4; jointVAL8states/scene descriptive',
                'Previously opened independent32states/scene and unseenanalytic6/7; joint and correspondingheldscenefold, K16/Q16'],
        protected_Q16_split_audit=read(HERE/'opened_Q16_overlap_audit.json'),
        bootstrap='20000 physical-state bootstrap replicates; all controller/eta/seed measurements cluster with physical state',
        mismatch='200 deterministic saved random derangements plus exact all-wrong-state expectation; same scene and same controller',
        primary_questions=['actual versus state-erased TRAIN supervision','correct state versus wrong same-scene state','all fixed eta, including bestperCfixed reselected inside bootstrap'],
        reverse='Reuse symmetric discovery-only all-pair criteria;Q4/Q2 split2/2 or1/1,gap.5,min2eachdirection;Q16 split8/8,gap3/8,min3eachdirection. No eligiblepairs means not estimable, never zeroaccuracy.',
        safety='Existing per-seed collision/numerical/deadlock/timeout flags retained. No substitution of a differentcontroller Safety policy; no formal B15 claim for K4/Q2-Q4 diagnostics.',
        models='Fixed mean probabilities of all3source-selected seeds, no test-based modelor seed selection',
        final_status='All opened DEVELOPMENT. Any positive scientific confirmation requires frozenmethod and untouchednewstates.',
        reused_engines={str(p):sha(p) for p in (MAT/'review.py',ROOT/'controller_program_input_probe_v1/review.py')},
        computational_aliases=ALIASES))


def matrix():
    p=read(HERE/'evaluation_protocol.json');d=dict(np.load(HERE/'dataset.npz'))
    root=HERE/'matrix_review';root.mkdir(exist_ok=True)
    for name in ('dataset.npz','raw_labels.npz'):
        path=root/name
        if not path.exists():path.symlink_to(MAT/name)
    if not (root/'evaluation_protocol.json').exists():(root/'evaluation_protocol.json').symlink_to(HERE/'evaluation_protocol.json')
    source=np.flatnonzero(d['source']=='matrix_goal_and_flow32')
    source=source[np.argsort(d['source_row'][source])]
    original=np.load(MAT/'dataset.npz');np.testing.assert_array_equal(d['source_row'][source],np.arange(len(original['scene'])))
    np.testing.assert_array_equal(d['state_uid'][source],original['state_uid'])
    for field in ('success','failure','unknown'):np.testing.assert_array_equal(d[field][source],original[field])
    for fold in FOLDS:
        for alias,arm in ALIASES.items():
            for seed in SEEDS:
                src=HERE/'fits'/fold/arm/f'seed{seed}';out=root/'fits'/fold/alias/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
                record=read(src/'complete.json');assert record['checkpoint_sha256']==sha(src/'checkpoint.msgpack')
                q=np.load(src/'predictions.npz')['probability'][source][:,[0,5,10,15]]
                np.savez_compressed(out/'predictions.npz',probability=q,eta_indices=np.array([0,5,10,15]))
                write(out/'complete.json',record)
    code=(MAT/'review.py').read_text()
    assert sha(MAT/'review.py')==p['reused_engines'][str(MAT/'review.py')]
    old="('field_same_all_labels','encounter_flow_eta_prior','encounter_goal_only')"
    assert code.count(old)==1
    code=code.replace(old,"('field_same_all_labels','encounter_flow_eta_prior')")
    (root/'frozen_reused_engine.py').write_text(code)
    mod=types.ModuleType('diagnostics.controller_ff_factorial_repair_v1.reused_union_review')
    mod.__package__='diagnostics.controller_ff_factorial_repair_v1';exec(compile(code,str(root/'frozen_reused_engine.py'),'exec'),mod.__dict__)
    mod.HERE=root;mod.CONDITIONS=tuple(ALIASES)
    # Float32 roundoff of constant eta-only is not a meaningful state correlation.
    def corr(a,b):
        a,b=np.asarray(a).ravel(),np.asarray(b).ravel();good=np.isfinite(a)&np.isfinite(b);a,b=a[good],b[good]
        if len(a)<3 or min(a.std(),b.std())<1e-7:return None
        return float(np.corrcoef(a,b)[0,1])
    mod.corr=corr;mod.run()
    result=read(root/'development_results.json')
    def rename(v):
        if isinstance(v,dict):return {k:rename(a) for k,a in v.items()}
        if isinstance(v,list):return [rename(a) for a in v]
        if isinstance(v,str):
            for a,b in ALIASES.items():v=v.replace(a,b)
        return v
    result=rename(result);result['source_training']='Compatible97,280outcome union; exactscope/countsinprotocol; priorcontrol erases allTRAIN stateassociations, notonlyFlow.'
    result['method_protocol_sha256']=sha(HERE/'protocol.json')
    write(HERE/'matrix_results.json',result)


def q16(index):
    import jax,jax.numpy as jnp
    from flax import serialization
    from diagnostics.controller_q_diversity_v1.train import Critic,compact_field
    from diagnostics.controller_ff_crossed_cache_full_v1.study import full_physical
    from diagnostics.orthoflow3_unified_representation_v1.representation import batch
    from diagnostics.controller_program_input_probe_v1.review import Data,analyze_panel
    scene=('toy_give_way','ring_exchange')[index];held=('held_toy','held_ring')[index]
    original=Data(np.load(DIV/'data/final/dataset.npz'));dp=read(DIV/'protocol.json');states={s['state_uid']:s for s in dp['states']}
    physical=[]
    for sid,sc in zip(original['state_uid'],original['scene']):
        physical.append(full_physical(str(sc),states[str(sid)]['physical'],dp['environments'][str(sc)]['config']['config']))
    entities=batch(physical);raw=np.asarray(compact_field(jnp.asarray(np.concatenate((original['nominal'],original['field']),-1))))
    p=read(HERE/'protocol.json');eta=jnp.asarray((np.asarray(p['eta'])-p['eta_input_center'])/np.asarray(p['eta_input_radius']),jnp.float32)
    outputs=[];arrays={};saved={};inferences={}
    for fold in ('joint',held):
        predictions={}
        for arm in ARMS:
            values=[]
            for seed in SEEDS:
                path=HERE/'fits'/fold/arm/f'seed{seed}';complete=read(path/'complete.json')
                assert sha(path/'checkpoint.msgpack')==complete['checkpoint_sha256']
                norm=read(path/'normalization.json')
                x={k:jnp.asarray(v,jnp.float32) for k,v in entities.items()}
                if arm in ('union_native_slots','union_slots_state_erased'):
                    x['agents']=x['agents'].at[...,3:7].set(jnp.eye(4)[None]*x['agent_mask'][...,None])
                c=jnp.asarray(np.clip((raw-np.asarray(norm['mean']))/np.asarray(norm['scale']),-10,10)*entities['agent_mask'][...,None],jnp.float32)
                net=Critic('eta_only' if arm=='eta_only' else 'state_field_compact')
                initial=net.init(jax.random.PRNGKey(seed),{k:v[:1] for k,v in x.items()},c[:1],eta[:1])
                par=serialization.from_bytes(initial,(path/'checkpoint.msgpack').read_bytes())
                @jax.jit
                def predict(ii,jj):return net.apply(par,{k:v[ii] for k,v in x.items()},c[ii],eta[jj])
                ii=np.repeat(np.arange(len(raw)),16);jj=np.tile(np.arange(16),len(raw));zs=[]
                for start in range(0,len(ii),256):zs.append(np.asarray(predict(jnp.asarray(ii[start:start+256]),jnp.asarray(jj[start:start+256]))))
                z=np.concatenate(zs).reshape(-1,16);values.append(1/(1+np.exp(-np.clip(z,-80,80))))
            name='state_field_compact' if arm=='union_current_entity' else arm
            predictions[name]=np.asarray(values);inferences[fold+'__'+arm]=np.asarray(values)
        ids=np.flatnonzero(original['scene']==scene)
        outputs.extend(analyze_panel(original,ids,predictions,
            dict(variant=fold,panel='opened_independent_Q16_unseen_controller',scene=scene,controller_group='analytic6_7'),arrays,saved))
    for r in outputs:
        if r['arm']=='state_field_compact':r['arm']='union_current_entity'
    write(HERE/f'q16_{scene}_results.json',dict(rows=outputs,new_rollouts=0,
        scope='Previously opened independent32physicalstates/scene,unseenanalytic6/7;zero physicalTRAIN/VAL overlap. Joint seen-scene versus source-only held-scenefold. DEVELOPMENT,notfreshconfirmation.',
        note='Keys state_field_compact in comparisons/bootstrap refer to union_current_entity; alloldlabels reused; source normalization/checkpoint frozen.'))
    np.savez_compressed(HERE/f'q16_{scene}_raw_predictions.npz',**inferences)
    np.savez_compressed(HERE/f'q16_{scene}_bootstrap.npz',**arrays)
    np.savez_compressed(HERE/f'q16_{scene}_predictions_and_controls.npz',**saved)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','matrix','q16']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='matrix':matrix()
    else:q16(a.index)
