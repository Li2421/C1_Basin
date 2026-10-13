"""Combine compatible actual-FF labels; query only missing static controller inputs."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from diagnostics.controller_four_scene_mechanism_v1.study import read,write,freeze,sha
from diagnostics.controller_cross_scene_root_v2.union_cache_audit import digest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
DIV=ROOT/'controller_q_diversity_v1'
MAT=ROOT/'controller_ff_factorial_repair_v1'
RING=ROOT/'state_nuisance_rootcause_v1/real_state_expansion'
AUDIT=ROOT/'controller_cross_scene_root_v2'
SOURCES={'analytic_and_native128':DIV/'data/source_augmented128/dataset.npz',
         'matrix_goal_and_flow32':MAT/'dataset.npz',
         'native_ring512':ROOT/'controller_entity_coverage_check_v1/dataset.npz'}
FOLDS={'joint':None,'held_toy':'toy_give_way','held_db':'double_bottleneck',
       'held_four':'four_way_intersection','held_ring':'ring_exchange'}
ARMS=('eta_only','union_current_entity','union_native_slots','union_slots_state_erased')
SEEDS=(17,23,41)


def prepare():
    HERE.mkdir(exist_ok=True)
    audit=read(AUDIT/'UNION_CACHE_COMPATIBILITY_AUDIT.json')
    assert not audit['train_validation_physical_conflicts']
    assert audit['exact_native_ring512_controller_payload_equal'] and audit['exact_native_ring512_environment_equal']
    dp,rp=read(DIV/'protocol.json'),read(RING/'protocol.json')
    fp=read(ROOT/'controller_four_scene_mechanism_v1/protocol.json')
    sp={'analytic_and_native128':{s['state_uid']:s for s in dp['states']},
        'matrix_goal_and_flow32':{s['state_uid']:s for s in fp['states']},
        'native_ring512':{s['state_uid']:s for s in rp['states']}}
    rows=read(AUDIT/'UNION_CACHE_ROW_INDEX.json')
    groups={}
    for row in rows:groups.setdefault((row['physical_hash'],row['controller']),[]).append(row)
    priority={'native_ring512':3,'analytic_and_native128':2,'matrix_goal_and_flow32':1}
    selected=[];discarded=[]
    for group in groups.values():
        group=sorted(group,key=lambda r:(-r['outcomes'],-priority[r['source']],r['row']))
        r=dict(group[0]);r['state']=sp[r['source']][r['state_uid']]
        r['inherited_dataset']=str(SOURCES[r['source']]);selected.append(r)
        discarded.extend(group[1:])
    selected.sort(key=lambda r:(r['scene'],r['physical_hash'],r['controller']))
    rd=np.load(SOURCES['native_ring512'])
    protected={s['scenario']+'__'+digest(s['physical']) for s,split in zip(rp['states'],rd['split']) if split=='test'}
    assert not (protected & {r['physical_hash'] for r in selected})
    assert all(r['state']['physical']['timestep']==0 for r in selected)
    missing=[r for r in selected if r['source']=='native_ring512' and not (DIV/'features'/r['state_uid']/'flow.npz').exists()]
    freeze(HERE/'rows.json',selected);freeze(HERE/'omitted_duplicate_rows.json',discarded)
    freeze(HERE/'missing_static_features.json',missing)
    freeze(HERE/'protocol.json',dict(
        schema='actual_ff_cached_union_root_diagnostic_v1',new_rollouts=0,
        question='After retaining native API roles, does combining complementary existing actualFF state/controller support improve transferable state-conditioned Q? Compare real versus state-erased labels on exactly the same union.',
        history_audit=str(AUDIT/'UNION_CACHE_COMPATIBILITY_AUDIT.json'),
        reused_experiments=['controller_ff_crossed_cache_full_v1','controller_ff_factorial_repair_v1','controller_ff_slot_repair_v1','controller_q_diversity_v1','controller_entity_symmetry_probe_v1'],
        genuine_gap='Earliernativeaugmentationused32Toy/Ringstates,notRing512;earlieranalytic128and10matrix-controllercohortsweretrainedseparately. No prior combined actualFF cache+slot+state-erasure experiment found in the historical code/evidence index.',
        scope='DEVELOPMENT mechanism diagnostic. Existing labels already opened; no final confirmation claim. If promising, freeze method and generate genuinely untouched confirmation.',
        sources={k:dict(path=str(p),sha256=sha(p)) for k,p in SOURCES.items()},
        protected_original_ring_test_states=len(protected),physical_train_validation_conflicts=0,
        dedup='For duplicate physicalstate/controller use one complete cohort, larger existing trial count then fixed sourcepriority; NEVER sum aggregates with unverified seed overlap. All original seed evidence retained.',
        duplicate_rows_omitted=len(discarded),rows=len(selected),missing_static_features=len(missing),
        candidates='Exactly original shared16 rawfloat64 eta; per-row observed K4 orK16 labels retained without inventing missing outcomes. Evaluation factorial continues its frozen4indices. Restore raweta from originalprotocol, never mix normalized oldarrays.',
        eta=dp['eta'],eta_input_center=[.875,0.,.375],eta_input_radius=[.375,.5,.375],
        physical='Existing full_physical maskedentity encoding, deployment constants restored, no sceneID. Slot intervention writes four APIonehot entries into otherwisezero channels3:7. All other features unchanged.',
        controller='Existing compact64(static nominal14 + field-query50) only. Same exact feature function and frozencontroller. No H20,physicalrolloutpreview,or success-derived input. Extra initializer field was a negative diagnostic, not adopted here.',
        static_feature_code={str(p):sha(p) for p in (DIV/'features.py',DIV/'runtime.py',DIV/'train.py')},
        static_feature_safety='Environment.step forbidden; only missing currentnativeRing descriptors queried;128compatibleexisting descriptors reused.',
        folds=FOLDS,arms=ARMS,seeds=SEEDS,
        source_only=['gradientlabels','normalization','state-erasedprior','checkpointselection'],
        train_sampling='Uniform source scene, then controller, then physicalstate, then observed eta. Per-cell binomialNLL divided by known trials. Eta-only sees identical real labels/draws; state-erased control retains eachscene/controller/eta mean but removes TRAIN state associations.',
        normalization='Same unionSOURCE TRAIN featuremean/scale for allarms within a fold; no heldscene/VAL data. Fixed physical/eta units.',
        selection='SOURCE VAL: mean per-scene/controller pessimistic selectionregret, then strongpairaccuracy(gap>=.25), then state×eta-centered error, NLL, earliest step. Test never selects model/seed.',
        training=dict(steps=6000,batch_per_scene=120,validation_every=200,lr=.0008,weight_decay=.0001,clip_grad=5.),
        evaluation='All four held scenes, original natural state panels; source joint VAL is diagnostic not independent confirmation. Reuse all-state/fixed-policy/S-mismatch/C-mismatch/reversal/unknown/collision analysis. Report everyfold, never choose winning subset.',
        literature_reuse=['https://arxiv.org/abs/2010.09536','https://proceedings.mlr.press/v119/raileanu20a.html','https://arxiv.org/abs/2502.11949'],
        literature_boundary='Policy-conditionedvaluework motivates crossedpolicy/environment support; neither its results nor our cacheunion guarantees unseen-scene transfer. No newliterature claim beyond retained primary-source review.',
        old_formal_data_excluded=audit['excluded_old_formal_sources']))
    print(dict(selected_rows=len(selected),duplicate_rows=len(discarded),missing_static_features=len(missing),new_rollouts=0),flush=True)


def features(index,workers):
    import jax
    from diagnostics.controller_q_diversity_v1.runtime import native_runtime,controller_context
    from diagnostics.controller_q_diversity_v1.features import field_context
    p=read(HERE/'protocol.json')
    for path,h in p['static_feature_code'].items():assert sha(path)==h
    rt=native_runtime('ring_exchange');jax.config.update('jax_enable_x64',False)
    def forbidden(*args,**kwargs):raise RuntimeError('Physical simulator step forbidden in static feature extraction')
    rt.env_cls.step=forbidden
    count=0
    for r in read(HERE/'missing_static_features.json')[index::workers]:
        path=HERE/'features'/r['state_uid']/'flow.npz'
        if path.exists():continue
        nominal=controller_context(rt,r['state']);field=field_context(rt,r['state'])
        path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,nominal=nominal,field=field)
        tmp.replace(path);count+=1
    write(HERE/'feature_jobs'/f'{index:02d}.json',dict(index=index,workers=workers,created=count,physical_steps=0,new_rollouts=0,
        protocol_sha256=sha(HERE/'protocol.json')))
    print(dict(index=index,created=count,physical_steps=0),flush=True)


def assemble():
    import jax.numpy as jnp
    from diagnostics.controller_q_diversity_v1.train import compact_field
    from diagnostics.controller_ff_crossed_cache_full_v1.study import full_physical
    from diagnostics.orthoflow3_unified_representation_v1.representation import batch
    p=read(HERE/'protocol.json');rows=read(HERE/'rows.json')
    ds={k:dict(np.load(path)) for k,path in SOURCES.items()}
    for name,path in SOURCES.items():assert sha(path)==p['sources'][name]['sha256']
    dp=read(DIV/'protocol.json');mp=read(MAT/'protocol.json');rp=read(RING/'protocol.json')
    physical={};features=[];labels={k:[] for k in ('success','failure','unknown')}
    for r in rows:
        source=r['source'];d=ds[source];i=r['row'];sid=r['state_uid'];sc=r['scene']
        if r['physical_hash'] not in physical:
            env=(rp['environment'] if source=='native_ring512' else
                 dp['environments'][sc] if source=='analytic_and_native128' else mp['environments'][sc])
            physical[r['physical_hash']]=full_physical(sc,r['state']['physical'],env['config']['config'])
        if source=='native_ring512':
            old=DIV/'features'/sid/'flow.npz';path=old if old.exists() else HERE/'features'/sid/'flow.npz'
            a=np.load(path);f=np.asarray(compact_field(jnp.asarray(np.concatenate((a['nominal'],a['field']),-1))))
            f=np.pad(f,((0,4-len(f)),(0,0)))
        elif source=='analytic_and_native128':
            f=np.asarray(compact_field(jnp.asarray(np.concatenate((d['nominal'][i],d['field'][i]),-1))))
        else:f=d['feature'][i]
        assert f.shape==(4,64) and np.isfinite(f).all()
        features.append(f)
        for k in labels:labels[k].append(d[k][i])
    x=batch([physical[r['physical_hash']] for r in rows])
    data={'entity_'+k:v for k,v in x.items()}
    data.update(feature=np.asarray(features,np.float32),eta=np.asarray(p['eta'],np.float64),
        **{k:np.asarray(v) for k,v in labels.items()},
        **{k:np.asarray([r[k] for r in rows]) for k in ('scene','split','state_uid','physical_hash','controller','source')},
        source_row=np.asarray([r['row'] for r in rows]))
    assert not (set(data['physical_hash'][data['split']=='train'])&set(data['physical_hash'][data['split']=='validation']))
    assert np.all(data['entity_agents'][...,3:7]==0)
    np.savez_compressed(HERE/'dataset.npz',**data)
    write(HERE/'dataset_audit.json',dict(sha256=sha(HERE/'dataset.npz'),physical_states=len(physical),rows=len(rows),
        labels=int(sum(np.sum(data[k]) for k in ('success','failure','unknown'))),unknown=int(data['unknown'].sum()),
        physical_train_validation_overlap=0,new_rollouts=0,missing_feature_files=0,
        source_seed_evidence='Inherited exact source datasets/protocols preserve per-state/per-controller/per-eta/per-seed outcomes; no synthesized labels except prospective TRAIN-only prior control.'))
    print(read(HERE/'dataset_audit.json'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','features','assemble']);p.add_argument('--index',type=int,default=0);p.add_argument('--workers',type=int,default=24);a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='features':features(a.index,a.workers)
    else:assemble()
