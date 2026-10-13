"""Read-only support/identity audit before deciding on a combined-cache fit."""
import hashlib,json,sqlite3
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent


def read(p): return json.loads(p.read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def run():
    diversity=ROOT/'controller_q_diversity_v1'
    dp=read(diversity/'protocol.json')
    rp=read(ROOT/'state_nuisance_rootcause_v1/real_state_expansion/protocol.json')
    mp=read(ROOT/'controller_ff_factorial_repair_v1/protocol.json')
    fp=read(ROOT/'controller_four_scene_mechanism_v1/protocol.json')
    raw_eta=np.asarray(dp['eta'],np.float64)
    assert np.array_equal(raw_eta,np.asarray(rp['eta'],np.float64))
    assert rp['controller']==dp['controllers']['ring_exchange']['flow']
    assert rp['environment']['config']==dp['environments']['ring_exchange']['config']
    for key in ('execution_callable','flow_checkpoint_path'):
        path=Path(rp['controller']['payload'][key])
        assert hashlib.sha256(path.read_bytes()).hexdigest()==rp['controller']['payload'][key.replace('_path','')+'_sha256']
    source_states={s['state_uid']:s for s in dp['states']}
    matrix_states={s['state_uid']:s for s in fp['states']}
    arrays={
        'analytic_and_native128':dict(np.load(diversity/'data/source_augmented128/dataset.npz')),
        'matrix_goal_and_flow32':dict(np.load(ROOT/'controller_ff_factorial_repair_v1/dataset.npz')),
        'native_ring512':dict(np.load(ROOT/'controller_entity_coverage_check_v1/dataset.npz')),
    }
    feature_paths={s['state_uid']:(diversity/'features'/s['state_uid']/'flow.npz') for s in rp['states']}
    summary=[];cohorts=[];roles={};states_by_source={}
    for name,d in arrays.items():
        states=(source_states if name=='analytic_and_native128' else matrix_states if name=='matrix_goal_and_flow32' else {s['state_uid']:s for s in rp['states']})
        rows=[]
        for i,(sid,split) in enumerate(zip(d['state_uid'],d['split'])):
            if split not in ('train','validation'):continue
            s=states[str(sid)];scene=s['scenario'];physical=s['physical']
            # Preserve API agent order and all physical components; no approximate matching.
            key=scene+'__'+digest(physical)
            roles.setdefault(key,set()).add(str(split))
            states_by_source.setdefault(name,set()).add(key)
            if name=='native_ring512':controller='native_flow';raw_candidates=raw_eta
            else:
                controller=str(d['controller'][i]);controller='native_flow' if controller=='flow' else controller
                raw_candidates=raw_eta if name=='analytic_and_native128' else d['eta'][i]
            assert np.array_equal(raw_candidates,raw_eta)
            # Labels remain tied to their original seed records; no pooling counts from unknown seed overlaps.
            count=(d['success'][i]+d['failure'][i]+d['unknown'][i]).sum()
            rows.append(dict(source=name,row=i,scene=scene,split=str(split),state_uid=str(sid),physical_hash=key,
                             controller=controller,outcomes=int(count)))
        cohorts.extend(rows)
        for sc in sorted({r['scene'] for r in rows}):
            for split in ('train','validation'):
                group=[r for r in rows if r['scene']==sc and r['split']==split]
                if not group:continue
                summary.append(dict(source=name,scene=sc,split=split,states=len({r['physical_hash'] for r in group}),
                    controllers=sorted({r['controller'] for r in group}),state_controller_rows=len(group),
                    outcomes=sum(r['outcomes'] for r in group),
                    state_count_per_controller={c:len({r['physical_hash'] for r in group if r['controller']==c}) for c in sorted({r['controller'] for r in group})}))
    conflicts={k:sorted(v) for k,v in roles.items() if len(v)>1}
    candidate_groups={}
    for r in cohorts:candidate_groups.setdefault((r['physical_hash'],r['controller']),[]).append(r)
    duplicates=[group for group in candidate_groups.values() if len(group)>1]
    pairs={}
    for a in states_by_source:
        for b in states_by_source:
            if a<b:pairs[a+'__'+b]=len(states_by_source[a]&states_by_source[b])
    native_rows=[r for r in cohorts if r['source']=='native_ring512']
    missing_native_features=[r for r in native_rows if not feature_paths[r['state_uid']].exists()]
    # Older files labelled "LOSO/FF" cannot be merged merely by human-facing placement names.
    excluded=[]
    with sqlite3.connect('file:shared_rollout_db/rollout.sqlite?mode=ro',uri=True) as db:
        for scene in ('toy_give_way','double_bottleneck','four_way_intersection','ring_exchange'):
            cases=read(ROOT/'physical_state_critic_formal_v1/loso'/scene/'train/cases.json')
            uid=cases[0]['controller_uid'];row=db.execute('SELECT config_json FROM controller_config WHERE controller_uid=?',(uid,)).fetchone()
            cfg=json.loads(row[0]) if row else {}
            excluded.append(dict(source='older_formal_loso_'+scene,controller_uid=uid,
                current_native_uid=dp.get('controllers',{}).get(scene,{}).get('flow',{}).get('controller_uid'),
                fingerprint_fields={k:cfg[k] for k in ('chain','conditioning','horizon','success_semantics') if k in cfg},
                decision='NOT_ACCEPTED_FOR_DIRECT_CURRENT_FF_UNION',
                reason='Different execution identity and conditioning/evaluator contract. Similar scene/Flow weights/FF naming is insufficient. Preserve labels for their actual pipeline; no assertion of equivalence.'))
    result=dict(status='AUDIT_ONLY; no combined fit or rollout started',new_rollouts=0,new_static_queries=0,
        exact_native_ring512_controller_payload_equal=True,exact_native_ring512_environment_equal=True,
        native_runtime_and_weights_still_match=True,raw_eta_float64_identical=True,
        eta_encoding_warning='Ring physical_inputs/coverage dataset eta is normalized; diversity eta is raw float32; factorial eta is per-row rawfloat64. Any union MUST use original protocol rawfloat64 identities and one explicit new train-only normalization. Never concatenate these eta arrays as if identical.',
        summaries=summary,physical_overlap_by_source=pairs,train_validation_physical_conflicts=conflicts,
        duplicate_state_controller_groups=duplicates,missing_native_static_features=len(missing_native_features),
        available_native_static_features=len(native_rows)-len(missing_native_features),
        excluded_old_formal_sources=excluded,
        proposed_split_rule='Physical payload hash, not UID. If a physical state ever served as VAL in an included cohort, preserve it as VAL everywhere; source-scene folds still exclude all held-scene labels. Prior opened TEST is never promoted in this audit.',
        dedup_rule='Do not count duplicate aggregate labels twice. Verify raw seed identities before merging; otherwise choose one prespecified compatible source record per physical-state/controller/eta and retain full provenance.',
        historical_gap='Prior native augmentation used32Toy/Ring and56DB/Four states; it did not jointly use this nativeRing512 cohort with the crossed controller data. This audit does not itself prove a union is the right repair.')
    (HERE/'UNION_CACHE_COMPATIBILITY_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    (HERE/'UNION_CACHE_ROW_INDEX.json').write_text(json.dumps(cohorts,indent=2)+'\n')
    print(dict(source_summaries=len(summary),split_conflicts=len(conflicts),duplicate_groups=len(duplicates),
               physical_overlaps=pairs,missing_native_features=len(missing_native_features),new_rollouts=0))


if __name__=='__main__':run()
