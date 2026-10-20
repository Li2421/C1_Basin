"""Preserve probe-state heterogeneity using existing outcome-free trajectories."""
from common import *
from features import one_trace,NAMES

ARMS=('safety_baseline','F13','F10')
COMPONENTS=('progress_50','progress_100','progress_200','progress_300','freeze','moving','activation','rewrite','mixed')
EXTRA_NAMES=tuple(f'{arm}_{component}_{stat}' for arm in ARMS for stat in ('std','q10','q90') for component in COMPONENTS)


def main(cohort,phase):
    if cohort=='locked':assert (HERE/'FROZEN_PIPELINE.receipt.json').exists()
    base_path=HERE/f'{cohort.upper()}_FEATURES.npz'
    for path,digest in read(HERE/f'{cohort.upper()}_FEATURES.receipt.json')['files'].items():assert sha(path)==digest
    with np.load(base_path,allow_pickle=False) as z:base={k:z[k] for k in z.files}
    p=read(HERE/f'{phase}_protocol.json');cases=read(HERE/f'{phase}_CASES.json');rows=[]
    for name,original in zip(base['names'],base['features']):
        if not np.isfinite(original).any():rows.append(np.full(len(EXTRA_NAMES),np.nan));continue
        payload=p['profiles'][str(name)]['payload']
        manifest=read(payload['adapter_manifest_path']) if 'adapter_manifest_path' in payload else p['native_manifest']
        config=manifest['scenario_config'];values={a:[] for a in ARMS}
        for c in cases:
            if c['base_controller_name']!=name:continue
            record=read(HERE/phase/'raw'/f"{c['rollout_uid']}.json")
            assert record['complete'] and not record['terminal_labels_read']
            assert not set(record)&{'success','collision','timeout','outcome'}
            assert sha(record['trace_path'])==record['trace_sha256']
            with np.load(record['trace_path'],allow_pickle=False) as trace:
                value=one_trace(trace,config['dt'],config['max_speed'],config['goal_tolerance'])
            values[c['arm']].append([*value['progress'],*[value[k] for k in ('freeze','moving','activation','rewrite','mixed')]])
        extra=[]
        for arm in ARMS:
            a=np.asarray(values[arm]);assert a.shape==(16,len(COMPONENTS))
            extra.extend(a.std(0));extra.extend(np.quantile(a,.1,axis=0));extra.extend(np.quantile(a,.9,axis=0))
        rows.append(extra)
    features=np.concatenate((base['features'],np.asarray(rows)),axis=1)
    assert np.array_equal(features[:,:len(NAMES)],base['features'],equal_nan=True)
    assert np.array_equal(np.isfinite(features).any(1),np.isfinite(base['features']).any(1))
    target=HERE/f'{cohort.upper()}_DISTRIBUTIONAL_FEATURES.npz';assert not target.exists()
    np.savez_compressed(target,features=features,r5=base['r5'],names=base['names'],types=base['types'],feature_names=(*NAMES,*EXTRA_NAMES))
    seal(f'{cohort.upper()}_DISTRIBUTIONAL_FEATURES.receipt.json',[target,base_path,HERE/'distributional_features.py',
        HERE/f'{cohort.upper()}_PROBE_TRAJECTORY_INDEX.json'],new_rollouts=0,task_labels_read=False,
        base_features_preserved=True,unavailable_probe_rule_unchanged=True)
    print(cohort,features.shape,'features from existing probes; no new physics')


if __name__=='__main__':main(*sys.argv[1:])
