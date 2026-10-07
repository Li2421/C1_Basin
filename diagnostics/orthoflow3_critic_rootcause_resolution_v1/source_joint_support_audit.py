"""Read-only canonical cache audit; never opens held-target success labels."""
import collections
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[1]
OLD = BASE / 'diagnostics/orthoflow3_controller_training_repair_v1'
CURRENT = ROOT / 'db_transfer_v1'
OUT = ROOT / 'source_joint_support_v1'
read = lambda p: json.loads(Path(p).read_text())


def dump(name, data):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False)+'\n')


def family(s):
    value = s.get('rollout_id', s.get('family', s.get('source_group', '')))
    # Only the known parent/trajectory alias is interchangeable. A numeric
    # seed is not a family identity across different generation namespaces.
    return value.rsplit(':', 1)[-1] if value.startswith('basin_v1_parent:') else value


def key(r):
    return tuple(r[k] for k in ('state_uid', 'eta_uid', 'controller_uid'))


def main():
    current = read(CURRENT/'pairs.json')
    states = read(CURRENT/'states.json')
    byuid = {s['state_uid']:s for s in states}
    famsplit = collections.defaultdict(set)
    for s in states:
        if s['scenario']=='ring_exchange':
            famsplit[family(s)].add(s['split'])
    assert all(len(v)==1 for v in famsplit.values()), 'Existing canonical-family split overlap'
    oldstates = read(OLD/'states.json')
    oldpairs = read(OLD/'pairs.json')
    provenance = read(OLD/'dataset_provenance.json')
    profiles = read(OLD/'protocol.json')['profiles']
    profile = {p['name']:p for p in profiles}
    physical = {s['state_uid']:s['physical'] for s in read(OLD/'physical.json')}
    currentkeys = {key(r) for r in current}
    oldstate = {s['uid']:s for s in oldstates}
    audit=[]
    for s in oldstates:
        canonical_family=family(s)
        assert not famsplit[canonical_family] or famsplit[canonical_family]=={s['split']}, (canonical_family,s['split'],famsplit[canonical_family])
        p=physical[s['uid']]
        assert s['physical']['timestep']==0 and p['remaining_fraction']==1.
        for f in ('positions','velocities','goals'):
            np.testing.assert_array_equal(p[f], s['physical'][f])
        if s['uid'] in byuid:
            assert family(byuid[s['uid']])==canonical_family
            for f in ('positions','velocities','goals','flow','remaining_fraction'):
                np.testing.assert_allclose(byuid[s['uid']]['physical'][f], p[f], atol=1e-7,rtol=0)
        audit.append(dict(state_uid=s['uid'],family=canonical_family,split=s['split'],
                          existing_same_family=bool(famsplit[canonical_family]),existing_same_state=s['uid'] in byuid))
    # Query only recorded canonical identities; old invalid float64 paths cannot
    # be promoted by filename or by a similar physical state.
    c=sqlite3.connect(f'file:{BASE}/shared_rollout_db/rollout.sqlite?mode=ro',uri=True)
    c.row_factory=sqlite3.Row
    exclusions=[]; rows=[]; config={}; allroll=set()
    for pr in provenance:
        p=oldpairs[pr['pair_index']]; ctl=profile[pr['controller']]['controller_uid']
        if ctl not in config:
            rr=dict(c.execute('SELECT * FROM controller_config WHERE controller_uid=?',(ctl,)).fetchone())
            assert rr['compatibility_quality'] in ('EXACT_PROFILE','EXACT_REUSE')
            payload=json.loads(rr['config_json'])
            assert payload['flow_checkpoint_sha256']==profile[pr['controller']]['sha256']
            config[ctl]=dict(row=rr,payload=payload,profile=profile[pr['controller']])
        records=[]
        for rid in pr['rollout_uids']:
            record=c.execute('SELECT * FROM rollout WHERE rollout_uid=?',(rid,)).fetchone()
            assert record is not None,rid
            r=dict(record)
            assert key(r)==(p['state_uid'],p['eta_uid'],ctl)
            assert rid not in allroll; allroll.add(rid)
            if r['compatibility_quality']!='EXACT_REUSE' or r['conflict_quarantined']:
                exclusions.append(dict(rollout_uid=rid,reason='incompatible_or_quarantined'));continue
            records.append(r)
        assert len({r['seed_key'] for r in records})==len(records)
        valid=[r for r in records if not r['numerical_failure']]
        if not valid:continue
        et=dict(c.execute('SELECT * FROM eta WHERE eta_uid=?',(p['eta_uid'],)).fetchone())
        exact=[et[f'eta{i}'] for i in (1,2,3)]
        assert exact==p['eta']
        observed16=[r for r in valid if json.loads(r['seed_key']).get('future_index',999)<16]
        state=oldstate[p['state_uid']]
        rows.append(dict(state_uid=p['state_uid'],eta_uid=p['eta_uid'],controller_uid=ctl,
            eta=exact,scenario='ring_exchange',split=state['split'],family=family(state),
            pool='intervention',origin='cached_native_true_t0_wide_controller',
            s=sum(r['success'] for r in valid),f=sum(not r['success'] for r in valid),
            s16=sum(r['success'] for r in observed16),f16=sum(not r['success'] for r in observed16),
            numerical=sum(bool(r['numerical_failure']) for r in records),
            rollout_uids=[r['rollout_uid'] for r in records],seed_keys=[r['seed_key'] for r in records],
            existing_pair=(p['state_uid'],p['eta_uid'],ctl) in currentkeys))
    for r in rows:
        r['B15']=r['s16']>=15;r['nonB15']=r['f16']>=2
    # Audit the exact previously common eta separately. Its coordinate is
    # already source TRAIN, never a newly chosen target-failure eta.
    common='eta_f694b09341f3da3c16c1afc65fdc6266404401eeb971c209f9c35679c5f86f65'
    rr=[r for r in current if r['eta_uid']==common and r['split']=='train']
    exposure=[]
    for (sc,ctl),gg in __import__('itertools').groupby(sorted(rr,key=lambda r:(r['scenario'],r['controller_uid'])),lambda r:(r['scenario'],r['controller_uid'])):
        gg=list(gg); stages=[byuid[r['state_uid']]['physical']['remaining_fraction'] for r in gg]
        exposure.append(dict(scenario=sc,controller_uid=ctl,pairs=len(gg),B15=sum(r['B15'] for r in gg),
                             true_t0=sum(x==1. for x in stages),remaining_fraction_quantiles=np.quantile(stages,[0,.5,1]).tolist()))
    summaries=[]
    for p in profiles:
        for split in ('train','validation'):
            rr=[r for r in rows if r['controller_uid']==p['controller_uid'] and r['split']==split]
            summaries.append(dict(controller=p['name'],split=split,pairs=len(rr),new_pairs=sum(not r['existing_pair'] for r in rr),
               states=len({r['state_uid'] for r in rr}),etas=len({r['eta_uid'] for r in rr}),
               trials=sum(r['s']+r['f'] for r in rr),numerical=sum(r['numerical'] for r in rr),
               B15=sum(r['B15'] for r in rr),nonB15=sum(r['nonB15'] for r in rr),
               clear_failure=sum(r['s']/(r['s']+r['f'])<=.5 for r in rr)))
    currentcontrollers=read(CURRENT/'controllers.json')
    controllers={}
    for ctl,v in config.items():
        p=v['profile'];path=p['path'] or next(vv['base_path'] for vv in currentcontrollers.values() if vv['scenario']=='ring_exchange')
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==p['sha256']
        controllers[ctl]=dict(scenario='ring_exchange',path=path,sha256=p['sha256'],config=v['payload'],base_path=next(vv['base_path'] for vv in currentcontrollers.values() if vv['scenario']=='ring_exchange'))
    dump('cache_audit.json',dict(new_rollouts=0,compatible_pairs=len(rows),old_dataset_pairs=len(provenance),
        observed_continuations=sum(r['s']+r['f'] for r in rows),numerical=sum(r['numerical'] for r in rows),
        excluded=exclusions,canonical_family_overlap=sum(s['existing_same_family'] for s in audit),
        exact_state_overlap=sum(s['existing_same_state'] for s in audit),source_family_split_conflicts=0,
        summaries=summaries,common_eta_source_exposure=exposure,
        common_eta_available_in_cached_crossed_panel=any(r['eta_uid']==common for r in rows)))
    dump('cached_source_rows.json',rows);dump('cached_source_states.json',[
        dict(state_uid=s['uid'],scenario='ring_exchange',split=s['split'],family=family(s),physical=physical[s['uid']]) for s in oldstates])
    dump('cached_source_controllers.json',controllers);dump('family_audit.json',audit)
    c.close()
    print(json.dumps(read(OUT/'cache_audit.json')))


if __name__=='__main__':
    main()
