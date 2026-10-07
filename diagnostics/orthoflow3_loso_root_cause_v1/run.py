"""Frozen-protocol LOSO extension. No rollout execution; first fold immutable."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
import json,sys,argparse,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from diagnostics.orthoflow3_cross_scene_zero_shot_v1 import build_source as old
from diagnostics.orthoflow3_unified_representation_v1 import representation as rep

ROOT=old.ROOT; OUT=Path(__file__).resolve().parent; FIRST=old.OUT
SCENES=('toy_giveway','double_bottleneck','four_way_intersection','ring_exchange')
FOLDS={'toy':'toy_giveway','db':'double_bottleneck','four':'four_way_intersection'}
load=old.load;sha=old.sha
def dump_at(base,name,value):
    p=base/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
def dump(name,value):dump_at(OUT,name,value)

def prepare():
    assert not (OUT/'protocol.json').exists(),'Protocol already frozen'
    protocol={'source_rule':'first-fold exact full-count >=16 for new scenes; unchanged Toy structured+wide',
      'source_exclusion':'all-scene-generator-derived proposals excluded to avoid target-label acquisition ancestry',
      'first_fold_decision_sha256':sha(FIRST/'final_decision.json'),
      'architecture':rep.VERSION,'architecture_sha256':sha(rep.__file__),
      'loss':'W1 pair equal BCE; scene equal minibatches','seeds':[17,23,41],
      'training':'same first-fold 4000 max steps, 100 eval, patience10 after1500; AdamW .001 wd .0001',
      'selection':'source scene mean VAL NLL; no target tuning',
      'target_pools':'frozen independent Toy48, DB24, Four24, Ring60; stochastic K16 only',
      'diagnostic':'joint-four adds target TRAIN labels; not zero-shot; same architecture/training',
      'new_rollout':0,'schema_caveat':'historically all-scene-informed fixed physical schema; target-label-free fitting, not target-naive schema discovery',
      'generator':'unchanged; target-supervised historical proposals, critic-only zero-shot'}
    dump('protocol.json',protocol)
    states=load(FIRST/'source_states.json');rows=pq.read_table(FIRST/'source_pairs.parquet').to_pylist()
    data=ROOT/'datasets/orthoflow3_basin_dataset_v2_audited'
    ss={r['state_uid']:r for r in pq.read_table(data/'states.parquet',filters=[('scenario','=','ring_exchange')]).to_pylist()}
    labels=pq.read_table(data/'eta_labels.parquet',filters=[('scenario','=','ring_exchange')]).to_pylist()
    c=old.con();ex=Counter();used={};seen=set()
    for r in labels:
        if r['split'] not in ('train','validation'):continue
        if r['seed_count']<16 or r['numerical_failure_count']:ex['partial_or_numerical']+=1;continue
        key=(r['state_uid'],r['eta_uid'],r['controller_uid'])
        if key in seen:continue
        rr=c.execute("SELECT rollout_uid,seed_key,success FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=? AND compatibility_quality='EXACT_REUSE' AND conflict_quarantined=0 AND numerical_failure=0",key).fetchall()
        if len(rr)<r['seed_count']:ex['db_incomplete']+=1;continue
        ct=dict(c.execute('SELECT * FROM controller_config WHERE controller_uid=?',(key[2],)).fetchone())
        if ct['compatibility_quality'] not in ('EXACT_REUSE','EXACT_PROFILE'):ex['controller']+=1;continue
        assert 'current' in r['label_semantics_version'] or 'v2' in r['label_semantics_version'],r['label_semantics_version']
        s=ss[key[0]];used[key[0]]={'state_uid':key[0],'scenario':'ring_exchange','split':s['split'],'family':s['parent_episode_id'],'physical':rep.parse(s)}
        rows.append(dict(state_uid=key[0],eta_uid=key[1],controller_uid=key[2],eta=rep.decode(r['eta_raw']),scenario='ring_exchange',split=r['split'],origin='historical_v2_full_count',requested_n=r['seed_count'],n=len(rr),k=sum(x['success'] for x in rr),q=sum(x['success'] for x in rr)/len(rr),rollout_uids=[x['rollout_uid'] for x in rr],seed_keys=[x['seed_key'] for x in rr]))
        seen.add(key)
    c.close();states+=list(used.values());dump('ring_exclusions.json',dict(ex))
    dump('all_source_states.json',states);pq.write_table(pa.Table.from_pylist(rows),OUT/'all_source_pairs.parquet')
    for fold,target in {**FOLDS,'joint':None}.items():
        d=OUT/fold;d.mkdir(exist_ok=True);scenes=[s for s in SCENES if s!=target]
        st=[s for s in states if s['scenario'] in scenes];ix={s['state_uid']:i for i,s in enumerate(st)}
        rr=[dict(r,state_index=ix[r['state_uid']]) for r in rows if r['scenario'] in scenes]
        for sc in scenes:
            tr={s['family'] for s in st if s['scenario']==sc and s['split']=='train'};va={s['family'] for s in st if s['scenario']==sc and s['split']=='validation'}
            assert tr and va and not tr&va,(sc,tr&va)
        dump_at(d,'source_states.json',st);pq.write_table(pa.Table.from_pylist(rr),d/'source_pairs.parquet')
        np.savez_compressed(d/'source_entities.npz',**rep.batch([rep.entities(s['physical']) for s in st]))
        e=np.array([r['eta'] for r in rr if r['split']=='train']);lo=e.min(0);hi=e.max(0)
        dump_at(d,'normalization.json',{'eta_center':((lo+hi)/2).tolist(),'eta_radius':np.maximum((hi-lo)/2,1e-6).tolist(),'fit_scenes':scenes,'fit_split':'train','physical':'fixed units'})
        counts={sc:{sp:{'states':sum(s['scenario']==sc and s['split']==sp for s in st),'pairs':sum(r['scenario']==sc and r['split']==sp for r in rr),'failure':sum(r['scenario']==sc and r['split']==sp and r['q']<=.5 for r in rr),'robust_rate':sum(r['scenario']==sc and r['split']==sp and r['q']>=15/16 for r in rr)} for sp in ('train','validation')} for sc in scenes}
        dump_at(d,'source_data_manifest.json',{'counts':counts,'target_scene':target,'source_only':target is not None,'pairs_sha256':sha(d/'source_pairs.parquet'),'source_family_overlap':0})
        dump_at(d,'protocol.json',{**protocol,'target_scene':target,'sources':scenes})
    dump('working_state.json',{'stage':'datasets_frozen','new_rollouts':0})
    print(json.dumps({f:load(OUT/f/'source_data_manifest.json')['counts'] for f in ('toy','db','four','joint')},indent=2))

def configure(fold):
    from diagnostics.orthoflow3_cross_scene_zero_shot_v1 import train as tr,baselines as ba
    d=OUT/fold;scenes=load(d/'protocol.json')['sources']
    def writer(name,value):
        if isinstance(value,dict) and 'target_labels_used' in value and fold not in FOLDS:
            value={**value,'target_labels_used':True,'confirmation_labels_used':False,'interpretation':'target-scene TRAIN/VAL supervised diagnostic, never zero-shot'}
        dump_at(d,name,value)
    tr.OUT=d;tr.SOURCES=scenes;tr.dump=writer
    ba.OUT=d;ba.SOURCES=scenes;ba.dump=writer
    return tr,ba

def train(fold):
    import jax,jax.numpy as jnp
    tr,ba=configure(fold);d=OUT/fold
    if (d/'models_frozen.json').exists():return
    # Reuse first-fold structural test with three source scenes; extra scene
    # is tested in another fold. No target labels used in structural checks.
    scenes=tr.SOURCES
    if len(scenes)>=3:
        tr.SOURCES=scenes[:3];tr.structural();tr.SOURCES=scenes
    else:
        # Identical parser/model already checked for this scene in a LOSO fold.
        checks=[load(OUT/f/'structural_tests.json') for f in FOLDS]
        assert all(r['passed'] for r in checks)
        dump_at(d,'structural_tests.json',{'passed':True,'inherited_identical_architecture_checks':[str(OUT/f/'structural_tests.json') for f in FOLDS]})
    for kind in ('shared','eta_only'):
        for seed in (17,23,41):
            if not (d/kind/f'seed{seed}/training.json').exists():tr.train(kind,seed)
    tr.freeze()
    if fold in FOLDS:ba.fit()
    dump('working_state.json',{'stage':'training','last_completed':fold,'new_rollouts':0})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','train']);ap.add_argument('--fold',default='toy');a=ap.parse_args()
    prepare() if a.action=='prepare' else train(a.fold)
