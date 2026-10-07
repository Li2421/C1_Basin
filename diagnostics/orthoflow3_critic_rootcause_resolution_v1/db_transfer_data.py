"""Frozen DB-fold source index. No target outcomes and no task rollouts.

Historical stopped paths retain their observed counts; numerical attempts are
unknown. Context caches are derived inputs, not success evidence.
"""
import argparse, collections, copy, hashlib, json, os, sqlite3
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[1]
OUT = ROOT/'db_transfer_v1'
OLD = BASE/'diagnostics/orthoflow3_loso_partial_count_v1'
INTER = BASE/'diagnostics/orthoflow3_controller_intervention_generalization_v1'
TARGET = BASE/'diagnostics/orthoflow3_loso_root_cause_v1/targets/db'
SCENES = ('toy_giveway', 'four_way_intersection', 'ring_exchange')
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2, sort_keys=True, allow_nan=False)+'\n')

def clearance(physical, positions):
    x=np.asarray(positions); rad=physical['radius']; out=[]
    for i in range(len(x)):
        for j in range(i):
            out.append(np.linalg.norm(x[i]-x[j])-rad[i]-rad[j]-physical['agent_margin'])
        for o in physical['obstacles']:
            a,b=np.asarray(o['a']),np.asarray(o['b']); d=b-a
            if o['curvature']:
                v=o['interior_sign']*(np.linalg.norm(x[i]-a)-o['radius'])-rad[i]
            else:
                u=np.clip(np.dot(x[i]-a,d)/max(np.dot(d,d),1e-20),0,1)
                v=np.linalg.norm(x[i]-a-u*d)-o['radius']-rad[i]
            out.append(v-physical['wall_margin'])
    return float(min(out))

def goal_design(p):
    """One common safe radius, source-designed; never consult outcomes."""
    d=np.asarray(p['goals'])-p['positions']; n=np.linalg.norm(d,axis=1)
    # Match the physical encoder's deterministic fallback at exact goals.
    for i in np.flatnonzero(n<1e-10):
        d[i]=np.asarray(p['velocities'])[i]
        if np.linalg.norm(d[i])<1e-10: d[i]=np.asarray(p['flow'])[i]
        if np.linalg.norm(d[i])<1e-10: d[i]=p['policy_axis']
    d/=np.linalg.norm(d,axis=1,keepdims=True)
    per=np.stack([-d[:,1],d[:,0]],axis=-1)
    offsets=np.array([d,-d,per,-per]); goal=np.asarray(p['goals'])
    for power in range(7):
        radius=.30/(2**power)
        margins=[clearance(p,goal+radius*v) for v in offsets]
        if min(margins)>1e-4: return radius,offsets,min(margins)
    return None,offsets,None

def prepare():
    assert not (OUT/'protocol.json').exists(), 'Frozen design already exists'
    OUT.mkdir(exist_ok=True)
    states0=read(OLD/'states.json'); oldrows=pq.read_table(OLD/'all_pairs.parquet').to_pylist()
    states={s['state_uid']:copy.deepcopy(s) for s in states0 if s['scenario'] in SCENES}
    rows=[]; bystate=collections.defaultdict(list)
    for r in oldrows:
        if r['scenario'] in SCENES: bystate[r['state_uid']].append(r)
    key=lambda r:hashlib.sha256(('db-transfer-source-eta-panel-v1|'+r['eta_uid']).encode()).hexdigest()
    for uid,rr in bystate.items():
        for r in sorted(rr,key=key)[:32]:
            rows.append({k:r[k] for k in ('state_uid','eta_uid','controller_uid','eta','scenario','split','rollout_uids')})
            rows[-1].update(family=states[uid]['family'],pool='historical_wide',origin='audited_partial_count')
    # Reuse earlier exact controller interventions, including four crossed Ring programs.
    ir=read(INTER/'training_rows.json'); iz=np.load(INTER/'training_dataset.npz')
    toy=BASE/'diagnostics/orthoflow3_controller_conditioning_probe_v1'; tp=read(toy/'protocol.json')
    toy_manifest={r['state_uid']:r for r in read(toy/'pair_manifest.json')}
    missing_toy=sorted({r['state_uid'] for r in ir if r['scene']=='toy_giveway' and r['state_uid'] not in states})
    if missing_toy:
        import jax
        from diagnostics.orthoflow3_controller_context_loso_v1.context import Runtime
        rt=Runtime('toy_giveway')
        template=next(s['physical'] for s in states.values() if s['scenario']=='toy_giveway')
        for uid in missing_toy:
            m=toy_manifest[uid]; env=rt.make(); env.reset(np.asarray(m['initial_positions']))
            flow=rt.flow(env,jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(42),int(m['rollout_id'])),0))
            r=next(r for r in ir if r['state_uid']==uid)
            p=copy.deepcopy(template); p.update(positions=env.positions.tolist(),velocities=env.velocities.tolist(),goals=env.goals.tolist(),flow=flow.tolist())
            states[uid]=dict(state_uid=uid,scenario='toy_giveway',split=r['split'],family=r['family'],physical=p)
    for i,r in enumerate(ir):
        if r['scene'] not in SCENES or r['role']=='canonical_historical':continue
        rows.append(dict(state_uid=r['state_uid'],eta_uid=r['eta_uid'],controller_uid=r['controller_uid'],eta=iz['eta'][i].astype(float).tolist(),
            scenario=r['scene'],split=r['split'],family=r['family'],pool='intervention',origin=r['role']))
    dense=ROOT/'state_breadth_training'; dr=read(dense/'states.json'); dp=read(dense/'pairs.json'); dz=np.load(dense/'dataset.npz')
    physical={p['state_uid']:p['physical'] for folder in ('controller_function_support','motion_state_breadth') for p in read(ROOT/folder/'physical.json')}
    profiles=read(ROOT/'controller_function_support/protocol.json')['profiles']
    profiles+=read(ROOT/'phase_factorial_support/protocol.json')['profiles']+read(ROOT/'motion_factorial_support/protocol.json')['profiles']
    assert len(profiles)==16
    for s in dr:
        assert s['uid'] not in states
        states[s['uid']]=dict(state_uid=s['uid'],scenario='ring_exchange',split=s['split'],family=s['source_group'],physical=physical[s['uid']])
    cached={}
    for ci,profile in enumerate(profiles):
        for j,p in enumerate(dp):
            if dz['success'][ci,j]+dz['failure'][ci,j]<=0:continue
            uid=p['state_uid'];r=dict(state_uid=uid,eta_uid=p['eta_uid'],controller_uid=profile['controller_uid'],eta=p['eta'],scenario='ring_exchange',split=states[uid]['split'],family=states[uid]['family'],pool='intervention',origin='dense_controller_factorial',cache_indices=[ci,j])
            rows.append(r)
    basepaths={'toy_giveway':tp['flow_paths']['0'],
      'four_way_intersection':str(BASE/'diagnostics/four_way_intersection_stage1/base_u_v13_broad_global_source_balanced_macflow/best.pkl'),
      'ring_exchange':str(BASE/'diagnostics/ring_exchange_stage1/base_u_v10_local_macflow/best.pkl')}
    pathsha={sha(p):p for p in basepaths.values()}
    pathsha.update({sha(tp['flow_paths']['1']):tp['flow_paths']['1']})
    for p in read(INTER/'balanced_expansion/protocol.json')['profiles'].values():pathsha[p['alternate_flow_sha256']]=p['alternate_path']
    for p in profiles:pathsha[p['sha256']]=p['path']
    c=sqlite3.connect(f'file:{BASE}/shared_rollout_db/rollout.sqlite?mode=ro',uri=True);c.row_factory=sqlite3.Row
    seen=set();out=[];controllers={};excluded=[];rollouts=0
    for r in rows:
        key=tuple(r[k] for k in ('state_uid','eta_uid','controller_uid'))
        if key in seen:continue
        seen.add(key)
        ctl=dict(c.execute('SELECT * FROM controller_config WHERE controller_uid=?',(key[2],)).fetchone())
        assert ctl['compatibility_quality'] in ('EXACT_REUSE','EXACT_PROFILE')
        payload=json.loads(ctl['config_json']); flowsha=payload.get('flow_checkpoint_sha256') or ctl['flow_checkpoint_sha256']
        assert flowsha in pathsha,(key[2],flowsha)
        controllers[key[2]]=dict(scenario=r['scenario'],path=pathsha[flowsha],sha256=flowsha,config=payload,base_path=basepaths[r['scenario']])
        records=[dict(a) for a in c.execute('SELECT * FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=?',key)]
        if 'rollout_uids' in r:
            allowed=set(r['rollout_uids']); records=[a for a in records if a['rollout_uid'] in allowed];assert len(records)==len(allowed)
        else:records=[a for a in records if json.loads(a['seed_key']).get('future_index',999)<16]
        assert records and all(a['compatibility_quality']=='EXACT_REUSE' and not a['conflict_quarantined'] for a in records),key
        seedkeys=[a['seed_key'] for a in records];assert len(set(seedkeys))==len(seedkeys)
        valid=[a for a in records if not a['numerical_failure']]
        if not valid:excluded.append(dict(key=key,reason='all_numerical'));continue
        # Recover exact canonical eta, never use a float32 model input as a cache key.
        et=dict(c.execute('SELECT * FROM eta WHERE eta_uid=?',(key[1],)).fetchone())
        exact=[et[f'eta{i}'] for i in (1,2,3)]
        assert np.allclose(r['eta'],exact,atol=1e-6,rtol=0)
        r['eta']=exact;r['s']=sum(a['success'] for a in valid);r['f']=len(valid)-r['s'];r['numerical']=len(records)-len(valid)
        r['rollout_uids']=[a['rollout_uid'] for a in records];r['seed_keys']=[a['seed_key'] for a in records]
        first=[a for a in valid if json.loads(a['seed_key']).get('future_index',999)<16]
        r['s16']=sum(a['success'] for a in first);r['f16']=len(first)-r['s16'];r['B15']=r['s16']>=15;r['nonB15']=r['f16']>=2
        r['state_index']=None;out.append(r);rollouts+=len(valid)
    c.close()
    sourceids={r['state_uid'] for r in out};states=[s for s in states.values() if s['state_uid'] in sourceids]
    si={s['state_uid']:i for i,s in enumerate(states)}
    for r in out:r['state_index']=si[r['state_uid']]
    target=read(TARGET/'manifest.json');assert not sourceids&{r['state_uid'] for r in target}
    for sc in SCENES:
        a={s['family'] for s in states if s['scenario']==sc and s['split']=='train'};b={s['family'] for s in states if s['scenario']==sc and s['split']=='validation'};assert not a&b
    design=[]
    for s in states:
        radius,offsets,margin=goal_design(s['physical']);assert radius is not None,(s['state_uid'],'no legal goal query')
        design.append(dict(state_uid=s['state_uid'],scenario=s['scenario'],radius=radius,min_clearance=margin))
    from diagnostics.orthoflow3_unified_representation_v1 import representation as rep
    x=rep.batch([rep.entities(s['physical']) for s in states]);np.savez_compressed(OUT/'source_entities.npz',**x)
    write(OUT/'states.json',states);write(OUT/'pairs.json',out);write(OUT/'controllers.json',controllers);write(OUT/'goal_legality_audit.json',design)
    # K16 coordinates/physical states only: never copy archived prediction fields.
    target_clean=[{k:r[k] for k in ('state_uid','controller_uid','eta','family')} for r in target]
    write(OUT/'target_input_manifest.json',target_clean)
    summary=[]
    for sc in SCENES:
        for split in ('train','validation'):
            rr=[r for r in out if r['scenario']==sc and r['split']==split]
            summary.append(dict(scene=sc,split=split,states=len({r['state_uid'] for r in rr}),families=len({r['family'] for r in rr}),pairs=len(rr),observed_trials=sum(r['s']+r['f'] for r in rr),B15=sum(r['B15'] for r in rr),nonB15=sum(r['nonB15'] for r in rr),partial=sum(r['s']+r['f']<16 for r in rr),controllers=len({r['controller_uid'] for r in rr}),eta=len({r['eta_uid'] for r in rr})))
    protocol=dict(question='Does the independently useful physical controller response transfer to the discriminative held-out DB scene?',
      sources=list(SCENES),target='double_bottleneck',test='Original frozen24 true-t0 K16 DB pool; reused benchmark, not untouched new confirmation',
      source_rule='At most32 exact eta per historical state, outcome-blind shared eta UID hash order; retain compatible stopped paths. Union all prior source controller interventions and native160family Ring matrix, dedup canonical keys.',
      context='H20 nominal+eta response24; mean active-agent response16; near-goal rest16+moving16; safe radius and validity flags. No scene/controller IDs.',
      geometry_fix='Source Toy0.30m probes often outside walls. Common radius0.30/2^j,j=0..6 until all four goal offsets satisfy analytical physical clearances>1e-4. Record radius; no outcome-based adjustment. Ring cached0.30 features exactly retained where legal.',
      training='Pure observed-count NLL with scene balancing and within-scene historical/intervention balanced sampling; constant per-scene/pool TRAIN trial divisor. No false fullQ labels.',
      models=['eta_only','no_context','additive_nominal_context','full_context','full_context_freeze25'],
      additive_definition='A(h,nominal eta-independent C)+B(eta); never give eta-conditioned channels to A and call it additive',
      seeds=[17,23,41],selection='Source-only scene/pool averaged observed-count VAL NLL; compare two full-context training variants on source VAL only; no target labels before freeze',
      fit_steps=2000,eval_every=100,optimizer='AdamW lr0.0008 wd0.0001 clip5; all arms same pair order',
      normalization='Rebuild source TRAIN-only eta/context statistics. Physical entity representation uses fixed physical units, no fitted target statistics.',
      target_labels_used=False,new_rollouts=0,generator_modified=False,
      limitations=['Ring-informed context design is allowed in DB-source Ring; cannot relabel this design as strict Ring feature-unseen zero-shot','Original target pool and historical target-informed physical schema already analyzed; no target label tuning in this round','Not every saturated scene must improve'],
      hashes={str(p.relative_to(BASE)):sha(p) for p in (Path(__file__),OLD/'all_pairs.parquet',OLD/'states.json',dense/'dataset.npz',TARGET/'manifest.json',TARGET/'physical.json')})
    write(OUT/'protocol.json',protocol);write(OUT/'data_audit.json',dict(summary=summary,source_pairs=len(out),source_states=len(states),observed_trials=rollouts,excluded=excluded,duplicate_keys_removed=len(rows)-len(seen),source_group_overlap=0,target_state_overlap=0,new_rollouts=0))
    print(json.dumps(dict(pairs=len(out),states=len(states),summary=summary,new_rollouts=0)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare']);a=p.parse_args();prepare()
