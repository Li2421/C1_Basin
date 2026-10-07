"""Source-only physical data build. Never opens Ring labels or checkpoints."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('OMP_NUM_THREADS','2')
import json,hashlib,sqlite3,sys
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from diagnostics.orthoflow3_unified_representation_v1 import representation as rep

ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
SOURCES=('toy_giveway','double_bottleneck','four_way_intersection')
def load(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(n,x):
    p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def con():
    c=sqlite3.connect(f'file:{ROOT}/shared_rollout_db/rollout.sqlite?mode=ro',uri=True);c.row_factory=sqlite3.Row;return c

def toy_scene(meta,h):
    sys.path.insert(0,'/home/zhihan/research/02_C1_Toy_GiveWay')
    from single_integrator.environment import Config,GiveWayEnv
    cfg=Config(**load(ROOT/'diagnostics/gphi_fixed_d_eta_predictor_v1/integrity_audit.json')['environment'])
    env=GiveWayEnv(cfg);st=np.load(meta['state_file'],allow_pickle=False)
    assert int(st['step'])==0
    p=st['positions'];v=st['velocities'];g=p+h[28:32].reshape(2,2)
    np.testing.assert_allclose(p,h[20:24].reshape(2,2),atol=2e-6,rtol=0)
    np.testing.assert_allclose(v,h[24:28].reshape(2,2),atol=2e-6,rtol=0)
    np.testing.assert_allclose(g,env.goals,atol=2e-6,rtol=0)
    # Archived Toy u_flow is speed bounded, not raw. Recover raw reference below
    # from the frozen policy/RNG before any model training; no environment step.
    flow=h[40:44].reshape(2,2)
    return {'positions':p.tolist(),'velocities':v.tolist(),'goals':env.goals.tolist(),'flow':flow.tolist(),
      'radius':[cfg.agent_radius]*2,'goal_tolerance':[cfg.goal_tolerance]*2,
      'obstacles':[{'a':a.tolist(),'b':b.tolist(),'radius':cfg.wall_radius,'curvature':0.,'interior_sign':0.} for a,b in env.walls],
      'policy_origin':[0.,0.],'policy_axis':[1.,0.],'policy_axis_required':True,
      'flow_committed':True,'remaining_fraction':1.,'remaining_seconds':cfg.max_steps*cfg.dt,
      'max_seconds':cfg.max_steps*cfg.dt,'dt':cfg.dt,'max_speed':cfg.max_speed,
      'wall_margin':cfg.wall_collision_margin,'agent_margin':cfg.agent_collision_margin,
      'monitor_active':cfg.terminate_on_deadlock,'progress_window_seconds':cfg.progress_window_seconds,
      'deadlock_hold_seconds':cfg.deadlock_hold_seconds,'progress_epsilon':cfg.progress_epsilon,
      'speed_epsilon_fraction':cfg.speed_epsilon_fraction,'monitor_elapsed':0.,'monitor_history_empty':True}

def recover_flow(states):
    import jax,jax.numpy as jnp
    jax.config.update('jax_enable_x64',True)
    from single_integrator.evaluate import load_policy
    from single_integrator.environment import bounded_nominal
    path=Path('/home/zhihan/research/02_C1_Toy_GiveWay/baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl')
    policy,_=load_policy(path)
    sample=jax.jit(jax.vmap(lambda obs,key:policy.sample_actions(obs[None],seed=key)[0]))
    errors=[]
    for r in states:
        s=r['physical'];p,v,g=(np.asarray(s[k]) for k in ('positions','velocities','goals'))
        obs=np.array([np.r_[p[i],v[i],g[i]-p[i],p[1-i]-p[i],v[1-i]-v[i]] for i in range(2)],np.float32)
        m=r['toy_metadata'];key=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(m['flow_seed']),m['rng_namespace']),0)
        raw=np.asarray(sample(jnp.asarray(obs[None]),key[None]))[0]
        err=float(np.max(np.abs(bounded_nominal(raw,s['max_speed'])-np.asarray(s['flow']))))
        if err>2e-5:raise ValueError(('Toy frozen reference mismatch',r['state_uid'],err))
        errors.append(err);s['flow']=raw.astype(float).tolist()
    dump('toy_reference_replay.json',{'states':len(states),'max_saved_bounded_flow_error':max(errors),'policy_sha256':sha(path),'environment_steps':0,'new_rollouts':0})

def build():
    protocol=load(OUT/'protocol.json');assert protocol['target_scene']=='ring_exchange'
    src=ROOT/'diagnostics/orthoflow3_structured_continuous_q_data_v1'
    struct=pq.read_table(src/'structured_pair_table.parquet').to_pylist()
    wide=pq.read_table(src/'sparse_matched_control.parquet').to_pylist()
    metadata={r['state_id']:r for r in load(src/'toy_state_split.json')['states']}
    chosen=[dict(r,split='train' if r['matrix_partition']=='TRAIN_TRAIN' else 'validation',origin='toy_structured') for r in struct if r['matrix_partition'] in ('TRAIN_TRAIN','VAL_VAL')]
    chosen += [dict(r,eta=[r['eta1'],r['eta2'],r['eta3']],split='train',origin='toy_wide') for r in wide]
    states={};pairs={};excluded=Counter()
    toy_ctl='ctl_df736b67f6410260d87812e0a76946af7152d147c9a0a25189d1908a92567b34'
    for r in chosen:
        sid=r['state_uid'];m=metadata[r['state_id']]
        if sid not in states:
            states[sid]={'state_uid':sid,'scenario':'toy_giveway','split':r['split'],'family':m['source_group'],
                'physical':toy_scene(m,np.asarray(r['h_raw'])),'toy_metadata':m}
        key=(sid,r['eta_uid'],r.get('controller_uid',toy_ctl))
        pairs.setdefault(key,dict(state_uid=sid,eta_uid=r['eta_uid'],controller_uid=key[2],eta=r['eta'],scenario='toy_giveway',split=r['split'],origin=r['origin'],requested_n=int(r['n_requested'] if 'n_requested' in r else r['n_trials'])))
    recover_flow(list(states.values()))
    data=ROOT/'datasets/orthoflow3_basin_dataset_v2_audited'
    source_rows=pq.read_table(data/'states.parquet',filters=[('scenario','in',list(SOURCES[1:]))]).to_pylist()
    by={r['state_uid']:r for r in source_rows}
    labels=pq.read_table(data/'eta_labels.parquet',filters=[('scenario','in',list(SOURCES[1:]))]).to_pylist()
    for r in labels:
        # Outcome-dependent early-stopped k/n is not an unbiased fixed-budget Q.
        if r['seed_count']<16 or r['numerical_failure_count']:
            excluded['partial_or_numerical_'+r['scenario']]+=1;continue
        sid=r['state_uid'];s=by[sid]
        if sid not in states:states[sid]={'state_uid':sid,'scenario':r['scenario'],'split':s['split'],'family':s['parent_episode_id'],'physical':rep.parse(s)}
        key=(sid,r['eta_uid'],r['controller_uid'])
        pairs.setdefault(key,dict(state_uid=sid,eta_uid=r['eta_uid'],controller_uid=r['controller_uid'],eta=rep.decode(r['eta_raw']),scenario=r['scenario'],split=s['split'],origin='historical_v2_full_count',requested_n=int(r['seed_count'])))
    c=con();final=[];controllers={}
    for key,r in pairs.items():
        rr=c.execute("SELECT rollout_uid,seed_key,success FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=? AND compatibility_quality='EXACT_REUSE' AND conflict_quarantined=0 AND numerical_failure=0",key).fetchall()
        if len(rr)<r['requested_n']:
            excluded['db_incomplete_'+r['scenario']]+=1;continue
        ct=dict(c.execute('SELECT * FROM controller_config WHERE controller_uid=?',(r['controller_uid'],)).fetchone())
        if ct['compatibility_quality'] not in ('EXACT_PROFILE','EXACT_REUSE'):
            excluded['controller_quality_'+str(ct['compatibility_quality'])]+=1;continue
        controllers[r['controller_uid']]=ct
        final.append({**r,'n':len(rr),'k':sum(x['success'] for x in rr),'q':sum(x['success'] for x in rr)/len(rr),'rollout_uids':[x['rollout_uid'] for x in rr],'seed_keys':[x['seed_key'] for x in rr]})
    c.close();used={r['state_uid'] for r in final};states=[r for sid,r in states.items() if sid in used]
    states.sort(key=lambda r:(r['scenario'],r['state_uid']));index={r['state_uid']:i for i,r in enumerate(states)}
    for r in final:r['state_index']=index[r['state_uid']]
    for sc in SOURCES:
        tr={r['family'] for r in states if r['scenario']==sc and r['split']=='train'}
        va={r['family'] for r in states if r['scenario']==sc and r['split']=='validation'}
        assert tr and va and not tr&va,(sc,'source family leakage')
    x=rep.batch([rep.entities(r['physical']) for r in states]);np.savez_compressed(OUT/'source_entities.npz',**x)
    pq.write_table(pa.Table.from_pylist(final),OUT/'source_pairs.parquet');dump('source_states.json',states)
    train_eta=np.asarray([r['eta'] for r in final if r['split']=='train'])
    low=train_eta.min(0);high=train_eta.max(0)
    dump('normalization.json',{'physical':'inherited fixed units; no fitted target statistics','eta_center':((low+high)/2).tolist(),'eta_radius':np.maximum((high-low)/2,1e-6).tolist(),'fit_scenes':SOURCES,'fit_split':'train'})
    counts={}
    for sc in SOURCES:
        counts[sc]={}
        for split in ('train','validation'):
            z=[r for r in final if r['scenario']==sc and r['split']==split];ss=[r for r in states if r['scenario']==sc and r['split']==split]
            counts[sc][split]={'states':len(ss),'families':len({r['family'] for r in ss}),'pairs':len(z),'eta':len({r['eta_uid'] for r in z}),'empirical_rate_ge15of16':sum(r['q']>=15/16 for r in z),'clear_failure':sum(r['q']<=.5 for r in z),'intermediate':sum(.5<r['q']<15/16 for r in z),'timestep_zero':sum(r['physical']['remaining_fraction']==1 for r in ss)}
    dump('source_data_manifest.json',{'counts':counts,'excluded':dict(excluded),'controllers':controllers,'state_family_overlap':0,'ring_rows':0,'ring_checkpoint_initialized':False,'target_labels_opened':False,'source_proposal_lineage':'Ring-trained source proposals excluded','new_rollout':0,'pairs_sha256':sha(OUT/'source_pairs.parquet'),'entities_sha256':sha(OUT/'source_entities.npz'),'representation_sha256':sha(rep.__file__),'protocol_sha256':sha(OUT/'protocol.json')})
    print(json.dumps(counts,indent=2));return states,x

if __name__=='__main__':build()
