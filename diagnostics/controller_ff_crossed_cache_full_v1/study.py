"""Reuse actual FF labels to test mechanism coverage, without new rollouts.

Historical DB commits its initial Flow random draw; Four resumes a physical
snapshot with an empty monitor. Those semantics are retained and represented,
not relabelled as the current t0 stochastic execution program.
"""
import argparse,copy,gzip,json,sqlite3,sys
from collections import Counter
from pathlib import Path
import numpy as np

from diagnostics.controller_four_scene_mechanism_v1.study import HERE as FOUR,read,write,freeze,sha
from diagnostics.controller_q_diversity_v1.runtime import FIELD

HERE=Path(__file__).resolve().parent
OLD=FIELD/'field_pipeline_v1'
DIVERSITY=HERE.parent/'controller_q_diversity_v1'
SCENES=('toy_give_way','double_bottleneck','four_way_intersection','ring_exchange')


def prepare():
    four=read(FOUR/'protocol.json');old=read(OLD/'protocol.json')
    checks={k:sha(FIELD/k)==v for k,v in old['controller_hashes'].items()}
    assert all(checks.values()),checks
    rows=[];d=np.load(FOUR/'combined.npz')
    state={s['state_uid']:s for s in four['states']}
    for i in range(len(d['scene'])):
        rows.append(dict(state=state[str(d['state_uid'][i])],controller=str(d['controller'][i]),
            kind='matrix',source_index=i,split=str(d['split'][i])))
    native=np.load(DIVERSITY/'data/source_augmented128/dataset.npz')
    chosen=[]
    for sc in ('toy_give_way','ring_exchange'):
        for sp in ('train','validation'):
            target=[s for s in four['states'] if s['scenario']==sc and s['study_split']==sp]
            for s in target:
                ii=np.flatnonzero((native['state_uid']==s['state_uid'])&(native['controller']=='flow'))
                assert len(ii)==1,(s['state_uid'],len(ii))
                i=int(ii[0]);assert native['split'][i]==sp
                rows.append(dict(state=s,controller='native_flow',kind='native_current',source_index=i,split=sp))
                chosen.append(i)
    c=sqlite3.connect(f'file:{OLD}/field_rollout.sqlite?mode=ro',uri=True)
    c.row_factory=sqlite3.Row
    profiles={**read(OLD/'preflight.json')['controller_uids'],
              'four_way_intersection':read(OLD/'precision_preflight.json')['controller_uids']['four_way_intersection']}
    raw=[];duplicates=0
    for r in old['states']:
        s=r['state'];sc=s['scenario']
        if sc not in ('double_bottleneck','four_way_intersection'):continue
        eta=np.asarray(r['etas'],np.float64);keys={e.tobytes():j for j,e in enumerate(eta)}
        assert len(eta)==12 and len(keys)==12
        y=np.full((12,16),-2,np.int8);collision=np.zeros_like(y,bool)
        phase='source' if sc=='double_bottleneck' else 'precision'
        for row in c.execute('SELECT * FROM continuation WHERE state_uid=? AND controller_uid=? AND phase=?',
                             (s['uid'],profiles[sc],phase)):
            row=dict(row);j=keys.get(np.asarray(json.loads(row['eta_json']),np.float64).tobytes())
            if j is None:continue
            k=row['seed'];assert 0<=k<16
            value=-1 if row['numerical_unresolved'] else row['success']
            if y[j,k]!=-2:
                assert y[j,k]==value and collision[j,k]==bool(row['collision']);duplicates+=1
            else:y[j,k]=value;collision[j,k]=bool(row['collision']);raw.append(row)
        assert np.all(y!=-2),(s['uid'],y)
        s=copy.deepcopy(s);s['state_uid']=s['uid'];s['study_split']=s['metadata']['split']
        rows.append(dict(state=s,controller='native_flow',kind='native_historical',split=s['study_split'],
                         controller_uid=profiles[sc],eta=eta.tolist(),outcome=y.tolist(),collision=collision.tolist(),
                         phase=phase,committed_initial_key=sc=='double_bottleneck',
                         empty_monitor=True,jax_x64=sc=='double_bottleneck'))
    c.close()
    # State IDs differ across databases: check the physical payload, not IDs.
    physical_splits={}
    for r in rows:
        s=r['state'];key=(s['scenario'],json.dumps(s['physical'],sort_keys=True))
        assert key not in physical_splits or physical_splits[key]==r['split'],'TRAIN/VAL physical overlap'
        physical_splits[key]=r['split']
    freeze(HERE/'rows.json',rows)
    with gzip.open(HERE/'reused_historical_seed_outcomes.jsonl.gz','wt') as f:
        for row in raw:f.write(json.dumps(row,sort_keys=True)+'\n')
    protocol=dict(schema='crossed_native_ff_cache_full_v1',new_rollouts=0,
        hypothesis='Current homogeneous simple-controller panels lack useful navigation mechanisms; task-capable native controllers may provide missing source state/controller/eta support.',
        conditions=['matrix_only','native_eta_prior','native_actual'],
        control='native_eta_prior replaces only source-TRAIN native targets by same-scene exact-eta mean across observed physical states. All inputs, optimization budget and source-validation labels are otherwise identical. This retains scene/controller eta priors including historical candidate-proposal selection. Rare eta with only one source state cannot remove the incidental state association; audit the counts.',
        representation='Existing physical masked entities, with deployment-known time/dynamics/empty-monitor constants restored; static current/future PROGRAM branch compact fields. No scene/controller ID, no physical rollout or H20 input.',
        historical_semantics='DB fixed initial Flow draw is queried as the known current program branch, future base Flow stochastic. Four continues saved timestep with empty monitor and correct remaining horizon. Never treat either as fresh-t0 labels.',
        candidates='ALL12 original historical candidates per DB/Four state; original K16 for Toy/Ring native cache; four pre-frozen matrix probes. No proposal change or new labels. Diagnostic coverage study, not a formal benchmark pool.',
        anchor_probe_superseded='The common-anchor-only DB panel had zero success and erased its useful native supervision. Retain that negative-control artifact; this full-cache stage restores all original candidate coverage.',
        source_selection='Historical source DB/Four states were selected using older terminal-label strata. Valid source/development evidence, not natural-distribution final confirmation.',
        normalization='Source TRAIN only',checkpoint='Source VAL: equal-weight matrix and native probability MSE; matrix controller-contrast MSE weight0.5, then source selection regret and NLL.',
        steps=6000,batch_size=120,seeds=[17,23,41],learning_rate=.0008,weight_decay=.0001,validation_every=200,
        source_only=['gradients','normalization','native prior estimation','checkpoint selection'],
        folds=['joint','held_toy','held_db','held_four','held_ring'],
        target_status='DEVELOPMENT only. Opened target labels do not enter fits. New method claims need untouched fresh confirmation.',
        controller_code_matches=checks,physical_train_val_overlap=0,
        reused_historical_unique_outcomes=len(raw),duplicate_phase_rows=duplicates,
        reused_current_native_outcomes=int(np.sum(native['success'][chosen]+native['failure'][chosen]+native['unknown'][chosen])),
        rows_by_kind=dict(Counter(r['kind'] for r in rows)),
        literature=['https://proceedings.mlr.press/v119/raileanu20a.html','https://arxiv.org/abs/1602.02867'],
        literature_boundary='PDVF factorizes policy/environment and uses target interactions; VIN uses explicit planning structure. Neither proves that current local descriptors identify arbitrary zero-shot scene values.')
    freeze(HERE/'protocol.json',protocol)
    print(json.dumps({k:protocol[k] for k in ('new_rollouts','reused_historical_unique_outcomes','reused_current_native_outcomes','rows_by_kind')}),flush=True)


def full_physical(scene,physical,config,committed=False):
    from diagnostics.physical_state_critic_formal_v1 import physical_entities as phys
    from diagnostics.orthoflow3_unified_representation_v1 import representation as rep
    p=copy.deepcopy(physical);t=p['timestep'];p['timestep']=0
    s=phys.physical_scene(scene,p);cfg=config
    s.update(remaining_fraction=1-t/cfg['max_steps'],remaining_seconds=(cfg['max_steps']-t)*cfg['dt'],
        max_seconds=cfg['max_steps']*cfg['dt'],dt=cfg['dt'],max_speed=cfg['max_speed'],flow_committed=committed,
        wall_margin=cfg.get('wall_collision_margin',cfg.get('collision_margin',0.)),agent_margin=cfg.get('agent_collision_margin',0.),
        monitor_active=bool(cfg.get('terminate_on_deadlock',False)),progress_window_seconds=cfg.get('progress_window_seconds',0.),
        deadlock_hold_seconds=cfg.get('deadlock_hold_seconds',0.),progress_epsilon=cfg.get('progress_epsilon',0.),
        speed_epsilon_fraction=cfg.get('speed_epsilon_fraction',0.),monitor_elapsed=0.,monitor_history_empty=True)
    x=rep.entities(s);x['agents'][:,3:7]=0.;x['agents'][:,10:15]=0.
    return x


def features(index,workers):
    import jax
    sys.path.insert(0,str(FIELD))
    from field_pipeline_v1 import runner,precision_audit
    from diagnostics.controller_q_diversity_v1.runtime import controller_context,native_runtime
    from diagnostics.controller_q_diversity_v1.features import field_context
    from diagnostics.controller_q_diversity_v1.train import compact_field
    rows=read(HERE/'rows.json');p=read(FOUR/'protocol.json');rtcache={}
    oldenv=runner.make_env
    for i in range(index,len(rows),workers):
        r=rows[i];s=r['state'];sc=s['scenario'];dest=HERE/'features'/f'{i:04d}.npz'
        if dest.exists():continue
        if r['kind']=='native_historical':
            if sc not in rtcache:
                rtcache[sc]=runner.make_runtime(sc) if sc=='double_bottleneck' else native_runtime(sc)
            rt=rtcache[sc];jax.config.update('jax_enable_x64',r['jax_x64'])
            expected=runner.uid(sc) if sc=='double_bottleneck' else precision_audit.controller_uid(sc)
            assert expected==r['controller_uid'],(expected,r['controller_uid'])
            class Program:
                def __init__(self,fixed):self.fixed=fixed
                def __getattr__(self,k):return getattr(rt,k)
                def make_env(self,st):return oldenv(rt,st)
                def nominal(self,env,key,traced=False):
                    if self.fixed:key=runner.key_for({'state':s,'seed':0},0)
                    return rt.nominal(env,key,traced)
            def no_step(*a,**kw):raise RuntimeError('Physical step forbidden during descriptor extraction')
            rt.env_cls.step=no_step
            pieces=[]
            for fixed in (r['committed_initial_key'],False):
                prog=Program(fixed)
                raw=np.concatenate((controller_context(prog,s),field_context(prog,s)),-1)
                pieces.append(np.asarray(compact_field(raw)))
            feature=np.concatenate(pieces,-1)
            config=rt.config.to_dict()
        else:
            if r['kind']=='matrix':a=np.load(FOUR/'features'/s['state_uid']/(r['controller']+'.npz'))
            else:a=np.load(DIVERSITY/'features'/s['state_uid']/'flow.npz')
            f=np.asarray(compact_field(np.concatenate((a['nominal'],a['field']),-1)))
            feature=np.concatenate((f,f),-1)
            config=p['environments'][sc]['config']['config']
        physical=full_physical(sc,s['physical'],config,r.get('committed_initial_key',False))
        dest.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest,controller=feature,**{'entity_'+k:v for k,v in physical.items()})
    write(HERE/f'features_worker{index}.json',dict(complete=True,physical_steps=0,new_rollouts=0))


def assemble():
    from diagnostics.orthoflow3_unified_representation_v1 import representation as rep
    rows=read(HERE/'rows.json');a=np.load(FOUR/'combined.npz');b=np.load(DIVERSITY/'data/source_augmented128/dataset.npz')
    entities=[];feature=[];out=[]
    for i,r in enumerate(rows):
        x=np.load(HERE/'features'/f'{i:04d}.npz');entities.append({k[7:]:x[k] for k in x.files if k.startswith('entity_')})
        feature.append(np.pad(x['controller'],((0,4-len(x['controller'])),(0,0))))
        if r['kind']=='native_historical':
            y=np.asarray(r['outcome']);eta=np.pad(r['eta'],((0,4),(0,0)))
            s=np.pad((y==1).sum(-1),(0,4));f=np.pad((y==0).sum(-1),(0,4));u=np.pad((y==-1).sum(-1),(0,4))
        else:
            d=a if r['kind']=='matrix' else b;j=r['source_index'];eta=d['eta'];s=d['success'][j];f=d['failure'][j];u=d['unknown'][j]
        out.append(dict(eta=eta,success=s,failure=f,unknown=u,scene=r['state']['scenario'],
            state_uid=r['state']['state_uid'],controller=r['controller'],kind=r['kind'],split=r['split']))
    data={'entity_'+k:v for k,v in rep.batch(entities).items()};data['feature']=np.asarray(feature)
    for k in out[0]:data[k]=np.asarray([r[k] for r in out])
    dest=HERE/'dataset.npz'
    if not dest.exists():np.savez_compressed(dest,**data)
    freeze(HERE/'dataset_audit.json',dict(dataset_sha256=sha(dest),rows=len(rows),new_rollouts=0,
        label_outcomes=int((data['success']+data['failure']+data['unknown']).sum()),physical_features_have_no_ID=True,
        conditional_first_program_and_remaining_horizon_retained=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','features','assemble']);p.add_argument('--index',type=int,default=0);p.add_argument('--workers',type=int,default=8);args=p.parse_args()
    if args.stage=='features':features(args.index,args.workers)
    else:globals()[args.stage]()
