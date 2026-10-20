"""Outcome-blind, deterministic Round20 geometry/state construction."""
from core import *
from shared_rollout_db.src.rollout_db import uid,canonical,connect
from collections import Counter

VARIANTS={'toy_044':('toy_give_way','corridor_width',.44),'toy_050':('toy_give_way','corridor_width',.50),'toy_056':('toy_give_way','corridor_width',.56),'db_060':('double_bottleneck','bottleneck_width',.60),'db_075':('double_bottleneck','bottleneck_width',.75),'db_090':('double_bottleneck','bottleneck_width',.90),'ring_075':('ring_exchange','obstacle_radius',.75),'ring_105':('ring_exchange','obstacle_radius',1.05)}
ORIGINAL={sc+'_original':(sc,None,None) for sc in SCENES}

def signature(v):
    try:
        a=[np.asarray(v[k],np.float64) for k in ('positions','velocities','goals')]
        if any(z.ndim!=2 or z.shape[1]!=2 or not np.isfinite(z).all() for z in a):return None
        return hashlib.sha256(canonical(dict(data=[(z.shape,z.tobytes().hex()) for z in a],timestep=float(v.get('timestep',0)))).encode()).hexdigest()
    except (KeyError,TypeError,ValueError):return None

def build():
    prereg();gate();old=read(E8/'protocol.json');from dataclasses import replace
    from single_integrator.environment import Config as ToyConfig
    from double_bottleneck.environment import Config as DBConfig
    from double_bottleneck.scenario import build_layout,INITIAL_REGIMES
    from diagnostics.double_bottleneck_initial_state_coverage.tools.generate_pools import _sample_spec
    from ring_exchange.environment import LocalFrameConfig,sample_instance
    from four_way_intersection.environment import Config as FourConfig
    from four_way_intersection.scenario import sample_initial_state,build_layout as four_layout
    cls={'toy_give_way':ToyConfig,'double_bottleneck':DBConfig,'ring_exchange':LocalFrameConfig,'four_way_intersection':FourConfig}
    geometries={};scenarios={};states=[];geometry_audit=[]
    for variant,(sc,key,val) in dict(VARIANTS,**ORIGINAL).items():
        original=old['environments'][sc]['config']['config'];cfg=cls[sc](**original);cfg=replace(cfg,**{key:val}) if key else cfg;cfgd=cfg.to_dict();geometries[variant]={key:val} if key else {}
        sid=uid('scenario',dict(parent=old['environments'][sc]['scenario_uid'],config=cfgd));scenarios[variant]=dict(scene=sc,scenario_uid=sid,config=cfgd)
        if sc=='toy_give_way':clearance=cfg.corridor_width-2*(cfg.agent_radius+cfg.wall_radius+cfg.wall_collision_margin)
        elif sc=='double_bottleneck':clearance=cfg.bottleneck_width-2*(cfg.agent_radius+cfg.wall_radius+cfg.wall_collision_margin)
        elif sc=='ring_exchange':clearance=cfg.outer_radius-cfg.obstacle_radius-6*cfg.agent_radius
        else:clearance=None
        geometry_audit.append(dict(variant=variant,config=cfgd,clearance=clearance,valid=(clearance is None or clearance>0)))
        for split,count in (('pilot',16),('probe',16),('evaluation',40)) if key else (('probe',16),):
            for i in range(count):
                alias=f'round20_{variant}_{split}_{i}';seed=int.from_bytes(hashlib.sha256(alias.encode()).digest()[:8],'little');rng=np.random.default_rng(seed)
                if sc=='toy_give_way':
                    x=rng.uniform(.55,1.05,2);y=rng.uniform(-.025,.025,2);pos=np.array([[-x[0],y[0]],[x[1],y[1]]],np.float32).astype(float);physical=dict(positions=pos.tolist(),velocities=[[0.,0.],[0.,0.]],goals=[[1.09,0.],[-1.09,0.]],timestep=0)
                elif sc=='double_bottleneck':
                    regime=INITIAL_REGIMES[i%3];spec=_sample_spec('test',regime,i,rng);physical=dict(positions=spec.positions.tolist(),velocities=spec.initial_velocities.tolist(),goals=build_layout(cfg).goals.tolist(),timestep=0)
                elif sc=='ring_exchange':
                    ins=sample_instance(seed,'test',cfg);physical=dict(positions=ins.positions.tolist(),velocities=ins.velocities.tolist(),goals=ins.goals.tolist(),timestep=0)
                else:
                    pos,vel,_=sample_initial_state(cfg,'development',i,base_seed=seed);physical=dict(positions=pos.tolist(),velocities=vel.tolist(),goals=four_layout(cfg).goals.tolist(),timestep=0)
                content=uid('content',dict(physical=physical,variant=variant));states.append(dict(uid=alias,state_uid=uid('state',dict(scenario=sid,content=content)),content_hash=content,scenario=sc,scenario_uid=sid,variant=variant,rng_namespace=int(hashlib.sha256(alias.encode()).hexdigest()[:8],16),physical=physical,family=alias,study_split=split,study_index=i,metadata=dict(draw_seed=seed,split=split,outcome_blind=True)))
    assert len(states)==8*72+4*16 and all(g['valid'] for g in geometry_audit)
    sig={};collisions=[]
    for s in states:
        h=signature(s['physical']);assert h is not None
        if h in sig:collisions.append(dict(type='new_overlap',a=sig[h],b=s['uid']))
        sig[h]=s['uid']
    with connect(True) as db:
        oldrows=db.execute('SELECT state_uid,content_hash,physical_state_json FROM state').fetchall()
    for r in oldrows:
        if r['state_uid'] in {s['state_uid'] for s in states} or r['content_hash'] in {s['content_hash'] for s in states}:collisions.append(dict(type='DB_identity',old=r['state_uid']))
        try:h=signature(json.loads(r['physical_state_json'])) if r['physical_state_json'] else None
        except ValueError:h=None
        if h in sig:collisions.append(dict(type='DB_physical',old=r['state_uid'],new=sig[h]))
    freeze(HERE/'GEOMETRIES.json',dict(geometries=geometries,scenarios=scenarios,checks=geometry_audit));freeze(HERE/'FRESH_STATES.json',states);freeze(HERE/'STATE_DEDUP_AUDIT.json',dict(utc=utc(),passed=not collisions,new=len(states),historical_DB_rows=len(oldrows),collisions=collisions,generator_sha256=sha(__file__)))
    assert not collisions,'STOP: state overlap; no adaptive replacement'
    print('Fresh states and geometry pass',len(states))
if __name__=='__main__':build()
