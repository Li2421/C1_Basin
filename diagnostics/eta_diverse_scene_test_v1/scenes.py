"""Specified four one-knob scenes; immutable base physical configurations."""
from common import *
from design import OLD,NEW_NAMES
from bottleneck_family.scenario import Config,Geometry
import numpy as np,math
from single_integrator.environment import segment_distance

def make_new(scene, ctl, state, action, seed=0):
    cfg = scene['config']
    if state['N'] != cfg['num_agents']:
        cfg = dict(cfg, num_agents=state['N'])
    scenario = dict(version='cross_scene_si_v1', config=cfg, walls=scene['walls'],
                    bottleneck_config=scene.get('bottleneck_config'), factor=scene['factor'],
                    plant='frozen_swap_step', scene_definition_sha256=scene['definition_sha256'])
    sources = {**{row['target']: row['target_sha256'] for row in read(HERE / 'BACKEND_BUILD.json')['sources']},
               **{str(HERE / n): sha(HERE / n) for n in ('scene_environment.py', 'controllers_new.py', 'runtime_new.py')}}
    payload = dict(nominal=ctl, execution_sources=sources, safety_layer='distributed', sensing_radius=.5,
        correction_semantics='contact_conflict_relative_gate_20261021', correction_basis='orthoflow3_local_contact_gated',
        arrived_flag_broadcast=True, precision='jax_x64_false', restoration='speed_only_then_safety_minimax')
    c = make_case(state, 'cross_scene_si_v1', action['key'], scenario, payload, seed, action['eta_a'],
        dict(scene=scene['name'], sampler_seed=scene.get('sampler_seed'), index=state['index'],
             physical_source=state.get('physical_source'), source_state_uid=state.get('source_state_uid')))
    c['cross_scene_new'] = True
    return c

def base_geometry_report(scene):
    rows = []
    cfg = scene['config']; r = cfg['agent_radius']
    geom = Geometry(Config(**scene['bottleneck_config'])) if scene.get('bottleneck_config') else None
    for s in scene['states']:
        p = np.asarray(s['positions']); g = np.asarray(s['goals']); ii, jj = np.triu_indices(s['N'], 1)
        sep = bool(np.all(np.linalg.norm(p[ii] - p[jj], axis=-1) > 2*r))
        goal_sep = bool(np.all(np.linalg.norm(g[ii] - g[jj], axis=-1) >= 2*r))
        valid = sep and goal_sep and np.isfinite(p).all() and np.isfinite(g).all()
        paths = [None]*s['N']
        if geom:
            valid = bool(valid and geom.valid_points(p).all() and geom.valid_points(g).all())
            paths = [geom.path(a, b) for a, b in zip(p, g)]
            paths_ok = all(v is not None for v in paths)
        else:
            paths_ok = True; paths = [np.stack([a, b]) for a, b in zip(p, g)]
        rows.append(dict(index=s['index'], geometric_valid=bool(valid), single_robot_paths=paths_ok,
            path_witnesses=[None if v is None else v.tolist() for v in paths], joint_solvable='not_certified',
            controller_success='not_evaluated', deadlock_resolution='not_established', eligible=bool(valid and paths_ok),
            capacity='N valid starts and goals witnessed; no joint certification', invalid_state_replaced=False,
            shared_bottleneck_pairs=((s['N']+1)//2)*(s['N']//2) if geom else None))
    return dict(scene=scene['name'], family=scene['family'], factor=scene['factor'], config=cfg, states=rows,
        unchanged_radius_speed_dt=True, topology_certificate='Single robot only',
        min_opening_over_diameter=scene['min_opening_over_diameter'])

def features(scene):
    c = scene['config']; n = c['num_agents']; r = c['agent_radius']; speed = c['max_speed']; walls = scene['walls']
    state_features = []; longest = []
    for s in scene['states']:
        p = np.asarray(s['positions']); g = np.asarray(s['goals']); d = g-p
        longest.append(float(np.linalg.norm(d, axis=-1).max()/speed))
        i, j = np.triu_indices(n, 1); rel = p[i]-p[j]; delta = d[i]-d[j]
        t = np.clip(-np.sum(rel*delta, axis=-1)/np.maximum(np.sum(delta*delta, axis=-1), 1e-30), 0., 1.)
        near = np.linalg.norm(rel+t[:, None]*delta, axis=-1)
        conflict = near < 3*r
        # Segment intersection includes collinear overlap, computed independently of shared arrival time.
        def cross(a,b):return a[...,0]*b[...,1]-a[...,1]*b[...,0]
        ab = cross(d[i], d[j]); offset = p[j]-p[i]
        ta = cross(offset, d[j])/np.where(abs(ab)>1e-14, ab, 1.)
        tb = cross(offset, d[i])/np.where(abs(ab)>1e-14, ab, 1.)
        intersect = (abs(ab)>1e-14)&(ta>=0)&(ta<=1)&(tb>=0)&(tb<=1)
        collinear = (abs(ab)<=1e-14)&(abs(cross(offset,d[i]))<=1e-14)
        along1 = np.sum(offset*d[i],axis=-1)/np.maximum(np.sum(d[i]*d[i],axis=-1),1e-30)
        along2 = np.sum((g[j]-p[i])*d[i],axis=-1)/np.maximum(np.sum(d[i]*d[i],axis=-1),1e-30)
        intersect |= collinear & (np.minimum(along1,along2)<=1)&(np.maximum(along1,along2)>=0)
        cosine = np.sum(d[i]*d[j],axis=-1)/np.maximum(np.linalg.norm(d[i],axis=-1)*np.linalg.norm(d[j],axis=-1),1e-30)
        head_on = np.mean(cosine[conflict]<-.5) if conflict.any() else 0.
        mid = (p+g)/2; mid_distance = np.linalg.norm(mid[i]-mid[j],axis=-1)
        # Midpoint crowding is per robot mean count within 3r; distances above are divided by r.
        state_features.append([near.min()/r, near.mean()/r, 2*conflict.sum()/n,
            2*intersect.sum()/n, head_on, 2*np.sum(mid_distance<3*r)/n])
    free_area = scene['free_area']
    density = 0. if free_area is None else n*math.pi*r*r/free_area
    return [float(n), r, speed*c['dt']/r, c['max_steps']*c['dt']/max(longest), density,
        float(len(walls)), scene['min_opening_over_diameter'], float(scene['serial_walls']),
        float(scene.get('moving',False)), .25/speed if scene.get('moving') else 0.,
        .3/r if scene.get('moving') else 0., float(scene['diametric'])] + np.mean(state_features,axis=0).tolist()

def geometry_report(scene):
    report=base_geometry_report(scene)
    report['unchanged_radius_speed_dt']=scene['name']!='xsc_div_size_r014'
    report['only_prespecified_knob_changed']=True;report['base_scene']='xsc_swap_n6'
    report['joint_solvable']='未认证';report['deadlock_resolution']='未认证'
    return report

def prepare():
    assert read(HERE/'REPLAY_CHECK.json')['passed']
    all_scenes=read_evidence(OLD/'SCENES.json');base=next(s for s in all_scenes if s['name']=='xsc_swap_n6')
    assert len(base['states'])==12 and base['config']['agent_radius']==.05 and base['config']['dt']==.05 and base['config']['max_speed']==.5
    longest=max(np.linalg.norm(np.asarray(s['goals'])-s['positions'],axis=-1).max()for s in base['states'])/.5
    assert base['config']['max_steps']==math.ceil(4*longest/.05)
    scenes=[];reports={}
    for name,title in NEW_NAMES.items():
        cfg=copy.deepcopy(base['config']);cfg['scene']=name;states=copy.deepcopy(base['states'])
        walls=[];bottleneck=None;factor='robot_radius'if name=='xsc_div_size_r014'else'time_limit'if name=='xsc_div_time_f20'else'central_obstacle'if name=='xsc_div_center_s024'else'offset_obstacle'
        if name=='xsc_div_size_r014':cfg['agent_radius']=.14
        elif name=='xsc_div_time_f20':cfg['max_steps']=int(math.ceil(2*longest/.05))
        else:
            rectangle=(-.24,-.24,.24,.24)if name=='xsc_div_center_s024'else(.30-.16,-.16,.30+.16,.16)
            try:
                bc=Config(num_agents=6,half_length=2.,half_height=2.,barrier_x=(1.7,),openings=(((0.,3.6),),),barrier_thickness=.1,
                    extra_rectangles=(rectangle,),agent_radius=.05,dt=.05,max_speed=.5,max_steps=base['config']['max_steps'])
                geom=Geometry(bc);walls=geom.walls.tolist();bottleneck=bc.to_dict()
            except Exception as exc:
                write(HERE/'STOP.receipt.json',dict(utc=utc(),reason='OBSTACLE_CONSTRUCTION_REJECTED',scene=name,type=type(exc).__name__,message=str(exc)));raise
        for original,state in zip(base['states'],states):
            state['scene']=name;state['physical_source']=str(OLD/'SCENES.json');state['source_state_uid']=uid('physical',{k:original[k]for k in('positions','velocities','goals')})
            if bottleneck:state['bottleneck_config']=copy.deepcopy(bottleneck)
            for field in('positions','goals','velocities'):assert canonical(state[field])==canonical(original[field])
        scene=dict(name=name,title=title,family='eta_diverse',factor=factor,config=cfg,walls=walls,states=states,
            sampler_seed=base['sampler_seed'],base_scene='xsc_swap_n6',base_scene_definition_sha256=base['definition_sha256'],base_states_unchanged=True,
            free_area=None if bottleneck is None else 16.-sum((r[2]-r[0])*(r[3]-r[1])for r in geom.rectangles),
            min_opening_over_diameter=10. if bottleneck is None else min(w for openings in bc.openings for _,w in openings)/(2*cfg['agent_radius']),
            serial_walls=0 if bottleneck is None else 1,diametric=True,moving=False)
        if bottleneck:scene['bottleneck_config']=bottleneck
        scene['definition_sha256']=uid('scene_definition',scene);scene['features']=features(scene)
        reports[name]=geometry_report(scene);scenes.append(scene)
    write(HERE/'SCENES.json',scenes);write(HERE/'GEOMETRY_BASE_REPORTS.json',reports);write(HERE/'GEOMETRY_REPORTS.json',reports)
    write(HERE/'SCENE_CONSTRUCTION_CHECK.json',dict(passed=True,new_scenes=4,states_per_scene=12,no_resampling=True,
        base_config=base['config'],longest_straight_time=float(longest),time_limit_steps=math.ceil(2*longest/.05),
        only_physical_knob_changes={s['name']:{k:dict(base=base['config'][k],new=v)for k,v in s['config'].items()if k!='scene'and v!=base['config'][k]}for s in scenes},
        geometric_valid_counts={n:sum(s['geometric_valid']for s in r['states'])for n,r in reports.items()},single_robot_path_counts={n:sum(s['single_robot_paths']for s in r['states'])for n,r in reports.items()}))
    print('Four scenes constructed from unchanged 12-state base.',flush=True)
    return scenes
