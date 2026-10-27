"""Prespecified physical scenes and outcome-free scene descriptors."""
from common import *
import math
import numpy as np
from dataclasses import replace
from bottleneck_family.scenario import Config, Geometry, build_instance
from bottleneck_family.__main__ import preset
from legacy_states import LiteratureConfig
from swap_geometry import config as swap_config
from single_integrator.environment import segment_distance

SEED = 2026102411
GAP_SEED = 2026102412
FEATURE_NAMES = ['N', 'radius', 'step_over_radius', 'horizon_over_straight_time', 'density',
    'wall_segments', 'min_opening_over_diameter', 'serial_walls', 'moving', 'mover_speed_over_speed',
    'mover_radius_over_radius', 'diametric', 'closest_approach_min', 'closest_approach_mean',
    'conflict_pairs_per_robot', 'intersections_per_robot', 'head_on_fraction', 'midpoint_crowding']


def literature_opening(name,walls,radius):
    if not walls:return 10.
    if name=='lit_circle_five':
        # The closed128-gon has inward support radius11m, hence width22m.
        return 22./(2*radius)
    w=np.asarray(walls,float);components=list(range(len(w)))
    def root(i):
        while components[i]!=i:i=components[i]
        return i
    for i in range(len(w)):
        for j in range(i):
            if np.min(np.linalg.norm(w[i,:,None,:]-w[j,None,:,:],axis=-1))<1e-12:
                components[root(i)]=root(j)
    gaps=[float(segment_distance(w[i,0],w[i,1],w[j,0],w[j,1]))for i in range(len(w))for j in range(i)if root(i)!=root(j)]
    assert gaps,('No defined opening descriptor',name)
    return min(gaps)/(2*radius)


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


def geometry_report(scene):
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


def prepare():
    assert read(HERE / 'REPLAY_CHECK.json')['passed'], 'Replay gate required'
    if (HERE / 'SCENES.json').exists():
        return read(HERE / 'SCENES.json')
    write(HERE / 'SAMPLING_FROZEN.receipt.json', dict(utc=utc(), diametric_seed=SEED, bottleneck_seed=GAP_SEED,
        no_resampling_invalid_states=True, diametric_formula='radius=N/8, plus N8 radius .7; goals fixed opposite unperturbed starts',
        initial_perturbation='Uniform [-.1,.1]^2, state0 unperturbed', horizon='max(2000,800*N), explicit axis4b prescription; Config.max_steps=10000 is a generic default, not an alternate N schedule',
        free_area='Physical workspace area minus rectangles; unbounded plane => infinity, density0',
        closest_approach='Shared normalized straight-path progress t in [0,1]', midpoint_crowding='Mean count of other path midpoints within 3r',
        seeds_for_analysis=2026102413, feature_names=FEATURE_NAMES))
    selection = read_evidence(SOURCE / 'SCREEN_SELECTION_PRIMARY_20261023.json')
    templates = read(HERE / 'HISTORY_CASES.json'); scenes = []
    for name, row in selection.items():
        template = next(c for c in templates if c['state']['scene']==name)
        cfg = template['scenario_payload']['config']; walls = template['scenario_payload'].get('walls')
        if walls is None:
            from states import walls as old_walls
            walls = old_walls(name).tolist()
        if name=='lit_circle_five':
            v=np.asarray(walls)[:,0];area=abs(np.sum(v[:,0]*np.roll(v[:,1],-1)-v[:,1]*np.roll(v[:,0],-1)))/2
        else:area=None
        scenes.append(dict(name=name,family='literature',factor='historical',config=cfg,walls=walls,
            states=row['training_states'],training_controllers=row['training_controllers'],free_area=area,
            min_opening_over_diameter=literature_opening(name,walls,cfg['agent_radius']),serial_walls=0,diametric=name.startswith('lit_swap8'),moving=name.endswith('moving')))
    for n,rad in [(4,.5),(6,.75),(10,1.25),(12,1.5),(8,.7)]:
        name=f'xsc_swap_n{n}' if n!=8 else 'xsc_swap_n8_dense'
        rng=np.random.default_rng(np.random.SeedSequence([SEED,n]));theta=np.arange(n)*2*np.pi/n
        initial=rad*np.stack([np.cos(theta),np.sin(theta)],-1);goals=-initial;states=[]
        for index in range(12):
            p=initial.copy() if index==0 else initial+rng.uniform(-.1,.1,initial.shape)
            states.append(dict(scene=name,index=index,N=n,positions=p.tolist(),goals=goals.tolist(),velocities=np.zeros_like(p).tolist()))
        longest=max(np.linalg.norm(np.asarray(s['goals'])-s['positions'],axis=-1).max() for s in states)/.5
        cfg=LiteratureConfig(name,n,.05,.05,.5,int(math.ceil(4*longest/.05))).to_dict()
        scenes.append(dict(name=name,family='diametric',factor='density' if n==8 else 'agent_count',config=cfg,walls=[],states=states,
            sampler_seed=SEED,free_area=None,min_opening_over_diameter=10.,serial_walls=0,diametric=True))
    manifest_path=ROOT/'datasets/gap1_eta_scaling_v4_global128/design_manifest.json';master=read_evidence(manifest_path)
    prefix_states=sorted([s for s in master['states']if s['N']==2 and s.get('physical_split',s['split'])=='train'],key=lambda s:s['state_uid'])[:8]
    # Seven training scenes plus one strictly held extrapolation scene.
    for kind,n,num in [('gap',2,8),('gap',4,8),('gap',6,8),('wide',4,8),('double',4,8),('offset',4,8),('three_doors',4,8),('extrap',10,12)]:
        name=f'xsc_gap_n{n}' if kind in ('gap','extrap') else f'xsc_{kind}_n4'
        base=Config(**master['scenario_info']['2']['config']) if kind in ('gap','extrap') else preset(kind,n)
        base=replace(base,num_agents=n,max_steps=max(2000,800*n),split='train');geom=Geometry(base);states=[]
        physical=sorted([s for s in master['states'] if s['N']==n and s.get('physical_split',s['split'])=='train'],key=lambda x:x['state_uid'])[:num] if n in (2,10) else None
        for index in range(num):
            generation_seed=prefix_states[index]['episode_seed']if kind=='gap'and n in (4,6)else GAP_SEED+index
            cfg=replace(base,seed=generation_seed)
            if physical is not None:
                p=np.asarray(physical[index]['positions']);g=np.asarray(physical[index]['goals']);source=physical[index]['state_uid']
            else:
                instance=build_instance(cfg);p=instance.positions;g=instance.goals;source=None
            states.append(dict(scene=name,index=index,N=n,positions=p.tolist(),goals=g.tolist(),velocities=np.zeros_like(p).tolist(),
                bottleneck_config=base.to_dict(),source_state_uid=source,physical_source=str(manifest_path) if physical is not None else 'bottleneck_family.scenario.build_instance'))
            if kind=='gap'and n in (4,6):
                assert np.array_equal(p[:2],prefix_states[index]['positions'])and np.array_equal(g[:2],prefix_states[index]['goals'])
                states[-1]['generation_seed']=generation_seed
        si=LiteratureConfig(name,n,base.agent_radius,base.dt,base.max_speed,base.max_steps,base.goal_tolerance,base.wall_radius,base.wall_collision_margin,base.agent_collision_margin).to_dict()
        area=4*base.half_length*base.half_height-sum((x1-x0)*(y1-y0)for x0,y0,x1,y1 in geom.rectangles)
        scenes.append(dict(name=name,family='extrapolation' if kind=='extrap' else 'bottleneck',factor='agent_count' if kind in ('gap','extrap') else 'geometry',
            config=si,walls=geom.walls.tolist(),states=states,bottleneck_config=base.to_dict(),sampler_seed=GAP_SEED,
            free_area=float(area),min_opening_over_diameter=min(w for os in base.openings for _,w in os)/(2*base.agent_radius),
            serial_walls=len(base.barrier_x),diametric=False))
        if kind=='gap'and n in (4,6):
            scenes[-1]['sampler_seed']=dict(rule='Corresponding prespecified N2 source episode_seed',
                seeds=[s['episode_seed']for s in prefix_states])
    for scene in scenes:
        scene['definition_sha256']=uid('scene_definition',scene)
        scene['features']=features(scene)
        if scene['family']!='literature':
            write(HERE/'geometry'/f"{scene['name']}.json",geometry_report(scene))
    assert len(scenes)==19 and sum(s['family']!='extrapolation'for s in scenes)==18
    write(HERE/'SCENES.json',scenes)
    write(HERE/'FEATURES.json',dict(names=FEATURE_NAMES,scenes={s['name']:s['features']for s in scenes}))
    return scenes


if __name__=='__main__':
    prepare()
