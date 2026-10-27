"""Specified four one-knob scenes; immutable base physical configurations."""
from common import *
from design import OLD,NEW_NAMES,SCENE_NAMES
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
 r=base_geometry_report(scene);r.update(joint_solvable='未认证',deadlock_resolution='未认证');return r

def prepare():
 from design import OLD_NAMES
 from legacy_states import LiteratureConfig
 assert read(HERE/'REPLAY_CHECK.json')['passed']
 old={s['name']:s for s in read_evidence(OLD/'SCENES.json')};out=[];checks={};SEED=2026102411
 for name,title in SCENE_NAMES.items():
  if name in OLD_NAMES:
   scene=copy.deepcopy(old[name]);scene['title']=title
   if name.startswith('xsc_swap_'):
    n=scene['config']['num_agents'];theta=np.arange(n)*2*np.pi/n;initial=(n/8)*np.stack([np.cos(theta),np.sin(theta)],-1);goals=-initial
    rng=np.random.default_rng(np.random.SeedSequence([SEED,n]));states=[]
    for index in range(28):
     p=initial.copy()if index==0 else initial+rng.uniform(-.1,.1,initial.shape)
     states.append(dict(scene=name,index=index,N=n,positions=p.tolist(),goals=goals.tolist(),velocities=np.zeros_like(p).tolist()))
    assert canonical(states[:12])==canonical(scene['states']);scene['states']=states
    checks[name]=dict(old12_exactly_equal=True,config_unchanged=True,definition_identity_preserved=True,new_indices=list(range(12,28)))
   scene['features']=features(scene);out.append(scene);continue
  if name.startswith('xsc_het_plus_'):
   w=.2 if name.endswith('020')else .4;n=4;r=.05;f=4.
   bc=Config(num_agents=4,half_length=1.6,half_height=1.6,barrier_x=(1.35,),openings=(((0.,2.8),),),barrier_thickness=.1,
    extra_rectangles=((w,w,1.6,1.6),(-1.6,w,-w,1.6),(-1.6,-1.6,-w,-w),(w,-1.6,1.6,-w)),agent_radius=.05,dt=.05,max_speed=.5,max_steps=400)
   geom=Geometry(bc);theta=np.arange(n)*2*np.pi/n;radial=np.stack([np.cos(theta),np.sin(theta)],-1);tangent=np.stack([-np.sin(theta),np.cos(theta)],-1)
   initial=radial.copy();goals=initial[(np.arange(n)+2)%n];rng=np.random.default_rng(np.random.SeedSequence([SEED,4]));states=[]
   for index in range(24):
    if index==0:p=initial.copy()
    else:
     along=rng.uniform(-.1,.1,4);side=rng.uniform(-.02,.02,4);p=initial+along[:,None]*radial+side[:,None]*tangent
    states.append(dict(scene=name,index=index,N=n,positions=p.tolist(),goals=goals.tolist(),velocities=np.zeros_like(p).tolist(),bottleneck_config=bc.to_dict()))
   area=3.2**2-4*(1.6-w)**2;opening=w/.05;diametric=True
  else:
   n=10;r=.13;f=2.5
   bc=Config(num_agents=10,half_length=1.2,half_height=1.2,barrier_x=(.95,),openings=(((0.,2.),),),barrier_thickness=.1,extra_rectangles=(),agent_radius=.05,dt=.05,max_speed=.5,max_steps=400)
   geom=Geometry(bc);rng=np.random.default_rng(np.random.SeedSequence([SEED,10,13]));clear=.13+.035;states=[];attempts=0
   from single_integrator.environment import point_segment_distance
   def usable(p):return bool(np.all(point_segment_distance(p,geom.walls[:,0],geom.walls[:,1])>clear))
   while len(states)<24:
    attempts+=1;p=[];goals=[]
    for _ in range(2000):
     x=rng.uniform(-1.2,1.2,2)
     if usable(x)and(not p or np.all(np.linalg.norm(np.asarray(p)-x,axis=-1)>2*.13+.1)):p.append(x)
     if len(p)==10:break
    if len(p)!=10:continue
    for _ in range(4000):
     x=rng.uniform(-1.2,1.2,2);i=len(goals)
     if usable(x)and np.linalg.norm(x-p[i])>=1.0 and(not goals or np.all(np.linalg.norm(np.asarray(goals)-x,axis=-1)>2*.13+.1)):goals.append(x)
     if len(goals)==10:break
    if len(goals)!=10:continue
    states.append(dict(scene=name,index=len(states),N=n,positions=np.asarray(p).tolist(),goals=np.asarray(goals).tolist(),velocities=np.zeros((n,2)).tolist(),bottleneck_config=bc.to_dict()))
   area=2.4**2-.04;opening=10.;diametric=False;checks[name]=dict(accepted=24,attempts=attempts,clearance=clear,minimum_start_goal_separation=1.)
  longest=max(np.linalg.norm(np.asarray(s['goals'])-s['positions'],axis=-1).max()for s in states)/.5
  cfg=LiteratureConfig(name,n,r,.05,.5,int(math.ceil(f*longest/.05))).to_dict()
  scene=dict(name=name,title=title,family='hetero_new',factor='cross_width'if n==4 else'box_random',config=cfg,walls=geom.walls.tolist(),states=states,bottleneck_config=bc.to_dict(),sampler_seed=[SEED,n]if n==4 else[SEED,10,13],free_area=area,min_opening_over_diameter=opening,serial_walls=0,diametric=diametric,moving=False)
  scene['definition_sha256']=uid('scene_definition',scene);scene['features']=features(scene);out.append(scene)
 reports={s['name']:geometry_report(s)for s in out};write(HERE/'SCENES.json',out);write(HERE/'GEOMETRY_BASE_REPORTS.json',reports);write(HERE/'GEOMETRY_REPORTS.json',reports)
 write(HERE/'SCENE_CONSTRUCTION_CHECK.json',dict(passed=True,checks=checks,counts={s['name']:len(s['states'])for s in out},new_state_rules='new hetero8..23; diametric12..27; literature existing16',no_invalid_state_replaced=True))
 return out
if __name__=='__main__':prepare()
