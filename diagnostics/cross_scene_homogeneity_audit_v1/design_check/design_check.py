"""Outcome-free design check: build candidate test scenes, compute frozen-procedure choices. No rollouts."""
import sys, json, math, numpy as np
from dataclasses import replace
sys.path.insert(0, '/home/zhihan/research/Basin_C1')
sys.path.insert(0, '/home/zhihan/.claude/jobs/be9710ae/tmp/homogeneity')
from bottleneck_family.scenario import Config, Geometry, build_instance
from bottleneck_family.__main__ import preset
from homogeneity import load, select, KEYS

def features(scene):   # copied verbatim from cross_scene_correction_v1/scenes.py
    c = scene['config']; n = c['num_agents']; r = c['agent_radius']; speed = c['max_speed']; walls = scene['walls']
    state_features = []; longest = []
    for s in scene['states']:
        p = np.asarray(s['positions']); g = np.asarray(s['goals']); d = g-p
        longest.append(float(np.linalg.norm(d, axis=-1).max()/speed))
        i, j = np.triu_indices(n, 1); rel = p[i]-p[j]; delta = d[i]-d[j]
        t = np.clip(-np.sum(rel*delta, axis=-1)/np.maximum(np.sum(delta*delta, axis=-1), 1e-30), 0., 1.)
        near = np.linalg.norm(rel+t[:, None]*delta, axis=-1)
        conflict = near < 3*r
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
        state_features.append([near.min()/r, near.mean()/r, 2*conflict.sum()/n,
            2*intersect.sum()/n, head_on, 2*np.sum(mid_distance<3*r)/n])
    free_area = scene['free_area']
    density = 0. if free_area is None else n*math.pi*r*r/free_area
    return [float(n), r, speed*c['dt']/r, c['max_steps']*c['dt']/max(longest), density,
        float(len(walls)), scene['min_opening_over_diameter'], float(scene['serial_walls']),
        float(scene.get('moving',False)), .25/speed if scene.get('moving') else 0.,
        .3/r if scene.get('moving') else 0., float(scene['diametric'])] + np.mean(state_features,axis=0).tolist()

LAYOUTS = [  # name, N, Config overrides ('preset:clutter' uses the master preset)
 ('xsp_long_tunnel_n4', 4, dict(barrier_thickness=3.0, openings=(((0.0,0.5),),))),
 ('xsp_clutter_n4', 4, 'preset:clutter'),
 ('xsp_wall_side_door_n4', 4, dict(openings=(((3.0,0.5),),))),
 ('xsp_zigzag_three_n4', 4, dict(barrier_x=(-3.0,0.0,3.0), openings=(((1.0,0.5),),((-1.0,0.5),),((1.0,0.5),)))),
 ('xsp_chicane_n4', 4, dict(barrier_x=(-0.6,0.6), openings=(((-0.5,0.5),),((0.5,0.5),)))),
 ('xsp_narrow_door_n4', 4, dict(openings=(((0.0,0.4),),))),
 ('xsp_low_room_n6', 6, dict(half_height=2.0, openings=(((0.0,0.5),),))),
 ('xsp_chamber_pillars_n6', 6, dict(barrier_x=(-1.5,1.5), openings=(((0.0,0.5),),((0.0,0.5),)), extra_rectangles=((-0.4,0.6,0.4,1.4),(-0.4,-1.4,0.4,-0.6)))),
 ('xsp_wide_wall_door_n6', 6, dict(openings=(((2.8,1.2),),))),
 ('xsp_unequal_doors_n6', 6, dict(openings=(((-2.0,0.4),(0.0,0.6),(2.0,0.9)),))),
 ('xsp_offset_tunnel_n6', 6, dict(barrier_thickness=2.0, openings=(((-2.0,0.5),),))),
]
SEED0 = 2026102600
scenes = []
for name, n, kw in LAYOUTS:
    base = preset('clutter', n) if kw == 'preset:clutter' else replace(Config(num_agents=n), **kw)
    base = replace(base, num_agents=n, max_steps=max(2000, 800*n), split='train'); geom = Geometry(base); states = []
    for k in range(16):
        inst = build_instance(replace(base, seed=SEED0+k))
        states.append(dict(positions=np.asarray(inst.positions).tolist(), goals=np.asarray(inst.goals).tolist()))
    area = 4*base.half_length*base.half_height - sum((x1-x0)*(y1-y0) for x0,y0,x1,y1 in geom.rectangles)
    scenes.append(dict(name=name, config=dict(num_agents=n, agent_radius=base.agent_radius, max_speed=base.max_speed, dt=base.dt, max_steps=base.max_steps),
        walls=geom.walls.tolist(), states=states, free_area=float(area),
        min_opening_over_diameter=min(w for os in base.openings for _,w in os)/(2*base.agent_radius), serial_walls=len(base.barrier_x), diametric=False))
# crowded random N=14: v2 revised split, training part, N=14 (first 16 in file order); walls/opening rule as v1's crowded scene
V2 = '/home/zhihan/research/Basin_C1/diagnostics/baseline_literature_scenes_distributed_v2/LEARNING_SPLIT_REVISED_20261021_lit_crowded_random.json'
v1 = {s['name']: s for s in json.load(open('/home/zhihan/research/Basin_C1/diagnostics/cross_scene_correction_v1/SCENES.json'))}
c8 = v1['lit_crowded_random']
st14 = [s for s in json.load(open(V2))['training'] if s['N'] == 14][:16]
scenes.append(dict(name='lit_crowded_random_n14', config=dict(c8['config'], num_agents=14, max_steps=333), walls=c8['walls'], states=st14,
    free_area=c8['free_area'], min_opening_over_diameter=c8['min_opening_over_diameter'], serial_walls=0, diametric=False))
print('crowded v1 config:', {k: c8['config'][k] for k in ('agent_radius','dt','max_speed','max_steps')}, 'n8 states', len(c8['states']), 'n14 states', len(st14))

names, fam, f, y, q, etas = load()
dedup = ['lit_narrow_passage','lit_crowded_random','lit_circle_five','lit_swap8_obstacles','xsc_swap_n4','xsc_swap_n12','xsc_swap_n8_dense','xsc_gap_n4','xsc_gap_n6','xsc_double_n4','xsc_offset_n4']
pools = {'dedup11': [names.index(n) for n in dedup], 'all18': list(range(len(names)))}
F = {s['name']: np.asarray(features(s)) for s in scenes}
X = f
print('\nfeature range check (new scene features outside training min/max):')
lo, hi = X.min(0), X.max(0)
FN = json.load(open('/home/zhihan/research/Basin_C1/diagnostics/cross_scene_correction_v1/FEATURES.json'))['names']
for s in scenes:
    out = [FN[k] for k in range(len(FN)) if not (lo[k]-1e-9 <= F[s['name']][k] <= hi[k]+1e-9)]
    print(f"  {s['name']:26s} outside: {out}")
print('\nfrozen-procedure choices (outcome-free):')
rows = {}
for s in scenes:
    r = {}
    for p, idx in pools.items():
        c = select(f[idx], q[idx], F[s['name']], etas)
        r[p] = dict(knn=KEYS[c['knn']], ridge=KEYS[c['ridge']], global_fixed=KEYS[c['global_fixed']], k=c['k'], penalty=c['penalty'])
    rows[s['name']] = r
    print(f"  {s['name']:26s} dedup11: knn {r['dedup11']['knn']:5s} ridge {r['dedup11']['ridge']:5s} global {r['dedup11']['global_fixed']} | all18: knn {r['all18']['knn']:5s} ridge {r['all18']['ridge']:5s} global {r['all18']['global_fixed']}")
json.dump(dict(features={k: v.tolist() for k, v in F.items()}, choices=rows), open('DESIGN_CHECK.json', 'w'), indent=1)
