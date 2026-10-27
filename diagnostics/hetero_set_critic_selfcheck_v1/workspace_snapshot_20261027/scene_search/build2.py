"""Second batch of scratch candidate scenes: bounded dense random crossings (circle-five-like
need for low repulsion?) and intermediate plus-intersection widths."""
import sys, json, math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'harness'))
import bootstrap  # noqa
from common import uid
from legacy_states import LiteratureConfig
from bottleneck_family.scenario import Config, Geometry
sys.path.insert(0, str(HERE))
from explore import swap_states, SEED

scenes = []


def finish(name, factor, states, r, f, bc):
    longest = max(np.linalg.norm(np.asarray(s['goals']) - np.asarray(s['positions']), axis=-1).max() for s in states) / .5
    cfg = LiteratureConfig(name, states[0]['N'], r, .05, .5, int(math.ceil(f * longest / .05))).to_dict()
    geom = Geometry(Config(**bc)); walls = geom.walls.tolist()
    for s in states: s['scene'] = name; s['bottleneck_config'] = dict(bc)
    scene = dict(name=name, family='eta_diverse_explore', factor=factor, config=cfg, walls=walls, states=states,
                 sampler_seed=SEED, scratch_exploration=True, bottleneck_config=dict(bc))
    scene['definition_sha256'] = uid('scene_definition', scene)
    scenes.append(scene)


def seg_dist(p, a, b):
    ab = b - a; t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-30), 0, 1); return np.linalg.norm(p - (a + t * ab))


# bounded dense random crossing: square arena [-H,H]^2, N robots, random starts and goals
H = 1.2
for r in (.12, .18):
    bc = Config(num_agents=5, half_length=H, half_height=H, barrier_x=(H - .25,), openings=(((0.0, 2 * H - .4),),),
                barrier_thickness=.1, extra_rectangles=(), agent_radius=.05, dt=.05, max_speed=.5, max_steps=400).to_dict()
    walls = np.asarray(Geometry(Config(**bc)).walls, float)
    clear = r + .005 + .03

    def ok(p):
        return all(seg_dist(p, w[0], w[1]) > clear for w in walls)
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 5, int(r * 100)]))
    base_states = []
    while len(base_states) < 12:
        pts = []
        for _ in range(2000):
            q = rng.uniform(-H, H, 2)
            if ok(q) and all(np.linalg.norm(q - x) > 2 * r + .1 for x in pts): pts.append(q)
            if len(pts) == 5: break
        goals = []
        for _ in range(4000):
            q = rng.uniform(-H, H, 2)
            i = len(goals)
            if i < 5 and ok(q) and all(np.linalg.norm(q - x) > 2 * r + .1 for x in goals) and np.linalg.norm(q - pts[i]) >= 1.0: goals.append(q)
            if len(goals) == 5: break
        if len(pts) == 5 and len(goals) == 5:
            base_states.append(dict(index=len(base_states), N=5, positions=np.asarray(pts).tolist(), goals=np.asarray(goals).tolist(),
                                    velocities=np.zeros((5, 2)).tolist()))
    for f in (2.5, 4.0):
        st = json.loads(json.dumps(base_states))
        finish(f'xsc_x_box_r{int(round(r*100)):03d}_f{int(f*10):02d}', 'bounded_dense_crossing', st, r, f, bc)

# intermediate plus-intersection widths
Hp = 1.6
for w in (.20, .32):
    bc = Config(num_agents=4, half_length=Hp, half_height=Hp, barrier_x=(Hp - .25,), openings=(((0.0, 2 * Hp - .4),),),
                barrier_thickness=.1, extra_rectangles=((w, w, Hp, Hp), (-Hp, w, -w, Hp), (-Hp, -Hp, -w, -w), (w, -Hp, Hp, -w)),
                agent_radius=.05, dt=.05, max_speed=.5, max_steps=400).to_dict()
    lateral = max(0., min(.02, w - .05 - 4 / 600 - .005 - .005))
    finish(f'xsc_x_plus_w{int(round(w*100)):03d}', 'corridor_half_width', swap_states(4, 1.0, .1, lateral=lateral), .05, 4.0, bc)

json.dump(scenes, open(HERE / 'scenes_explore2.json', 'w'))
print('built', [(s['name'], s['config']['max_steps']) for s in scenes])
