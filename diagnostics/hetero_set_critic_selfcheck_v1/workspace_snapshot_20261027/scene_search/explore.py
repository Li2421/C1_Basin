"""Scratch scene search (not evidence): build candidate scenes, run them through the
verified isolated harness, append results to JSONL. Official labels are produced later
by an implementation agent; nothing here touches the shared rollout database."""
import sys, os, json, math, time, argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import numpy as np

HERE = Path(__file__).resolve().parent
HARNESS = HERE / 'harness'
SEED = 2026102411
CTL_TYPES = ('centralized_sampling', 'gradient_mpc', 'distributed_shooting')


def swap_states(N, R, perturb, shift=None, lateral=None, count=12):
    """Robots on a circle (or on axis arms for N=4 corridors), goals on the unperturbed circle."""
    rng = np.random.default_rng(np.random.SeedSequence([SEED, N]))
    theta = np.arange(N) * 2 * np.pi / N
    initial = R * np.stack([np.cos(theta), np.sin(theta)], -1)
    shift = N // 2 if shift is None else shift
    goals = initial[(np.arange(N) + shift) % N]
    states = []
    for index in range(count):
        if index == 0:
            p = initial.copy()
        elif lateral is None:
            p = initial + rng.uniform(-perturb, perturb, initial.shape)
        else:
            along = rng.uniform(-perturb, perturb, N); side = rng.uniform(-lateral, lateral, N)
            radial = initial / R; tangent = np.stack([-radial[:, 1], radial[:, 0]], -1)
            p = initial + along[:, None] * radial + side[:, None] * tangent
        states.append(dict(index=index, N=N, positions=p.tolist(), goals=goals.tolist(), velocities=np.zeros_like(p).tolist()))
    return states


def build_scenes():
    sys.path.insert(0, str(HARNESS))
    import bootstrap  # noqa
    from common import uid
    from legacy_states import LiteratureConfig
    from bottleneck_family.scenario import Config, Geometry
    scenes = []

    def finish(name, factor, states, r, f, bc=None):
        longest = max(np.linalg.norm(np.asarray(s['goals']) - np.asarray(s['positions']), axis=-1).max() for s in states) / .5
        cfg = LiteratureConfig(name, states[0]['N'], r, .05, .5, int(math.ceil(f * longest / .05))).to_dict()
        walls = []
        if bc is not None:
            geom = Geometry(Config(**bc)); walls = geom.walls.tolist()
            for s in states: s['bottleneck_config'] = dict(bc)
        for s in states: s['scene'] = name
        scene = dict(name=name, family='eta_diverse_explore', factor=factor, config=cfg, walls=walls, states=states,
                     sampler_seed=SEED, scratch_exploration=True)
        if bc is not None: scene['bottleneck_config'] = dict(bc)
        scene['definition_sha256'] = uid('scene_definition', scene)
        scenes.append(scene)

    # T group: open 6-robot circle swap (radius .75), time factor x robot radius
    for f, r in [(3.0, .05), (2.5, .05), (3.0, .14), (2.5, .14)]:
        finish(f'xsc_x_time_f{int(f*10):02d}_r{int(round(r*100)):03d}', 'time_x_radius', swap_states(6, .75, .1), r, f)
    # C group: plus-shaped intersection, 4 robots on the arms at distance 1.0, corridor half-width w
    H = 1.6
    for w in (.10, .16, .25, .40):
        bc = Config(num_agents=4, half_length=H, half_height=H, barrier_x=(H - .25,), openings=(((0.0, 2 * H - .4),),),
                    barrier_thickness=.1, extra_rectangles=((w, w, H, H), (-H, w, -w, H), (-H, -H, -w, -w), (w, -H, H, -w)),
                    agent_radius=.05, dt=.05, max_speed=.5, max_steps=400).to_dict()
        lateral = max(0., min(.02, w - .05 - 4 / 600 - .005 - .005))
        finish(f'xsc_x_plus_w{int(round(w*100)):03d}', 'corridor_half_width', swap_states(4, 1.0, .1, lateral=lateral), .05, 4.0, bc)
    return scenes


_W = {}


def _init():
    sys.path.insert(0, str(HARNESS))
    os.environ['JAX_PLATFORMS'] = 'cpu'
    import bootstrap  # noqa
    from common import actions
    from design import controllers
    from scenes import make_new
    from execution import execute
    _W.update(actions=actions(), controllers=controllers(), make_new=make_new, execute=execute,
              scenes={s['name']: s for s in json.load(open(HERE / os.environ.get('SCENE_FILE', 'scenes_explore.json')))})


def _run(task):
    name, ci, si, ai, solo_agent, seed = task
    scene = _W['scenes'][name]; state = scene['states'][si]
    if solo_agent is not None:
        i = solo_agent
        state = dict(state, N=1, parent_N=state['N'], agent=i, positions=[state['positions'][i]], goals=[state['goals'][i]],
                     velocities=[state['velocities'][i]], index=state['index'] * 100 + i)
    c = _W['make_new'](scene, _W['controllers'][ci], state, _W['actions'][ai], seed)
    t = time.time(); r, _ = _W['execute'](c)
    return dict(scene=name, controller=ci, state=si, action=_W['actions'][ai]['key'], solo=solo_agent, seed=seed,
                success=int(r['success']), outcome=r['outcome'], length=int(r['episode_length']),
                numerical=int(r['numerical_failure']), collision=int(r['collision']), seconds=round(time.time() - t, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['build', 'capability', 'sweep'])
    ap.add_argument('--scenes', default='')
    ap.add_argument('--states', type=int, default=8)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--out', default='results.jsonl')
    ap.add_argument('--scenefile', default='scenes_explore.json')
    a = ap.parse_args()
    if a.stage == 'build':
        scenes = build_scenes(); json.dump(scenes, open(HERE / 'scenes_explore.json', 'w'))
        print('built', [s['name'] for s in scenes]); return
    os.environ['SCENE_FILE'] = a.scenefile
    scenes = json.load(open(HERE / a.scenefile))
    wanted = [s for s in scenes if not a.scenes or s['name'] in a.scenes.split(',')]
    done = set()
    out = HERE / a.out
    if out.exists():
        for line in open(out):
            d = json.loads(line); done.add((d['scene'], d['controller'], d['state'], d['action'], d['solo'], d['seed']))
    tasks = []
    for s in wanted:
        if a.stage == 'capability':
            for ci in range(3):
                for si in range(4):
                    for agent in range(s['states'][si]['N']):
                        for seed in (0, 1): tasks.append((s['name'], ci, si, 0, agent, seed))
        else:
            for ci in range(3):
                for si in range(min(a.states, len(s['states']))):
                    for ai in range(49): tasks.append((s['name'], ci, si, ai, None, 0))
    keys = ['safety'] + [f'eta{i}' for i in range(16)] + [f'sE{i:02d}' for i in range(32)]  # order of common.actions()
    todo = [t for t in tasks if (t[0], t[1], t[2], keys[t[3]], t[4], t[5]) not in done]
    print(f'{len(todo)} tasks ({len(tasks) - len(todo)} already done)', flush=True)
    t0 = time.time(); n = 0
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn'), initializer=_init) as pool, open(out, 'a') as f:
        futures = [pool.submit(_run, t) for t in todo]
        for fu in as_completed(futures):
            d = fu.result(); f.write(json.dumps(d) + '\n'); f.flush(); n += 1
            if n % 500 == 0: print(n, round(time.time() - t0, 1), 's', flush=True)
    print('done', n, round(time.time() - t0, 1), 's', flush=True)


if __name__ == '__main__':
    main()
