"""Learned planner as a harness nominal controller.  install() makes runtime_new.Nominal return LearnedActor for rows
whose type_code starts with 'learned_'; every other row keeps the original controller, so all existing episode loops
(eta menu, pipeline eta, right-hand rules, nominal only) run learned planners unchanged."""
import os, sys, json, hashlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import learned_obs as LO

_POLICIES = {}


def policy(ckpt):
    if ckpt not in _POLICIES:
        import learned_models as LM
        _POLICIES[ckpt] = LM.Policy(ckpt)
    return _POLICIES[ckpt]


def rng_for(key):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(np.asarray(key, np.uint32).tobytes()).digest()[:8], 'little'))


def episode_bounds(env):
    walls = np.asarray(env.walls, float).reshape(-1, 2, 2)
    lo, hi = LO.bounds(walls, env.positions, env.goals)
    return walls, lo - 0.3, hi + 0.3


class LearnedActor:
    def __init__(self, row):
        self.row = row; self.policy = policy(row['checkpoint']); self.geo = None

    def action(self, env, key):
        c = env.config
        if self.geo is None:
            self.geo = episode_bounds(env)
        walls, lo, hi = self.geo
        obs, _ = LO.observe(c.scene, env.positions, env.velocities, env.goals, walls, c.agent_radius, env.instance.geometry.margin,
                            c.max_speed, c.goal_tolerance, lo, hi)
        return self.policy(obs, rng_for(key)) * c.max_speed


def install():
    import runtime_new as RN
    if getattr(RN, '_learned_installed', False):
        return
    orig = RN.Nominal

    def factory(row):
        return LearnedActor(row) if str(row.get('type_code', '')).startswith('learned_') else orig(row)
    RN.Nominal = factory
    RN._learned_installed = True


def row(name, ckpt):
    meta = json.load(open(os.path.join(ckpt, 'meta.json')))
    sha = hashlib.sha256(open(os.path.join(ckpt, 'params.msgpack'), 'rb').read()).hexdigest()
    return dict(controller=name, type='learned planner (' + meta['kind'] + ')', type_code='learned_' + meta['kind'], family='learned',
                cohort='hetset_learned_v1', parameters='[]', checkpoint=os.path.abspath(ckpt), checkpoint_sha256=sha, recipe='imitation of prioritized_astar0')
