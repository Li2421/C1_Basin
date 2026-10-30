"""Literal old nominal/RHR loops with output hooks; no new RHR mechanism."""
from core import *
import ast
import time
import traceback
from learned_setup import install

HETERO = HERE / 'inputs/diagnostics/hetero_set_critic_selfcheck_v1/workspace_snapshot_20261027/hetset'


def original_function(filename, name):
    path = HETERO / filename
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(functions) == 1
    namespace = dict(hashlib=hashlib)
    if filename == 'rhr.py':
        rule = [node for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'RULE' for t in node.targets)]
        assert len(rule) == 1
        exec(compile(ast.Module(body=rule, type_ignores=[]), str(path), 'exec'), namespace)
        assert namespace['RULE'] == dict(steps=40, threshold=.05, speed=.35)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def run(request):
    install()
    engine = request['controller_payload']['engine']
    goal_rhr = engine == 'copied_goal_rhr'
    assert engine in ('copied_goal_rhr', 'copied_nominal_only')
    assert not goal_rhr or request['benchmark'] == 'A', 'B uses the existing runtime reference interface.'
    function = original_function('rhr.py' if goal_rhr else 'nominal.py', 'run_rhr' if goal_rhr else 'run_nominal')
    old_env, old_actor = RN.SwapEnv, RN.Nominal
    created = []

    class RecordedEnv(old_env):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.saved_positions = [self.positions.copy()]
            self.saved_velocities = [self.velocities.copy()]
            self.saved_actions = []
            self.saved_clearance = []
            self.saved_seconds = []
            self.record_clock = time.monotonic()
            created.append(self)

        def step(self, action):
            result = super().step(action)
            now = time.monotonic()
            self.saved_seconds.append(now - self.record_clock)
            self.saved_positions.append(self.positions.copy())
            self.saved_velocities.append(self.velocities.copy())
            self.saved_actions.append(np.asarray(action).copy())
            self.saved_clearance.append(result[2]['swept_clearance'])
            self.record_clock = time.monotonic()
            return result

    def actor(row):
        value = old_actor(row)
        if created:
            created[-1].record_clock = time.monotonic()
        if request['benchmark'] == 'B' and request['scenario_payload']['route_guidance']:
            # Same declared route-goal input hook as the frozen B main loop.
            # The literal no-safety loop below still supplies the bounded action
            # to the unchanged plant and tests the unchanged terminal goals.
            from routes import route_geometry, guidance
            original_action = value.action

            def action(env, key):
                goals = env.goals.copy()
                guides, _, _ = guidance(route_geometry(env.instance.geometry), env.positions, goals, True)
                if value.kind not in ('prioritized_astar', 'conflict_based_search'):
                    env.goals = guides
                try:
                    return original_action(env, key)
                finally:
                    env.goals = goals
            value.action = action
        return value

    start = time.monotonic()
    error = None
    RN.SwapEnv, RN.Nominal = RecordedEnv, actor
    try:
        result = function(request, variant='goal') if goal_rhr else function(request)
    except Exception as exc:
        if isinstance(exc, PermissionError):
            raise
        error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        result = dict(success=0, outcome='unknown')
    finally:
        RN.SwapEnv, RN.Nominal = old_env, old_actor
    config = request['scenario_payload']['config']
    count = request['state']['N']
    if created:
        env = created[-1]
        arrays = dict(positions=np.asarray(env.saved_positions), velocities=np.asarray(env.saved_velocities),
                      actions=np.asarray(env.saved_actions).reshape(-1, count, 2), goals=np.asarray(request['state']['goals']),
                      swept_clearance=np.asarray(env.saved_clearance), step_seconds=np.asarray(env.saved_seconds))
        collision = int(env.collided)
        length = len(env.saved_actions)
        final = env.positions
    else:
        arrays = dict(positions=np.asarray([request['state']['positions']]), velocities=np.asarray([request['state']['velocities']]),
                      actions=np.zeros((0, count, 2)), goals=np.asarray(request['state']['goals']),
                      swept_clearance=np.zeros(0), step_seconds=np.zeros(0))
        collision, length = 0, 0
        final = np.asarray(request['state']['positions'])
    result.update(collision=collision, timeout=int(result['outcome'] == 'timeout'), numerical_failure=int(error is not None),
                  error=error, episode_length=length, elapsed_seconds=time.monotonic() - start,
                  final_goal_distances=np.linalg.norm(np.asarray(request['state']['goals']) - final, axis=1),
                  minimum_swept_clearance=float(arrays['swept_clearance'].min()) if length else None,
                  safety_layer='distributed' if goal_rhr else 'none', deadlock=0,
                  profiling_scope='Output-hook timings for literal archived function; not archived original wall-clock measurements.',
                  literal_source=str((HETERO / ('rhr.py' if goal_rhr else 'nominal.py')).relative_to(HERE)))
    return result, arrays
