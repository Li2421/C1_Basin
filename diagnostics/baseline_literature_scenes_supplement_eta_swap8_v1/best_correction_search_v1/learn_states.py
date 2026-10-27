"""Circle states 32-99 for the state-conditioned selector test: v1 pool (same sampler as v2's 0-47), validated with the
sealed flow_scene.validate; cases built from the same controller's template with the state, provenance and RNG identity
replaced (v2 rule: RNG from the state's own uid)."""
from pathlib import Path
import sys
S2 = Path(__file__).resolve().parent
sys.path.insert(0, str(S2))
from ceiling_common import *  # noqa: F401,F403
from ceiling_circle import make_circle, CIRCLE, evidence2
from case_factory import history
from flow_scene import validate
from registry import case
from functools import lru_cache

V1_POOL = ROOT / 'diagnostics/baseline2_literature_scenes_v1/FLOW_STATES_UNCHECKED.json'
TRAIN_NEW = tuple(range(32, 80))
HELDOUT_NEW = tuple(range(80, 100))


@lru_cache(None)
def pool():
    evidence2(V1_POOL)
    states = {s['index']: s for s in read(V1_POOL) if s['scene'] == CIRCLE}
    split = read(ROOT / 'diagnostics/baseline_literature_scenes_distributed_v2/LEARNING_SPLIT_lit_circle_five.json')
    for s in split['training'] + split['evaluation']:
        assert {k: states[s['index']][k] for k in ('positions', 'goals', 'velocities', 'N')} == {k: s[k] for k in ('positions', 'goals', 'velocities', 'N')}, s['index']
    return states


def new_states(indices):
    out = []
    for i in indices:
        s = {k: pool()[i][k] for k in ('scene', 'index', 'N', 'positions', 'goals', 'velocities')}
        v = validate(s); assert v['eligible'], ('ineligible circle state', i, v)
        out.append(s)
    return out


def make_circle_any(ctl, s, a, seed=0):
    if s['index'] < 32:
        return make_circle(ctl, s, a, seed)
    _, templates = history(); template = templates[canonical([CIRCLE, ctl['controller'], s['N'], 16])]
    payload = {k: v for k, v in template['controller_payload'].items() if k not in ('method', 'eta_b', 'theta')}
    if np.any(a['eta_a']) or np.any(a['eta_b']):
        from audit_supplement import V2
        payload.update(correction_semantics='contact_conflict_relative_gate_20261021', correction_revision_receipt_sha256=sha(V2 / 'CORRECTION_REVISION.receipt.json'),
                       correction_basis='orthoflow3_local_contact_gated', arrived_flag_broadcast=True)
    provenance = dict(template['state_provenance'], state_index=s['index'])
    c = case(s, 'flow', a['key'], template['scenario_payload'], payload, seed, a['eta_a'], a['eta_b'], a['theta'], provenance)
    c.update(safety_layer='distributed', rng_state_uid=c['state_uid'], requested_action=a['key']); return c
