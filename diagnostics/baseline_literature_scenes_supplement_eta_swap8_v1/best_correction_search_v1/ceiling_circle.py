"""Circle-five cases for the ceiling search: same branch as the sealed case_factory.make for old scenes,
with historical evidence logged under the search directory instead of the sealed parent."""
from pathlib import Path
import sys
S2 = Path(__file__).resolve().parent
sys.path.insert(0, str(S2))
from ceiling_common import *  # noqa: F401,F403
from case_factory import semantic, history
from audit_supplement import V2
from registry import case
import copy

CIRCLE = 'lit_circle_five'


def evidence2(p):
    p = Path(p).resolve(); h = sha(p)
    with (S2 / 'EVIDENCE_INPUTS.jsonl').open('a') as f:
        f.write(json.dumps(dict(path=str(p), sha256=h)) + '\n')
    return h


def make_circle(ctl, s, a, seed=0):
    old, templates = history(); key = semantic(CIRCLE, ctl, s, seed, a)
    if key in old:
        p = Path(old[key]['path']); evidence2(p); raw = read(p); c = copy.deepcopy(raw['case'])
        c.update(requested_action=a['key'], reuse_source=str(p), reuse_source_sha256=sha(p)); return c
    template = templates[canonical([CIRCLE, ctl['controller'], s['N'], s['index']])]
    payload = {k: v for k, v in template['controller_payload'].items() if k not in ('method', 'eta_b', 'theta')}
    if np.any(a['eta_a']) or np.any(a['eta_b']):
        payload.update(correction_semantics='contact_conflict_relative_gate_20261021', correction_revision_receipt_sha256=sha(V2 / 'CORRECTION_REVISION.receipt.json'),
                       correction_basis='orthoflow3_local_contact_gated', arrived_flag_broadcast=True)
    c = case(s, 'flow', a['key'], template['scenario_payload'], payload, seed, a['eta_a'], a['eta_b'], a['theta'], template['state_provenance'])
    c.update(safety_layer='distributed', rng_state_uid=template.get('rng_state_uid', template['state_uid']), requested_action=a['key']); return c


def make_any(scene, ctl, s, a, seed=0):
    return make_circle(ctl, s, a, seed) if scene == CIRCLE else make(scene, ctl, s, a, seed)
