"""Stage 0: exact reproduction of archived DB goal-reference rollouts at the
original width (no override) and through the override path at width=0.50.

Compares outcome flags, episode_length, first_action (exact float equality) and
the full env summary with the archived raw records. Writes reproduction.json.
"""
import json, glob, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parent / 'controller_four_scene_mechanism_v1'


def main(n=8):
    import jax
    from .runtime import base_runtime, controller
    proto = json.loads((ARCHIVE / 'protocol.json').read_text())
    states = {s['state_uid']: s for s in proto['states']}
    recs = []
    for f in sorted(glob.glob(str(ARCHIVE / 'raw' / '*.json'))):
        d = json.loads(Path(f).read_text())
        if d['case']['scene'] == 'double_bottleneck':
            recs.append(d)
    # deterministic, outcome-blind spread over controllers / eta / seeds
    recs.sort(key=lambda d: d['case']['rollout_uid'])
    picked, seen = [], set()
    for d in recs:
        k = (d['case']['controller_name'], d['case']['eta_index'])
        if k not in seen:
            seen.add(k); picked.append(d)
        if len(picked) >= n:
            break
    payload_x64 = proto['controllers']['double_bottleneck']['isotropic']['payload']['jax_enable_x64']
    out = []
    bases = {'native': base_runtime(None), 'override_0.50': base_runtime(0.50)}
    for d in picked:
        c = d['case']; s = states[c['state_uid']]
        for label, base in bases.items():
            jax.config.update('jax_enable_x64', bool(payload_x64))
            rt = controller(base, c['controller_name'])
            r = rollout_and_compare(rt, s, c, d)
            r['path'] = label
            out.append(r)
            print(json.dumps({k: r[k] for k in ('path', 'controller', 'eta_index', 'seed', 'all_equal', 'seconds')}), flush=True)
    res = dict(n_cases=len(picked), all_equal=all(r['all_equal'] for r in out), rows=out,
               mean_seconds=float(np.mean([r['seconds'] for r in out])))
    (HERE / 'reproduction.json').write_text(json.dumps(res, indent=2) + '\n')
    print(json.dumps(dict(all_equal=res['all_equal'], mean_seconds=res['mean_seconds'])), flush=True)


def rollout_and_compare(rt, s, c, d):
    from .runtime import rollout
    r = rollout(rt, s, c['eta'], c['seed'])
    checks = {k: r[k] == d[k] for k in ('success', 'deadlock', 'timeout', 'collision', 'numerical_failure', 'episode_length', 'outcome')}
    checks['first_action'] = r['first_action'] == d['first_action']
    checks['summary'] = json.loads(json.dumps(r['summary'])) == d['summary']
    return dict(controller=c['controller_name'], eta_index=c['eta_index'], seed=c['seed'],
                checks=checks, all_equal=all(checks.values()), seconds=r['elapsed_seconds'],
                archived_outcome=d['outcome'], archived_length=d['episode_length'])


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
