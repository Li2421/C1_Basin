"""Reproduction gate: archived Toy flow_isotropic / flow_anisotropic factorial records at width 0.40,
through the native runtime and through the width-override path. Writes reproduction.json."""
import glob, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parent / 'controller_ff_factorial_repair_v1'


def main(n_per=3):
    import jax
    from .runtime import base_runtime, controller, rollout, CONTROLLERS
    proto = json.loads((ARCHIVE / 'protocol.json').read_text())
    states = {s['state_uid']: s for s in proto['states']}
    recs = []
    for f in sorted(glob.glob(str(ARCHIVE / 'raw' / '*.json'))):
        d = json.loads(Path(f).read_text())
        if d['case']['scene'] == 'toy_give_way' and d['case']['controller_name'] in CONTROLLERS:
            recs.append(d)
    recs.sort(key=lambda d: d['case']['rollout_uid'])
    picked = []
    for name in CONTROLLERS:
        seen = set()
        for d in recs:
            c = d['case']
            if c['controller_name'] == name and c['eta_index'] not in seen:
                seen.add(c['eta_index']); picked.append(d)
            if len(seen) >= n_per:
                break
    x64 = proto['controllers']['toy_give_way']['flow_isotropic']['payload']['jax_enable_x64']
    bases = {'native': base_runtime(None), 'override_0.40': base_runtime(0.40)}
    rows = []
    for d in picked:
        c = d['case']
        for label, base in bases.items():
            jax.config.update('jax_enable_x64', bool(x64))
            r = rollout(controller(base, c['controller_name']), states[c['state_uid']], c['eta'], c['seed'])
            checks = {k: r[k] == d[k] for k in ('success', 'deadlock', 'timeout', 'collision', 'numerical_failure', 'episode_length', 'outcome')}
            checks['first_action'] = r['first_action'] == d['first_action']
            checks['summary'] = json.loads(json.dumps(r['summary'])) == d['summary']
            rows.append(dict(path=label, controller=c['controller_name'], eta_index=c['eta_index'], seed=c['seed'],
                             archived_outcome=d['outcome'], checks=checks, all_equal=all(checks.values()), seconds=r['elapsed_seconds']))
            print(json.dumps({k: rows[-1][k] for k in ('path', 'controller', 'eta_index', 'archived_outcome', 'all_equal', 'seconds')}), flush=True)
    out = dict(all_equal=all(r['all_equal'] for r in rows), n=len(rows), mean_seconds=float(np.mean([r['seconds'] for r in rows])), rows=rows)
    (HERE / 'reproduction.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(dict(all_equal=out['all_equal'], mean_seconds=out['mean_seconds'])), flush=True)


if __name__ == '__main__':
    main()
