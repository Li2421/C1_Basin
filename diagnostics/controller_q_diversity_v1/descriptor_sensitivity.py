"""Source-input-only sensitivity of the exact deployed controller descriptor.

PVN/PeVFA motivate checking the representation independently of return fitting.
This inexpensive screen queries functions, never simulator steps or success
labels. A local Jacobian is not a global identifiability proof.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import jax
from .experiment import HERE, read, write, freeze, sha
from .runtime import native_runtime, AnalyticField, controller_context
from .contact_features import measure

OUT = HERE/'checks/descriptor_sensitivity'
LOWER = np.array([.55, .2, -.85, 0., 0., 1.5])
UPPER = np.array([1.1, .9, .85, .6, .8, 4.5])


def prepare():
    p = read(HERE/'protocol.json')
    cases = []
    for s in p['states']:
        if s['study_split'] == 'final' or s['study_index'] >= 2:
            continue
        names = range(4) if s['study_split'] == 'train' else (4, 5)
        cases.extend(dict(scene=s['scenario'], state_uid=s['state_uid'],
                          split=s['study_split'], controller=f'analytic{k}') for k in names)
    OUT.mkdir(parents=True, exist_ok=True)
    primary = read(HERE/'models_frozen.json')['primary']
    fit = next(r for r in read(HERE/'models_frozen.json')['records']
               if r['kind'] == primary and r['seed'] == 17)
    freeze(OUT/'protocol.json', dict(cases=cases, normalization=str(Path(fit['path'])/'normalization.json'),
        finite_difference_relative_steps=[.001, .005], success_labels=False,
        final_states=False, simulator_steps=0, new_rollouts=0, model_changes=False,
        literature='literature/REVIEW.txt: independent policy-behavior representation tests',
        warning='Local finite-difference sensitivity, not global injectivity or Q sufficiency',
        code_sha256=sha(__file__)))
    print('Frozen source-only descriptor sensitivity cases', len(cases))


def run(scene):
    spec = read(OUT/'protocol.json');p = read(HERE/'protocol.json')
    assert spec['code_sha256'] == sha(__file__)
    states = {s['state_uid']:s for s in p['states']}
    norm = read(Path(spec['normalization']))
    mean, scale = np.array(norm['mean']), np.array(norm['scale'])
    base = native_runtime(scene)
    records, arrays = [], {}
    for ci, case in enumerate(c for c in spec['cases'] if c['scene'] == scene):
        s = states[case['state_uid']];name = case['controller']
        ctl = p['controllers'][scene][name]['payload']
        jax.config.update('jax_enable_x64', bool(ctl['jax_enable_x64']))
        theta = np.asarray(ctl['controller_parameters'])
        def query(parameters):
            rt = AnalyticField(base, parameters)
            nominal = controller_context(rt, s)
            contact, _ = measure(rt, s)
            return np.concatenate((nominal, contact), -1)
        def levels(raw):
            standardized = np.clip((raw-mean)/scale, -10., 10.)
            pooled = np.r_[standardized.mean(0), standardized.max(0)]
            clipped = np.clip(pooled, -3., 3.)
            return {'raw':raw.ravel(), 'standardized':standardized.ravel(),
                    'pooled':pooled, 'primary_phi':np.r_[clipped, clipped**2]}
        actual = query(theta)
        nominal = np.load(HERE/'features'/case['state_uid']/(name+'.npz'))['nominal']
        contact = np.load(HERE/'contact_features'/case['state_uid']/(name+'.npz'))['interaction']
        cached = np.concatenate((nominal, contact), -1)
        error = float(abs(actual-cached).max())
        np.testing.assert_allclose(actual, cached, atol=2e-6, rtol=2e-6)
        lv = levels(actual)
        record = dict(**case, cached_query_max_error=error,
            pooled_clipped_fraction=float(np.mean(abs(lv['pooled'])>=3.)),
            physical_agents=len(actual), sensitivities=[])
        for step in spec['finite_difference_relative_steps']:
            columns = {k:[] for k in lv}
            for j in range(6):
                left, right = theta.copy(), theta.copy()
                left[j] = max(LOWER[j], theta[j]-step*(UPPER[j]-LOWER[j]))
                right[j] = min(UPPER[j], theta[j]+step*(UPPER[j]-LOWER[j]))
                a, b = levels(query(left)), levels(query(right))
                width = (right[j]-left[j])/(UPPER[j]-LOWER[j])
                for key in columns:
                    columns[key].append((b[key]-a[key])/width)
            for key, values in columns.items():
                jac = np.stack(values, -1)/np.sqrt(len(values[0]))
                singular = np.linalg.svd(jac, compute_uv=False)
                record['sensitivities'].append(dict(representation=key, step=step,
                    singular_values=singular.tolist(), rank_at_absolute_1e_4=int(sum(singular>1e-4)),
                    condition=float(singular[0]/max(singular[-1],1e-15))))
                arrays[f'case{ci}_{key}_step{step}'] = jac
        records.append(record)
    np.savez_compressed(OUT/(scene+'_jacobians.npz'), **arrays)
    write(OUT/(scene+'.json'), dict(records=records, success_labels_read=False,
        simulator_steps=0, new_rollouts=0, no_model_changes=True,
        interpretation='Finite-difference local sensitivity only; does not establish global sufficiency for rollout Q'))
    print(scene, 'source descriptor sensitivity complete', len(records))


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('action', choices=['prepare','run'])
    parser.add_argument('--scene', choices=['toy_give_way','ring_exchange']);args = parser.parse_args()
    prepare() if args.action == 'prepare' else run(args.scene)
