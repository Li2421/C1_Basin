"""Supplemental, pre-outcome frozen decomposition of Q and eta interactions.

No model selection or fitting. Orthogonal projections separate controller
strength and state difficulty from the interactions required for eta selection.
Independent seed halves estimate component energy without squared-label bias.
Numerical censoring bounds for prediction skill need no missing-at-random claim.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
AXES = ('state', 'controller', 'eta')
COMPONENTS = [tuple(v) for n in range(1, 4)
              for v in itertools.combinations(AXES, n)]


def project(x, components):
    """Projection onto one balanced-panel ANOVA component; last axes S,C,E."""
    z = np.asarray(x, float)
    for name, axis in zip(AXES, (-3, -2, -1)):
        if name not in components:
            z = z.mean(axis, keepdims=True)
    for name, axis in zip(AXES, (-3, -2, -1)):
        if name in components:
            z = z - z.mean(axis, keepdims=True)
    return z


def summarize(x):
    return [float(v) for v in np.quantile(x, [.025, .975])]


def self_check():
    rng = np.random.default_rng(29)
    y = rng.normal(size=(9, 2, 16))
    pred = rng.normal(size=y.shape)
    parts = [project(y, c) for c in COMPONENTS]
    np.testing.assert_allclose(sum(parts) + y.mean(), y, atol=1e-12)
    np.testing.assert_allclose(sum(np.mean(p**2) for p in parts), y.var(), atol=1e-12)
    for c in COMPONENTS:
        p = project(pred, c)
        t = project(y, c)
        np.testing.assert_allclose(np.mean(t*t-(t-p)**2), np.mean(2*y*p-p*p), atol=1e-12)
    # Controller strength cannot generate state-by-eta interaction.
    strength = np.broadcast_to(np.array([.1, .9])[None, :, None], y.shape)
    assert np.max(abs(project(strength, ('state', 'eta')))) < 1e-12
    assert np.max(abs(project(strength, AXES))) < 1e-12
    print('Orthogonal decomposition contracts passed')


def run():
    assert (ROOT/'evaluation_complete.json').exists()
    d = np.load(ROOT/'data/final/dataset.npz')
    predictions = np.load(ROOT/'confirmation_predictions.npz')
    frozen = json.loads((ROOT/'models_frozen.json').read_text())
    models = sorted(set(r['kind'] for r in frozen['records']))
    energies, skills = [], []
    bootstrap_arrays = {}
    for scene in sorted(set(d['scene'])):
        ids = np.flatnonzero(d['scene'] == scene)
        n = len(ids)//2
        success = d['seed_success'][ids].reshape(n, 2, 16, 16)
        unknown = d['seed_unknown'][ids].reshape(n, 2, 16, 16)
        disc, ev = success[..., :8].mean(-1), success[..., 8:].mean(-1)
        lo, hi = success.mean(-1), (success+unknown).mean(-1)
        boot = np.random.default_rng(2026100952).integers(n, size=(20000, n))
        for component in COMPONENTS:
            name = '_x_'.join(component)
            energy = np.mean(project(disc, component)*project(ev, component))
            samples = np.mean(project(disc[boot], component)*project(ev[boot], component), axis=(-3,-2,-1))
            energies.append(dict(scene=str(scene),component=name,physical_states=n,
                cross_half_energy=float(energy),CI95=summarize(samples),
                estimand='Verified operational success; negative estimates retained, not true negative variance'))
            bootstrap_arrays[str(scene)+'_'+name+'_energy'] = samples
            for model in models:
                pred = predictions[model][:,ids].mean(0).reshape(n,2,16)
                projected = project(pred, component)
                a, b = 2*lo*projected-projected**2, 2*hi*projected-projected**2
                p = project(pred[boot], component)
                sa, sb = 2*lo[boot]*p-p**2, 2*hi[boot]*p-p**2
                lower = np.minimum(sa,sb).mean((-3,-2,-1))
                upper = np.maximum(sa,sb).mean((-3,-2,-1))
                skill = sa.mean((-3,-2,-1))
                skills.append(dict(scene=str(scene),model=model,component=name,physical_states=n,
                    verified_success_skill=float(a.mean()),verified_success_CI95=summarize(skill),
                    censor_worst_skill=float(np.minimum(a,b).mean()),censor_worst_CI95=summarize(lower),
                    censor_best_skill=float(np.maximum(a,b).mean()),censor_best_CI95=summarize(upper)))
                prefix = str(scene)+'_'+model+'_'+name
                bootstrap_arrays[prefix+'_skill'] = skill
                bootstrap_arrays[prefix+'_worst'] = lower
                bootstrap_arrays[prefix+'_best'] = upper
    out = dict(energies=energies,skills=skills,primary=frozen['primary'],
        scope='Supplemental descriptive mechanism analysis, no final-data model or metric selection',
        sampling='20000 physical-state bootstraps, both held controllers paired; conditional on these two controllers',
        energy_note='Independent discovery/evaluation seed halves; all attempts denominator. Numerical unknown is unresolved physical success, so energy is operational, while skill has full-support censoring bounds.',
        skill_note='Squared-error reduction versus zero projected interaction: 2<E[Q],P(pred)>-||P(pred)||^2. Positive skill requires predicting the actual interaction, not merely a controller main effect.',
        final_models_unchanged=True)
    (ROOT/'interaction_decomposition.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(ROOT/'interaction_decomposition_bootstrap.npz',**bootstrap_arrays)
    print('Supplemental Q interaction decomposition complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--check',action='store_true')
    args = parser.parse_args()
    self_check() if args.check else run()
