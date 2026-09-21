"""CPU audit of the generic R_CERT aggregation; no robot certificates."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from single_integrator.c1.risk.r_cert import aggregate, atom_penalty


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('audit output must be new')
    jax.config.update('jax_enable_x64', True)

    scales = [np.array([[.7, 1.3], [.9, .6]]),
              np.array([[1.1, .8], [.5, 1.4]])]
    base = jnp.array([-.4, .2, .3, -.1, 2., 2.5, 1.7, 3.])
    direction = jnp.sin(jnp.arange(8, dtype=jnp.float64)+.3)

    def stable_score(vector):
        return aggregate([vector[:4].reshape(2, 2), vector[4:].reshape(2, 2)],
                         scales, kappa=.37)['R_CERT']

    ad = float(jnp.vdot(jax.grad(stable_score)(base), direction))
    finite_differences = []
    for step in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        fd = float((stable_score(base+step*direction)-stable_score(base-step*direction))/(2*step))
        finite_differences.append(dict(step=step, finite_difference=fd,
            absolute_error=abs(fd-ad), relative_error=abs(fd-ad)/max(abs(ad), 1e-30)))

    # R_CERT=max(h(-c0),h(-c1)) at c0=c1=0. The direction (1,-1)
    # switches the selected event at the tie, so no ordinary derivative exists.
    tie = jnp.zeros(2)
    tie_direction = jnp.array([1., -1.])
    tie_score = lambda vector: aggregate(
        [vector[0].reshape(1, 1), vector[1].reshape(1, 1)],
        [np.ones((1, 1)), np.ones((1, 1))], kappa=.2)['R_CERT']
    step = 1e-6
    centre = float(tie_score(tie))
    tie_report = dict(
        value=centre,
        jax_selected_subgradient=float(jnp.vdot(jax.grad(tie_score)(tie), tie_direction)),
        right_directional_difference=float((tie_score(tie+step*tie_direction)-centre)/step),
        left_directional_difference=float((tie_score(tie-step*tie_direction)-centre)/(-step)),
        classification='outer_max_event_tie_nondifferentiable')

    margins = jnp.array([-100., -10., 0., 10., 100.])
    atom_gradient = jax.vmap(jax.grad(lambda c: atom_penalty(c, 1.)))(margins)
    saturation = [dict(margin=float(c), penalty=float(p), derivative=float(g))
                  for c,p,g in zip(margins, atom_penalty(margins, jnp.ones_like(margins)), atom_gradient)]

    result = dict(scope='generic aggregation only; no C1 certificate margins or rollout',
        stable_region=dict(value=float(stable_score(base)), autodiff_directional=ad,
                           finite_differences=finite_differences),
        exact_outer_max_tie=tie_report,
        atom_saturation=saturation,
        numerical=dict(all_finite=bool(np.isfinite(np.array(
            [stable_score(base), ad, *[x['finite_difference'] for x in finite_differences],
             *[x['penalty'] for x in saturation], *[x['derivative'] for x in saturation]])).all())),
        limitations=[
            'No task-specific certificate inclusion is established.',
            'No projection, closed-loop trajectory, or operational outcome is evaluated.',
            'The exact outer max remains nonsmooth at event ties; no smoothing or straight-through rule is used.'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
