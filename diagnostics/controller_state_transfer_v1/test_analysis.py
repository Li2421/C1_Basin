"""Code-path test on synthetic labels (planted, width-consistent state x eta structure). Not a result."""
import numpy as np
from . import analysis as A


def main():
    A.ENT_STEPS, A.MLP_STEPS, A.SEEDS, A.PERM = 500, 500, (17,), 10
    p = A.read(A.HERE / 'protocol.json')
    arch = {s['state_uid']: s for s in A.read(A.ARCHIVE / 'protocol.json')['states']}
    states = p['train_states'] + p['new_states']; phys = [arch[u]['physical'] for u in states]
    W = sorted(float(w) for w in p['widths']); roles = {float(w): r for w, r in p['widths'].items()}
    Xs = np.array([np.concatenate([np.ravel(ph[k]) for k in ('positions', 'goals', 'velocities')]) for ph in phys])
    eta = np.array([p['eta'][str(j)] for j in p['eta_indices']], float)
    rng = np.random.default_rng(0)
    x0 = Xs[:, 0]                                                            # agent-0 start x: planted state variable
    z = (x0 - x0.mean()) / x0.std()
    prob = np.zeros((len(W), len(states), 2, 8))
    for wi, w in enumerate(W):
        for e in range(8):
            prob[wi, :, :, e] = (0.5 + 0.3 * np.tanh(z * (1 if e % 2 else -1)) + 0.1 * (w - 0.5))[:, None]
    Y = (rng.random(prob.shape + (4,)) < prob[..., None]).astype(float)
    D = dict(Y=Y, W=W, roles=roles, phys=phys, Xs=Xs, eta=eta, n_train_states=len(p['train_states']), missing=0, numerical_failures=0)
    A.load = lambda: D
    import sys; sys.argv = ['x', 'results_synthetic_test.json']
    A.main()


if __name__ == '__main__':
    main()
