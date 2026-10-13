"""Read-only check that scalar pooling did not conceal a useful local target.

This does not revise the frozen learning target or select a best component.
All active agent components and all four uncoupled controllers are retained.
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def centered(x, axes):
    for axis in axes:
        x = x - x.mean(axis, keepdims=True)
    return x


def covariance(a, b):
    return np.mean(a * b) - np.mean(a.mean(0) * b.mean(0))


def run():
    raw = dict(np.load(HERE / 'raw_labels.npz'))
    source = json.loads((ROOT / 'controller_ff_factorial_repair_v1/protocol.json').read_text())
    physical = {s['state_uid']: s['physical'] for s in source['states']}
    bootstrap = {}; rows = []; saved = {}
    for scene in sorted(set(raw['scene'])):
        config = source['environments'][scene]['config']['config']
        vmax = config['max_speed']
        for split in ('train', 'validation'):
            ix = np.flatnonzero((raw['scene'] == scene) & (raw['split'] == split))
            ix = np.array(sorted(ix, key=lambda i: (raw['state_uid'][i], raw['controller'][i])))
            ids = raw['state_uid'][ix][::4]; n = len(ids)
            assert len(ix) == n * 4
            actions = raw['first_action'][ix][:, [0, 5, 10, 15]]
            seen = raw['observed'][ix][:, [0, 5, 10, 15]].astype(bool)
            count = seen.sum(-1)
            assert (count == count[0, 0]).all()
            nt = int(count[0, 0]); assert nt in (2, 4)
            agents = len(physical[str(ids[0])]['positions'])
            action = []
            for i, sid in zip(ix, raw['state_uid'][ix]):
                p = physical[str(sid)]
                e = np.asarray(p['goals']) - np.asarray(p['positions'])
                e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
                frame = np.stack((e, np.stack((-e[:, 1], e[:, 0]), -1)), -1)
                a = raw['first_action'][i, [0, 5, 10, 15], :nt, :agents]
                action.append(np.einsum('ekan,anm->ekam', a, frame) / vmax)
            action = np.asarray(action).reshape(n, 4, 4, nt, agents, 2)
            assert np.isfinite(action).all()
            tag = str(scene) + '__' + split
            saved[tag + '__states'] = ids
            saved[tag + '__goal_frame_action'] = action
            indices = np.random.default_rng(2026101029 + n).integers(n, size=(20000, n))
            bootstrap[tag + '__indices'] = indices
            targets = {
                'full_vector': action.reshape(n, 4, 4, nt, agents * 2),
                'pooled_45_degree': ((action[..., 0] + action[..., 1]).mean(-1) / np.sqrt(2))[..., None],
            }
            for target, value in targets.items():
                a, b = value[:, :, :, :nt//2].mean(3), value[:, :, :, nt//2:nt].mean(3)
                for interaction, axes in (('state_eta', (2,)), ('state_controller_eta', (1, 2))):
                    aa, bb = centered(a, axes), centered(b, axes)
                    point = covariance(aa, bb)
                    energy_a = covariance(aa, aa); energy_b = covariance(bb, bb)
                    values = []
                    for start in range(0, len(indices), 500):
                        ib = indices[start:start+500]
                        x, y = aa[ib], bb[ib]
                        values.append((x*y).mean(axis=tuple(range(1, x.ndim))) - (x.mean(1)*y.mean(1)).mean(axis=tuple(range(1, x.ndim-1))))
                    values = np.concatenate(values)
                    key = tag + '__' + target + '__' + interaction
                    bootstrap[key] = values
                    rows.append(dict(scene=str(scene), split=split, N=n, rollout_seeds=nt,
                        target=target, interaction=interaction, signal_energy=float(point),
                        CI95=np.quantile(values, [.025, .975]).tolist(),
                        half_energy=[float(energy_a), float(energy_b)],
                        half_reliability=float(point / np.sqrt(energy_a*energy_b)) if energy_a*energy_b > 0 else None))
    result = dict(history='Reuse all exact verified first-action records; prior scalar signal audit exists. No prior full-vector half-signal audit found. No new labels, queries, or fits.',
        scope='DEVELOPMENT target informativeness only; all active goal-frame action components retained, no best-coordinate selection; not a trained-model transfer result.',
        statistic='Independent-seed-half covariance after eta centering, additionally controller centering for triple interaction; physical-state recentering inside all 20000 bootstrap draws. State_eta retains controller-specific interaction, so averages energy across controllers rather than cancelling them.',
        rows=rows, new_rollout=0, new_query=0, new_fit=0)
    (HERE/'vector_signal_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    np.savez_compressed(HERE/'vector_signal_bootstrap.npz', **bootstrap)
    np.savez_compressed(HERE/'goal_frame_actions.npz', **saved)
    for r in rows:
        if r['target'] == 'full_vector':
            print(r['scene'], r['split'], r['interaction'], f"energy={r['signal_energy']:.4g}", 'CI=' + str([round(v, 8) for v in r['CI95']]), f"reliability={r['half_reliability']:.3f}")


if __name__ == '__main__':
    run()
