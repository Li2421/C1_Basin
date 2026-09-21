"""Original Flow-BC pair splits, without trajectory-state or synthetic starts."""
import hashlib
import json
import numpy as np

from single_integrator.environment import GiveWayEnv


def load_dataset_starts(root, split, plant):
    metadata = json.loads((root / 'environment.json').read_text())
    if split not in ('train', 'val'):
        raise ValueError('C1 training/selection permits only train or val splits')
    lo, hi = metadata['split_pairs'][split]
    expected = {'train': (0, 199), 'val': (200, 224)}[split]
    if (lo, hi) != expected:
        raise ValueError('dataset split differs from frozen Flow-BC protocol')
    starts = []
    env = GiveWayEnv(plant)
    for pair in range(lo, hi + 1):
        modes = []
        for mode in range(2):
            with np.load(root / 'raw' / f'episode_{2 * pair + mode:04d}.npz', allow_pickle=False) as data:
                if int(data['pair_id']) != pair or int(data['mode']) != mode:
                    raise ValueError('dataset pair/mode provenance mismatch')
                modes.append(np.asarray(data['initial_positions'], dtype=np.float64))
        if not np.array_equal(*modes):
            raise ValueError('expert modes must share the same pair initial state')
        env.reset(modes[0])
        starts.append(modes[0])
    starts = np.asarray(starts)
    digest = hashlib.sha256(starts.astype('<f8').tobytes()).hexdigest()
    return starts, dict(protocol='flow_bc_pair_initial_states_v1', split=split,
                        pair_range=[lo, hi], size=len(starts), initial_positions_sha256=digest)


def load_fixed_inputs(path, horizon):
    """Validate before invoking the model; pin exact content for resume."""
    with np.load(path, allow_pickle=False) as data:
        starts = np.asarray(data['initial_positions'], dtype=np.float64)
        noise = np.asarray(data['noise'])
    if (starts.ndim != 3 or starts.shape[1:] != (2, 2) or len(starts) == 0
            or noise.shape != (len(starts), horizon, 4)):
        raise ValueError('fixed inputs require nonempty [B,2,2] starts and [B,horizon,4] noise')
    if not np.isfinite(starts).all() or not np.isfinite(noise).all():
        raise ValueError('fixed inputs must be finite')
    return starts, noise, hashlib.sha256(path.read_bytes()).hexdigest()


def require_population(starts, population, name):
    # Exactly once per pair gives the same uniform empirical measure used by
    # training. Extra repeats, omitted pairs and foreign starts are rejected.
    a = sorted(tuple(x.ravel()) for x in starts)
    b = sorted(tuple(x.ravel()) for x in population)
    if a != b:
        raise ValueError(f'{name} must contain each original split initial state exactly once')
