"""Evaluate every member of a frozen calibration set, including zero risks."""
import hashlib
import json
import numpy as np


def reuse_fixed_calibration(reference, path, horizon, batch_size, expected, evaluate):
    """Reuse an explicitly supplied unchanged zero-residual calibration.

    Match full model/plant/risk/input provenance and independently replay the
    first batch. Reference risks concern initialization, never trained weights.
    """
    if batch_size < 1:
        raise ValueError('calibration batch size must be positive')
    metadata = json.loads(reference.read_text())
    for key, value in expected.items():
        if metadata[key] != value:
            raise ValueError(f'calibration reference differs: {key}')
    training = metadata['training']
    calibration = training['calibration']
    with np.load(path, allow_pickle=False) as data:
        starts, noise = data['initial_positions'], data['noise']
    if (starts.ndim != 3 or starts.shape[1:] != (2, 2) or len(starts) == 0
            or not np.isfinite(starts).all() or not np.isfinite(noise).all()
            or noise.shape != (len(starts), horizon, 4)
            or training['horizon'] != horizon
            or calibration['batch_size'] != batch_size
            or calibration['size'] != len(starts)
            or calibration['sha256'] != hashlib.sha256(path.read_bytes()).hexdigest()
            or calibration['protocol'] != 'frozen_set_episode_mean'):
        raise ValueError('calibration reference input/protocol differs')
    risks = np.asarray(calibration['risks'], dtype=float)
    if risks.shape != (len(starts),) or not np.isfinite(risks).all() or (risks < 0).any():
        raise ValueError('invalid cached calibration risks')
    mean = float(np.mean(risks))
    if mean <= 0 or mean != training['baseline_J_live']:
        raise ValueError('cached calibration mean differs')
    actual = np.asarray(evaluate(starts[:batch_size], noise[:batch_size]))
    np.testing.assert_allclose(actual, risks[:batch_size], rtol=0, atol=1e-10,
                               err_msg='zero-residual calibration replay differs')
    return mean, calibration


def calibrate_fixed_set(path, horizon, batch_size, evaluate, progress=None):
    if batch_size < 1:
        raise ValueError('calibration batch size must be positive')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path, allow_pickle=False) as data:
        starts = np.asarray(data['initial_positions'], dtype=np.float64)
        noise = np.asarray(data['noise'])
    if (starts.ndim != 3 or starts.shape[1:] != (2, 2) or len(starts) == 0
            or noise.shape != (len(starts), horizon, 4)):
        raise ValueError('calibration requires nonempty [B,2,2] starts and [B,horizon,4] noise')
    if not np.isfinite(starts).all() or not np.isfinite(noise).all():
        raise ValueError('calibration inputs must be finite')
    risks = []
    for offset in range(0, len(starts), batch_size):
        batch = starts[offset:offset + batch_size]
        values = np.asarray(evaluate(batch, noise[offset:offset + batch_size]), dtype=float)
        if values.shape != (len(batch),) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError('calibration must return one finite nonnegative risk per episode')
        risks.extend(values.tolist())
        if progress is not None:
            progress(len(risks), len(starts))
    mean = float(np.mean(risks))
    if mean <= 0:
        raise ValueError('fixed calibration risk is zero; cannot calibrate a positive constraint')
    return mean, dict(protocol='frozen_set_episode_mean', sha256=digest,
                      size=len(starts), batch_size=batch_size, risks=risks)
