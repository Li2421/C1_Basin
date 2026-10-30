"""NumPy Threefry-2x32 (20 rounds) reproducing jax.random.fold_in for legacy uint32[2] keys."""
import numpy as np
M = np.uint64(0xFFFFFFFF)
R = ((13, 15, 26, 6), (17, 29, 16, 24))
def _rotl(x, d): return ((x << np.uint64(d)) | (x >> np.uint64(32 - d))) & M
def threefry2x32(k0, k1, x0, x1):
    k0 = np.uint64(k0); k1 = np.uint64(k1); x0 = np.uint64(x0); x1 = np.uint64(x1)
    ks = (k0, k1, (k0 ^ k1 ^ np.uint64(0x1BD11BDA)) & M)
    x0 = (x0 + ks[0]) & M; x1 = (x1 + ks[1]) & M
    for i in range(5):
        for r in R[i % 2]:
            x0 = (x0 + x1) & M; x1 = _rotl(x1, r); x1 = x1 ^ x0
        x0 = (x0 + ks[(i + 1) % 3]) & M; x1 = (x1 + ks[(i + 2) % 3] + np.uint64(i + 1)) & M
    return np.array([x0, x1], dtype=np.uint32)
def fold_in(key, data):
    k = np.asarray(key, dtype=np.uint32).reshape(2)
    d = int(data) & 0xFFFFFFFF
    return threefry2x32(int(k[0]), int(k[1]), 0, d)
