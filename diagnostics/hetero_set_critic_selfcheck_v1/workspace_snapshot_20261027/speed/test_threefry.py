import os; os.environ['JAX_PLATFORMS'] = 'cpu'
import numpy as np, jax, time
from np_threefry import fold_in
rng = np.random.default_rng(0); bad = 0; n = 0
for _ in range(3000):
    key = rng.integers(0, 2**32, 2, dtype=np.uint64).astype(np.uint32); data = int(rng.integers(0, 2**32))
    a = np.asarray(jax.random.fold_in(jax.numpy.asarray(key), data), dtype=np.uint32); b = fold_in(key, data); n += 1; bad += not np.array_equal(a, b)
# the exact chain used by runtime_new: PRNGKey(2026102004) -> fold_in(hash) -> fold_in(seed) -> fold_in(step) -> (distributed) fold_in(agent)
root = jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(2026102004), 0xDEADBEEF), 0)
rn = fold_in(fold_in(np.asarray(jax.random.PRNGKey(2026102004)), 0xDEADBEEF), 0)
chain_ok = np.array_equal(np.asarray(root, np.uint32), rn)
for t in range(300):
    for i in range(10):
        chain_ok &= np.array_equal(np.asarray(jax.random.fold_in(jax.random.fold_in(root, t), i), np.uint32), fold_in(fold_in(rn, t), i))
t0 = time.time(); [jax.random.fold_in(root, t) for t in range(2000)]; tj = time.time() - t0
t0 = time.time(); [fold_in(rn, t) for t in range(2000)]; tn = time.time() - t0
print(f'random inputs: {n - bad}/{n} identical; runtime chain (300 steps x 10 agents) identical: {bool(chain_ok)}; per call JAX {tj/2000*1e6:.0f} us vs NumPy {tn/2000*1e6:.0f} us')
