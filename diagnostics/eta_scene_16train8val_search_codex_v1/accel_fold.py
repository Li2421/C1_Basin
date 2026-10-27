import jax,accel_mode
from np_threefry import fold_in as _fast
def fold_in(key,data):
 return (_fast if accel_mode.ENABLED else jax.random.fold_in)(key,data)
