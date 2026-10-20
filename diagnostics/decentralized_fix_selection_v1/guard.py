from core import *
import new_benchmark_common.macflow
def guard():
 import shared_rollout_db,shared_rollout_db.src.cache_writer as w,jax
 assert Path(shared_rollout_db.__file__).resolve().is_relative_to(ROOT/'shared_rollout_db')
 assert (w.ROOT/'journals').resolve()==ROOT/'shared_rollout_db/journals'
 jax.config.update('jax_enable_x64',False)
 assert not jax.config.x64_enabled
 return {'package':shared_rollout_db.__file__,'journals':str(w.ROOT/'journals'),'jax_enable_x64':bool(jax.config.x64_enabled)}
