from common import *
from design import profiles
from learned import LearnedNominal
import jax
jax.config.update('jax_enable_x64',False)
p=profiles(['deterministic','mixture','diffusion']);state=read(HERE/'PILOT_STATES.json')[0];checks=[]
for name,cp in p['profiles'].items():
 rt=LearnedNominal(cp['payload']);env=rt.make_env(state);a=rt.nominal(env,jax.random.PRNGKey(380))[1];b=rt.nominal(env,jax.random.PRNGKey(380))[1];assert a.shape==(4,2) and np.isfinite(a).all() and np.linalg.norm(a,axis=-1).max()<=rt.config.max_speed+1e-12;assert a.tobytes()==b.tobytes()
 if cp['payload']['controller_class']=='deterministic':assert a.tobytes()==rt.nominal(env,jax.random.PRNGKey(381))[1].tobytes()
 assert env.step_count==0;checks.append(name)
write(HERE/'LEARNED_UNIT_TESTS.json',dict(utc=utc(),passed=True,controllers=checks,shape_speed_key_determinism=True,deterministic_ignores_key=True,physical_steps=0,x64=bool(jax.config.x64_enabled)))
seal('TRAINED_POLICIES_FROZEN.receipt.json',[HERE/'TRAINING_CONFIGS.json',HERE/'FRAME_TESTS.json',HERE/'LEARNED_UNIT_TESTS.json',*list((HERE/'policies').glob('*/COMPLETE.receipt.json'))],before_any_learned_pilot_labels=True)
print('18 learned policies pass nominal tests')
