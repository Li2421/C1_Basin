from common import *
from critic_model import *
from frozen_networks import Critic as FrozenCritic
with np.load(HERE/'CONTROLLER_FEATURES.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
with np.load(HERE/'STATE_INPUTS.npz',allow_pickle=False) as z:data.update(h=z['h'][:48],c=z['c'][:48],action=z['action'][:48])
source=sorted(set(data['types'])-{'Flow'});tr=source_transform(data,source);ar=transform_arrays(data,tr);old,norm=original_parameters()
rows=[]
for precision in ('default','highest'):
 with jax.default_matmul_precision(precision):
  expected=np.asarray(FrozenCritic().apply({'params':old},jnp.asarray((data['h'][:8]-norm['h_mean'])/np.asarray(norm['h_std'])),jnp.asarray((data['c'][:8]-norm['c_mean'])/np.asarray(norm['c_std'])),jnp.asarray(ar['action'][:8,0,:3]),method=FrozenCritic().ring))
  pa=initialize('full',17,'warm',tr,(old,norm))
  actual=np.asarray(network('full').apply(pa,jnp.asarray(ar['h'][:8]),jnp.asarray(ar['c'][:8]),jnp.asarray(ar['controller'][:8]),jnp.asarray(ar['action'][:8,0])))
  rows.append(dict(precision=precision,logit_error=float(np.max(abs(actual-expected))),probability_error=float(np.max(abs(np.asarray(jax.nn.sigmoid(jnp.asarray(actual)))-np.asarray(jax.nn.sigmoid(jnp.asarray(expected)))))),expected=expected.tolist(),actual=actual.tolist()))
backend=jax.default_backend();write(HERE/f'WARM_PRECISION_DIAGNOSIS_{backend}.json',dict(utc=utc(),backend=backend,rows=rows,x64=jax.config.x64_enabled,new_rollouts=0))
print(backend,json.dumps(rows),flush=True)
