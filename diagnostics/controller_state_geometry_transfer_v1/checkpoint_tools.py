"""Replay saved parameters or explicitly refit into a NEW output directory."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent

def dataset(kind):
 z=np.load(HERE/'data'/('legacy_diagnostic.npz' if kind=='legacy_diagnostic' else 'factors.npz'))
 D={k:z[k] for k in z.files if not k.startswith('entity_')};D['X']={k[7:]:z[k] for k in z.files if k.startswith('entity_')}
 if 'eta' not in D:
  p=json.loads((HERE/'factors/protocol.json').read_text());D['eta']=np.array([p['eta'][str(e)] for e in p['eta_indices']])
 return D

def replay(path):
 import jax,jax.numpy as jnp,flax.linen as nn
 from flax import serialization
 from diagnostics.controller_q_diversity_v1.train import Critic
 kind=path.relative_to(HERE/'models').parts[0];D=dataset(kind);m=json.loads(path.with_suffix('.json').read_text());f=np.load(path.with_suffix('.npz'));arm=m['arm'];tr=f['tr'];S=D['Y'].shape[1];C=D['Y'].shape[2]
 # The factor runtime imports poc.field_controller, which enables x64 globally.
 # Encoder's default-dtype eye promotes pair pooling even with float32 weights.
 # Saved predictions + actual import path independently document the mode.
 jax.config.update('jax_enable_x64',kind!='legacy_diagnostic')
 assert f['pred'].dtype==(np.dtype('float32') if kind=='legacy_diagnostic' or arm in ('MLP','MECH') else np.dtype('float64'))
 X={k:jnp.asarray(v,jnp.float32) for k,v in D['X'].items()};eta=jnp.asarray(D['eta'],jnp.float32);onehot=jnp.eye(C,dtype=jnp.float32);g=tr[:,0]*S+tr[:,1]
 if arm in ('ENT_raw','ENT_mech','ENT_mech_no_margin','ENT_mech_phys'):
  cols=(D['raw'] if arm=='ENT_raw' else D['mech']).copy()
  if arm=='ENT_mech_no_margin':cols[:,1]=0
  mu=cols[g].mean(0);sd=np.maximum(cols[g].std(0),.01)
  if arm=='ENT_mech_phys':sd=np.array([1.,60.,1.,1.,1.,4.,10.,1.])
  X['globals']=jnp.concatenate([X['globals'],jnp.asarray((cols-mu)/sd,jnp.float32)],-1)
 if arm in ('MLP','MECH'):
  cols=D['flat'] if arm=='MLP' else D['mech'];mu=cols[g].mean(0);sd=np.maximum(cols[g].std(0),.01);cont=jnp.asarray((cols-mu)/sd,jnp.float32)
  class MLP(nn.Module):
   @nn.compact
   def __call__(self,x,c,e):
    z=jnp.concatenate([x,c,e],-1)
    for _ in range(3):z=nn.silu(nn.Dense(128)(z))
    return nn.Dense(1)(z)[:,0]
  net=MLP()
  def inputs(ix):return cont[ix[:,0]*S+ix[:,1]],onehot[ix[:,2]],eta[ix[:,3]]
 else:
  net=Critic('state_field_compact');na=X['agents'].shape[1]
  def inputs(ix):
   ii=ix[:,0]*S+ix[:,1];x={k:v[ii] for k,v in X.items()};c=jnp.broadcast_to(onehot[ix[:,2]][:,None,:],(len(ii),na,C))*x['agent_mask'][...,None]
   if arm=='STATE':c=jnp.zeros_like(c)
   return x,c,eta[ix[:,3]]
 params=serialization.msgpack_restore(path.read_bytes());predict=jax.jit(lambda par,ix:jax.nn.sigmoid(net.apply(par,*inputs(ix))))
 ix=np.indices(D['Y'].shape[:4]).reshape(4,-1).T
 pred=np.concatenate([np.asarray(predict(params,jnp.asarray(ix[i:i+512],jnp.int32))) for i in range(0,len(ix),512)]).reshape(f['pred'].shape)
 assert np.isfinite(pred).all();err=float(np.max(abs(pred-f['pred'])));backend=jax.default_backend();out=HERE/'replay_predictions'/(backend+'_recorded')/path.relative_to(HERE/'models');out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out.with_suffix('.npz'),pred=pred)
 return {'checkpoint':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'backend':backend,'jax_enable_x64':bool(jax.config.x64_enabled),'devices':[str(d) for d in jax.devices()],'cells':int(pred.size),'max_abs_error':err,'mean_abs_error':float(np.mean(abs(pred-f['pred']))),'within_strict_tolerance':bool(np.allclose(pred,f['pred'],rtol=2e-5,atol=2e-6)),'argmax_changed_all_cells':int((pred.argmax(-1)!=f['pred'].argmax(-1)).sum())}

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['replay','refit']);p.add_argument('--checkpoint');p.add_argument('--outdir');p.add_argument('--all',action='store_true');a=p.parse_args()
 if a.mode=='refit':
  assert a.checkpoint and a.outdir;path=Path(a.checkpoint).resolve();out=Path(a.outdir).resolve();assert not out.exists(),'Fresh output directory required; preserve prior evidence'
  from .learn import fit
  import jax
  kind=path.relative_to(HERE/'models').parts[0];jax.config.update('jax_enable_x64',kind!='legacy_diagnostic');f=np.load(path.with_suffix('.npz'));m=json.loads(path.with_suffix('.json').read_text());fit(dataset(kind),f['tr'],f['y'],m['arm'],m['seed'],out,steps=m['steps']);return
 paths=[Path(a.checkpoint).resolve()] if a.checkpoint else [HERE/'models'/p for p in ['legacy_diagnostic/MANY/ENT_17.msgpack','legacy_diagnostic/MANY/STATE_17.msgpack','legacy_diagnostic/MANY/MLP_17.msgpack','factors/FAMILY/ENT_raw_17.msgpack','factors/FAMILY/ENT_mech_17.msgpack','factors/FAMILY/MECH_17.msgpack','normalization/TIME/ENT_mech_phys_17.msgpack']]
 if a.all:paths=sorted((HERE/'models').rglob('*.msgpack'))
 rows=[]
 for path in paths:
  r=replay(path);rows.append(r);print(r,flush=True)
 if not a.checkpoint:(HERE/f"CHECKPOINT_REPLAY_{rows[0]['backend']}_recorded.json").write_text(json.dumps({'completed':True,'strict_numeric_reproduction':all(r['within_strict_tolerance'] for r in rows),'scope':'all 138 checkpoints' if a.all else 'seven representative architectures','precision':'matching original import paths; legacy x64 false, factor/normalization x64 true','checks':rows},indent=2)+'\n')
if __name__=='__main__':main()
