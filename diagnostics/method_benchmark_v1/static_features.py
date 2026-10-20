from core import *
import time
def main(index,workers):
 prereg();p=read(HERE/'INPUT_IDENTITIES.json');states=p['common_states']
 import jax
 from diagnostics.controller_q_transfer_round3_v1.flow_adapter import RingFlowAdapter
 from diagnostics.controller_q_diversity_v1.runtime import native_runtime,controller_context
 from diagnostics.controller_q_diversity_v1.features import field_context
 from diagnostics.controller_q_diversity_v1.train import compact_field
 from ring_exchange.environment import RingExchangeEnv
 jax.config.update('jax_enable_x64',False);assert not jax.config.x64_enabled
 def forbidden(*a,**kw):raise RuntimeError('R3 static query called env.step')
 RingExchangeEnv.step=forbidden
 for name in sorted(p['profiles'])[index::workers]:
  out=HERE/'r3'/f'{name}.npz';receipt=out.with_suffix('.json');out.parent.mkdir(exist_ok=True)
  if out.exists() and receipt.exists():continue
  v=p['profiles'][name];old=D/'controller_loco_real_flow_v1/static_features'/f'{name}.npz';oldrec=old.with_name(name+'_receipt.json');reused=False;t0=time.time()
  if old.exists() and oldrec.exists() and not read(oldrec)['jax_enable_x64']:
   oldp=read(D/'controller_loco_real_flow_v1/protocol.json')['controllers']['ring_exchange'].get(name)
   z=np.load(old);lookup={str(s):i for i,s in enumerate(z['state_uids'])}
   if oldp and oldp['payload']['flow_checkpoint_sha256']==v['flow_checkpoint_sha256'] and all(s['state_uid'] in lookup for s in states):
    x=z['R3'][[lookup[s['state_uid']] for s in states]];reused=True
  if not reused:
   rt=native_runtime('ring_exchange') if name=='base_u_v10_local_macflow' else RingFlowAdapter(v['flow_checkpoint_path'],read(v['adapter_manifest_path']),v['observation_frame'])
   jax.config.update('jax_enable_x64',False);assert not jax.config.x64_enabled;x=[];errors=[]
   for i,s in enumerate(states):
    try:a=np.asarray(compact_field(np.concatenate((controller_context(rt,s),field_context(rt,s)),axis=-1)),np.float32).reshape(256)
    except Exception as e:a=np.full(256,np.nan,np.float32);errors.append(dict(state=i,type=type(e).__name__,message=str(e)))
    x.append(a)
   x=np.asarray(x)
  else:errors=read(oldrec).get('errors',[])
  assert x.shape==(128,256)
  np.savez_compressed(out,R3=x,state_uids=[s['state_uid'] for s in states])
  write(receipt,dict(utc=utc(),controller=name,checkpoint_sha256=sha(v['flow_checkpoint_path']),features_sha256=sha(out),reused=str(old) if reused else None,source_sha256=sha(old) if reused else None,errors=errors,jax_enable_x64=False,simulator_steps=0,test_labels_read=False,seconds=time.time()-t0));print(name,'reused',reused,'seconds',time.time()-t0,flush=True)
if __name__=='__main__':main(int(sys.argv[1]),int(sys.argv[2]))
