from core import *
from guard import guard
from student import Student,inputs,encoding,target
from data import dataset
from audit_utils import source_inventory
import jax,jax.numpy as jnp
G=guard()
def distillation(index,variant):
 ck=HERE/'students'/variant/str(index)/'round3.pkl';info=read(ck.with_suffix('.json'));assert sha(ck)==info['checkpoint_sha256'];s=Student(index,variant,ck);x,o,_=dataset(index,'dev');ix=np.random.default_rng(2026103033).integers(0,len(x),2048);keys=jax.random.split(jax.random.PRNGKey(2026103034),2048);xx=jnp.asarray(x[ix]);oo=jnp.asarray(o[ix]);u=jax.jit(jax.vmap(s._base.native_fn))(oo,keys);z=jax.jit(jax.vmap(lambda ob,k:s._base.trace_fn(ob,k)[0].reshape(4,2)))(oo,keys);pred=np.asarray(s.forward(inputs(xx,s.mean,s.scale),encoding(index,variant,xx,oo,z)));uu=np.asarray(target(index,u,oo));error=np.linalg.norm(pred-uu,axis=-1);n=np.linalg.norm(uu,axis=-1);cos=(pred*uu).sum(-1)/np.maximum(n*np.linalg.norm(pred,axis=-1),1e-12);mask=n>=.1;cosine=float(cos[mask].mean()) if mask.any() else None;frac=float((error<=.05).mean());valid=info['selected_step']!=0 and cosine is not None and cosine>=.9 and frac>=.8
 dest=HERE/'gates'/f'{variant}_{index}.npz';dest.parent.mkdir(exist_ok=True);assert not dest.exists();np.savez_compressed(dest,prediction=pred,target=uu,noise=np.asarray(z),indices=ix,keys=np.asarray(keys));row=dict(index=index,variant=variant,name=f'{variant}__{index}',checkpoint=str(ck),checkpoint_sha256=sha(ck),selected_step=info['selected_step'],mean_cosine_ge_10=cosine,coverage_ge_10=float(mask.mean()),within_05_mps=frac,rmse=float(np.sqrt(np.mean((pred-uu)**2))),valid=bool(valid),prediction_sha256=sha(dest),guard=guard(),evaluation_labels_read=False);write(dest.with_suffix('.json'),row);source_inventory(f'imports_gate_{variant}_{index}.json');print(row,flush=True)
def freeze():
 rows=[read(p) for p in sorted((HERE/'gates').glob('V*.json'))];assert rows
 write(HERE/'DISTILLATION_GATE.json',dict(utc=utc(),controllers=rows,valid_names=[r['name'] for r in rows if r['valid']],excluded=[r for r in rows if not r['valid']]));seal('TRAINING_FROZEN.receipt.json',[HERE/'DISTILLATION_GATE.json']+[Path(r['checkpoint']) for r in rows]+[p for p in (HERE/'gates').glob('*') if p.is_file()],before_capability_gate=True,before_main_labels=True)
if __name__=='__main__':
 if sys.argv[1]=='freeze':freeze()
 else:distillation(int(sys.argv[1]),sys.argv[2])
