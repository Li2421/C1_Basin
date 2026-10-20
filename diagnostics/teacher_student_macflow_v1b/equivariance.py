"""Pre-training descriptive diagnosis, no simulation steps."""
from core import *
from audit_utils import source_inventory
from data import teacher
import jax,jax.numpy as jnp
jax.config.update('jax_enable_x64',False)
def main():
    assert (HERE/'PREREGISTRATION.receipt.json').exists()
    assert not (HERE/'EQUIVARIANCE.json').exists()
    results=[]
    for i in read(HERE/'GLOBAL_INDICES.json'):
        r=read(HERE/'TEACHERS.json')[i];rt=teacher(i);rng=np.random.default_rng(2026101501+i);x=np.load(D/'teacher_student_macflow_v1/data'/r['name']/'train_teacher.npy',mmap_mode='r');indices=rng.integers(0,len(x),256);o=np.array(x[indices]);angle=rng.uniform(-np.pi,np.pi,256);cs=np.cos(angle);sn=np.sin(angle);rot=np.stack((np.stack((cs,-sn),-1),np.stack((sn,cs),-1)),-2)
        rotated=o.copy();rotated[...,:20]=np.einsum('bij,bakj->baki',rot,o[...,:20].reshape(256,4,10,2)).reshape(256,4,20)
        keys=jax.random.split(jax.random.PRNGKey(2026101502+i),256*16).reshape(256,16,2)
        fn=jax.jit(jax.vmap(lambda obs,ks:jax.vmap(lambda k:rt.native_fn(obs,k))(ks).mean(0)))
        a=np.asarray(fn(jnp.asarray(o),keys));b=np.asarray(fn(jnp.asarray(rotated),keys));ra=np.einsum('bij,baj->bai',rot,a);error=b-ra
        report=dict(teacher=r['name'],states=256,noise_samples=16,indices=indices,angles=angle,mean_action_rms=float(np.sqrt(np.mean(a*a))),equivariance_error_rms=float(np.sqrt(np.mean(error*error))),relative_rms=float(np.sqrt(np.mean(error*error))/max(np.sqrt(np.mean(a*a)),1e-12)),relative_mean_norm=float(np.linalg.norm(error,axis=-1).mean()/max(np.linalg.norm(a,axis=-1).mean(),1e-12)),descriptive_only=True)
        results.append(report);print('equivariance',i,report['relative_rms'],flush=True)
    write(HERE/'EQUIVARIANCE.json',dict(utc=utc(),controllers=results,training_started=False,simulation_steps=0,jax_enable_x64=bool(jax.config.x64_enabled)));seal('EQUIVARIANCE.receipt.json',[HERE/'EQUIVARIANCE.json',HERE/'equivariance.py'],before_training=True);source_inventory('imports_equivariance.json')
if __name__=='__main__':main()
