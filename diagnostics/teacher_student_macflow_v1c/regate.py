from core import *
from student import Student,inputs
from coordinates import to_local_jax
from audit_utils import source_inventory
import jax,jax.numpy as jnp
jax.config.update('jax_enable_x64',False)
def main():
    assert not (HERE/'REGATE.json').exists();results=[];valid=[];files=[]
    for i,r in enumerate(read(HERE/'TEACHERS.json')):
        info=read(HERE/'STUDENTS.json')[str(i)];assert sha(info['checkpoint'])==info['checkpoint_sha256'];s=Student(i);base=s._base;data=D/'teacher_student_macflow_v1/data'/r['name'];x=np.load(data/'dev_local.npy',mmap_mode='r');o=np.load(data/'dev_teacher.npy',mmap_mode='r');indices=np.random.default_rng(2026103033).integers(0,len(x),2048);keys=jax.random.split(jax.random.PRNGKey(2026103034),2048);jo=jnp.asarray(o[indices]);u=jax.jit(jax.vmap(base.native_fn))(jo,keys);z=jax.jit(jax.vmap(lambda obs,key:base.trace_fn(obs,key)[0].reshape(4,2)))(jo,keys)
        if i not in [8,10,12,13]:u=to_local_jax(u,jo[...,:2]);z=to_local_jax(z,jo[...,:2])
        prediction=np.asarray(s.forward(inputs(jnp.asarray(x[indices]),s.mean,s.scale),z));u=np.asarray(u);norm=np.linalg.norm(u,axis=-1);error=np.linalg.norm(prediction-u,axis=-1);cos=(prediction*u).sum(-1)/np.maximum(norm*np.linalg.norm(prediction,axis=-1),1e-12);metrics={}
        for threshold in [.05,.10,.20]:
            mask=norm>=threshold;metrics[str(threshold)]=dict(samples=int(mask.sum()),coverage=float(mask.mean()),mean_cosine=float(cos[mask].mean()) if mask.any() else None)
        m=metrics['0.1']['mean_cosine'];fraction=float((error<=.05).mean());regate=info['best_step']!=0 and m is not None and m>=.90 and fraction>=.80;previous=info['previously_valid'];effective=previous or regate
        row=dict(index=i,teacher=r['name'],best_step=info['best_step'],checkpoint_sha256=sha(info['checkpoint']),previously_valid=previous,regate_applied=not previous,passes_revised_metrics=bool(regate),valid=bool(effective),all_sample_cosine=float(cos.mean()),within_05_mps=fraction,action_rmse=float(np.sqrt(np.mean((prediction-u)**2))),filtered=metrics);results.append(row)
        if not previous and effective:valid.append(r['name'])
        p=HERE/'dev_predictions'/f"{r['name']}.npz";p.parent.mkdir(exist_ok=True);np.savez_compressed(p,prediction=prediction,target=u,local_noise=np.asarray(z),indices=indices,keys=np.asarray(keys));files.append(p);print('regate',i,metrics['0.1'],effective,flush=True)
    write(HERE/'REGATE.json',dict(utc=utc(),controllers=results,new_valid=valid,n_new_valid=len(valid),valid_names=[r['teacher'] for r in results if r['valid']],jax_enable_x64=bool(jax.config.x64_enabled),simulation_steps=0,closed_loop_labels_read=False));seal('REGATE_FROZEN.receipt.json',[HERE/'REGATE.json',HERE/'regate.py',HERE/'student.py',HERE/'coordinates.py',HERE/'STUDENTS.json',HERE/'PREREGISTRATION.md']+files,before_new_candidate_closed_loop=True,training_steps=0);source_inventory('imports_regate.json')
    lines=['# Revised distillation gate (frozen checkpoints only)','', '| Student | Previously valid | Step | All cosine | Cosine ≥.05/.10/.20 | ≤.05m/s | New decision |','|---|---|---:|---:|---|---:|---|']
    for r in results:lines.append(f"| {r['teacher']} | {r['previously_valid']} | {r['best_step']} | {r['all_sample_cosine']:.5f} | "+'/'.join(f"{r['filtered'][str(t)]['mean_cosine']:.5f}" if r['filtered'][str(t)]['mean_cosine'] is not None else 'undefined' for t in [.05,.1,.2])+f" | {r['within_05_mps']:.2%} | {'unchanged' if r['previously_valid'] else r['valid']} |")
    (HERE/'REGATE.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
