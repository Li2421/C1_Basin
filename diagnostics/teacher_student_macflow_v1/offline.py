from core import *
from audit_utils import source_inventory
from student import Student,inputs
from actions import nominal_only
from data import teacher
import jax,jax.numpy as jnp,pickle
jax.config.update('jax_enable_x64',False)
def main():
    results=[];files=[]
    for i,r in enumerate(read(HERE/'TEACHERS.json')):
        folder=HERE/'students'/r['name'];c=read(folder/'COMPLETE.json');assert sha(folder/'best.pkl')==c['checkpoint_sha256'];s=Student(i);base=s._base;env=s.make_env(read(HERE/'EVAL_STATES.json')[0]);o=s.nominal(env,jax.random.PRNGKey(11))[0];obs=jnp.asarray(inputs(jnp.asarray(o),s.mean,s.scale));noise=jax.random.normal(jax.random.PRNGKey(12),(4,2));first=np.asarray(s.forward(obs,noise));second=np.asarray(s.forward(obs,noise));assert first.tobytes()==second.tobytes()
        goals=env.goals.copy();obs_goals=[];own_noise=[]
        for agent in range(4):
            env.goals[:]=goals;env.goals[np.arange(4)!=agent]+=.123
            from ring_exchange.local_frame import local_observation
            perturbed=local_observation(env.positions,env.velocities,env.goals,env.config);assert perturbed[agent].tobytes()==o[agent].tobytes()
            po=np.asarray(inputs(jnp.asarray(perturbed),s.mean,s.scale));a=np.asarray(s.forward(po[agent],noise[agent]));assert a.tobytes()==np.asarray(s.forward(obs[agent],noise[agent])).tobytes();obs_goals.append(True)
            other=np.array(noise);other[np.arange(4)!=agent]+=17.;b=np.asarray(s.forward(obs[agent],jnp.asarray(other[agent])));assert b.tobytes()==a.tobytes();own_noise.append(True)
        env.goals[:]=goals;key=jax.random.PRNGKey(31);a,_=nominal_only(s,env,key,np.zeros(3),np.zeros(4,int));assert a.tobytes()==s.nominal(env,key)[1].tobytes()
        data=HERE/'data'/r['name'];x=np.load(data/'dev_local.npy',mmap_mode='r');o_teacher=np.load(data/'dev_teacher.npy',mmap_mode='r');rng=np.random.default_rng(2026103033);indices=rng.integers(0,len(x),2048);keys=jax.random.split(jax.random.PRNGKey(2026103034),2048);target=np.asarray(jax.jit(jax.vmap(base.native_fn))(jnp.asarray(o_teacher[indices]),keys));z=jax.vmap(lambda obs,k:base.trace_fn(obs,k)[0].reshape(4,2))(jnp.asarray(o_teacher[indices]),keys);prediction=np.asarray(s.forward(inputs(jnp.asarray(x[indices]),s.mean,s.scale),z));diff=prediction-target;cos=(prediction*target).sum(-1)/(np.maximum(np.linalg.norm(prediction,axis=-1)*np.linalg.norm(target,axis=-1),1e-12));report=dict(teacher=r['name'],best_step=c['best_step'],selected_fixed_dev_mse=c['dev_mse'],independent_dev_mse=float(np.mean(diff**2)),action_rmse=float(np.sqrt(np.mean(diff**2))),mean_cosine=float(cos.mean()),within_05_mps=float((np.linalg.norm(diff,axis=-1)<=.05).mean()),other_goals_invariant=all(obs_goals),own_noise_only=all(own_noise),deterministic=True,nominal_only_matches_student_bitwise=True)
        results.append(report);files.extend([folder/'best.pkl',folder/'COMPLETE.json']);print('offline PASS',i,report['independent_dev_mse'],flush=True)
    write(HERE/'DISTILLATION_REPORT.json',dict(utc=utc(),controllers=results,closed_loop_labels_read=False,jax_enable_x64=bool(jax.config.x64_enabled)));seal('TRAINING_FROZEN.receipt.json',files+[HERE/'DISTILLATION_REPORT.json',HERE/'TRAINING_CONFIG.json',HERE/'student.py',HERE/'train.py'],before_student_rollouts=True);source_inventory('imports_offline.json')
if __name__=='__main__':main()
