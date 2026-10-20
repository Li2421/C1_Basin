from core import *
from audit_utils import source_inventory
from student import Student,inputs,network
from coordinates import to_local_jax
from data import teacher
from ring_exchange.local_frame import world_actions_to_local,local_actions_to_world,local_observation
import jax,jax.numpy as jnp,pickle
jax.config.update('jax_enable_x64',False)
def main():
    assert not (HERE/'TRAINING_FROZEN.receipt.json').exists()
    rng=np.random.default_rng(2026101503);p=rng.normal(size=(4,2));v=rng.normal(size=(4,2));expected=world_actions_to_local(v,p);actual=np.asarray(to_local_jax(jnp.asarray(v),jnp.asarray(p)));assert np.max(abs(expected-actual))<1e-6
    for i in range(4):
        q=p.copy();q[np.arange(4)!=i]+=10;assert world_actions_to_local(v,q)[i].tobytes()==expected[i].tobytes();np.testing.assert_array_equal(np.asarray(to_local_jax(jnp.asarray(v),jnp.asarray(q)))[i],actual[i])
    tests=dict(row_own_position_only=True,jax_numpy_max_abs_error=float(abs(actual-expected).max()))
    results=[];files=[]
    for i in read(HERE/'GLOBAL_INDICES.json'):
        r=read(HERE/'TEACHERS.json')[i];folder=HERE/'students'/r['name'];c=read(folder/'COMPLETE.json');assert sha(folder/'best.pkl')==c['checkpoint_sha256'];s=Student(i);base=s._base;env=s.make_env(read(HERE/'EVAL_STATES.json')[0]);obs=local_observation(env.positions,env.velocities,env.goals,env.config);o=inputs(jnp.asarray(obs),s.mean,s.scale);noise=jax.random.normal(jax.random.PRNGKey(12),(4,2));first=np.asarray(s.forward(o,noise));second=np.asarray(s.forward(o,noise));assert first.tobytes()==second.tobytes();goals=env.goals.copy()
        for agent in range(4):
            env.goals[:]=goals;env.goals[np.arange(4)!=agent]+=.123;pert=local_observation(env.positions,env.velocities,env.goals,env.config);assert pert[agent].tobytes()==obs[agent].tobytes();po=inputs(jnp.asarray(pert),s.mean,s.scale);assert np.asarray(s.forward(po[agent],noise[agent])).tobytes()==np.asarray(s.forward(o[agent],noise[agent])).tobytes()
        env.goals[:]=goals
        data=D/'teacher_student_macflow_v1/data'/r['name'];x=np.load(data/'dev_local.npy',mmap_mode='r');ot=np.load(data/'dev_teacher.npy',mmap_mode='r');indices=np.random.default_rng(2026103033).integers(0,len(x),2048);keys=jax.random.split(jax.random.PRNGKey(2026103034),2048);jo=jnp.asarray(ot[indices]);target=to_local_jax(jax.jit(jax.vmap(base.native_fn))(jo,keys),jo[...,:2]);z=to_local_jax(jax.jit(jax.vmap(lambda obs,k:base.trace_fn(obs,k)[0].reshape(4,2)))(jo,keys),jo[...,:2]);prediction=np.asarray(s.forward(inputs(jnp.asarray(x[indices]),s.mean,s.scale),z));target=np.asarray(target);diff=prediction-target;cos=(prediction*target).sum(-1)/np.maximum(np.linalg.norm(prediction,axis=-1)*np.linalg.norm(target,axis=-1),1e-12);fraction=float((np.linalg.norm(diff,axis=-1)<=.05).mean());cosine=float(cos.mean());valid=c['best_step']!=0 and cosine>=.90 and fraction>=.80
        report=dict(teacher=r['name'],best_step=c['best_step'],independent_dev_mse=float(np.mean(diff**2)),action_rmse=float(np.sqrt(np.mean(diff**2))),mean_cosine=cosine,within_05_mps=fraction,valid=bool(valid),other_goals_invariant=True,deterministic=True)
        results.append(report);files.extend([folder/'best.pkl',folder/'COMPLETE.json']);print('distillation',i,valid,cosine,fraction,flush=True)
    # Local teacher uses identity label/noise; original local student deployment chart is identical.
    r=read(HERE/'TEACHERS.json')[10];base=teacher(10);env=base.make_env(read(HERE/'EVAL_STATES.json')[0]);key=jax.random.PRNGKey(19);obs=base.observation(env);u=np.asarray(base.native_fn(obs,key));xi=np.asarray(base.trace_fn(obs,key)[0]).reshape(4,2);np.testing.assert_array_equal(base.to_world(u,env),local_actions_to_world(u,env.positions))
    ck=pickle.loads((D/'teacher_student_macflow_v1/students'/r['name']/'best.pkl').read_bytes());model=network();ox=inputs(jnp.asarray(local_observation(env.positions,env.velocities,env.goals,env.config)),ck['mean'],ck['scale']);raw=np.stack([np.asarray(model.apply({'params':ck['params']},ox[i],jax.random.normal(jax.random.fold_in(key,i),(2,),dtype=jnp.float32))) for i in range(4)]);raw=np.clip(raw,-1.,1.);raw*=np.minimum(1.,env.config.max_speed/np.maximum(np.linalg.norm(raw,axis=1,keepdims=True),1e-12));np.testing.assert_array_equal(base.to_world(np.asarray(raw,np.float64),env),local_actions_to_world(np.asarray(raw,np.float64),env.positions));tests.update(local_teacher_identity_labels_noise=True,local_teacher_deployment_matches_round30=True,local_teacher_checkpoint_sha256=sha(D/'teacher_student_macflow_v1/students'/r['name']/'best.pkl'))
    write(HERE/'UNIT_TESTS.json',dict(status='PASS',tests=tests,all_students_other_goals_invariant=True,all_students_deterministic=True,jax_enable_x64=bool(jax.config.x64_enabled),simulation_steps=0));valid_names=[r['teacher'] for r in results if r['valid']];write(HERE/'DISTILLATION_REPORT.json',dict(utc=utc(),controllers=results,valid_names=valid_names,n_valid=len(valid_names),closed_loop_labels_read=False,jax_enable_x64=bool(jax.config.x64_enabled)));seal('TRAINING_FROZEN.receipt.json',files+[HERE/'DISTILLATION_REPORT.json',HERE/'UNIT_TESTS.json',HERE/'TRAINING_CONFIG.json',HERE/'student.py',HERE/'train.py',HERE/'coordinates.py'],before_student_rollouts=True);source_inventory('imports_offline.json')
    lines=['# Distillation validity gate','', '| Teacher | Step | Cosine | <= .05m/s | Valid |','|---|---:|---:|---:|---|']+[f"| {r['teacher']} | {r['best_step']} | {r['mean_cosine']:.5f} | {r['within_05_mps']:.3%} | {r['valid']} |" for r in results];(HERE/'DISTILLATION_REPORT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
