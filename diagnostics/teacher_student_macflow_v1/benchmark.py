from core import *
from audit_utils import source_inventory
from student import Student,inputs
from pipeline import LocalSafety
import jax,jax.numpy as jnp,time
jax.config.update('jax_enable_x64',False)
def timing(fn,n):
    for _ in range(30):fn()
    samples=[]
    for _ in range(n):
        t=time.perf_counter_ns();fn();samples.append((time.perf_counter_ns()-t)/1e6)
    return dict(mean_ms=float(np.mean(samples)),median_ms=float(np.median(samples)),p95_ms=float(np.quantile(samples,.95)),n=n)
def main():
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});rows=[]
    for i,r in enumerate(read(HERE/'TEACHERS.json')):
        s=Student(i);rt=s._base;env=s.make_env(read(HERE/'EVAL_STATES.json')[0]);obs=jnp.asarray(rt.observation(env));own=jnp.asarray(inputs(jnp.asarray(s.nominal(env,jax.random.PRNGKey(0))[0]),s.mean,s.scale))[0];z=jnp.zeros(2,jnp.float32);key=jax.random.PRNGKey(10)
        a=timing(lambda:s.forward(own,z).block_until_ready(),1000);b=timing(lambda:rt.native_fn(obs,key).block_until_ready(),200);fs=[LocalSafety(env.positions,i,env.config,1.5) for i in range(4)];nominal=rt.nominal(env,key)[1]
        def qp():
            for i,f in enumerate(fs):f.project(nominal[i:i+1])
        c=timing(qp,200);rows.append(dict(controller=r['name'],student_per_robot=a,teacher_joint_10_euler=b,QP_four_local_projections=c,student_speedup_per_robot_vs_teacher_joint=b['mean_ms']/a['mean_ms']));print('timing',i,flush=True)
    write(HERE/'INFERENCE_TIMING.json',dict(utc=utc(),cpu_affinity=list(os.sched_getaffinity(0)),jax_enable_x64=bool(jax.config.x64_enabled),backend=jax.default_backend(),scope='JIT warmed, prepared input network timing; teacher includes joint noise and10Euler, student one robot; QP four single-robot solves on first fixed eval state, no rollout',expert_reference=dict(mean_planning_seconds=.075,privileged_all_goals=True,source='round28'),controllers=rows));source_inventory('imports_timing.json')
if __name__=='__main__':main()
