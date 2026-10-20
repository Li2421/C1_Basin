from core import *
from guard import guard
from student import *
from audit_utils import source_inventory
from ring_exchange.local_frame import world_actions_to_local
G=guard()
def main():
 rng=np.random.default_rng(2026101606);base=teacher(10);env=base.env_cls(base.config);p=np.array([[2.,.2],[-.3,2.2],[-2.1,-.1],[.1,-2.]]);v=rng.normal(size=(4,2))*.1;g=-p;env.reset(p,velocities=v,goals=g);x=local_observation(p,v,g,env.config);z=rng.normal(size=(4,2)).astype(np.float32);q=np.array([[np.cos(.713),-np.sin(.713)],[np.sin(.713),np.cos(.713)]]);rx=local_observation(p@q.T,v@q.T,g@q.T,env.config)
 np.testing.assert_allclose(x,rx,atol=1e-6)
 loc=np.asarray(joint_local_noise(jnp.asarray(x),jnp.asarray(z)));rloc=np.asarray(joint_local_noise(jnp.asarray(rx),jnp.asarray(z)));np.testing.assert_allclose(loc,rloc,atol=1e-6)
 world=local_actions_to_world(z,p);expected=[]
 for i in range(4):
  radial=p[i]/np.linalg.norm(p[i]);tangent=np.array([-radial[1],radial[0]])
  expected.append(np.stack((world@radial,world@tangent),-1).ravel())
 expected=np.asarray(expected)
 np.testing.assert_allclose(loc,expected,atol=1e-6)
 glob=np.asarray(joint_global_noise(jnp.asarray(z),jnp.asarray(p)));rglob=np.asarray(joint_global_noise(jnp.asarray(z@q.T),jnp.asarray(p@q.T)));np.testing.assert_allclose(glob,rglob,atol=1e-6)
 # Alter unrelated observer rows: row i encoding is exactly unchanged.
 for i in range(4):
  xx=x.copy();xx[np.arange(4)!=i]+=5;np.testing.assert_array_equal(np.asarray(joint_local_noise(jnp.asarray(xx),jnp.asarray(z)))[i],loc[i])
  gg=g.copy();gg[np.arange(4)!=i]+=2;xx=local_observation(p,v,gg,env.config);np.testing.assert_array_equal(xx[i],x[i])
  model=network();par=model.init(jax.random.PRNGKey(1),jnp.zeros(27),jnp.zeros(8))['params'];fn=jax.jit(lambda a,b:model.apply({'params':par},a,b));a=fn(inputs(jnp.asarray(x),0,1)[i],loc[i]);b=fn(inputs(jnp.asarray(xx),0,1)[i],joint_local_noise(jnp.asarray(xx),jnp.asarray(z))[i]);np.testing.assert_array_equal(a,b)
 checks=[]
 for i in range(14):
  rt=teacher(i);oo=np.load(D/'teacher_student_macflow_v1/data'/read(HERE/'TEACHERS.json')[i]['name']/'dev_teacher.npy',mmap_mode='r')[0];k=public_key(jax.random.PRNGKey(93));actual=np.asarray(rt.trace_fn(jnp.asarray(oo),k)[0]).reshape(4,2);draw=np.asarray(jax.random.normal(k,(1,8),dtype=jnp.float32)).reshape(4,2);np.testing.assert_array_equal(actual,draw);checks.append(dict(index=i,bitwise_equal=True))
 write(HERE/'UNIT_TESTS.json',dict(utc=utc(),status='PASS',guard=guard(),local_rotation=True,global_coupled_vector_rotation=True,global_fixed_key_rotation_max_difference=float(abs(glob-np.asarray(joint_global_noise(jnp.asarray(z),jnp.asarray(p@q.T)))).max()),source_frames_from_observer_row_only=True,other_goals_output_invariant=True,teacher_noise_bitwise=checks));source_inventory('imports_tests.json');print('tests PASS',flush=True)
if __name__=='__main__':main()
