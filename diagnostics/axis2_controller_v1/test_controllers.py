from common import *
from controllers import NovelNominal,GaussianNominal,closest_distance
import jax,copy
jax.config.update('jax_enable_x64',False)
p=read(D/'other_controller_classes_v1/main_protocol.json');state=p['states'][0];manifest=p['native_manifest'];cfg=read(HERE/'DESIGN_CONFIG.json');checks=[]
def check(rt,name):
 env=rt.make_env(state);key=jax.random.PRNGKey(7);a=rt.nominal(env,key)[1];b=rt.nominal(env,key)[1];assert a.shape==(4,2) and np.isfinite(a).all() and np.linalg.norm(a,axis=-1).max()<=rt.config.max_speed+1e-12;assert a.tobytes()==b.tobytes();assert env.step_count==0 and not env.done;checks.append(name)
for kind in ['DM','VO','PT']:
 for i,param in enumerate(cfg[kind]):
  param=param+cfg['PT_repulsion'] if kind=='PT' else param;check(NovelNominal(kind,param,manifest),kind+str(i))
q=read(D/'controller_universal_eta_schedule_v1/protocol.json')
for row in cfg['G']:
 v=copy.deepcopy(q['base_profiles'][row['recipe']]['payload']);v['flow_checkpoint_path']=str(HERE/'gaussian'/row['name']/'best.msgpack');v['flow_checkpoint_sha256']=sha(v['flow_checkpoint_path']);check(GaussianNominal(v),row['name'])
assert closest_distance(np.array([1.,0.]),np.array([-1.,0.]),2.)==0
assert closest_distance(np.array([1.,0.]),np.zeros(2),2.)==1
write(HERE/'UNIT_TESTS.json',dict(utc=utc(),passed=True,controllers_checked=checks,speed_and_shape=True,same_key_determinism=True,simulation_steps_executed=0,no_task_labels_generated=True,relative_motion_collision_test=True,jax_enable_x64=bool(jax.config.x64_enabled)))
print('PASS24 nominal sources, speed bounds, deterministic keys, analytic closest approach; no env.step')
