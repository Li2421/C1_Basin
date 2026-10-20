from common import *
import unittest,tempfile
from dataclasses import replace
from policies39 import Policy,loss,decode
from training39 import normalization,earliest_best
from dagger39 import collect_episode,aggregate,validate_sources,round_specs
from geometry import GeneratedGeometry
from single_agent import SingleAgentGeneratedEnv
from bottleneck_family.scenario import Config
import jax,jax.numpy as jnp,optax

class TrainingTests(unittest.TestCase):
 def test_both_networks_use_39_inputs_and_gradients(self):
  x=np.random.default_rng(17).normal(size=(8,39)).astype(np.float32);y=np.zeros((8,2),np.float32);normal=normalization(x,y)
  for kind in ['deterministic','mixture']:
   net=Policy(kind);pa=net.init(jax.random.PRNGKey(17),jnp.zeros((1,39),jnp.float32));gradient=jax.grad(lambda p:loss(net,kind,p,jnp.asarray(x),jnp.asarray(y)))(pa)
   self.assertTrue(all(np.isfinite(np.asarray(g)).all() for g in jax.tree.leaves(gradient)))
   self.assertEqual(pa['params']['Dense_0']['kernel'].shape,(39,256))
   updated=jax.tree.map(lambda p,g:p-.0003*g,pa,gradient);self.assertTrue(np.isfinite(float(loss(net,kind,updated,jnp.asarray(x),jnp.asarray(y)))))
   u=np.asarray(decode(net,kind,pa,jnp.asarray(x),jax.random.PRNGKey(23),normal));self.assertEqual(u.shape,(8,2));self.assertLessEqual(float(np.linalg.norm(u,axis=1).max()),.5000001)
  with self.assertRaises(AssertionError):normalization(x[:,:35],y)
  self.assertEqual(earliest_best([dict(step=250,dev_loss=1),dict(step=0,dev_loss=1),dict(step=500,dev_loss=2)]),0)
  self.assertFalse(jax.config.x64_enabled)
 def test_all_three_rounds_train_only_and_fixed_dev(self):
  with tempfile.TemporaryDirectory(dir=HERE/'tmp') as directory:
   records=[]
   for role,round_index in [('round0_train',0),('round0_dev',0),('relabel_train',1),('relabel_train',2),('relabel_train',3)]:
    p=Path(directory)/f'{role}_{round_index}.npz';np.savez(p,observations=np.full((2,39),round_index,np.float32),labels=np.zeros((2,2),np.float32),actions=np.zeros((2,1,2)),positions=np.zeros((3,1,2)))
    records.append(dict(role=role,round=round_index,geometry_seed=123,trace_path=str(p),trace_sha256=sha(p)))
   for round_index in range(4):
    arrays,_=aggregate(records[:1],records[1:2],records[2:2+round_index],round_index,{123})
    self.assertEqual(len(arrays['train_X']),2*(round_index+1));self.assertTrue(np.array_equal(arrays['dev_X'],np.zeros((2,39))))
   with self.assertRaises(AssertionError):validate_sources([dict(records[0],geometry_seed=999)],{123})
   with self.assertRaises(AssertionError):aggregate(records[:1],records[1:2],records[2:],2,{123})
  spec=round_specs('B');self.assertEqual(spec['relabel_rounds'],[1,2,3]);self.assertEqual(spec['relabel_count_per_policy_round'],16)
 def test_collision_prestep_labeled_without_postterminal_state(self):
  # Legal fixture; policy deliberately hits the wall. Expert labels differ.
  geo=GeneratedGeometry(Config(num_agents=4,max_steps=10),[(-.3,-4,.3,-.3),(-.3,.3,.3,4)])
  p=np.array([[-.5,1.]]);g=np.array([[1.,1.]])
  self.assertTrue(geo.valid_points(np.r_[p,g]).all());self.assertIsNotNone(geo.path(p[0],g[0]))
  env=SingleAgentGeneratedEnv(geo,p,g)
  record=collect_episode(env,lambda env,fields,key:np.array([[.5,0.]]),jax.random.PRNGKey(17))
  self.assertEqual(record['outcome'],'collision');self.assertEqual(len(record['positions']),len(record['labels'])+1);self.assertEqual(len(record['labels']),len(record['actions']))
  self.assertLessEqual(record['wall_clearance'][-1],0);self.assertFalse(np.array_equal(record['labels'],record['actions'][:,0]))
  self.assertEqual(record['observations'].shape[1],39)

if __name__=='__main__':
 suite=unittest.defaultTestLoader.loadTestsFromTestCase(TrainingTests);result=unittest.TextTestRunner(verbosity=2).run(suite)
 write(HERE/'TRAINING39_UNIT_TESTS.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,errors=len(result.errors),failures=len(result.failures),x64_enabled=bool(jax.config.x64_enabled),checks=['39-dimensional parameter shape and finite gradients for both policies','world velocity bound','train geometry contamination rejection','all rounds1,2,3 cumulative expert labels','fixed development data','earliest development loss selection','collision pre-step labeled; no post-terminal example']))
 assert result.wasSuccessful()
