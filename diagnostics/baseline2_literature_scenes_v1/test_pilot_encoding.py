from experiment import *
from pilot_policies import observation,PilotEnv,Policy
import numpy as np
import jax
import jax.numpy as jnp
import unittest

class Encoding(unittest.TestCase):
    def test_global_coordinates_and_agent_slot(self):
        s=read(HERE/'FLOW_STATES_lit_narrow_passage.json')[0];full=PilotEnv(s);x,slots=observation(full,3);self.assertEqual(slots,[0,1,2]);self.assertEqual(x.shape,(57,))
        for i in range(3):
            row=x.reshape(3,-1)[i];np.testing.assert_array_equal(row[:2],np.asarray(s['positions'][i],np.float32));np.testing.assert_array_equal(row[4:6],np.asarray(s['goals'][i],np.float32)-np.asarray(s['positions'][i],np.float32))
        solo=dict(s,N=1,positions=[s['positions'][2]],goals=[s['goals'][2]],velocities=[[0.,0.]],parent_N=3,agent=2);env=PilotEnv(solo);y,slots=observation(env,3)
        self.assertEqual(env.num_agents,1);self.assertEqual(slots,[2]);np.testing.assert_array_equal(y.reshape(3,-1)[:2],0);np.testing.assert_array_equal(y.reshape(3,-1)[2,:6],x.reshape(3,-1)[2,:6]);np.testing.assert_array_equal(y.reshape(3,-1)[2,6:14],0)

    def test_only_input_output_counts_change(self):
        for n in (2,3,5):
            for family in ('deterministic','mixture'):
                net=Policy(family,n);X=jnp.zeros((1,n*(4*n+7)));params=net.init(jax.random.PRNGKey(17),X);out=net.apply(params,X)
                self.assertEqual(out[1].shape if family=='mixture' else out.shape,(1,5,2*n) if family=='mixture' else (1,2*n))
                self.assertEqual(params['params']['Dense_1']['kernel'].shape,(256,256))

if __name__=='__main__':unittest.main()
