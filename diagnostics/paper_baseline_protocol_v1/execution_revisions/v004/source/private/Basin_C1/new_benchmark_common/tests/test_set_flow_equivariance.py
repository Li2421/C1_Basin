import unittest

import jax.numpy as jnp
import numpy as np

from new_benchmark_common.macflow import JointMACFlowAgent, SetActorVectorField, get_config


class SetFlowEquivarianceTests(unittest.TestCase):
    def test_joint_field_relabels_agents_without_changing_actions(self):
        n = 5
        config = get_config(agent_order=tuple(f'a{i}' for i in range(n)),
            num_agents=n, obs_dim=8, act_dim=2, normalize=False,
            architecture='set_attention', set_width=32, set_layers=1,
            motion_loss_weight=3.0)
        observation = jnp.arange(2 * n * 8,dtype=jnp.float32).reshape(2,n,8) / 17
        action = jnp.arange(2 * n * 2,dtype=jnp.float32).reshape(2,n,2) / 31
        agent = JointMACFlowAgent.create(11,observation,action,config)
        self.assertIsInstance(agent.network.model_def.modules['actor_bc_flow'],SetActorVectorField)
        times = jnp.asarray([[.25],[.75]],dtype=jnp.float32)
        field = agent.network.select('actor_bc_flow')
        output = field(observation.reshape(2,-1),action.reshape(2,-1),times,
                       params=agent.network.params).reshape(2,n,2)
        permutation = np.asarray([4,1,3,0,2])
        relabeled = field(observation[:,permutation].reshape(2,-1),
                          action[:,permutation].reshape(2,-1),times,
                          params=agent.network.params).reshape(2,n,2)
        np.testing.assert_allclose(np.asarray(relabeled),np.asarray(output[:,permutation]),
                                   rtol=1e-5,atol=1e-5)


if __name__ == '__main__':
    unittest.main()
