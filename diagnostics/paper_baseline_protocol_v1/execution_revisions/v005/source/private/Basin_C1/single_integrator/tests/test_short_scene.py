import unittest
import numpy as np
from single_integrator.environment import Config,GiveWayEnv
from single_integrator.expert import Expert


class ShortSceneTests(unittest.TestCase):
    def test_bottleneck_unchanged_and_ends_moved(self):
        original=GiveWayEnv()
        short=GiveWayEnv(Config(corridor_half_length=1.3))
        np.testing.assert_array_equal(original.walls[5:],short.walls[5:])
        np.testing.assert_array_equal(original.walls[:,:,1],short.walls[:,:,1])
        np.testing.assert_allclose(short.goals,[[1.09,0],[-1.09,0]])
        self.assertTrue(short.outside(np.array([[1.31,0],[0,.69]])).all())
        for mode in [0,1]:
            short.reset([[-.85,0],[.85,0]])
            expert=Expert(short,mode)
            for _ in range(short.config.max_steps):
                _,_,done,info=short.step(expert.action())
                self.assertFalse(info['wall_collision'] or info['agent_collision'])
                if done:break
            self.assertTrue(info['task_success'])


if __name__=='__main__':unittest.main()
