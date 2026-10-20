from common import *
from conservative_reward import ConservativeEnv,reward_delta
from training_env import TrainingEnv
import jax
import jax.numpy as jnp


def main():
    config=read(D/'crosstype1_learned_zoo_selector_v1/IPPO_TRAINING_CONFIGS.json')[0]['environment']
    old=TrainingEnv(config);new=ConservativeEnv(config)
    state=old.reset(jnp.asarray([[2.,0.],[0.,2.],[-2.,0.],[0.,-2.]]),jnp.zeros((4,2)),jnp.asarray([[-2.,0.],[0.,-2.],[2.,0.],[0.,2.]]))
    a=old.step(state,jnp.zeros((4,2)));b=new.step(state,jnp.zeros((4,2)))
    assert all(np.asarray(x).tobytes()==np.asarray(y).tobytes() for x,y in zip(jax.tree.leaves((a[0],a[2],a[3])),jax.tree.leaves((b[0],b[2],b[3]))))
    assert np.array_equal(np.asarray(a[1]),np.asarray(b[1]))
    info=dict(collided=jnp.asarray([True,False,False,False]),swept_agent_distances=jnp.ones(6),swept_obstacle_distance=jnp.ones(4),swept_outer_distance=jnp.ones(4))
    assert np.array_equal(np.asarray(reward_delta(info,jnp.ones(4,bool))),[-45,-15,-15,-15])
    info['collided']=jnp.zeros(4,bool)
    for gap,expected in [(0.,-.05),(.1,-.025),(.2,0.),(.3,0.)]:
        info['swept_obstacle_distance']=jnp.full(4,gap)
        assert np.allclose(reward_delta(info,jnp.ones(4,bool)),expected,atol=1e-8)
    assert np.all(np.asarray(reward_delta(info,jnp.zeros(4,bool)))==0)
    write(HERE/'CONSERVATIVE_REWARD_TESTS.json',dict(utc=utc(),passed=True,plant_identical=True,own_collision=-60,other_collision=-20,proximity_threshold=.2,proximity_coefficient=.05,training_env_source_sha256=sha(HERE/'training_env.py')))
    print('Reward tests passed; dynamics/events bitwise unchanged')


if __name__=='__main__':main()
