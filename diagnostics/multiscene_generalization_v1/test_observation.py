from common import *
from types import SimpleNamespace
from bottleneck_family.scenario import Config
from geometry import GeneratedGeometry,GeneratedEnv
from single_agent import SingleAgentGeneratedEnv
from observation import observe

def main():
    geo=GeneratedGeometry(Config(num_agents=4),[])
    p=np.array([[-2.,-.8],[-.3,1.1],[1.4,.2],[2.2,-1.8]])
    env=GeneratedEnv(geo,p,p*.5);env.velocities=np.array([[.1,.2],[.2,-.1],[-.1,.1],[-.2,-.1]])
    x=observe(env);assert x.shape==(4,35)
    assert np.array_equal(x[:,:2],env.velocities.astype(np.float32))
    assert np.array_equal(x[:,2:4],(env.goals-env.positions).astype(np.float32))
    assert np.all((x[:,4:20]>=0)&(x[:,4:20]<=3))
    assert np.all(x[:,20:].reshape(4,3,5)[:,:,-1]==1)
    rotate=lambda a:np.stack([-a[...,1],a[...,0]],axis=-1)
    shifted=SimpleNamespace(positions=p+[.7,-.4],velocities=env.velocities,goals=env.goals+[.7,-.4],walls=env.walls+[.7,-.4])
    assert np.allclose(x,observe(shifted),rtol=0,atol=1e-6)
    turned=SimpleNamespace(positions=rotate(p),velocities=rotate(env.velocities),goals=rotate(env.goals),walls=rotate(env.walls))
    y=observe(turned)
    assert np.allclose(y[:,:2],rotate(x[:,:2]),atol=1e-6)
    assert np.allclose(y[:,2:4],rotate(x[:,2:4]),atol=1e-6)
    assert np.allclose(y[:,4:20],np.roll(x[:,4:20],4,axis=1),atol=1e-6)
    solo=SingleAgentGeneratedEnv(geo,p[:1],env.goals[:1]);sx=observe(solo)
    assert np.array_equal(sx[:,20:],np.zeros((1,15),np.float32))
    from generated_runtime import clockwise_goal
    delta=env.goals-env.positions;cw=clockwise_goal(env.positions,env.goals)
    assert np.allclose((cw*delta).sum(1),0)
    assert np.all(delta[:,0]*cw[:,1]-delta[:,1]*cw[:,0]<0)
    write(HERE/'OBSERVATION_UNIT_TESTS.json',dict(passed=True,dimension=35,world_velocity_and_goal=True,rays16=True,neighbors3_with_masks=True,translation_and_rotation_chart=True,single_agent_padding=True,clockwise_goal_tangent=True))

if __name__=='__main__':main()
