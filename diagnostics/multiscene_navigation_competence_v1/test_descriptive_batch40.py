from common import *
from descriptive_batch40 import evaluate
from dagger39 import collect_episode
from geometry import GeneratedGeometry
from single_agent import SingleAgentGeneratedEnv
from bottleneck_family.scenario import Config
from dataclasses import replace
from policies39 import Policy,load_policy
from flax import serialization
import jax,jax.numpy as jnp,tempfile

def main():
    geo=GeneratedGeometry(Config(num_agents=4),[(-.3,-4,.3,-.3),(-.3,.3,.3,4)])
    fixtures=[([-0.5,1.],[1.,1.],[.5,0.],10),([-2.,0.],[-1.9,0.],[.5,0.],10),([-2.,-1.],[-1.,-1.],[0.,0.],3)]
    def env(f):
        p,g,u,h=f;assert geo.valid_points(np.asarray([p,g])).all();assert geo.path(np.asarray(p),np.asarray(g)) is not None
        return SingleAgentGeneratedEnv(geo,[p],[g],replace(geo.config,max_steps=h))
    class Constant:
        def __init__(self,u):self.u=np.asarray([u])
        def __call__(self,env,fields,key):return self.u.copy()
        def from_observation(self,env,observed,key):return self.u.copy()
    keys=[jax.random.PRNGKey(i) for i in range(3)];checks=[]
    def compare(actors,label):
        a=[collect_episode(env(f),p,k) for f,p,k in zip(fixtures,actors,keys)];b=evaluate([env(f) for f in fixtures],actors,keys)
        for i,(old,new) in enumerate(zip(a,b)):
            assert old['outcome']==new['outcome'];assert new['error'] is None
            for name in ['actions','positions','wall_clearance']:assert old[name].dtype==new[name].dtype and old[name].tobytes()==new[name].tobytes(),(label,i,name)
            checks.append(dict(actor=label,case=i,outcome=new['outcome'],steps=len(new['actions']),all_arrays_bitwise=True))
        return b
    b=compare([Constant(f[2]) for f in fixtures],'constant')
    assert [r['outcome'] for r in b]==['collision','success','timeout'];assert b[0]['wall_clearance'][-1]<0
    with tempfile.TemporaryDirectory(dir=HERE/'tmp') as folder:
        for kind in ['deterministic','mixture']:
            p=Path(folder)/kind;p.mkdir();net=Policy(kind);params=net.init(jax.random.PRNGKey(17),jnp.zeros((1,39),jnp.float32));(p/'best.msgpack').write_bytes(serialization.to_bytes(params));(p/'CONFIG.json').write_text(json.dumps(dict(family=kind,observation_dimensions=39)));(p/'NORMALIZATION.json').write_text(json.dumps(dict(obs_mean=[0]*39,obs_scale=[1]*39,act_mean=[0]*2,act_scale=[1]*2)))
            policy=load_policy(p/'best.msgpack');compare([policy]*3,kind)
    write(HERE/'DESCRIPTIVE_BATCH40_TEST.json',dict(passed=True,checks=checks,untrained_unit_checkpoints_only=True,no_expert_labels_in_evaluation=True,independent_native_termination=True,x64_enabled=jax.config.jax_enable_x64))
    print(checks)

if __name__=='__main__':main()
