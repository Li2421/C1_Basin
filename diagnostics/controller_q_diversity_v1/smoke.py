"""Non-rollout execution checks; never supplies learning labels."""
from .experiment import HERE,read,write,sha
from .runtime import native_runtime,AnalyticField,controller_context,FIELD
import numpy as np
import jax

p=read(HERE/'protocol.json')
assert sha(FIELD/'poc/field_controller.py')==p['FF_operator_sha256']
results=[]
for scene in p['controllers']:
    state=next(s for s in p['states'] if s['scenario']==scene)
    base=native_runtime(scene)
    assert base.steps==10
    for name in ['flow','analytic0']:
        rt=base if name=='flow' else AnalyticField(base,p['controllers'][scene][name]['payload']['controller_parameters'])
        env=rt.make_env(state);key=jax.random.PRNGKey(174)
        obs,safe,flow,ref,delta,path=rt.prepare(env,key,np.asarray(p['eta'][0]),traced=True)
        action,trace=rt.integrate(env,obs,key,delta,safe,traced=True)
        direct,cert,n=rt.action(env,key,np.asarray(p['eta'][0]),'field')
        np.testing.assert_allclose(action,direct,rtol=0,atol=1e-12)
        assert len(trace['steps'])==10
        safe.assert_safe(action)
        context=controller_context(rt,state)
        assert context.shape==(len(env.positions),14) and np.isfinite(context).all()
        results.append(dict(scene=scene,controller=name,context_shape=list(context.shape),
             context_l2=float(np.linalg.norm(context)),action=action.tolist(),steps=len(trace['steps']),
             config=vars(rt.config)))
write(HERE/'execution_smoke.json',results)
print('Execution operator and non-rollout context checks passed',flush=True)
