"""Remove device round trips from analytic policy function calls only.

The original frozen Runtime.integrate remains the execution entry point.
Its two jnp.asarray calls serialize arrays immediately converted back to NumPy
by AnalyticField.velocity_fn, which ignores pseudo-time. This adapter supplies
NumPy as that function's 'jnp' global without modifying the source module or
any other runtime. All integration, projection and arithmetic stay original.
"""
from __future__ import annotations
import types
import numpy as np
from .runtime import AnalyticField


class HostArrays:
    @staticmethod
    def asarray(value):
        import jax
        x=np.asarray(value)
        # Reproduce jnp.asarray's native scene-specific canonical dtype.
        if not jax.config.x64_enabled:
            if x.dtype==np.float64:x=x.astype(np.float32)
            elif x.dtype==np.int64:x=x.astype(np.int32)
            elif x.dtype==np.uint64:x=x.astype(np.uint32)
        return x


class NumpyAnalyticField(AnalyticField):
    def integrate(self,env,obs,key,delta,safety,**kwargs):
        from poc.field_controller import Runtime
        # Clone the function with isolated globals; no global monkey patch.
        if not hasattr(self,'_numpy_integrate'):
            globals_copy=dict(Runtime.integrate.__globals__)
            globals_copy['jnp']=HostArrays
            self._numpy_integrate=types.FunctionType(Runtime.integrate.__code__,globals_copy,
                                                     Runtime.integrate.__name__,Runtime.integrate.__defaults__)
            self._numpy_integrate.__kwdefaults__=Runtime.integrate.__kwdefaults__
        return self._numpy_integrate(self,env,obs,key,delta,safety,**kwargs)


def audit():
    import time
    from .experiment import HERE,read,write,sha
    from .runtime import native_runtime
    from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for
    p=read(HERE/'protocol.json');rows=[]
    for scene in p['controllers']:
        base=native_runtime(scene)
        states=[s for s in p['states'] if s['scenario']==scene and s['study_split']=='train'][:8]
        for ci in range(4):
            params=p['controllers'][scene][f'analytic{ci}']['payload']['controller_parameters']
            old=AnalyticField(base,params);new=NumpyAnalyticField(base,params)
            for s in states:
                for j in range(0,16,3):
                    outputs=[];times=[];errors=[]
                    for rt in [old,new]:
                        env=rt.make_env(s);start=time.monotonic()
                        try:
                            out=rt.action(env,key_for(s,0,0),np.asarray(p['eta'][j]),'field')
                            outputs.append(out[0]);errors.append(None)
                        except Exception as ex:
                            outputs.append(None);errors.append((type(ex).__name__,str(ex)))
                        times.append(time.monotonic()-start)
                    assert errors[0]==errors[1]
                    if errors[0] is None:np.testing.assert_array_equal(outputs[0],outputs[1])
                    rows.append(dict(scene=scene,controller=ci,state=s['state_uid'],eta=j,
                                     exact_equal=True,error=errors[0],seconds=times))
    write(HERE/'numpy_backend_audit.json',dict(non_rollout=True,labels_generated=0,cases=len(rows),
            every_action_bitwise_equal=True,code_sha256=sha(__file__),rows=rows))
    a=np.asarray([r['seconds'] for r in rows])
    print(dict(cases=len(rows),old_seconds=a[:,0].sum(),new_seconds=a[:,1].sum(),
               median_speedup=np.median(a[:,0]/a[:,1])),flush=True)


if __name__=='__main__':audit()
