"""World-field convex blends inside FF, with exact legacy boundary paths.

Only source checkpoint functions are changed; no changed plant or safety.
See the preregistration for the explicit nominal-reference chart convention.
"""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from common import *
import numpy as np
from diagnostics.controller_q_transfer_round3_v1.flow_adapter import RingFlowAdapter
from diagnostics.controller_q_diversity_v1.runtime import native_runtime,bounded
from poc.field_controller import Runtime,admissible_step
import jax
import jax.numpy as jnp

def anchor(name,profiles):
    p=profiles[name]['payload']
    assert p['jax_enable_x64'] is False
    jax.config.update('jax_enable_x64',False)
    assert sha(p['flow_checkpoint_path'])==p['flow_checkpoint_sha256']
    if name=='base_u_v10_local_macflow':rt=native_runtime('ring_exchange')
    else:
        assert sha(p['adapter_manifest_path'])==p['adapter_manifest_sha256']
        rt=RingFlowAdapter(p['flow_checkpoint_path'],read(p['adapter_manifest_path']),p['observation_frame'])
    assert not jax.config.x64_enabled
    return rt

class BlendRuntime:
    def __init__(self,a,b,lam):
        assert 0<=lam<=1
        ca=a.config.to_dict();cb=b.config.to_dict()
        # Schema names encode observation chart, not different plant physics.
        assert {k:v for k,v in ca.items() if k!='schema'}=={k:v for k,v in cb.items() if k!='schema'}, 'Physical configurations must match'
        assert a.steps==b.steps==10
        self.a,self.b,self.lam=a,b,float(lam)
        self.base=a;self.scenario='ring_exchange';self.config=a.config;self.basis=a.basis
        self.steps=a.steps;self.agent=a.agent;self.scale=a.scale;self.mean=a.mean
        assert a.cbf==b.cbf
    def __getattr__(self,name):return getattr(self.base,name)
    def observation(self,env):
        self._env=env
        return (jnp.asarray(self.a.observation(env)),jnp.asarray(self.b.observation(env)))
    def initial(self,env,key):
        xi=np.asarray(jax.random.normal(key,(1,8))).reshape(-1,2)
        ha=self.a.to_world(self.a.mean+self.a.scale*xi,env)
        hb=self.b.to_world(self.b.mean+self.b.scale*xi,env)
        return (1-self.lam)*ha+self.lam*hb
    def field(self,env,obs,h,t):
        fields=[]
        for rt,o in zip((self.a,self.b),obs):
            z=(rt.to_local(h,env)-rt.mean)/rt.scale
            fz=np.asarray(rt.velocity_fn(o,jnp.asarray(z),jnp.asarray(t))).reshape(-1,2)
            fields.append(rt.to_world(rt.scale*fz,env))
        return (1-self.lam)*fields[0]+self.lam*fields[1]
    def terminal(self,env,h):
        values=[]
        for rt in (self.a,self.b):
            # Preserve each chart's clipping, float32 sampler speed-bound,
            # and final float64 world speed re-bound. Interior uses common h.
            local=jnp.clip(jnp.asarray(rt.to_local(h,env)),-1.,1.)
            local=local*jnp.minimum(1.,self.config.max_speed/jnp.maximum(jnp.linalg.norm(local,axis=1,keepdims=True),1e-30))
            values.append(rt.to_world(np.asarray(local,dtype=np.float64),env))
        return bounded((1-self.lam)*values[0]+self.lam*values[1],self.config.max_speed)
    def nominal_formula(self,env,key,traced=False):
        obs=self.observation(env);h=self.initial(env,key);path=[h.copy()]
        for k in range(self.steps):
            h=h+self.field(env,obs,h,k/self.steps)/self.steps;path.append(h.copy())
        return obs,self.terminal(env,h),np.asarray(path) if traced else None
    def nominal(self,env,key,traced=False):
        if self.lam in (0.,1.):
            rt=self.a if self.lam==0 else self.b
            _,flow,path=rt.nominal(env,key,traced)
            return self.observation(env),flow,path
        return self.nominal_formula(env,key,traced)
    def velocity_fn(self,obs,z,t):
        # Adapter needed by unchanged static field_context: use A's query chart
        # only as a coordinate transport; the returned field includes A and B.
        h=self.to_world(self.mean+self.scale*np.asarray(z),self._env)
        w=self.field(self._env,obs,h,float(t))
        return (self.to_local(w,self._env)/self.scale).ravel()
    def prepare(self,env,key,eta,traced=False):return Runtime.prepare(self,env,key,eta,traced)
    def integrate(self,env,obs,key,delta,safety,*,project=True,traced=False):
        h=self.initial(env,key);path=[h.copy()];fields=[];targets=[];records=[]
        for k in range(self.steps):
            s=k/self.steps;ds=1/self.steps;proposed=self.field(env,obs,h,s)+delta
            if project:new_h,w,target,d=admissible_step(h,proposed,s,ds,safety)
            else:new_h,w,target,d=h+ds*proposed,proposed,h+ds*proposed,{}
            if traced:
                fields.append(w);targets.append(target)
                records.append(dict(s=s,field_projection_norm=float(np.linalg.norm(w-proposed)),solver_status=d.get('status')))
            h=new_h;path.append(h.copy())
        if project:safety.assert_safe(h)
        return h,dict(path=np.asarray(path),fields=np.asarray(fields),targets=np.asarray(targets),steps=records)
    def action_formula(self,env,key,eta):return Runtime.action(self,env,key,eta,'field')
    def action(self,env,key,eta,chain='field'):
        assert chain=='field'
        if self.lam==0:return self.a.action(env,key,eta,chain)
        if self.lam==1:return self.b.action(env,key,eta,chain)
        return self.action_formula(env,key,eta)

def make(name,p,cache):
    prof=p['controllers']['ring_exchange'][name]['payload']
    if 'blend' not in prof:
        if name not in cache:cache[name]=anchor(name,p['controllers']['ring_exchange'])
        return cache[name]
    if name not in cache:
        q=prof['blend'];cache[name]=BlendRuntime(make(q['A'],p,cache),make(q['B'],p,cache),q['lambda'])
    return cache[name]

def audit():
    from ring_exchange.environment import RingExchangeEnv
    from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for
    p=read(PREVIOUS/'protocol.json');profiles=p['controllers']['ring_exchange'];cache={}
    old=RingExchangeEnv.step
    def forbidden(*args,**kwargs):raise RuntimeError('endpoint audit called env.step')
    RingExchangeEnv.step=forbidden;rows=[]
    try:
        for pair,A,B in [(0,'base_u_macflow','base_u_plus_w_v4_wide_macflow'),(1,'base_u_macflow','base_u_v9_local_macflow'),(2,'base_u_v10_local_macflow','base_u_plus_w_v4_wide_macflow'),(3,'base_u_v10_local_macflow','base_u_v9_local_macflow')]:
            for name in (A,B):
                if name not in cache:cache[name]=anchor(name,profiles)
            for lam,name in [(0.,A),(1.,B)]:
                rt=BlendRuntime(cache[A],cache[B],lam);errors=[];formula=[];near=[]
                for s in p['states'][:2]:
                    for seed in (0,1):
                        key=key_for(s,seed,0)
                        for eta in [np.zeros(3)]+[np.asarray(p['eta'][j]) for j in (2,4,10,13)]:
                            env=rt.make_env(s);ref=cache[name].action(env,key,eta,'field')[0]
                            got=rt.action(env,key,eta)[0];raw=rt.action_formula(env,key,eta)[0]
                            inside=BlendRuntime(cache[A],cache[B],1e-6 if lam==0 else 1-1e-6).action(env,key,eta)[0]
                            errors.append(float(np.max(np.abs(ref-got))));formula.append(float(np.max(np.abs(ref-raw))));near.append(float(np.max(np.abs(ref-inside))))
                row=dict(pair=pair,lambda_value=lam,anchor=name,exact_boundary_max_error=max(errors),unbranched_formula_max_error=max(formula),near_boundary_max_error=max(near),queries=len(errors))
                rows.append(row);print(json.dumps(row),flush=True)
    finally:RingExchangeEnv.step=old
    rec=dict(status='PASS' if max(r['exact_boundary_max_error'] for r in rows)<=1e-10 and max(r['unbranched_formula_max_error'] for r in rows)<=1e-4 else 'FAIL',jax_enable_x64=False,physical_steps=0,rows=rows,runtime_sha256=sha(__file__),numerical_boundary_branch='Explicit preservation of legacy float32 evaluation order, not evidence of exact floating-point continuity')
    write(HERE/'ENDPOINT_EQUIVALENCE_AUDIT.json',rec);assert rec['status']=='PASS',rec

if __name__=='__main__':audit()
