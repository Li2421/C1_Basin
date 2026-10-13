"""Controls that can detect swapped axes, spurious switching or lost masks."""
import numpy as np
import jax
import jax.numpy as jnp
from diagnostics.orthoflow3_unified_representation_v1 import representation as rep
from .experiment import HERE,read,write
from .train import Critic,KINDS,compact_field
from .evaluate import reversal,square_gain_bounds,center_known,masked_mean,interaction_skill_bounds


def run():
    n=8;truth=np.zeros((n,2,16))
    truth[:4,0,0]=1;truth[4:,0,1]=1
    truth[:4,1,1]=1;truth[4:,1,0]=1
    boot=np.random.default_rng(99).integers(n,size=(2000,n))
    perfect,_=reversal(truth,truth,truth,truth,truth,boot)
    fixed=np.broadcast_to(np.arange(16)[None,None,:],truth.shape)
    dummy,_=reversal(fixed,truth,truth,truth,truth,boot)
    assert perfect['switch_accuracy']==1 and perfect['micro_direction_accuracy']==1
    assert dummy['switch_accuracy']==0 and dummy['balanced_direction_accuracy']==.5
    rng=np.random.default_rng(97);lo=rng.random(100);hi=lo+(1-lo)*rng.random(100)
    model=rng.random(100);baseline=rng.random(100)
    a,b=square_gain_bounds(baseline,model,lo,hi)
    for t in np.linspace(0,1,101):
        q=lo+t*(hi-lo);v=(baseline-q)**2-(model-q)**2
        assert np.all(v>=a-1e-12) and np.all(v<=b+1e-12)
    prediction=rng.normal(size=(8,16));bottom=rng.normal(size=(8,16));top=bottom+rng.random((8,16))
    for eta_center in (False,True):
        bound_lo,bound_hi=interaction_skill_bounds(prediction,bottom,top,eta_center)
        for _ in range(101):
            truth=bottom+rng.random((8,16))*(top-bottom)
            projected_p=prediction-prediction.mean(0);projected_y=truth-truth.mean(0)
            if eta_center:
                projected_p-=projected_p.mean(1,keepdims=True);projected_y-=projected_y.mean(1,keepdims=True)
            skill=np.mean(projected_y**2-(projected_p-projected_y)**2)
            assert bound_lo-1e-12<=skill<=bound_hi+1e-12
        prior=np.broadcast_to(rng.random(16),(8,16))
        z0,z1=interaction_skill_bounds(prior,bottom,top,eta_center)
        assert abs(z0)<1e-12 and abs(z1)<1e-12
    # Compression preserves intercepts and the exact Jacobian of every affine
    # analytic controller. Independent time/pose intercepts remain available.
    intercept=rng.normal(size=(2,4,7,3,2));lam=np.array([2.,4.])[:,None,None,None,None,None]
    vectors=np.array([[0,0],[.5,0],[-.5,0],[0,.5],[0,-.5]])
    fields=intercept[...,None,:]-lam*vectors
    raw=np.concatenate((np.zeros((2,4,14)),fields.reshape(2,4,210)),-1)
    compact=np.asarray(compact_field(jnp.asarray(raw)))
    assert compact.shape==(2,4,64)
    np.testing.assert_allclose(compact[...,14:56],intercept.reshape(2,4,42),atol=2e-6)
    expected=np.broadcast_to(np.array([[-2,0,0,-2],[-4,0,0,-4]])[:,None,:],(2,4,4))
    np.testing.assert_allclose(compact[...,56:60],expected,atol=2e-6)
    np.testing.assert_allclose(compact[...,60:],0,atol=2e-10)
    observed=np.array([[0.,1.],[999.,3.],[2.,999.]])
    mask=np.array([[1,1],[0,1],[1,0]],bool)
    centered=center_known(observed,mask)
    np.testing.assert_allclose(masked_mean(centered,mask,axis=0),0,atol=1e-12)
    p=read(HERE/'protocol.json');items=[];context=[]
    for sc in p['controllers']:
        s=next(s for s in p['states'] if s['scenario']==sc)
        d=np.load(HERE/'features'/s['state_uid']/'analytic0.npz')
        items.append({k[7:]:d[k] for k in d.files if k.startswith('entity_')})
        c=np.concatenate((d['nominal'],d['field']),-1)
        context.append(np.pad(c,((0,4-len(c)),(0,0))))
    x={k:jnp.asarray(v) for k,v in rep.batch(items).items()};c=jnp.asarray(np.asarray(context))
    eta=jnp.asarray(np.asarray(p['eta'][:2],np.float32));perm=np.array([2,0,3,1]);xp=dict(x)
    xp['agents']=x['agents'][:,perm];xp['agent_mask']=x['agent_mask'][:,perm]
    xp['pairs']=x['pairs'][:,perm][:,:,perm];xp['obstacles']=x['obstacles'][:,perm]
    differences={}
    for kind in KINDS:
        net=Critic(kind);parameters=net.init(jax.random.PRNGKey(17),x,c,eta)
        v=np.asarray(net.apply(parameters,x,c,eta));w=np.asarray(net.apply(parameters,xp,c[:,perm],eta))
        np.testing.assert_allclose(v,w,rtol=2e-6,atol=2e-6)
        gradient=np.asarray(jax.grad(lambda cc:jnp.sum(net.apply(parameters,x,cc,eta)))(c))
        if kind in ('eta_only','state_only'):assert np.max(np.abs(gradient))==0
        else:assert np.linalg.norm(gradient)>0
        differences[kind]=dict(permutation_max_abs=float(np.max(np.abs(v-w))),controller_input_gradient=float(np.linalg.norm(gradient)))
    write(HERE/'analysis_contract_checks.json',dict(passed=True,perfect_switch=perfect,fixed_switch=dummy,
          interval_bound_grid_points=101,models=differences,success_labels_read=False,
          compact_affine_response_check=True,all_unknown_mask_check=True,
          full_support_interaction_bounds_check=True))
    print('Analytic truth, censor bounds, input gradients, and entity permutation checks passed',flush=True)


if __name__=='__main__':run()
