"""Predeclared analytic/unit tests; no locked task labels or rollouts."""
from common import *
import jax,jax.numpy as jnp

def transformer():
    from unseen_models import TransformerPolicy,gaussian_nll,sample,decode
    from flax import serialization
    assert jax.default_backend()=='gpu';rng=np.random.default_rng(193031);rows=[]
    with jax.default_matmul_precision('highest'):
        for layers in (2,4):
            net=TransformerPolicy(layers);x=jnp.asarray(rng.normal(size=(5,92)),jnp.float32);parameters=net.init(jax.random.PRNGKey(1),x);mean,logs=net.apply(parameters,x);assert mean.shape==(5,8) and logs.shape==(5,8);assert np.all(np.asarray(logs)>=-5) and np.all(np.asarray(logs)<=1.5)
            perm=np.asarray([2,0,3,1]);pm,pl=net.apply(parameters,x.reshape(5,4,23)[:,perm].reshape(5,92));assert np.allclose(pm.reshape(5,4,2),np.asarray(mean).reshape(5,4,2)[:,perm],rtol=1e-5,atol=2e-5);assert np.allclose(pl.reshape(5,4,2),np.asarray(logs).reshape(5,4,2)[:,perm],rtol=1e-5,atol=2e-5)
            grad=jax.grad(lambda p:gaussian_nll(*net.apply(p,x),jnp.zeros((5,8))).mean())(parameters);assert all(np.isfinite(a).all() for a in jax.tree_util.tree_leaves(grad));normal=dict(obs_mean=[0.]*92,obs_scale=[1.]*92,act_mean=[0.]*8,act_scale=[1.]*8,max_speed=.52);restored=serialization.from_bytes(parameters,serialization.to_bytes(parameters));a=np.asarray(sample(net,parameters,x[0].reshape(4,23),jax.random.PRNGKey(7),normal));b=np.asarray(sample(net,restored,x[0].reshape(4,23),jax.random.PRNGKey(7),normal));assert a.tobytes()==b.tobytes();assert np.max(np.linalg.norm(a,axis=-1))<=.52+1e-7;rows.append(dict(layers=layers,permutation_equivariance=True,finite_nll_gradient=True,serialization_same_key_bitwise=True,bounded_speed=True))
    write(HERE/'TRANSFORMER_UNIT_TESTS.json',dict(utc=utc(),passed=True,checks=rows,x64=bool(jax.config.x64_enabled),backend=jax.default_backend(),locked_outcomes_read=False));seal('TRANSFORMER_UNIT_TESTS.receipt.json',[HERE/'TRANSFORMER_UNIT_TESTS.json',HERE/'test_unseen.py',HERE/'unseen_models.py',HERE/'OPENING_CONFIG.receipt.json'])
    print('Transformer Gaussian/set/gradient/same-key/restore tests passed',flush=True)

def optimizers():
    from gradient_mpc import shooting_cost_jnp,linear_actions,GradientMPC,norm2
    from legacy_controllers import shooting_cost,Nominal
    from conflict_based_search import search,first_conflict
    from fast_astar import FastGrid
    from types import SimpleNamespace
    from mixed_controllers import make
    manifest=read(HERE/'BASE_PROTOCOL.json')['native_manifest'];cfg=Nominal('test',[],manifest).config;rng=np.random.default_rng(193032);errors={}
    for dtype,enabled,tol in [(np.float32,False,1e-6),(np.float64,True,1e-12)]:
        maximum=0.;count=0
        with jax.experimental.enable_x64(enabled):
            for horizon in (10,30):
                pos=rng.uniform(-4,4,(500,horizon,4,2)).astype(dtype);act=rng.uniform(-.52,.52,(500,horizon,4,2)).astype(dtype);goals=rng.uniform(-3,3,(4,2)).astype(dtype);previous=rng.uniform(-.52,.52,(4,2)).astype(dtype);expected=shooting_cost(pos,act,goals,cfg,100,previous);actual=np.asarray(shooting_cost_jnp(jnp.asarray(pos),jnp.asarray(act),jnp.asarray(goals),cfg,100,jnp.asarray(previous)));relative=np.max(np.abs(actual-expected)/np.maximum(np.abs(expected),np.finfo(dtype).tiny));maximum=max(maximum,float(relative));count+=500
        assert count==1000 and maximum<=tol,(dtype,maximum,tol);errors[str(dtype)]=maximum
    derivatives=[]
    with jax.experimental.enable_x64(True):
        for initial in (np.asarray([[2,0],[0,2],[-2,0],[0,-2.]],float),np.asarray([[1,0],[1.2,0],[0,1],[0,1.2]],float)):
            knots=rng.uniform(-.2,.2,(3,4,2));goals=-initial;previous=np.zeros((4,2))
            def loss(x):
                actions=linear_actions(x,10);positions=jnp.asarray(initial)[None]+cfg.dt*jnp.cumsum(actions,0);return shooting_cost_jnp(positions[None],actions[None],jnp.asarray(goals),cfg,100,jnp.asarray(previous))[0]
            actual=np.asarray(jax.grad(loss)(jnp.asarray(knots)));expected=np.empty_like(knots);eps=1e-5
            for idx in np.ndindex(knots.shape):
                a=knots.copy();b=knots.copy();a[idx]+=eps;b[idx]-=eps;expected[idx]=(float(loss(jnp.asarray(a)))-float(loss(jnp.asarray(b))))/(2*eps)
            assert np.allclose(actual,expected,rtol=5e-5,atol=1e-6);derivatives.append(float(np.max(abs(actual-expected))))
            assert np.isfinite(np.asarray(jax.grad(lambda a:norm2(a).sum())(jnp.zeros((4,2))))).all()
    jax.config.update('jax_enable_x64',False)
    initial=np.asarray([[2,0],[0,2],[-2,0],[0,-2.]],float);env=lambda:SimpleNamespace(positions=initial.copy(),goals=-initial,velocities=np.zeros((4,2)),config=cfg,step_count=0)
    a=GradientMPC([10,20,.05],manifest).nominal(env(),jax.random.PRNGKey(9))[1];b=GradientMPC([10,20,.05],manifest).nominal(env(),jax.random.PRNGKey(9))[1];assert a.tobytes()==b.tobytes() and np.isfinite(a).all() and np.max(np.linalg.norm(a,axis=-1))<=cfg.max_speed+1e-12
    starts=np.asarray([[0.,0.],[4.,0.]]);goals=starts[::-1].copy();args=dict(resolution=1.,quantum=1.,inner=0.,outer=10.,separation=.6,max_time=12,bounds=(0.,4.,-1.,1.));paths,meta=search(starts,goals,max_nodes=512,**args);assert not meta['fallback'] and first_conflict(paths,.6) is None
    # Each agent needs at least4 moves plus the original grid's final goal append.
    lower_bound=2*(4+1);assert meta['sum_of_costs']==lower_bound,(meta,paths)
    independent,cap=search(starts,goals,max_nodes=1,**args);assert cap['fallback'] and cap['expanded_nodes']==1 and first_conflict(independent,.6) is not None
    for i in range(2):
        ref=FastGrid(1.,1.,0.,10.,.6,12).search(starts[i],goals[i]);assert ref.tobytes()==independent[i].tobytes()
    single,one=search(np.asarray([[0.,0.]]),np.asarray([[2.,1.]]),1.,1.,0.,10.,.6,64,max_time=12);reference=FastGrid(1.,1.,0.,10.,.6,12).search([0.,0.],[2.,1.]);assert single[0].tobytes()==reference.tobytes() and not one['fallback']
    # Public cache injection reaches new classes without changing any factory/global.
    for kind in ('transformer_bc','gradient_mpc','conflict_based_search'):
        marker=object();assert make('unit',dict(profiles={'unit':dict(payload=dict(controller_class=kind))}),{'unit':marker}) is marker
    r=dict(utc=utc(),passed=True,cost_random_inputs_per_precision=1000,cost_max_relative_errors=errors,finite_difference_max_abs_errors=derivatives,finite_zero_norm_subgradient=True,gradient_same_key_bitwise=True,cbs_corridor_conflict_free=True,cbs_corridor_optimal_cost=lower_bound,cbs_corridor_lower_bound=lower_bound,cbs_cap1_independent_shortest_paths=True,cbs_single_agent_same_underlying_priority_grid=True,unmodified_factory_public_cache_injection=True,x64_restored=not bool(jax.config.x64_enabled),locked_outcomes_read=False)
    write(HERE/'OPTIMIZER_UNIT_TESTS.json',r);seal('OPTIMIZER_UNIT_TESTS.receipt.json',[HERE/n for n in ('OPTIMIZER_UNIT_TESTS.json','test_unseen.py','gradient_mpc.py','conflict_based_search.py','cbs_kernel.cpp','cbs_kernel.so','new_execution.py','OPENING_CONFIG.receipt.json')]);print(json.dumps(r),flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
