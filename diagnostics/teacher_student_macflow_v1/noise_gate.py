"""Frozen training prerequisite, using only original runtime public interfaces."""
from core import *
import traceback
import jax
import jax.numpy as jnp
from diagnostics.controller_q_transfer_round3_v1.flow_adapter import RingFlowAdapter
from diagnostics.controller_q_diversity_v1.runtime import native_runtime
from ring_exchange.environment import RingExchangeEnv
# All project imports are complete before the explicit precision restoration.
jax.config.update('jax_enable_x64',False)

def main():
    assert not (HERE/'NOISE_GATE.json').exists(),'Gate evidence is immutable; do not overwrite/retry.'
    assert sha(HERE/'PREREGISTRATION_SUPPLEMENT.md')==read(HERE/'PREREGISTRATION_SUPPLEMENT.receipt.json')['sha256']
    assert sha(HERE/'PREREGISTRATION.md')==read(HERE/'PREREGISTRATION.receipt.json')['sha256']
    teachers=read(HERE/'TEACHERS.json');p=read(D/'decentralized_pipeline_v1/main_protocol.json')
    checks=[];status='PASS';failure=None
    for r in teachers:
        n=r['name'];v=p['base_profiles'][n]['payload'];assert sha(r['checkpoint'])==r['checkpoint_sha256']
        rt=native_runtime('ring_exchange') if n=='base_u_v10_local_macflow' else RingFlowAdapter(v['flow_checkpoint_path'],read(v['adapter_manifest_path']),v['observation_frame'])
        jax.config.update('jax_enable_x64',False);assert not jax.config.x64_enabled
        manifest=Path(r['dataset_manifest_path']);rows=[f for f in read(manifest)['files'] if f['split']=='train'][:72]
        assert len(rows)==72
        for j,row in enumerate(rows):
            path=manifest.parent/row['file'];z=np.load(path);initial=json.loads(str(z['initial_state_json'].item()))
            env=RingExchangeEnv(rt.config);env.reset(z['states'][0],velocities=np.asarray(initial['velocities']),goals=np.asarray(initial['goals']))
            k=jax.random.fold_in(jax.random.PRNGKey(2026103030+r['index']),j)
            c=dict(teacher=n,trajectory=str(path),trajectory_sha256=sha(path),transition=0,key=np.asarray(k).tolist(),jax_enable_x64=bool(jax.config.x64_enabled))
            stage='observation_reconstruction'
            try:
                obs=jnp.asarray(rt.observation(env));stored=z['observations'][0];c['stored_observation_max_abs_difference']=float(np.abs(np.asarray(obs)-stored).max())
                stage='noise_bitwise'
                xi=np.asarray(rt.trace_fn(obs,k)).reshape(rt.steps+1,4,2)[0]
                independent=np.asarray(jax.random.normal(k,(1,rt.agent.config['joint_act_dim']))).reshape(4,2)
                c['noise_bitwise']=xi.dtype==independent.dtype and xi.tobytes()==independent.tobytes()
                assert c['noise_bitwise'],'trace initial noise differs bitwise from independent normal'
                stage='native_nominal_bitwise'
                raw=np.asarray(rt.native_fn(obs,k),dtype=np.float64)
                world=rt.to_world(raw,env).copy();world*=np.minimum(1.,rt.config.max_speed/np.maximum(np.linalg.norm(world,axis=1,keepdims=True),1e-30))
                _,nominal,_=rt.nominal(env,k,traced=False)
                c['native_nominal_bitwise']=world.dtype==nominal.dtype and world.tobytes()==nominal.tobytes()
                assert c['native_nominal_bitwise'],'native action differs bitwise after world transform and bound'
                stage='trace_endpoint_absolute_tolerance'
                zpath=np.asarray(rt.trace_fn(obs,k)).reshape(rt.steps+1,4,2)
                decoded=np.clip(zpath[-1]*rt.scale+rt.mean,-1.,1.)
                decoded*=np.minimum(1.,rt.config.max_speed/np.maximum(np.linalg.norm(decoded,axis=1,keepdims=True),1e-30))
                decoded=rt.to_world(decoded,env)
                c['trace_endpoint_max_absolute_error']=float(np.abs(decoded-nominal).max())
                assert c['trace_endpoint_max_absolute_error']<=1e-5,'trace endpoint exceeds 1e-5 m/s'
                c['passed']=True
            except Exception as e:
                c.update(passed=False,failed_stage=stage,error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc())
                failure=c;status='FAIL'
                # Save independent interface measurements as diagnostics, without retrying the failed assertion.
                obs=jnp.asarray(rt.observation(env));xi=np.asarray(rt.trace_fn(obs,k)).reshape(rt.steps+1,4,2)[0]
                independent=np.asarray(jax.random.normal(k,(1,rt.agent.config['joint_act_dim']))).reshape(4,2)
                raw=np.asarray(rt.native_fn(obs,k),dtype=np.float64);world=rt.to_world(raw,env).copy();world*=np.minimum(1.,rt.config.max_speed/np.maximum(np.linalg.norm(world,axis=1,keepdims=True),1e-30))
                _,nominal,_=rt.nominal(env,k,traced=False)
                c['noise_bitwise_diagnostic']=xi.tobytes()==independent.tobytes()
                c['native_nominal_bitwise_diagnostic']=world.tobytes()==nominal.tobytes()
                np.savez_compressed(HERE/'NOISE_GATE_FAILURE.npz',positions=env.positions,velocities=env.velocities,goals=env.goals,key=np.asarray(k),obs=np.asarray(obs),trace=np.asarray(rt.trace_fn(obs,k)),raw=raw,world=world,nominal=nominal)
            checks.append(c)
            if failure is not None:break
        if failure is not None:break
    errors=[c['trace_endpoint_max_absolute_error'] for c in checks if 'trace_endpoint_max_absolute_error' in c]
    write(HERE/'NOISE_GATE.json',dict(utc=utc(),status=status,trace_endpoint_max=max(errors) if errors else None,trace_endpoint_p999=float(np.quantile(errors,.999)) if errors else None,planned_samples=1008,completed_samples=len(checks),passed_samples=sum(c['passed'] for c in checks),checks=checks,failure=failure,simulation_steps=0,training_started=False))
    source_inventory('GATE_IMPORTED_SOURCES.json')
    seal('NOISE_GATE.receipt.json',[HERE/'NOISE_GATE.json',HERE/'noise_gate.py',HERE/'PREREGISTRATION.md'],stopped_on_first_failure=failure is not None)
    print(json.dumps(dict(status=status,completed=len(checks),failure_stage=failure.get('failed_stage') if failure else None)),flush=True)
    return 0 if status=='PASS' else 2
if __name__=='__main__':sys.exit(main())
