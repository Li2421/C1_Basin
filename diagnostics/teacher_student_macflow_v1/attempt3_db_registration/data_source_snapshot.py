"""Read immutable teacher datasets; rebuild physical states without rolling out."""
from core import *
import jax
from ring_exchange.environment import RingExchangeEnv
from ring_exchange.local_frame import local_observation
from diagnostics.controller_q_transfer_round3_v1.flow_adapter import RingFlowAdapter
from diagnostics.controller_q_diversity_v1.runtime import native_runtime
jax.config.update('jax_enable_x64',False)

def teacher(index):
    r=read(HERE/'TEACHERS.json')[index];p=read(D/'decentralized_pipeline_v1/main_protocol.json');v=p['base_profiles'][r['name']]['payload']
    assert sha(r['checkpoint'])==r['checkpoint_sha256']
    rt=native_runtime('ring_exchange') if r['name']=='base_u_v10_local_macflow' else RingFlowAdapter(v['flow_checkpoint_path'],read(v['adapter_manifest_path']),v['observation_frame'])
    jax.config.update('jax_enable_x64',False)
    return rt

def prepare(index):
    assert read(HERE/'NOISE_GATE.json')['status']=='PASS'
    r=read(HERE/'TEACHERS.json')[index];out=HERE/'data'/r['name'];out.mkdir(parents=True,exist_ok=True)
    assert not (out/'READY.json').exists()
    rt=teacher(index);manifest=Path(r['dataset_manifest_path']);m=read(manifest);env=RingExchangeEnv(rt.config);audit=[]
    for split in ['train','dev']:
        local=[];observed=[]
        for row in m['files']:
            if row['split']!=split:continue
            f=manifest.parent/row['file'];z=np.load(f);initial=json.loads(str(z['initial_state_json'].item()));pos=z['states'];L=len(z['actions']);vel=np.concatenate((np.asarray(initial['velocities'])[None],np.diff(pos,axis=0)/rt.config.dt),axis=0);diff=0.;action_diff=float(np.abs(vel[1:]-z['actions']).max())
            for t in range(L):
                env.reset(pos[t],velocities=vel[t],goals=np.asarray(initial['goals']))
                a=rt.observation(env);observed.append(a);local.append(local_observation(env.positions,env.velocities,env.goals,env.config));diff=max(diff,float(np.abs(a-z['observations'][t]).max()))
            audit.append(dict(file=str(f),sha256=sha(f),split=split,transitions=L,stored_observation_max_abs_difference=diff,stored_action_vs_reconstructed_velocity_max_abs_difference=action_diff))
        x=np.asarray(local,np.float32);o=np.asarray(observed,np.float32)
        np.save(out/f'{split}_local.npy',x);np.save(out/f'{split}_teacher.npy',o)
        if split=='train':np.savez(out/'normalization.npz',mean=x.reshape(-1,23).mean(0,dtype=np.float64).astype(np.float32),scale=np.maximum(x.reshape(-1,23).std(0,dtype=np.float64),.01).astype(np.float32))
    write(out/'DATA_AUDIT.json',audit);source_inventory(f'imports_data_{index}.json');write(out/'READY.json',dict(utc=utc(),teacher=r,files={p.name:sha(p) for p in out.iterdir() if p.is_file()},jax_enable_x64=bool(jax.config.x64_enabled)))
    print('data READY',r['name'],flush=True)
if __name__=='__main__':prepare(int(sys.argv[1]))
