"""Re-use independent static-action labels and existing checkpoint trajectories."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from .vector_study import HERE,PARENT,SEEDS,read,write,freeze,sha
ROOT=PARENT.parent;OLD=ROOT/'controller_ff_action_mediation_v1'
ARMS=('current','complete')

def cache():
    d=dict(np.load(HERE/'dataset.npz'));old=dict(np.load(OLD/'dataset.npz'))
    protocol=read(ROOT/'controller_four_scene_mechanism_v1/protocol.json')
    np.testing.assert_array_equal(np.asarray(protocol['eta'],np.float64),d['eta'])
    lookup={(str(s),str(c)):i for i,(s,c) in enumerate(zip(old['state_uid'],old['controller']))}
    ix=np.array([lookup[str(s),str(c).removeprefix('goal_')] for s,c in zip(d['state_uid'],d['controller'])])
    np.testing.assert_array_equal(old['feature'][ix],d['feature'][...,:64])
    physical={s['state_uid']:s['physical'] for s in protocol['states']}
    y=np.full((len(ix),16,4,8),np.nan);seen=np.zeros((len(ix),16,4),bool);files=[];failed=0
    for i,(sid,sc,controller) in enumerate(zip(d['state_uid'],d['scene'],d['controller'])):
        path=OLD/'features'/str(sid)/(str(controller).removeprefix('goal_')+'.npz')
        f=np.load(path);assert int(f['physical_steps'])==0
        a=f['raw_actions'];n=a.shape[2];p=physical[str(sid)]
        e=np.asarray(p['goals'])-np.asarray(p['positions']);e/=np.maximum(np.linalg.norm(e,axis=1,keepdims=True),1e-12)
        left=np.stack((-e[:,1],e[:,0]),-1);vmax=protocol['environments'][str(sc)]['config']['config']['max_speed']
        z=np.stack(((a*e).sum(-1),(a*left).sum(-1)),-1)/vmax
        good=f['query_success'].astype(bool);seen[i]=good;failed+=int((~good[[0,5,10,15]]).sum())
        assert np.isfinite(z[good]).all();assert abs(z[good]).max()<=1+1e-5
        y[i,:,:,:n*2]=.5+.5*np.clip(z.reshape(16,4,n*2),-1,1)
        files.append(dict(path=str(path),sha256=sha(path)))
    assert failed==0,'Do not silently drop independent static query failures'
    np.savez_compressed(HERE/'independent_static_actions.npz',vector_motion=y,observed=seen,state_uid=d['state_uid'],controller=d['controller'])
    freeze(HERE/'independent_static_audit.json',dict(
        history='These exact static actions were previously Qinput controls, never targets for Stage29. Reuse without any controller/simulator query.',
        state_controller_rows=len(ix),static_action_count=int(seen.sum()),query_failures=failed,
        actual_query_eta_float64_exact=True,compact64_features_exact=True,
        old_dataset_float32_eta_difference='Serialization only; actual old query reads exact float64 protocol eta, verified equal. No relaxed eta semantic match.',
        RNG='Old fixed root2026101204 fourfolds. Independent of Stage29 label seed roots; same four noises shared across physical states. Any state-bootstrap CI is conditional on this fixed independent noise panel, not a seed-population confidence claim.',
        source_files=files,new_query=0,new_fit=0,new_rollout=0))

def infer(index):
    import jax,jax.numpy as jnp
    from flax import serialization
    from .vector_net import Critic
    arm=ARMS[index//3];seed=SEEDS[index%3]
    d=dict(np.load(HERE/'dataset.npz'));base=HERE/'fits/joint'/('motion_'+arm)/f'seed{seed}'
    norm=read(base/'normalization.json');feature=d['feature'];mean=np.asarray(norm['mean']);scale=np.asarray(norm['scale'])
    c=np.concatenate((np.clip((feature[...,:64]-mean[:64])/scale[:64],-10,10),feature[...,64:]),-1)*d['entity_agent_mask'][...,None]
    if arm=='current':c[...,64:]=0
    x={k[7:]:jnp.asarray(v,jnp.float32) for k,v in d.items() if k.startswith('entity_')}
    x['agents']=x['agents'].at[...,3:7].set(jnp.eye(4)[None,:,:]*x['agent_mask'][...,None])
    c=jnp.asarray(c,jnp.float32);e=jnp.asarray((d['eta']-np.asarray(norm['eta_center']))/np.asarray(norm['eta_radius']),jnp.float32)
    net=Critic('state_field_compact',8)
    @jax.jit
    def predict(params,ii,jj):return jax.nn.sigmoid(net.apply(params,{k:v[ii] for k,v in x.items()},c[ii],e[jj]))
    ii=np.repeat(np.arange(len(d['scene'])),4);jj=np.tile([0,5,10,15],len(d['scene']))
    def run(params):
        chunks=[np.asarray(predict(params,jnp.asarray(ii[k:k+256]),jnp.asarray(jj[k:k+256]))) for k in range(0,len(ii),256)]
        return np.concatenate(chunks).reshape(len(d['scene']),4,8)
    par=serialization.msgpack_restore((base/'checkpoint.msgpack').read_bytes())
    chosen=run(par);expected=np.load(base/'predictions.npz')['probability'][:,[0,5,10,15]]
    np.testing.assert_allclose(chosen,expected,atol=2e-6,rtol=2e-6)
    output={'selected':chosen};hashes={}
    for step in range(1000,6001,1000):
        f=base/f'step{step}_state.msgpack';par=serialization.msgpack_restore(f.read_bytes())['params']
        output['step'+str(step)]=run(par);hashes[str(step)]=sha(f)
    dest=HERE/'trajectory';dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/f'{arm}_seed{seed}.npz',**output)
    write(dest/f'{arm}_seed{seed}.json',dict(checkpoint_hashes=hashes,selected_replay_max_error=float(abs(chosen-expected).max()),new_fit=0,new_rollout=0))
    print(dict(arm=arm,seed=seed,checkpoints=7,selected_replay=True))

def review():
    from .vector_review import center,ci
    d=dict(np.load(HERE/'dataset.npz'));ind=dict(np.load(HERE/'independent_static_actions.npz'))
    orig=dict(np.load(HERE/'motion_raw.npz'));rows=[];boots={}
    pred={a:[dict(np.load(HERE/'trajectory'/f'{a}_seed{s}.npz')) for s in SEEDS] for a in ARMS}
    for sc in sorted(set(d['scene'])):
        for split in ('train','validation'):
            ix=np.flatnonzero((d['scene']==sc)&(d['split']==split))
            ix=np.array(sorted(ix,key=lambda i:(d['state_uid'][i],d['controller'][i])))
            n=len(ix)//4;na=int(d['entity_agent_mask'][ix[0]].sum())*2
            tt=ind['vector_motion'][ix][:,[0,5,10,15],:,:na].reshape(n,4,4,4,na).mean(3)
            y=orig['vector_motion'][ix][:,[0,5,10,15],:,:na].reshape(n,4,4,4,na)
            nt=2 if split=='train' else 4;old=y[:,:,:,:nt].mean(3)
            ib=np.random.default_rng(2026101031+n).integers(n,size=(20000,n));tag=str(sc)+'__'+split;boots[tag+'__indices']=ib
            for arm,values in pred.items():
                for checkpoint in ('selected','step1000','step2000','step3000','step4000','step5000','step6000'):
                    p=np.mean([v[checkpoint] for v in values],0)[ix,:,:na].reshape(n,4,4,na)
                    for interaction,axes in (('state_eta',(0,2)),('state_controller_eta',(0,1,2))):
                        pc=center(p,axes);tc=center(tt,axes);oc=center(old,axes)
                        bs=[]
                        for start in range(0,len(ib),500):
                            jj=ib[start:start+500];pp=center(p[jj],tuple(a+1 for a in axes))
                            yy=tt[jj]
                            # Orthogonal projection means <Pp,Pt>=<Pp,t>.
                            bs.append((2*pp*yy-pp*pp).mean(axis=tuple(range(1,pp.ndim))))
                        bs=np.concatenate(bs);key=tag+'__'+arm+'__'+checkpoint+'__'+interaction;boots[key]=bs
                        rows.append(dict(scene=str(sc),split=split,N=n,arm=arm,checkpoint=checkpoint,interaction=interaction,
                            independent_skill=float(np.mean(2*pc*tc-pc*pc)),CI95_conditional_fixed_keys=ci(bs),
                            original_skill=float(np.mean(2*pc*oc-pc*pc)),
                            predicted_interaction_power=float(np.mean(pc*pc)),
                            independent_corr=float(np.corrcoef(pc.ravel(),tc.ravel())[0,1]) if min(pc.std(),tc.std())>1e-12 else None,
                            total_MSE_independent=float(np.mean((p-tt)**2))))
    write(HERE/'trajectory_results.json',dict(rows=rows,new_fit=0,new_query=0,new_rollout=0,
        boundary='Allstates SOURCE TRAIN/VAL; noheldscene fits. Independent cached four static noise keys are shared across states; confidence intervals conditional on these keys. Selected checkpoint unchanged. Epoch trajectory is diagnosis, not selection using evaluation labels.'))
    np.savez_compressed(HERE/'trajectory_bootstrap.npz',**boots)
    for r in rows:
        if r['arm']=='complete' and r['checkpoint'] in ('selected','step6000') and r['interaction']=='state_eta':
            print(r['scene'],r['split'],r['checkpoint'],round(r['independent_skill'],8),np.round(r['CI95_conditional_fixed_keys'],8).tolist())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['cache','infer','review']);p.add_argument('--index',type=int,default=0)
    a=p.parse_args();infer(a.index) if a.stage=='infer' else globals()[a.stage]()

