"""Full SOURCE TRAIN gradient audit at cached checkpoints; no fit or rollout."""
import argparse
from pathlib import Path
import numpy as np
from diagnostics.controller_cached_union_v1.study import HERE as BASE,SEEDS,read,write,freeze,sha
HERE=Path(__file__).resolve().parent
NATIVE=HERE.parent/'controller_native_exposure_v1'


def prepare():
    assert read(NATIVE/'completion.json')['all_joint_replay_hashes_exact']
    freeze(HERE/'protocol.json',dict(question='Localize the already-established native mixed-training penalty: other controllers in Ring versus other scenes, and whether their gradients oppose the native branch.',
        prior='Stage19same-drawother-taskremoval improvesnativeheld-stateQandinteraction. Stage20oldtarget-supportQgainnotadaptation. No new broadarchitecture/objective/data search.',
        reuse='Three Stage19acceptedjointreplayselected/finalcheckpoints,fullStage17SOURCE TRAIN labels andnormalization. Priorhistorysearchfoundnofullpopulationsplit-gradientaudit.',
        data_sha256=sha(BASE/'dataset.npz'),seeds=SEEDS,checkpoints=['selected','final'],
        groups=['native_ring512','other_ring_controllers','other_scenes'],
        weighting='Exactoriginaluniformscene->controller->physicalstate->eligibleeta objective. NLLpercell knowncounts. FullobservedTRAINpopulation,notminibatchdotproducts.',
        metrics='Gradientnorms,mass,cosine/dotmatrix,nativefirst-orderdescentunderjointgradient,per-layercontributions. Descriptive SOURCE mechanism;notphysical-stateTESTCIoractualAdamstep reconstruction.',
        caveat='Negativegradientalignment isnot itselfproof of incapacity, norof thecompleteLOSOfailure. No savedAdamstateisavailable;do not labelSGDdirectionasactualhistoricalAdamupdate.',
        decision='Only if TRAINgradients oppose native learning, testone standardPCGradrepair atunchangedlabels/input/budget;no hyperparameter sweep. Otherwiseinvestigateoptimizerdilutionwithoutclaiminggradientconflict.',
        literature=[dict(url='https://proceedings.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html',use='PCGrad motivates measuring taskgradientconflict before a surgicalrepair; it doesnotguaranteeLOSO.')],
        new_fits=0,new_rollouts=0,code_sha256=sha(__file__)))
    print('Frozen full-populationSOURCE gradientdiagnostic',flush=True)


def run(seed):
    import os,jax,jax.numpy as jnp
    from flax import serialization
    from jax.flatten_util import ravel_pytree
    from diagnostics.controller_q_diversity_v1.train import Critic
    p=read(HERE/'protocol.json');assert sha(__file__)==p['code_sha256'];assert sha(BASE/'dataset.npz')==p['data_sha256']
    d=dict(np.load(BASE/'dataset.npz'));tr=d['split']=='train';norm=read(NATIVE/'fits/joint_replay'/f'seed{seed}/normalization.json')
    x={k[7:]:jnp.asarray(v,jnp.float32) for k,v in d.items() if k.startswith('entity_')};x['agents']=x['agents'].at[...,3:7].set(jnp.eye(4)[None]*x['agent_mask'][...,None])
    c=jnp.asarray(np.clip((d['feature']-np.asarray(norm['mean'],np.float32))/np.asarray(norm['scale'],np.float32),-10,10)*d['entity_agent_mask'][...,None],jnp.float32)
    e=jnp.asarray((d['eta']-np.asarray(norm['eta_center']))/np.asarray(norm['eta_radius']),jnp.float32)
    s,f=[jnp.asarray(d[k],jnp.float32) for k in ['success','failure']]
    weights=np.zeros_like(d['success'],np.float64);scenes=np.unique(d['scene'][tr]);groups=[]
    for sc in scenes:
        ctls=np.unique(d['controller'][tr&(d['scene']==sc)])
        for ctl in ctls:
            ids=np.flatnonzero(tr&(d['scene']==sc)&(d['controller']==ctl))
            for i in ids:
                js=np.flatnonzero(d['success'][i]+d['failure'][i]>0);weights[i,js]=1/(len(scenes)*len(ctls)*len(ids)*len(js))
    assert abs(weights.sum()-1)<1e-12
    pools=[]
    native=(d['scene']=='ring_exchange')&(d['controller']=='native_flow')&(d['source']=='native_ring512')
    regions=[native,(d['scene']=='ring_exchange')&~native,d['scene']!='ring_exchange']
    for name,mask in zip(p['groups'],regions):
        ii,jj=np.nonzero((weights>0)&mask[:,None]);w=weights[ii,jj]
        pools.append((ii,jj,w));groups.append(dict(name=name,cells=len(ii),physical_states=len(set(d['state_uid'][ii])),objective_mass=float(w.sum())))
    net=Critic('state_field_compact');initial=net.init(jax.random.PRNGKey(seed),{k:v[:1] for k,v in x.items()},c[:1],e[:1]);_,unravel=ravel_pytree(initial)
    @jax.jit
    def contribution(par,ii,jj,w):
        def loss(pp):
            z=net.apply(pp,{k:v[ii] for k,v in x.items()},c[ii],e[jj]);ss,ff=s[ii,jj],f[ii,jj]
            return jnp.sum(w*(ss*jax.nn.softplus(-z)+ff*jax.nn.softplus(z))/jnp.maximum(ss+ff,1))
        value,gradient=jax.value_and_grad(loss)(par);flat,_=ravel_pytree(gradient)
        return value,flat
    rows=[];saved={}
    for which in p['checkpoints']:
        cp=NATIVE/'fits/joint_replay'/f'seed{seed}'/('checkpoint.msgpack' if which=='selected' else 'final_checkpoint.msgpack');par=serialization.from_bytes(initial,cp.read_bytes())
        gradients=[];losses=[]
        for ii,jj,w in pools:
            full=None;value=0.
            for start in range(0,len(ii),256):
                n=min(256,len(ii)-start);i=np.pad(ii[start:start+n],(0,256-n));j=np.pad(jj[start:start+n],(0,256-n));ww=np.pad(w[start:start+n],(0,256-n))
                l,g=contribution(par,jnp.asarray(i),jnp.asarray(j),jnp.asarray(ww,jnp.float32));a=np.asarray(g,np.float64)
                full=a if full is None else full+a;value+=float(l)
            gradients.append(full);losses.append(value)
        g=np.asarray(gradients);dot=g@g.T;norms=np.sqrt(np.diag(dot));cos=dot/np.maximum(norms[:,None]*norms[None],1e-30)
        tree=[unravel(jnp.asarray(v)) for v in g];layer={}
        for name in par['params']:
            z=np.asarray([np.asarray(ravel_pytree(v['params'][name])[0],np.float64) for v in tree]);layer[name]=(z@z.T).tolist()
        row=dict(seed=seed,checkpoint=which,checkpoint_sha256=sha(cp),groups=groups,weighted_loss=losses,gradient_norm=norms.tolist(),dot=dot.tolist(),cosine=cos.tolist(),
            native_dot_joint=float(dot[0].sum()),native_dot_other_ring=float(dot[0,1]),native_dot_other_scenes=float(dot[0,2]),layer_dot=layer,affinity=sorted(os.sched_getaffinity(0)))
        rows.append(row);saved[which+'__gradients']=g;print(dict(seed=seed,checkpoint=which,cosine_native_other_ring=cos[0,1],cosine_native_other_scenes=cos[0,2],native_dot_joint=dot[0].sum()),flush=True)
    write(HERE/f'seed{seed}_results.json',dict(rows=rows,new_fits=0,new_rollouts=0,scope='FullSOURCE TRAIN only;gradientdescriptive,noAdamstate'))
    np.savez_compressed(HERE/f'seed{seed}_gradients.npz',**saved)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','run']);ap.add_argument('--seed',type=int);a=ap.parse_args();prepare() if a.stage=='prepare' else run(a.seed)
