"""Fixed source-only fits for the actual-FF cache union and state-erased control."""
import argparse,time,os
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import serialization
from diagnostics.controller_q_diversity_v1.train import Critic
from .study import HERE,FOLDS,ARMS,SEEDS,read,write,freeze,sha

JOBS=[dict(fold=f,arm=a,seed=s) for f in FOLDS for a in ARMS for s in SEEDS]


def centered(x):
    return x-x.mean(0,keepdims=True)-x.mean(1,keepdims=True)+x.mean()


def metrics(z,d,ix):
    q=1/(1+np.exp(-np.clip(z,-80,80)))
    s,f,u=[d[k][ix] for k in ('success','failure','unknown')]
    count=s+f+u;observed=count>0;known=s+f>0
    lo=s/np.maximum(count,1);hi=(s+u)/np.maximum(count,1)
    selected=np.where(observed,q,-np.inf).argmax(-1)
    rr=np.arange(len(ix));reg=np.where(observed,hi-lo[rr,selected,None],-np.inf)
    reg[rr,selected]=0;reg=np.maximum(reg.max(-1),0)
    a,b=np.triu_indices(16,1)
    pos=(lo[:,a]-hi[:,b]>=.25)&observed[:,a]&observed[:,b]
    neg=(lo[:,b]-hi[:,a]>=.25)&observed[:,a]&observed[:,b]
    delta=q[:,a]-q[:,b];strong=pos|neg
    scores=np.where(abs(delta)<1e-7,.5,np.where(pos,delta>0,delta<0))
    by_scene=[];all_groups=[]
    for sc in np.unique(d['scene'][ix]):
        groups=[]
        for c in np.unique(d['controller'][ix][d['scene'][ix]==sc]):
            ii=np.flatnonzero((d['scene'][ix]==sc)&(d['controller'][ix]==c))
            support=observed[ii].all(0)
            nll=np.sum(s[ii]*np.logaddexp(0,-z[ii])+f[ii]*np.logaddexp(0,z[ii]))/max(np.sum(s[ii]+f[ii]),1)
            true=s[ii]/np.maximum(s[ii]+f[ii],1)
            g=dict(scene=str(sc),controller=str(c),N=len(ii),regret=float(reg[ii].mean()),
                ranking=float(scores[ii][strong[ii]].mean()) if strong[ii].any() else .5,
                state_eta_MSE=float(np.mean((centered(q[ii][:,support])-centered(lo[ii][:,support]))**2)),
                Q_MSE=float(np.mean((q[ii]-true)[known[ii]]**2)),nll=float(nll),
                selected_success=float(lo[ii,selected[ii]].mean()),strong_cases=int(strong[ii].sum()))
            groups.append(g);all_groups.append(g)
        by_scene.append({k:float(np.mean([g[k] for g in groups])) for k in ('regret','ranking','state_eta_MSE','Q_MSE','nll')})
    answer={k:float(np.mean([g[k] for g in by_scene])) for k in ('regret','ranking','state_eta_MSE','Q_MSE','nll')}
    answer['groups']=all_groups
    return answer


def fit(index):
    job=JOBS[index];p=read(HERE/'protocol.json');h=p['training']
    assert sha(__file__)==p['experiment']['code_sha256']
    if job['fold']!='joint':assert read(HERE/'source_gate.json')['passed']
    audit=read(HERE/'dataset_audit.json');assert sha(HERE/'dataset.npz')==audit['sha256']
    d=dict(np.load(HERE/'dataset.npz'));held=FOLDS[job['fold']]
    if job['arm'].startswith('motion_'):
        d['success']=d['motion_success'].copy();d['failure']=d['motion_failure'].copy()
    source=np.ones(len(d['scene']),bool) if held is None else d['scene']!=held
    tr=source&(d['split']=='train');va=np.flatnonzero(source&(d['split']=='validation'))
    assert not (set(d['physical_hash'][tr])&set(d['physical_hash'][va]))
    if held:assert held not in d['scene'][tr] and held not in d['scene'][va]
    out=HERE/'fits'/job['fold']/job['arm']/f"seed{job['seed']}";out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    freeze(out/'job.json',dict(**job,dataset_sha256=audit['sha256'],protocol_sha256=sha(HERE/'protocol.json'),
        code_sha256=sha(__file__),source_train_physical_hashes=sorted(set(d['physical_hash'][tr])),
        source_validation_physical_hashes=sorted(set(d['physical_hash'][va])),held_labels_used_for_fit_or_selection=False))
    mask=d['entity_agent_mask'].astype(bool);features=d['feature']
    values=features[tr][mask[tr]][:,:64];mean=np.r_[values.mean(0),np.zeros(8,dtype=np.float32)];scale=np.r_[np.maximum(values.std(0),.05),np.ones(8,dtype=np.float32)]
    freeze(out/'normalization.json',dict(mean=mean.tolist(),scale=scale.tolist(),source_train_only=True,clip=10.,
        eta_center=p['eta_input_center'],eta_radius=p['eta_input_radius']))
    normalized=np.concatenate((np.clip((features[...,:64]-mean[:64])/scale[:64],-10,10),features[...,64:]),-1)*mask[...,None]
    if not job['arm'].endswith('_complete'):normalized[...,64:]=0
    c=jnp.asarray(normalized,jnp.float32)
    x={k[7:]:jnp.asarray(v,jnp.float32) for k,v in d.items() if k.startswith('entity_')}
    if True:  # Preserve APIroles in EVERY controlled representation.
        assert np.all(np.asarray(x['agents'])[...,3:7]==0)
        x['agents']=x['agents'].at[...,3:7].set(jnp.eye(4)[None,:,:]*x['agent_mask'][...,None])
    e=jnp.asarray((d['eta']-np.asarray(p['eta_input_center']))/np.asarray(p['eta_input_radius']),jnp.float32)
    s,f=[d[k].astype(float).copy() for k in ('success','failure')];eligible=s+f>0
    controls=[]
    if job['arm']=='union_slots_state_erased':
        for sc in np.unique(d['scene'][tr]):
            for ctl in np.unique(d['controller'][tr&(d['scene']==sc)]):
                keep=tr&(d['scene']==sc)&(d['controller']==ctl);trials=s[keep]+f[keep];valid=trials>0
                prior=np.sum(s[keep]/np.maximum(trials,1),0)/np.maximum(valid.sum(0),1)
                s[keep]=trials*prior;f[keep]=trials*(1-prior)
                controls.append(dict(scene=str(sc),controller=str(ctl),states=int(keep.sum()),
                    observed_states=valid.sum(0).tolist(),prior=prior.tolist()))
    write(out/'label_control.json',dict(source_train_only=True,rows=controls))
    s,f=jnp.asarray(s,jnp.float32),jnp.asarray(f,jnp.float32)
    net=Critic('eta_only' if job['arm'].endswith('eta_only') else 'state_field_compact')
    gather=lambda ii:{k:v[ii] for k,v in x.items()}
    params=net.init(jax.random.PRNGKey(job['seed']),gather(jnp.array([0])),c[:1],e[:1])
    optim=optax.chain(optax.clip_by_global_norm(h['clip_grad']),optax.adamw(h['lr'],weight_decay=h['weight_decay']));opt=optim.init(params)
    @jax.jit
    def block(params,opt,draws):
        def step(carry,draw):
            par,state=carry;ii,jj=draw[:,0],draw[:,1]
            def loss(pp):
                z=net.apply(pp,gather(ii),c[ii],e[jj]);ss,ff=s[ii,jj],f[ii,jj]
                return jnp.mean((ss*jax.nn.softplus(-z)+ff*jax.nn.softplus(z))/jnp.maximum(ss+ff,1))
            value,grad=jax.value_and_grad(loss)(par);updates,state=optim.update(grad,state,par)
            return (optax.apply_updates(par,updates),state),value
        return jax.lax.scan(step,(params,opt),draws)
    @jax.jit
    def batch_predict(par,ii,jj):return net.apply(par,gather(ii),c[ii],e[jj])
    def infer(par,rows):
        ii=np.repeat(rows,16);jj=np.tile(np.arange(16),len(rows));parts=[]
        for start in range(0,len(ii),256):parts.append(np.asarray(batch_predict(par,jnp.asarray(ii[start:start+256]),jnp.asarray(jj[start:start+256]))))
        return np.concatenate(parts).reshape(-1,16)
    pools={};coverage=[];sampling_weights={}
    for sc in np.unique(d['scene'][tr]):
        pools[sc]=[]
        controllers=np.unique(d['controller'][tr&(d['scene']==sc)])
        ww=np.full(len(controllers),1/len(controllers))
        if 'native_flow' in controllers:
            assert len(controllers)>1
            ww=np.where(controllers=='native_flow',p['native_sampling_mass'],(1-p['native_sampling_mass'])/(len(controllers)-1))
        sampling_weights[sc]=ww
        for ctl in controllers:
            indices=np.flatnonzero(tr&(d['scene']==sc)&(d['controller']==ctl))
            rows_by_state=[]
            for i in indices:
                js=np.flatnonzero(eligible[i]);assert len(js)>0
                rows_by_state.append((int(i),js))
            pools[sc].append(rows_by_state)
            coverage.append(dict(scene=str(sc),controller=str(ctl),physical_states=len(indices),observed_pairs=int(eligible[indices].sum())))
    write(out/'sampling_audit.json',dict(groups=coverage,source_scenes=sorted(pools),
        total_draws=h['steps']*h['batch_per_scene']*len(pools),uniform_scene_controller_state_eta=False,native_mass=p['native_sampling_mass'],sampling_weights={str(k):v.tolist() for k,v in sampling_weights.items()},cpu_affinity=sorted(os.sched_getaffinity(0))))
    # Per-controller rows share a predetermined observed candidate panel (K4/K16).
    for groups in pools.values():
        for group in groups:
            # All-numerical eta can be absent from a row; retain its eligibility exactly.
            assert len(group)>0
    rng=np.random.default_rng(job['seed']);best=None;history=[];t0=time.monotonic()
    for step in range(h['validation_every'],h['steps']+1,h['validation_every']):
        chunks=[]
        for sc,groups in pools.items():
            shape=(h['validation_every'],h['batch_per_scene']);which=rng.choice(len(groups),size=shape,p=sampling_weights[sc])
            draw=np.empty(shape+(2,),np.int32)
            for group_index,group in enumerate(groups):
                picked=np.argwhere(which==group_index);state_choice=rng.integers(len(group),size=len(picked))
                for choice in np.unique(state_choice):
                    places=picked[state_choice==choice];i,js=group[int(choice)];eta_choice=js[rng.integers(len(js),size=len(places))]
                    draw[places[:,0],places[:,1],0]=i;draw[places[:,0],places[:,1],1]=eta_choice
            chunks.append(draw)
        (params,opt),loss=block(params,opt,jnp.asarray(np.concatenate(chunks,1)))
        z=infer(params,va);m=metrics(z,d,va)
        row=dict(step=step,training_nll=float(np.mean(loss)),**m);history.append(row)
        key=(m['regret'],-m['ranking'],m['state_eta_MSE'],m['nll'],step)
        if best is None or key<best:
            best=key;best_bytes=serialization.to_bytes(params);selected=row
        if step%1000==0:
            (out/f'step{step}_state.msgpack').write_bytes(serialization.to_bytes(dict(params=params,optimizer=opt)))
            write(out/'history.json',history)
            print(dict(**job,step=step,best_step=selected['step'],regret=m['regret'],ranking=m['ranking'],seconds=time.monotonic()-t0),flush=True)
    chosen=serialization.from_bytes(params,best_bytes);(out/'checkpoint.msgpack').write_bytes(best_bytes)
    z=infer(chosen,np.arange(len(d['scene'])))
    np.savez_compressed(out/'predictions.npz',probability=1/(1+np.exp(-np.clip(z,-80,80))),state_uid=d['state_uid'])
    write(out/'history.json',history)
    write(out/'complete.json',dict(**job,selected=selected,source_train_metrics=metrics(z[np.flatnonzero(tr)],d,np.flatnonzero(tr)),
        dataset_sha256=audit['sha256'],checkpoint_sha256=sha(out/'checkpoint.msgpack'),protocol_sha256=sha(HERE/'protocol.json'),
        code_sha256=sha(__file__),seconds=time.monotonic()-t0,held_scene_labels_used_for_fit_or_selection=False,new_rollouts=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);a=p.parse_args();fit(a.index)
