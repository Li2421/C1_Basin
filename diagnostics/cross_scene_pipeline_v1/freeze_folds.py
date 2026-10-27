"""Outcome-free proposals and every prospective choice, before any new rollout."""
from common import *
from networks import Generator,critic,stable_int,dist_params,eta_mean,eta_from_noise,CENTER,RADIUS
from flax import serialization
import jax,jax.numpy as jnp,numpy as np

OLD=ROOT/'diagnostics/cross_scene_correction_v1'

def load(folder,mode):
    row=read(folder/mode/'SELECTED.json')['selected'];assert sha(row['checkpoint'])==row['sha256']
    net=Generator()if mode=='generator'else critic(mode)
    args=[jnp.zeros((1,6)),jnp.zeros((1,12))]
    if mode!='generator':args.extend([jnp.zeros((1,8)),jnp.zeros((1,3))])
    template=net.init(jax.random.PRNGKey(0),*args)
    return net,serialization.from_bytes(template,Path(row['checkpoint']).read_bytes()),row

def scores(net,params,h,c,ct,etas):
    C=len(ct);S=len(h);A=etas.shape[1]
    hh=np.broadcast_to(h[None,:,None,:],(C,S,A,6));cc=np.broadcast_to(c[None,:,None,:],(C,S,A,12))
    xx=np.broadcast_to(ct[:,None,None,:],(C,S,A,8));ee=np.broadcast_to(etas[None],(C,S,A,3))
    return np.asarray(net.apply(params,jnp.asarray(hh),jnp.asarray(cc),jnp.asarray(xx),jnp.asarray((ee-CENTER)/RADIUS)))

def main():
    assert(HERE/'train_critic_COMPLETE.json').exists()
    inputs=read(HERE/'INPUTS.json');old_choices=read_evidence(OLD/'LOSO_CHOICES.json');etas=np.array(inputs['etas'],float);keys=inputs['eta_keys']
    counts={};all_paths=[]
    for fold in read(HERE/'FOLDS.json'):
        folder=HERE/'folds'/fold['name'];receipt=HERE/'FOLD_FROZEN'/f"{fold['name']}.receipt.json"
        if receipt.exists():
            for p,h in read(receipt)['files'].items():assert sha(p)==h,p
            continue
        models={m:load(folder,m)for m in fold['modes']}
        if fold['generator']:generator,gparams,grow=load(folder,'generator')
        choice_rows={};proposals={}
        for name in fold['targets']:
            d=inputs['scene_inputs'][name];norm=fold['normalization'];h=((np.array(d['h'])-norm['h_mean'])/norm['h_scale']).astype(np.float32)
            c=np.repeat(((np.array(d['c'])-norm['c_mean'])/norm['c_scale'])[None],len(h),axis=0).astype(np.float32);ct=np.array(d['controller_one_hot'],np.float32)
            fixed=np.broadcast_to(etas[None],(len(h),49,3)).copy()
            if fold['generator']:
                raw=np.asarray(generator.apply(gparams,jnp.asarray(h),jnp.asarray(c)))
                mean=np.asarray(eta_mean(jnp.asarray(raw)));rows=[];pp=[]
                for i,index in enumerate(d['state_indices']):
                    seed=stable_int('cross-scene-pipeline-proposals',grow['seed'],name,index);rng=np.random.default_rng(seed)
                    samples=np.asarray(eta_from_noise(jnp.asarray(raw[i])[None].repeat(4,0),jnp.asarray(rng.standard_normal((4,3)),jnp.float32)))
                    e=np.asarray([mean[i],*samples],float);pp.append(e)
                    rows.append(dict(index=index,proposal_seed=seed,generator_seed=grow['seed'],generator_mean=mean[i].tolist(),samples=samples.tolist(),
                        sigma=np.asarray(dist_params(jnp.asarray(raw[i]))[1]).tolist(),candidate_order=['generator_mean','sample_0','sample_1','sample_2','sample_3']))
                proposals[name]=rows;menu=np.concatenate([fixed,np.array(pp)],axis=1)
            else:menu=fixed
            net,p,_=models['full'];full=scores(net,p,h,c,ct,menu);best=np.argmax(full,axis=-1)
            row=dict(state_indices=d['state_indices'],controllers=d['controllers'],controller_types=d['controller_types'],
                candidate_etas=menu.tolist(),candidate_keys=keys+(['generator_mean','sample_0','sample_1','sample_2','sample_3']if fold['generator']else[]),
                critic49=np.argmax(full[:,:,:49],axis=-1).tolist(),full_logits=full.tolist(),
                global_fixed=fold['baselines'][name]['global_fixed'],knn=fold['baselines'][name]['knn'],
                strong_global=keys[old_choices[name]['global_fixed']],eta13='eta13',safety='safety')
            if fold['kind']=='full18':assert row['global_fixed']==row['strong_global']
            if fold['generator']:
                row.update(pipeline=best.tolist(),generator_mean=49)
                for mode in ('no_controller','scene_only'):
                    m,p,_=models[mode];ss=scores(m,p,h,c,ct,fixed);row[mode+'49']=np.argmax(ss,axis=-1).tolist();row[mode+'_logits']=ss.tolist()
            choice_rows[name]=row
        write(folder/'PROPOSALS.json',proposals);write(folder/'CHOICES.json',choice_rows)
        paths=[folder/'definition.json',folder/'PROPOSALS.json',folder/'CHOICES.json',HERE/'INPUTS.json',HERE/'SCENE_SET.json',
            HERE/'TRAINING_FROZEN.receipt.json',HERE/'freeze_folds.py',HERE/'networks.py',HERE/'models.py']
        for mode in fold['modes']+(['generator']if fold['generator']else[]):
            paths.append(folder/mode/'SELECTED.json')
            for seed in (17,23,41):paths.extend(folder/mode/f'seed{seed}'/n for n in ('FIT.receipt.json','history.json','checkpoint.msgpack'))
        receipt.parent.mkdir(parents=True,exist_ok=True)
        seal(str(receipt.relative_to(HERE)),paths,fold=fold['name'],sources=fold['sources'],targets=fold['targets'],
            no_heldout_labels_for_training_validation_seed_or_scaling=True,new_rollouts_before_freeze=0,
            tie_order='49 canonical candidates, generator mean, samples0-3; exact argmax first occurrence')
        counts[fold['name']]=sum(len(v['controllers'])*len(v['state_indices'])for v in choice_rows.values());all_paths.append(receipt)
        print('FROZEN',fold['name'],counts[fold['name']],'cells',flush=True)
    receipts=sorted((HERE/'FOLD_FROZEN').glob('*.receipt.json'));assert len(receipts)==31
    write(HERE/'ALL_FOLDS_FROZEN.receipt.json',dict(utc=utc(),files={str(p):sha(p)for p in receipts},folds=31,
        heldout_new_rollouts_before_all_folds_frozen=0,models_and_proposals_and_all_choices_frozen=True))

if __name__=='__main__':main()
