"""Reuse frozen predictions; source-selected models, opened DEVELOPMENT only."""
import argparse
import importlib
import numpy as np
from .run import HERE, OLD, VARIANTS, ARMS, SEEDS, install, read, write, sha
from diagnostics.controller_q_diversity_v1.factorial import SCENES, execution_constants
from diagnostics.controller_q_diversity_v1.evaluate import donor_bootstrap, reversal
from diagnostics.controller_entity_coverage_check_v1.review import corr
from diagnostics.controller_ff_factorial_repair_v1.review import (
    select, selection_difference, mse_gain, intervals, derangements, centered_axes)

BASES=('eta_only','state_field_compact','state_field_compact_bilinear')


class Data(dict):
    @property
    def files(self): return list(self)


def folders(variant, arm):
    return [(OLD/'fits'/variant/arm/f'seed{s}') if arm in BASES else
            (HERE/arm/'fits'/variant/'state_field_compact'/f'seed{s}') for s in SEEDS]


def probability(z): return 1/(1+np.exp(-np.clip(z,-80,80)))


def final_predictions(variant, arm, data):
    """Inference only on previously opened Q16 states, never model selection."""
    import jax
    import jax.numpy as jnp
    from flax import serialization
    from diagnostics.controller_q_diversity_v1 import train
    values=[]
    for seed, folder in zip(SEEDS,folders(variant,arm)):
        train=importlib.reload(train)
        kind=arm if arm in BASES else 'state_field_compact'
        if arm not in BASES: train=install(arm,'final')
        norm=read(folder/'normalization.json')
        x,c,e=train.inputs(data,norm); net=train.Critic(kind)
        initial=net.init(jax.random.PRNGKey(seed),{k:v[:1] for k,v in x.items()},c[:1],e[:1])
        p=serialization.from_bytes(initial,(folder/'checkpoint.msgpack').read_bytes())
        @jax.jit
        def batch(ii,jj): return net.apply(p,{k:v[ii] for k,v in x.items()},c[ii],e[jj])
        ii=np.repeat(np.arange(len(data['scene'])),16);jj=np.tile(np.arange(16),len(data['scene']))
        zs=[np.asarray(batch(jnp.asarray(ii[i:i+256]),jnp.asarray(jj[i:i+256]))) for i in range(0,len(ii),256)]
        values.append(probability(np.concatenate(zs).reshape(-1,16)))
    return np.asarray(values)


def cross_mean(a): return np.asarray(a).mean((1,2))


def analyze_panel(d, ids, pred, metadata, arrays, saved):
    ids=np.asarray(sorted(ids,key=lambda i:(d['state_uid'][i],d['controller'][i])))
    states=np.unique(d['state_uid'][ids]); controllers=np.unique(d['controller'][ids])
    n,nc=len(states),len(controllers);shape=(n,nc,16)
    np.testing.assert_array_equal(d['state_uid'][ids],np.repeat(states,nc))
    np.testing.assert_array_equal(d['controller'][ids],np.tile(controllers,n))
    trials=d['trials'][ids].reshape(n,nc)
    s,u=(d[k][ids].reshape(shape) for k in ('success','unknown'))
    lo=s/trials[...,None];hi=(s+u)/trials[...,None];mid=(lo+hi)/2
    ss,uu,cc=(d[k][ids].reshape(shape+(16,)) for k in ('seed_success','seed_unknown','seed_collision'))
    boot=np.random.default_rng(2026101016+n).integers(n,size=(20000,n))
    prefix='__'.join(str(metadata[k]) for k in ('variant','panel','scene','controller_group'))
    arrays[prefix+'__bootstrap_indices']=boot
    perm=derangements(n);saved[prefix+'__permutations']=perm
    saved[prefix+'__state_uid']=states;saved[prefix+'__controllers']=controllers
    saved[prefix+'__row_indices']=ids
    saved[prefix+'__truth_lower']=lo;saved[prefix+'__truth_upper']=hi
    pp={k:v[:,ids].mean(0).reshape(shape) for k,v in pred.items()}
    choices={k:q.argmax(-1) for k,q in pp.items()}
    results=[]
    for arm,q in pp.items():
        key=prefix+'__'+arm;chosen=choices[arm]
        freq=np.bincount(chosen.ravel(),minlength=16)/chosen.size
        value=select(lo,chosen);upper=select(hi,chosen)
        row=dict(**metadata,arm=arm,N=n,controllers=controllers.tolist(),trials_per_cell=np.unique(trials).tolist(),
            Q_MSE_midpoint=float(np.mean((q-mid)**2)),
            state_effect_corr=corr(centered_axes(q,(0,)),centered_axes(mid,(0,))),
            state_eta_corr=corr(centered_axes(q,(0,2)),centered_axes(mid,(0,2))),
            controller_effect_corr=corr(centered_axes(q,(1,)),centered_axes(mid,(1,))),
            controller_state_eta_corr=corr(centered_axes(q,(0,1,2)),centered_axes(mid,(0,1,2))),
            success_bounds=[float(value.mean()),float(upper.mean())],
            oracle_bounds=[float(lo.max(-1).mean()),float(hi.max(-1).mean())],
            regret_bounds=[float(np.maximum(lo.max(-1)-upper,0).mean()),float((hi.max(-1)-value).mean())],
            entropy_nats=float(-np.sum(freq[freq>0]*np.log(freq[freq>0]))),selection_frequency=freq.tolist(),
            all_fixed_success_lower=lo.mean((0,1)).tolist(),all_fixed_success_upper=hi.mean((0,1)).tolist(),
            best_per_controller_fixed_bounds=[float(lo.mean(0).max(-1).mean()),float(hi.mean(0).max(-1).mean())],
            numerical_unknown=int(u.sum()),collisions=int(cc.sum()),
            selected_collisions=int(select(cc.sum(-1),chosen).sum()),selected_unknown=int(select(u,chosen).sum()))
        saved[key+'__matched_predictions']=q
        # These arrays plus saved permutations reconstruct every mismatched prediction exactly.
        saved[key+'__mismatched_selections']=chosen[perm]
        saved[key+'__selected_eta']=chosen
        row['individual_seeds']=[]
        for seed,z in zip(SEEDS,pred[arm]):
            p=z[ids].reshape(shape)
            row['individual_seeds'].append(dict(seed=seed,Q_MSE=float(np.mean((p-mid)**2)),
                state_eta_corr=corr(centered_axes(p,(0,2)),centered_axes(mid,(0,2))),success=float(select(lo,p.argmax(-1)).mean())))
        if np.all(trials==16):
            row['B15_bounds']=[int((select(s,chosen)>=15).sum()),int((select(s+u,chosen)>=15).sum())]
        # Exact donor population; resample recipient and donor physical states jointly.
        ml,mu,ql,qu=(np.zeros((n,n)) for _ in range(4))
        for donor in range(n):
            a,b=selection_difference(lo,hi,chosen,np.broadcast_to(chosen[donor],chosen.shape))
            ml[:,donor]=a.mean(1);mu[:,donor]=b.mean(1)
            a,b=mse_gain(q[donor],q,lo,hi);ql[:,donor]=cross_mean(a);qu[:,donor]=cross_mean(b)
        for label,a,b in [('matched_minus_mismatch',ml,mu),('wrong_state_MSE_minus_matched',ql,qu)]:
            vl,al,okl=donor_bootstrap(a,boot);vu,bu,oku=donor_bootstrap(b,boot);good=okl&oku
            row[label]=dict(identification_bounds=[vl,vu],censor_CI95=[float(np.quantile(al[good],.025)),float(np.quantile(bu[good],.975))])
            arrays[key+'__'+label+'_lower']=al;arrays[key+'__'+label+'_upper']=bu
        cml,cmu,cl,cu=(np.zeros(n) for _ in range(4))
        for offset in range(1,nc):
            donor=np.roll(np.arange(nc),offset)
            a,b=mse_gain(q[:,donor],q,lo,hi);cl+=cross_mean(a)/(nc-1);cu+=cross_mean(b)/(nc-1)
            a,b=selection_difference(lo,hi,chosen,chosen[:,donor]);cml+=a.mean(1)/(nc-1);cmu+=b.mean(1)/(nc-1)
        row['correct_minus_wrong_controller']=intervals(cml,cmu,boot,key+'__C_selection',arrays)
        row['wrong_controller_MSE_minus_matched']=intervals(cl,cu,boot,key+'__C_Q',arrays)
        for baseline in ('eta_only','state_field_compact'):
            a,b=selection_difference(lo,hi,chosen,choices[baseline])
            row['minus_'+baseline]=intervals(a.mean(1),b.mean(1),boot,key+'__'+baseline,arrays)
            a,b=mse_gain(pp[baseline],q,lo,hi)
            row[baseline+'_Q_MSE_minus_model']=intervals(cross_mean(a),cross_mean(b),boot,key+'__Q_'+baseline,arrays)
        a,b=selection_difference(lo[...,None,:],hi[...,None,:],chosen[...,None],np.arange(16)[None,None,:])
        # Most demanding comparator: each controller may choose its own constant eta,
        # reselected inside each state bootstrap. Never used to construct our selector.
        al,bu=[],[]
        for start in range(0,len(boot),500):
            ix=boot[start:start+500];al.append(a[ix].mean(1).min(-1).mean(-1));bu.append(b[ix].mean(1).min(-1).mean(-1))
        al,bu=np.concatenate(al),np.concatenate(bu)
        row['minus_bootstrap_reselected_per_controller_fixed']=dict(
            identification_bounds=[float(a.mean(0).min(-1).mean()),float(b.mean(0).min(-1).mean())],
            censor_CI95=[float(np.quantile(al,.025)),float(np.quantile(bu,.975))])
        arrays[key+'__fixed_lower']=al;arrays[key+'__fixed_upper']=bu
        if np.all(trials>=4) and len(np.unique(trials))==1:
            half=int(trials[0,0]//2)
            dl=ss[...,:half].mean(-1);dh=(ss+uu)[...,:half].mean(-1)
            el=ss[...,half:2*half].mean(-1);eh=(ss+uu)[...,half:2*half].mean(-1)
            row['reversal'],records=reversal(np.where(abs(q-q.mean(0,keepdims=True))<1e-7,q.mean(0,keepdims=True),q),dl,dh,el,eh,boot)
            row['reversal_pairs']=records
        results.append(row)
    return results


def run(index):
    variant=VARIANTS[index];source=SCENES[variant]
    d=Data(np.load(OLD/'data'/variant/'dataset.npz'))
    f=Data(np.load(OLD/'data/final/dataset.npz'))
    for sc in np.unique(f['scene']): f['entity_globals'][f['scene']==sc]=execution_constants(str(sc))
    pred={};final={};provenance={};arrays={};saved={};rows=[]
    for arm in BASES+ARMS:
        paths=folders(variant,arm)
        for path in paths:
            r=read(path/'complete.json');assert r['dataset_sha256']==sha(OLD/'data'/variant/'dataset.npz')
            assert r['checkpoint_sha256']==sha(path/'checkpoint.msgpack')
        pred[arm]=np.asarray([probability(np.load(p/'source_predictions.npz')['logits']) for p in paths])
        final[arm]=final_predictions(variant,arm,f)
        provenance[arm]=[dict(path=str(p),**read(p/'complete.json')) for p in paths]
    np.savez_compressed(HERE/f'{variant}_raw_predictions.npz',**{'development__'+k:v for k,v in pred.items()},**{'opened_final__'+k:v for k,v in final.items()})
    for scene in sorted(set(d['scene'])):
        for split in ('train','validation'):
            for group,names in [('seen',[f'analytic{i}' for i in range(4)]),('unseen',['analytic4','analytic5'])]:
                ids=np.flatnonzero((d['scene']==scene)&(d['original_split']==split)&np.isin(d['controller'],names))
                if not len(ids): continue
                panel=('source_' if scene==source else 'held_scene_')+split
                rows.extend(analyze_panel(d,ids,pred,dict(variant=variant,scene=scene,panel=panel,controller_group=group),arrays,saved))
        ids=np.flatnonzero(f['scene']==scene)
        panel=('source_' if scene==source else 'held_scene_')+'opened_Q16'
        rows.extend(analyze_panel(f,ids,final,dict(variant=variant,scene=scene,panel=panel,controller_group='unseen6_7'),arrays,saved))
    write(HERE/f'{variant}_results.json',dict(rows=rows,checkpoint_provenance=provenance,new_rollouts=0,
        note='All DEVELOPMENT. Models/norm/checkpoints source-only. Original FINAL has been opened previously. No new method confirmation. Unknown outcomes retained as censor intervals; midpoint correlations sensitivity only. Exact wrong-S donor expectation and200 random derangements; bootstrap resamples physical states jointly as recipients/donors. All16 fixed policies, additionally per-controller fixed eta. Q4 reversal uses2/2;Q16 uses8/8;all qualifying pairs, symmetric3/8 threshold,>=3states each direction.',
        input_scope='Exact six-parameter analytic family, not arbitrary neuralFlow. Initialization physical mean/covariance/chart are known runtime metadata, not fitted from target labels.'))
    np.savez_compressed(HERE/f'{variant}_bootstrap.npz',**arrays)
    np.savez_compressed(HERE/f'{variant}_predictions_and_controls.npz',**saved)
    for row in rows:
        if row['panel']=='held_scene_opened_Q16':
            print({k:row[k] for k in ('variant','arm','N','Q_MSE_midpoint','state_eta_corr','success_bounds','matched_minus_mismatch','wrong_controller_MSE_minus_matched')},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);a=p.parse_args();run(a.index)
