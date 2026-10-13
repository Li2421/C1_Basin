"""Frozen all-fold falsification for the isolated-interaction diagnostic."""
import argparse,types
import numpy as np
from .study import HERE,BASE,FOLDS,SEEDS,read,write,freeze,sha

MAT=BASE.parent/'controller_ff_factorial_repair_v1'
ARMS=('eta_only','slot_bce','full_q_mse','interaction_mse')
ALIASES={'eta_only_same_all_labels':'eta_only','field_same_all_labels':'slot_bce',
    'encounter_same_all_labels':'interaction_mse','encounter_flow_eta_prior':'full_q_mse'}


def path(fold,arm,seed):
    if arm in ('eta_only','slot_bce'):
        return BASE/'fits'/fold/('eta_only' if arm=='eta_only' else 'union_native_slots')/f'seed{seed}'
    return HERE/'fits'/fold/arm/f'seed{seed}'


def prepare():
    freeze(HERE/'evaluation_protocol.json',dict(
        method_protocol_sha256=sha(HERE/'protocol.json'),new_rollouts=0,
        primary='Centered S x eta and S x controller x eta predictive skill. For the interaction arm use the raw learned residual, not its nuisance prior. Compare every arm against zero, interaction versus full-Q MSE and versus existing BCE.',
        secondary='Selector score for the interaction arm adds the frozen Stage17 state-erased prior. All fixed policies, correct/wrong state and controller, independent-half all-pair reversals, censor intervals and collisions retained. Raw score is not a calibrated probability.',
        panels='Four held scenes, all natural40physicalstates; source jointVAL8states per scene descriptive. Fixed all/goal/Flow-reference strata. Previously opened32states per scene unseenanalytic6/7 K16/Q16 additionally.',
        information_boundary='All DEVELOPMENT. No target data enters fitting, normalization, TRAIN projection, checkpoint or prior. The diagnostic design is informed by previously opened development failures.',
        statistics='20000 paired physical-state bootstrap; primary skill recentered inside every replicate; censor endpoints exact. All model seeds averaged, no chosen seed. Matrix legacy mismatch uses a fixed donor population; Q16 mismatch jointly resamples donor/recipient physical states. A positive matrix mismatch alone cannot confirm transfer.',
        projection='S_eta: center predicted response over state and eta within each controller. S_C_eta: first take every unordered C contrast, then the same centering. Gain = mean(2*truth_contrast*prediction - prediction^2); orthogonality permits uncentered truth. Bounds are affine endpoints.',
        literature=read(HERE/'protocol.json')['literature'],
        reused_engines={str(p):sha(p) for p in (MAT/'review.py',BASE.parent/'controller_program_input_probe_v1/review.py')},code_sha256=sha(__file__)))


def matrix():
    p=read(HERE/'evaluation_protocol.json');root=HERE/'matrix_review';root.mkdir(exist_ok=True)
    for name,src in [('dataset.npz',MAT/'dataset.npz'),('raw_labels.npz',MAT/'raw_labels.npz'),('evaluation_protocol.json',HERE/'evaluation_protocol.json')]:
        if not (root/name).exists():(root/name).symlink_to(src)
    d=dict(np.load(BASE/'dataset.npz'));ii=np.flatnonzero(d['source']=='matrix_goal_and_flow32');ii=ii[np.argsort(d['source_row'][ii])]
    np.testing.assert_array_equal(d['source_row'][ii],np.arange(1600))
    for fold in FOLDS:
        for alias,arm in ALIASES.items():
            for seed in SEEDS:
                src=path(fold,arm,seed);out=root/'fits'/fold/alias/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
                record=read(src/'complete.json');assert record['checkpoint_sha256']==sha(src/'checkpoint.msgpack')
                score=np.load(src/'predictions.npz')['probability'][ii][:,[0,5,10,15]]
                np.savez_compressed(out/'predictions.npz',probability=score,eta_indices=np.array([0,5,10,15]));write(out/'complete.json',record)
    assert sha(MAT/'review.py')==p['reused_engines'][str(MAT/'review.py')]
    code=(MAT/'review.py').read_text().replace("('field_same_all_labels','encounter_flow_eta_prior','encounter_goal_only')","('field_same_all_labels','encounter_flow_eta_prior')")
    mod=types.ModuleType('diagnostics.controller_ff_factorial_repair_v1.reused_interaction_review');mod.__package__='diagnostics.controller_ff_factorial_repair_v1'
    exec(compile(code,str(root/'frozen_reused_engine.py'),'exec'),mod.__dict__);(root/'frozen_reused_engine.py').write_text(code)
    mod.HERE=root;mod.CONDITIONS=tuple(ALIASES)
    def corr(a,b):
        a,b=np.asarray(a).ravel(),np.asarray(b).ravel();good=np.isfinite(a)&np.isfinite(b);a,b=a[good],b[good]
        if len(a)<3 or min(a.std(),b.std())<1e-7:return None
        return float(np.corrcoef(a,b)[0,1])
    mod.corr=corr;mod.run()
    def rename(v):
        if isinstance(v,dict):return {k:rename(x) for k,x in v.items()}
        if isinstance(v,list):return [rename(x) for x in v]
        if isinstance(v,str):
            for a,b in ALIASES.items():v=v.replace(a,b)
        return v
    result=rename(read(root/'development_results.json'));result['scope']=p['information_boundary'];result['prediction_semantics']=p['secondary']
    write(HERE/'matrix_results.json',result)


def interactions():
    from diagnostics.controller_cached_union_v1.interaction_audit import center,gain
    p=read(HERE/'evaluation_protocol.json');d=dict(np.load(BASE/'dataset.npz'));old=dict(np.load(MAT/'dataset.npz'))
    ii=np.flatnonzero(d['source']=='matrix_goal_and_flow32');ii=ii[np.argsort(d['source_row'][ii])]
    output=[];saved={};arrays={}
    comparisons=[(arm,'zero') for arm in ARMS]+[('interaction_mse','full_q_mse'),('interaction_mse','slot_bce')]
    for fold,held in FOLDS.items():
        pred={}
        for arm in ARMS:
            field='raw_network_output' if arm=='interaction_mse' else 'probability'
            pred[arm]=np.mean([np.load(path(fold,arm,s)/'predictions.npz')[field][ii][:,[0,5,10,15]] for s in SEEDS],0)
        for scene in (sorted(set(old['scene'])) if held is None else (held,)):
            mask=old['scene']==scene
            if held is None:mask&=old['split']=='validation'
            for reference in ('all','flow','goal'):
                use=mask.copy()
                if reference!='all':use&=old['kind']==reference
                ids=sorted(np.flatnonzero(use),key=lambda i:(old['state_uid'][i],old['controller'][i]));ids=np.asarray(ids)
                states=np.unique(old['state_uid'][ids]);controllers=np.unique(old['controller'][ids]);n,nc=len(states),len(controllers)
                assert len(ids)==n*nc;shape=(n,nc,4)
                s,f,u=[old[k][ids][:,[0,5,10,15]].reshape(shape) for k in ('success','failure','unknown')]
                lo=s/(s+f+u);hi=(s+u)/(s+f+u);qq={arm:q[ids].reshape(shape) for arm,q in pred.items()}
                a,b=np.triu_indices(nc,1);boot=np.random.default_rng(2026101018+n).integers(n,size=(20000,n))
                prefix=f'{fold}__{scene}__{reference}';saved[prefix+'__states']=states;saved[prefix+'__controllers']=controllers
                saved[prefix+'__truth_lower']=lo;saved[prefix+'__truth_upper']=hi;arrays[prefix+'__bootstrap_indices']=boot
                for label in ('S_eta','S_C_eta'):
                    if label=='S_C_eta':dl,du=lo[:,a]-hi[:,b],hi[:,a]-lo[:,b];q={arm:v[:,a]-v[:,b] for arm,v in qq.items()}
                    else:dl,du=lo,hi;q=qq
                    x={arm:center(v[None],True) for arm,v in q.items()};x['zero']=np.zeros_like(x[ARMS[0]])
                    point={pair:gain(x[pair[0]],x[pair[1]],dl[None],du[None]) for pair in comparisons}
                    samples={pair:[[],[]] for pair in comparisons}
                    for start in range(0,len(boot),250):
                        ix=boot[start:start+250];xx={arm:center(v[ix],True) for arm,v in q.items()};xx['zero']=np.zeros_like(xx[ARMS[0]])
                        for pair in comparisons:
                            l,h=gain(xx[pair[0]],xx[pair[1]],dl[ix],du[ix]);samples[pair][0].append(l);samples[pair][1].append(h)
                    for pair in comparisons:
                        l,h=[np.concatenate(v) for v in samples[pair]];pl,ph=point[pair]
                        key=prefix+'__'+label+'__'+pair[0]+'__minus__'+pair[1];arrays[key+'__lower']=l;arrays[key+'__upper']=h
                        output.append(dict(fold=fold,scene=scene,reference=reference,N=n,interaction=label,first=pair[0],second=pair[1],
                            skill_identification_bounds=[float(pl[0]),float(ph[0])],censor_CI95=[float(np.quantile(l,.025)),float(np.quantile(h,.975))],unknown=int(u.sum())))
                for arm,q in qq.items():saved[prefix+'__'+arm+'__raw_mechanism_predictions']=q
    write(HERE/'interaction_results.json',dict(rows=output,protocol_sha256=sha(HERE/'evaluation_protocol.json'),new_rollouts=0))
    np.savez_compressed(HERE/'interaction_bootstrap.npz',**arrays);np.savez_compressed(HERE/'interaction_truth_and_predictions.npz',**saved)


def q16(index):
    import jax,jax.numpy as jnp
    from flax import serialization
    from diagnostics.controller_q_diversity_v1.train import Critic,compact_field
    from diagnostics.controller_ff_crossed_cache_full_v1.study import full_physical
    from diagnostics.orthoflow3_unified_representation_v1.representation import batch
    from diagnostics.controller_program_input_probe_v1.review import Data,analyze_panel
    div=BASE.parent/'controller_q_diversity_v1';scene=('toy_give_way','ring_exchange')[index];held=('held_toy','held_ring')[index]
    data=Data(np.load(div/'data/final/dataset.npz'));dp=read(div/'protocol.json');states={s['state_uid']:s for s in dp['states']}
    physical=[full_physical(str(sc),states[str(sid)]['physical'],dp['environments'][str(sc)]['config']['config']) for sid,sc in zip(data['state_uid'],data['scene'])]
    x={k:jnp.asarray(v,jnp.float32) for k,v in batch(physical).items()};x['agents']=x['agents'].at[...,3:7].set(jnp.eye(4)[None]*x['agent_mask'][...,None])
    feature=np.asarray(compact_field(jnp.asarray(np.concatenate((data['nominal'],data['field']),-1))))
    old=np.load(BASE/f'q16_{scene}_raw_predictions.npz');result=[];arrays={};saved={};raws={}
    for fold in ('joint',held):
        pred={'eta_only':old[fold+'__eta_only'],'state_field_compact':old[fold+'__union_native_slots']}
        for arm in ('full_q_mse','interaction_mse'):
            values=[]
            for seed in SEEDS:
                folder=path(fold,arm,seed);norm=read(folder/'normalization.json');complete=read(folder/'complete.json')
                assert sha(folder/'checkpoint.msgpack')==complete['checkpoint_sha256']
                c=jnp.asarray(np.clip((feature-np.asarray(norm['mean']))/np.asarray(norm['scale']),-10,10))*x['agent_mask'][...,None]
                e=jnp.asarray((np.asarray(dp['eta'])-np.asarray(norm['eta_center']))/np.asarray(norm['eta_radius']),jnp.float32)
                net=Critic('state_field_compact');initial=net.init(jax.random.PRNGKey(seed),{k:v[:1] for k,v in x.items()},c[:1],e[:1])
                par=serialization.from_bytes(initial,(folder/'checkpoint.msgpack').read_bytes())
                @jax.jit
                def predict(ii,jj):return net.apply(par,{k:v[ii] for k,v in x.items()},c[ii],e[jj])
                ii=np.repeat(np.arange(len(feature)),16);jj=np.tile(np.arange(16),len(feature));z=[]
                for start in range(0,len(ii),256):z.append(np.asarray(predict(jnp.asarray(ii[start:start+256]),jnp.asarray(jj[start:start+256]))))
                values.append(np.concatenate(z).reshape(-1,16))
            raw=np.asarray(values);raws[fold+'__'+arm]=raw
            pred[arm]=raw+old[fold+'__union_slots_state_erased'] if arm=='interaction_mse' else raw
        ids=np.flatnonzero(data['scene']==scene)
        result.extend(analyze_panel(data,ids,pred,dict(variant=fold,panel='opened_independent_Q16_unseen_controller',scene=scene,controller_group='analytic6_7'),arrays,saved))
    for r in result:
        if r['arm']=='state_field_compact':r['arm']='slot_bce'
    write(HERE/f'q16_{scene}_results.json',dict(rows=result,new_rollouts=0,scope='Opened DEVELOPMENT only; state_field_compact comparison keys mean Stage17 slot_bce. MSE models output uncalibrated diagnostic scores.'))
    np.savez_compressed(HERE/f'q16_{scene}_bootstrap.npz',**arrays);np.savez_compressed(HERE/f'q16_{scene}_predictions.npz',**saved)
    np.savez_compressed(HERE/f'q16_{scene}_raw_network_output.npz',**raws)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','matrix','interactions','q16']);ap.add_argument('--index',type=int,default=0);a=ap.parse_args()
    {'prepare':prepare,'matrix':matrix,'interactions':interactions,'q16':lambda:q16(a.index)}[a.stage]()
