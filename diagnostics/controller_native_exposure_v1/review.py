"""Native branch hold-out audit; inference and cached-label comparisons only."""
import argparse
import numpy as np
from .study import HERE,BASE,OLD,ARMS,SEEDS,read,write,freeze,sha


def prepare():
    old=BASE.parent/'controller_program_input_probe_v1/review.py'
    freeze(HERE/'evaluation_protocol.json',dict(method_protocol_sha256=sha(HERE/'protocol.json'),
        scope='Opened nativeRing64 DEVELOPMENT states only. All labels reused. This diagnoses source native learning and cannot certify LOSO.',
        checkpoints='NativeVAL32-selected primary and final-step sensitivity, all3seeds averaged; no TEST choice.',
        panels=['test_all64 primary','test_fresh_origin48 secondary','test_original16 secondary'],
        comparisons=[['isolated_sparse_field','joint_replay'],['isolated_dense_field','isolated_sparse_field'],
            ['isolated_sparse_zero','isolated_sparse_field'],['isolated_dense_zero','isolated_dense_field'],
            ['isolated_dense_zero','isolated_sparse_zero']],
        metrics='Q MSE, state/eta centered correlation, interaction skill versus zero, selected success/B15/regret, paired MSE/selection differences, correct versus same-scene wrongstate, bootstrap-reselected all16fixed and symmetric discovery8/evaluation8 reversals.',
        statistics='20000 physical-state bootstrap. Same draws for paired comparisons. Mismatch donor and recipient states jointly resampled. Primary interaction skill recentered in every replicate. Censor bounds retained.',
        controller_metric='Only one controller in this panel: wrong-controller and cross-controller correlation are NOT ESTIMABLE, never interpreted as zero skill.',
        engine_sha256=sha(old),new_rollouts=0,code_sha256=sha(__file__)))


def infer():
    import jax,jax.numpy as jnp
    from flax import serialization
    from diagnostics.controller_q_diversity_v1.train import Critic,compact_field
    d=dict(np.load(OLD/'dataset.npz'));ix=np.flatnonzero(d['split']=='test');full=dict(np.load(BASE/'dataset.npz'))
    for seed in SEEDS:assert read(HERE/'fits/joint_replay'/f'seed{seed}/replay_integrity.json')['byte_identical']
    for arm in ARMS:
        for seed in SEEDS:assert (HERE/'fits'/arm/f'seed{seed}/complete.json').exists()
    p=read(BASE/'protocol.json')
    for file,h in p['static_feature_code'].items():assert sha(file)==h
    feature=[]
    for sid in d['state_uid'][ix]:
        z=np.load(HERE/'features'/str(sid)/'flow.npz');feature.append(np.asarray(compact_field(jnp.asarray(np.concatenate((z['nominal'],z['field']),-1)))))
    feature=np.asarray(feature,np.float32)
    x={k[7:]:d[k][ix] for k in d if k.startswith('entity_')}
    x['obstacles']=np.pad(x['obstacles'],((0,0),(0,0),(0,full['entity_obstacles'].shape[2]-x['obstacles'].shape[2]),(0,0)))
    x['obstacle_mask']=np.pad(x['obstacle_mask'],((0,0),(0,full['entity_obstacle_mask'].shape[1]-x['obstacle_mask'].shape[1])))
    x={k:jnp.asarray(v,jnp.float32) for k,v in x.items()};x['agents']=x['agents'].at[...,3:7].set(jnp.eye(4)[None]*x['agent_mask'][...,None])
    e=jnp.asarray((full['eta']-np.asarray(p['eta_input_center']))/np.asarray(p['eta_input_radius']),jnp.float32)
    outputs={};checkpoints={}
    for arm in ARMS+('original_union_global',):
        for which in ('selected','final') if arm!='original_union_global' else ('selected',):
            predictions=[]
            for seed in SEEDS:
                folder=BASE/'fits/joint/union_native_slots'/f'seed{seed}' if arm=='original_union_global' else HERE/'fits'/arm/f'seed{seed}'
                file=folder/('checkpoint.msgpack' if which=='selected' else 'final_checkpoint.msgpack')
                norm=read(folder/'normalization.json')
                c=np.clip((feature-np.asarray(norm['mean'],np.float32))/np.asarray(norm['scale'],np.float32),-10,10)
                if arm.endswith('_zero'):c=np.zeros_like(c)
                c=jnp.asarray(c,jnp.float32);net=Critic('state_field_compact')
                initial=net.init(jax.random.PRNGKey(seed),{k:v[:1] for k,v in x.items()},c[:1],e[:1]);par=serialization.from_bytes(initial,file.read_bytes())
                @jax.jit
                def predict(ii,jj):return net.apply(par,{k:v[ii] for k,v in x.items()},c[ii],e[jj])
                ii=np.repeat(np.arange(len(ix)),16);jj=np.tile(np.arange(16),len(ix));z=[]
                for start in range(0,len(ii),256):z.append(np.asarray(predict(jnp.asarray(ii[start:start+256]),jnp.asarray(jj[start:start+256]))))
                z=np.concatenate(z).reshape(-1,16);predictions.append(1/(1+np.exp(-np.clip(z,-80,80))))
                checkpoints[str(file)]=sha(file)
            outputs[which+'__'+arm]=np.asarray(predictions)
    for name,pattern in [('eta_only',OLD/'fits/eta_only/n512'),('old_physical_slot',BASE.parent/'controller_entity_symmetry_probe_v1/fits')]:
        outputs['selected__'+name]=np.asarray([np.load(pattern/f'seed{s}/predictions.npz')['probability'][ix] for s in SEEDS])
    np.savez_compressed(HERE/'test_predictions.npz',state_uid=d['state_uid'][ix],**outputs)
    write(HERE/'inference_audit.json',dict(test_states=len(ix),checkpoints=checkpoints,new_rollouts=0,physical_steps=0,static_feature_files=len(feature),test_labels_used_for_fit_or_selection=False))


def analyze():
    from diagnostics.controller_program_input_probe_v1.review import Data,analyze_panel
    from diagnostics.controller_ff_factorial_repair_v1.review import selection_difference,mse_gain,intervals
    protocol=read(HERE/'evaluation_protocol.json');d=dict(np.load(OLD/'dataset.npz'));ix=np.flatnonzero(d['split']=='test');out=d['raw_outcomes'][ix]
    assert out.shape==(64,16,16,3)
    data=Data(state_uid=d['state_uid'][ix],controller=np.array(['native_flow']*len(ix)),
        trials=np.full(len(ix),16),success=d['success'][ix],unknown=d['unknown'][ix],
        seed_success=((out[...,0]==1)&(out[...,1]==0)).astype(float),seed_unknown=(out[...,1]==1).astype(float),seed_collision=(out[...,2]==1).astype(int))
    np.testing.assert_array_equal(data['seed_success'].sum(-1),data['success']);np.testing.assert_array_equal(data['seed_unknown'].sum(-1),data['unknown'])
    archive=np.load(HERE/'test_predictions.npz');np.testing.assert_array_equal(archive['state_uid'],data['state_uid'])
    rows=[];comparisons=[];arrays={};saved={}
    for which in ('selected','final'):
        pred={a:archive[which+'__'+a] for a in ARMS}
        pred['eta_only']=archive['selected__eta_only'];pred['state_field_compact']=pred.pop('joint_replay')
        pred['old_physical_slot']=archive['selected__old_physical_slot'];pred['original_union_global']=archive['selected__original_union_global']
        for panel in ('test_all','test_fresh','test_original'):
            ids=np.arange(len(ix)) if panel=='test_all' else np.flatnonzero(d['origin'][ix]==panel[5:])
            metadata=dict(variant=which,panel=panel,scene='ring_exchange',controller_group='native_flow')
            rr=analyze_panel(data,ids,pred,metadata,arrays,saved)
            for r in rr:
                if r['arm']=='state_field_compact':r['arm']='joint_replay'
                for k in ('correct_minus_wrong_controller','wrong_controller_MSE_minus_matched','controller_effect_corr','controller_state_eta_corr'):r[k]=None
            rows.extend(rr)
            ids=np.asarray(sorted(ids,key=lambda i:data['state_uid'][i]));n=len(ids);boot=np.random.default_rng(2026101016+n).integers(n,size=(20000,n))
            lo=data['success'][ids]/16;hi=(data['success'][ids]+data['unknown'][ids])/16
            q={('joint_replay' if a=='state_field_compact' else a):v[:,ids].mean(0) for a,v in pred.items()}
            for first,second in protocol['comparisons']:
                key=which+'__'+panel+'__'+first+'__minus__'+second;a,b=selection_difference(lo,hi,q[first].argmax(-1),q[second].argmax(-1))
                selection=intervals(a,b,boot,key+'__selection',arrays)
                a,b=mse_gain(q[second],q[first],lo,hi);error=intervals(a.mean(1),b.mean(1),boot,key+'__Q_gain',arrays)
                comparisons.append(dict(**metadata,first=first,second=second,N=n,selection_gain=selection,Q_MSE_gain=error))
            for arm,v in q.items():
                def centered(z):return z-z.mean(-2,keepdims=True)-z.mean(-1,keepdims=True)+z.mean((-2,-1),keepdims=True)
                x=centered(v);a=2*x*lo-x*x;b=2*x*hi-x*x;pl,pu=np.minimum(a,b).mean(),np.maximum(a,b).mean();al=[];au=[]
                for start in range(0,len(boot),500):
                    ib=boot[start:start+500];x=centered(v[ib]);a=2*x*lo[ib]-x*x;b=2*x*hi[ib]-x*x;al.append(np.minimum(a,b).mean((1,2)));au.append(np.maximum(a,b).mean((1,2)))
                al,au=np.concatenate(al),np.concatenate(au);key=which+'__'+panel+'__'+arm+'__interaction_skill';arrays[key+'__lower']=al;arrays[key+'__upper']=au
                comparisons.append(dict(**metadata,first=arm,second='zero_interaction',N=n,interaction_skill=dict(identification_bounds=[float(pl),float(pu)],censor_CI95=[float(np.quantile(al,.025)),float(np.quantile(au,.975))])))
    write(HERE/'results.json',dict(rows=rows,comparisons=comparisons,new_rollouts=0,scope=protocol['scope'],
        alias='state_field_compact comparison/array keys mean joint_replay. Wrong-controller metrics not estimable. Primary test_all; old fresh/original strata secondary.'))
    np.savez_compressed(HERE/'bootstrap.npz',**arrays);np.savez_compressed(HERE/'predictions_and_controls.npz',**saved)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','infer','analyze']);a=ap.parse_args();{'prepare':prepare,'infer':infer,'analyze':analyze}[a.stage]()
