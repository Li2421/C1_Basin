"""Native TEST inference for the completed sampler intervention; no fitting."""
import argparse
import numpy as np
from .study import HERE, BASE, ROOT, SEEDS, read, write, freeze, sha

OLD = ROOT / 'controller_entity_coverage_check_v1'
EXPOSURE = ROOT / 'controller_native_exposure_v1'
ARMS = ('eta_only', 'union_native_slots', 'uniform_eta_only', 'uniform_native_slots')


def prepare():
    freeze(HERE/'native_evaluation_protocol.json', dict(
        scope='Opened native Ring64 DEVELOPMENT, SOURCE VAL-selected joint models; not LOSO or untouched confirmation.',
        history='Stage19 tested different isolated-task interventions on these labels. Stage22 sampling-restoration checkpoints have never been inferred here. Reuse old uniform predictions and all existing labels/descriptors; infer only3 new state checkpoints.',
        primary='All64 test states; fresh-origin48 and original16 descriptive sensitivity only.',
        source_selection='Original global SOURCE VAL rule for both new and old models, frozen before these TEST predictions.',
        comparison='New sampler versus original uniform sampler, same current inputs/labels/architecture/steps. Fair eta-only uses the matching new sampler.',
        statistics='20000 paired physical-state bootstrap; recipient and donor resampled jointly;200 saved derangements; all16 fixed eta including bootstrap-reselected best;8/8 symmetric strong-reversal rule reused unchanged.',
        controller_metric='Single controller: between-controller generalization NOT ESTIMABLE.',
        raw_labels=str(OLD/'dataset.npz'), raw_labels_sha256=sha(OLD/'dataset.npz'),
        reused_prediction_sha256=sha(EXPOSURE/'test_predictions.npz'),
        reused_engine_sha256=sha(ROOT/'controller_program_input_probe_v1/review.py'),
        new_rollouts=0, new_static_queries=0, code_sha256=sha(__file__)))


def infer():
    import jax
    import jax.numpy as jnp
    from flax import serialization
    from diagnostics.controller_q_diversity_v1.train import Critic, compact_field
    d=dict(np.load(OLD/'dataset.npz')); ix=np.flatnonzero(d['split']=='test')
    full=dict(np.load(BASE/'dataset.npz')); p=read(BASE/'protocol.json')
    prior=np.load(EXPOSURE/'test_predictions.npz')
    np.testing.assert_array_equal(prior['state_uid'], d['state_uid'][ix])
    assert not set(d['state_uid'][ix]) & set(full['state_uid'][np.isin(full['split'], ['train','validation'])])
    for f,h in p['static_feature_code'].items(): assert sha(f)==h
    features=[]
    for sid in d['state_uid'][ix]:
        z=np.load(EXPOSURE/'features'/str(sid)/'flow.npz')
        features.append(np.asarray(compact_field(jnp.asarray(np.concatenate((z['nominal'],z['field']),-1)))))
    features=np.asarray(features,np.float32)
    x={k[7:]:d[k][ix] for k in d if k.startswith('entity_')}
    x['obstacles']=np.pad(x['obstacles'],((0,0),(0,0),(0,full['entity_obstacles'].shape[2]-x['obstacles'].shape[2]),(0,0)))
    x['obstacle_mask']=np.pad(x['obstacle_mask'],((0,0),(0,full['entity_obstacle_mask'].shape[1]-x['obstacle_mask'].shape[1])))
    x={k:jnp.asarray(v,jnp.float32) for k,v in x.items()}
    x['agents']=x['agents'].at[...,3:7].set(jnp.eye(4)[None]*x['agent_mask'][...,None])
    e=jnp.asarray((full['eta']-np.asarray(p['eta_input_center']))/np.asarray(p['eta_input_radius']),jnp.float32)
    outputs={'uniform_native_slots':prior['selected__original_union_global']}; records=[]
    for arm in ARMS:
        if arm=='uniform_native_slots': continue
        values=[]
        for seed in SEEDS:
            folder=HERE/'fits/joint'/arm/f'seed{seed}'
            record=read(folder/'complete.json'); assert sha(folder/'checkpoint.msgpack')==record['checkpoint_sha256']
            records.append(dict(arm=arm,seed=seed,checkpoint_sha256=record['checkpoint_sha256']))
            if arm.endswith('eta_only'):
                cached=np.load(folder/'predictions.npz')['probability']
                np.testing.assert_array_equal(cached,np.broadcast_to(cached[:1],cached.shape))
                values.append(np.broadcast_to(cached[0],(len(ix),16)).copy()); continue
            norm=read(folder/'normalization.json')
            c=jnp.asarray(np.clip((features-np.asarray(norm['mean'],np.float32))/np.asarray(norm['scale'],np.float32),-10,10),jnp.float32)
            net=Critic('state_field_compact')
            initial=net.init(jax.random.PRNGKey(seed),{k:v[:1] for k,v in x.items()},c[:1],e[:1])
            params=serialization.from_bytes(initial,(folder/'checkpoint.msgpack').read_bytes())
            @jax.jit
            def predict(ii,jj): return net.apply(params,{k:v[ii] for k,v in x.items()},c[ii],e[jj])
            ii=np.repeat(np.arange(len(ix)),16); jj=np.tile(np.arange(16),len(ix)); logits=[]
            for start in range(0,len(ii),256):
                logits.append(np.asarray(predict(jnp.asarray(ii[start:start+256]),jnp.asarray(jj[start:start+256]))))
            z=np.concatenate(logits).reshape(-1,16);values.append(1/(1+np.exp(-np.clip(z,-80,80))))
        outputs[arm]=np.asarray(values)
    np.savez_compressed(HERE/'native_predictions.npz',state_uid=d['state_uid'][ix],**outputs)
    write(HERE/'native_inference_audit.json',dict(checkpoints=records,reused_uniform_state=True,reused_eta_predictions=True,physical_state_overlap=0,new_rollouts=0,new_static_queries=0))


def analyze():
    from diagnostics.controller_program_input_probe_v1.review import Data, analyze_panel
    from diagnostics.controller_ff_factorial_repair_v1.review import selection_difference, mse_gain, intervals
    d=dict(np.load(OLD/'dataset.npz')); ix=np.flatnonzero(d['split']=='test'); out=d['raw_outcomes'][ix]
    assert out.shape==(64,16,16,3)
    data=Data(state_uid=d['state_uid'][ix],controller=np.array(['native_flow']*len(ix)),trials=np.full(len(ix),16),
        success=d['success'][ix],unknown=d['unknown'][ix],seed_success=((out[...,0]==1)&(out[...,1]==0)).astype(float),
        seed_unknown=(out[...,1]==1).astype(float),seed_collision=(out[...,2]==1).astype(int))
    np.testing.assert_array_equal(data['success'],data['seed_success'].sum(-1))
    np.testing.assert_array_equal(data['unknown'],data['seed_unknown'].sum(-1))
    archive=np.load(HERE/'native_predictions.npz'); np.testing.assert_array_equal(archive['state_uid'],data['state_uid'])
    pred={('state_field_compact' if a=='union_native_slots' else a):archive[a] for a in ARMS}
    rows=[];paired=[];arrays={};saved={}
    def centered(z): return z-z.mean(-2,keepdims=True)-z.mean(-1,keepdims=True)+z.mean((-2,-1),keepdims=True)
    def igain(a,b,lo,hi):
        a,b=centered(a),centered(b);x,y=2*(a-b)*lo-a*a+b*b,2*(a-b)*hi-a*a+b*b
        return np.minimum(x,y).mean((-2,-1)),np.maximum(x,y).mean((-2,-1))
    for panel in ('test_all','test_fresh','test_original'):
        ids=np.arange(len(ix)) if panel=='test_all' else np.flatnonzero(d['origin'][ix]==panel[5:])
        meta=dict(variant='joint_source_selected',panel=panel,scene='ring_exchange',controller_group='native_flow')
        rr=analyze_panel(data,ids,pred,meta,arrays,saved)
        for r in rr:
            if r['arm']=='state_field_compact':r['arm']='union_native_slots'
            for k in ('correct_minus_wrong_controller','wrong_controller_MSE_minus_matched','controller_effect_corr','controller_state_eta_corr'):r[k]=None
        rows.extend(rr)
        ids=np.asarray(sorted(ids,key=lambda i:data['state_uid'][i]));n=len(ids)
        boot=np.random.default_rng(2026101016+n).integers(n,size=(20000,n))
        lo=data['success'][ids]/16;hi=(data['success']+data['unknown'])[ids]/16
        qa=archive['union_native_slots'][:,ids].mean(0);qb=archive['uniform_native_slots'][:,ids].mean(0)
        key=panel+'__new_minus_old';a,b=selection_difference(lo,hi,qa.argmax(-1),qb.argmax(-1))
        selection=intervals(a,b,boot,key+'__selection',arrays)
        a,b=mse_gain(qb,qa,lo,hi);qgain=intervals(a.mean(1),b.mean(1),boot,key+'__Q_gain',arrays)
        a,b=igain(qa,qb,lo,hi);al=[];au=[]
        for start in range(0,len(boot),500):
            ib=boot[start:start+500];l,u=igain(qa[ib],qb[ib],lo[ib],hi[ib]);al.append(l);au.append(u)
        al,au=np.concatenate(al),np.concatenate(au);arrays[key+'__interaction_lower']=al;arrays[key+'__interaction_upper']=au
        paired.append(dict(**meta,N=n,selection_gain=selection,Q_MSE_gain=qgain,interaction_MSE_gain=dict(
            identification_bounds=[float(a),float(b)],censor_CI95=[float(np.quantile(al,.025)),float(np.quantile(au,.975))])))
    write(HERE/'native_results.json',dict(rows=rows,paired_new_minus_uniform=paired,new_rollouts=0,
        alias='state_field_compact in comparison keys means new union_native_slots; native64 opened DEVELOPMENT only.'))
    np.savez_compressed(HERE/'native_bootstrap.npz',**arrays)
    np.savez_compressed(HERE/'native_predictions_and_controls.npz',**saved)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','infer','analyze']);a=p.parse_args()
    {'prepare':prepare,'infer':infer,'analyze':analyze}[a.stage]()
