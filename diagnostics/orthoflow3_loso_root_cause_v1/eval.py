"""No rollout: reconstruct frozen inputs, freeze scores, then read exact cache."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import json,sys,csv,copy,hashlib,argparse
import numpy as np
import jax,jax.numpy as jnp
from flax import serialization
from scipy.special import expit
from scipy.spatial.distance import cdist
from scipy.stats import binomtest
from .run import ROOT,OUT,FIRST,FOLDS,SCENES,load,sha,dump,dump_at,configure,old,rep
from diagnostics.orthoflow3_cross_scene_zero_shot_v1 import train as tr
jax.config.update('jax_default_matmul_precision','highest')
CONF=ROOT/'diagnostics/orthoflow3_k16_critic_canonicalization_v1/phase_a/confirmation'
TOY=ROOT/'diagnostics/orthoflow3_c1_generalization_v1/toy_replication'

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def inputs(fold):
    path=OUT/'targets'/fold
    if (path/'manifest.json').exists():return load(path/'manifest.json'),dict(np.load(path/'entities.npz'))
    path.mkdir(parents=True,exist_ok=True);physical=[];flats=[];result=[]
    if fold=='ring':
        # First fold inputs already numerically audited against archived policy.
        p=load(FIRST/'target_predictions.json');x=dict(np.load(FIRST/'target_entities.npz'))
        result=p['rows'];dump_at(path,'manifest.json',result);np.savez_compressed(path/'entities.npz',**x);return result,x
    if fold=='toy':
        jax.config.update('jax_enable_x64',True)
        sys.path.insert(0,'/home/zhihan/research/02_C1_Toy_GiveWay')
        from single_integrator.evaluate import load_policy
        from single_integrator.environment import bounded_nominal
        policy,_=load_policy('/home/zhihan/research/02_C1_Toy_GiveWay/baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl')
        pp=load(TOY/'frozen_proposals.json')['states'];h=np.load(TOY/'cohort_features.npz')['h_raw']
        template=next(s['physical'] for s in load(FIRST/'source_states.json') if s['scenario']=='toy_giveway')
        keys=list(csv.DictReader((TOY/'candidate_cache_keys.csv').open()));ctl={r['state_uid']:r['controller_uid'] for r in keys if r['kind'].startswith('sample_')}
        sample=jax.jit(lambda ob,key:policy.sample_actions(ob[None],seed=key)[0]);errors=[]
        for i,p in enumerate(pp):
            s=copy.deepcopy(template);pos=np.asarray(p['initial_positions']);v=h[i,24:28].reshape(2,2);g=np.asarray(s['goals'])
            np.testing.assert_allclose(pos,h[i,20:24].reshape(2,2),atol=1e-6)
            obs=np.array([np.r_[pos[a],v[a],g[a]-pos[a],pos[1-a]-pos[a],v[1-a]-v[a]] for a in range(2)],np.float32)
            key=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(42),p['rollout_id']),0)
            flow=np.asarray(sample(obs,key));err=float(np.max(abs(bounded_nominal(flow,s['max_speed'])-h[i,40:44].reshape(2,2))))
            assert err<2e-5,err;errors.append(err)
            s.update(positions=pos.tolist(),velocities=v.tolist(),flow=flow.tolist());physical.append(s)
            result.append({'state_uid':p['state_uid'],'controller_uid':ctl[p['state_uid']],'eta':[p['eta'][f'sample_{j}'] for j in range(16)],'family':p['source_group'],'scores':{'archived_target_reference':expit(p['critic_scores']).tolist()}})
        dump_at(path,'input_replay.json',{'bounded_flow_max_error':max(errors),'environment_steps':0})
        jax.config.update('jax_enable_x64',False)
    else:
        sc=FOLDS[fold];ss={s['state_uid']:s for s in load(CONF/'state_manifest.json')['states'] if s['scenario']==sc}
        pp=[p for p in load(CONF/'proposals.json')['states'] if p['scenario']==sc]
        if fold=='db':
            from diagnostics.double_bottleneck_eta_basis_redesign.tools import run_rollouts as db
            dataset=db.FlowBC4ADataset(CONF/'double_pool','all');policy,_=db.load_checkpoint(db.CHECKPOINT,dataset.environment_fingerprint)
        else:
            from new_benchmark_common.macflow import load_checkpoint,sample_bounded_actions
            from four_way_intersection.environment import FourWayIntersectionEnv,Config
            man=load(ROOT/'diagnostics/four_way_intersection_stage1/base_u_v13_broad_global_dataset/manifest.json');cfg=Config(**man['scenario_config'])
            policy,_=load_checkpoint(ROOT/'diagnostics/four_way_intersection_stage1/base_u_v13_broad_global_source_balanced_macflow/best.pkl',expected_environment_fingerprint=man['environment_fingerprint'])
        for p in pp:
            record=ss[p['state_uid']];m=record['metadata'];phy=record['physical']
            if fold=='db':
                ep=dataset.by_family[m['family_id']][0];env=db._initialize_env(db.Config(**dataset.config),ep)
                np.testing.assert_allclose(env.positions,phy['positions'],atol=1e-12)
                key=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(m['historical_seed']),m['rollout_id']),0)
                obs=env.observation();flow=np.asarray(policy.sample_actions(obs[None],key)[0],float);desc=dataset.config
            else:
                env=FourWayIntersectionEnv(cfg);env.reset(np.asarray(phy['positions']),np.asarray(phy['velocities']));obs=env.observation()
                token=int(hashlib.sha256(p['state_uid'].encode()).hexdigest()[:8],16);key=jax.random.fold_in(jax.random.PRNGKey(2026100106),token)
                flow=np.asarray(sample_bounded_actions(policy,obs[None],key)[0],float);flow*=np.minimum(1.,cfg.max_speed/np.maximum(np.linalg.norm(flow,axis=-1,keepdims=True),1e-30));desc={}
            flat=np.r_[obs.ravel(),flow.ravel()];flats.append(flat)
            s=rep.parse({'scenario':sc,'structured_state':{'positions':env.positions.tolist(),'velocities':env.velocities.tolist(),'goals':env.goals.tolist(),'timestep':0},'conditioning':{'flat':flat.tolist()},'environment_descriptor':desc});physical.append(s)
            result.append({'state_uid':p['state_uid'],'controller_uid':p['controller_uid'],'eta':p['etas'][1:],'family':m.get('family_id',m.get('draw_index')),'scores':{'archived_target_joint_reference':p['scores'][1:]}})
        np.savez_compressed(path/'native_features.npz',h=np.asarray(flats,np.float32))
        # Independent replay of the same archived critic detects wrong Flow RNG,
        # coordinates or candidate indexing; no target outcomes read.
        from diagnostics.orthoflow3_generator_critic_v1 import train_evaluate as ref
        from diagnostics.orthoflow3_ring_revision_v1.run_fresh import context_vector
        from new_benchmark_common.basin_dataset_v1 import _environment_descriptor
        from types import SimpleNamespace
        norm=load(ROOT/'diagnostics/orthoflow3_generator_critic_v1/normalization.json');model=ref.Critic();dims={s:len(norm['scenarios'][s]['h_mean']) for s in ref.SCENARIOS}
        init=ref.merge_initialized(model,dims,len(norm['environment_keys'])+3,critic=True);ck=load(CONF.parent/'critic_frozen.json')['checkpoint'];params=serialization.from_bytes(init,open(ck,'rb').read())
        desc=dataset.config if fold=='db' else _environment_descriptor(SimpleNamespace(name=sc,config=SimpleNamespace(**man['scenario_config']),kind='four'))
        ctx=context_vector(desc,norm,sc);hn=norm['scenarios'][sc];hh=np.repeat((np.asarray(flats,np.float32)-np.asarray(hn['h_mean'],np.float32))/np.asarray(hn['h_std'],np.float32),16,axis=0)
        eta=np.asarray([r['eta'] for r in result],np.float32).reshape(-1,3)
        scores=expit(np.asarray(model.apply(params,jnp.asarray(hh),jnp.asarray(np.repeat(ctx[None],len(hh),axis=0)),jnp.asarray((eta-ref.CENTER)/ref.RADIUS),method=getattr(model,ref.METHOD[sc])))).reshape(-1,16)
        error=float(np.max(abs(scores-np.array([p['scores'][1:] for p in pp]))));assert error<2e-4,(fold,error)
        dump_at(path,'input_replay.json',{'archived_reference_max_error':error,'environment_steps':0,'checkpoint':ck})
    x=rep.batch([rep.entities(s) for s in physical]);np.savez_compressed(path/'entities.npz',**x)
    dump_at(path,'physical.json',physical);dump_at(path,'manifest.json',result)
    return result,x

def score_model(folder,x,etas,kind='shared'):
    frozen=load(folder/'models_frozen.json');norm=load(folder/'normalization.json');e=(etas.reshape(-1,3)-np.array(norm['eta_center'],np.float32))/np.array(norm['eta_radius'],np.float32)
    ix=np.repeat(np.arange(len(etas)),16);model=tr.model_for(kind);init=model.init(jax.random.PRNGKey(0),tr.gather(x,[0]),jnp.zeros((1,3)));scores={}
    for r in frozen[kind]['runs']:
        assert sha(r['checkpoint'])==r['sha256'];params=serialization.from_bytes(init,open(r['checkpoint'],'rb').read())
        fn=jax.jit(lambda bx,be:jax.nn.sigmoid(model.apply(params,bx,be)))
        a=np.concatenate([np.asarray(fn(tr.gather(x,ix[i:i+128]),jnp.asarray(e[i:i+128]))) for i in range(0,len(ix),128)]).reshape(-1,16)
        scores[f'{kind}_seed{r["seed"]}']=a
    scores[kind]=scores[f'{kind}_seed{frozen[kind]["selected"]["seed"]}'];scores[kind+'_ensemble']=np.mean([scores[f'{kind}_seed{s}'] for s in (17,23,41)],axis=0)
    return scores

def predict(fold):
    folder=OUT/fold
    if (folder/'target_predictions.json').exists():return
    ps,x=inputs(fold);etas=np.array([p['eta'] for p in ps],np.float32);scores={}
    for kind in ('shared','eta_only'):scores.update(score_model(folder,x,etas,kind))
    norm=load(folder/'normalization.json');e=(etas.reshape(-1,3)-np.array(norm['eta_center'],np.float32))/np.array(norm['eta_radius'],np.float32)
    kern=np.load(folder/'eta_kernel.npz');linear=np.load(folder/'eta_linear.npz')
    scores['eta_kernel']=np.clip(float(kern['prior'])+np.exp(-cdist(e,kern['z'],'sqeuclidean')/(2*float(kern['bandwidth'])**2))@kern['alpha'],0,1).reshape(-1,16)
    scores['global_linear']=expit(np.column_stack([e,np.ones(len(e))])@linear['beta']).reshape(-1,16)
    frozen=load(folder/'models_frozen.json');base=load(folder/'eta_baselines_frozen.json');baseline=min([('eta_only',frozen['eta_only']['selected']['validation_nll']),('eta_kernel',base['kernel']['validation_nll'])],key=lambda v:v[1])[0]
    states=load(folder/'source_states.json');assert not {s['state_uid'] for s in states}&{p['state_uid'] for p in ps}
    result={'source_selected_eta_baseline':baseline,'target_outcomes_opened':False,'rows':[dict(p,scores={**p['scores'],**{k:v[i].astype(float).tolist() for k,v in scores.items()}}) for i,p in enumerate(ps)]}
    dump_at(folder,'target_predictions.json',result)

def evaluate(fold,diagnostic=False):
    folder=OUT/fold;pred=load(folder/'target_predictions.json');ps=pred['rows'];details=[];truth=[]
    from shared_rollout_db.src.rollout_db import eta_identity
    import subprocess
    req=[{'state_uid':p['state_uid'],'eta_uid':eta_identity(e)[0],'controller_uid':p['controller_uid'],'seed_keys':[json.dumps({'future_index':i},separators=(',',':')) for i in range(16)]} for p in ps for e in p['eta']]
    dump_at(folder,'planned_rollouts.json',{'requests':req,'execute_missing':False})
    subprocess.run([sys.executable,'-m','shared_rollout_db.plan','--manifest',str(folder/'planned_rollouts.json'),'--output',str(folder/'cache_preflight.json')],cwd=ROOT,check=True,capture_output=True)
    c=old.con()
    for p in ps:
        t={'state_uid':p['state_uid'],'q':[],'lower':[],'upper':[],'robust':[],'unknown_seeds':[],'collisions':[]}
        for e in p['eta']:
            rows=c.execute("SELECT seed_key,success,numerical_failure,collision FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=? AND compatibility_quality='EXACT_REUSE' AND conflict_quarantined=0",(p['state_uid'],eta_identity(e)[0],p['controller_uid'])).fetchall()
            by={json.loads(r['seed_key']).get('future_index'):r for r in rows};good=[by[i] for i in range(16) if i in by and not by[i]['numerical_failure']]
            k=sum(r['success'] for r in good);u=16-len(good);t['q'].append(k/16 if not u else None);t['lower'].append(k/16);t['upper'].append((k+u)/16);t['robust'].append(True if k>=15 else False if k+u<15 else None);t['unknown_seeds'].append(u);t['collisions'].append(sum(r['collision'] for r in good))
        truth.append(t)
        for name,scores in p['scores'].items():
            j=int(np.argmax(scores));details.append({'method':name,'state_uid':p['state_uid'],'selected_index':j,'b15':t['robust'][j],'oracle_b15':any(v is True for v in t['robust']),'p':scores[j],'q':t['q'][j],'Q_lower':t['lower'][j],'Q_upper':t['upper'][j],'regret_lower':max(0,max(t['lower'])-t['upper'][j]),'regret_upper':max(t['upper'])-t['lower'][j],'severe_false_positive':scores[j]>=.95 and t['upper'][j]<=.5,'collision':t['collisions'][j],'numerical_seeds':t['unknown_seeds'][j]})
    c.close();dump_at(folder,'target_truth.json',truth);write_csv(folder/'target_selection.csv',details)
    summaries={}
    for name in ps[0]['scores']:
        rr=[r for r in details if r['method']==name];scores=np.array([p['scores'][name] for p in ps]).ravel();q=np.array([np.nan if v is None else v for t in truth for v in t['q']]);ok=np.isfinite(q)
        summaries[name]={'states':len(rr),'b15':sum(r['b15'] is True for r in rr),'unresolved':sum(r['b15'] is None for r in rr),'oracle_b15':sum(r['oracle_b15'] for r in rr),'Q_lower':float(np.mean([r['Q_lower'] for r in rr])),'Q_upper':float(np.mean([r['Q_upper'] for r in rr])),'regret_lower':float(np.mean([r['regret_lower'] for r in rr])),'regret_upper':float(np.mean([r['regret_upper'] for r in rr])),'severe_false_positive':sum(r['severe_false_positive'] for r in rr),'collision':sum(r['collision'] for r in rr),'probability_metrics':tr.metrics(scores[ok],q[ok])}
    paired={};rng=np.random.default_rng(20261002)
    for other in (pred['source_selected_eta_baseline'],'global_linear'):
        a=[r for r in details if r['method']=='shared'];b=[r for r in details if r['method']==other];diff=np.array([int(x['b15'])-int(y['b15']) for x,y in zip(a,b) if x['b15'] is not None and y['b15'] is not None]);res=int(sum(diff>0));br=int(sum(diff<0));boot=diff[rng.integers(0,len(diff),(10000,len(diff)))].mean(1)
        paired[other]={'rescue':res,'break':br,'resolved':len(diff),'paired_95CI':np.quantile(boot,[.025,.975]).tolist(),'exact_p':float(binomtest(res,res+br).pvalue) if res+br else 1.}
    main=summaries['shared'];base=summaries[pred['source_selected_eta_baseline']]
    verdict='FOLD_UNDERDISCRIMINATIVE' if min(main['b15'],base['b15'])/len(ps)>=.95 else 'CROSS_SCENE_STATE_CONDITIONING_NOT_YET_SUPPORTED'
    if paired[pred['source_selected_eta_baseline']]['paired_95CI'][0]>0 and main['b15']/max(1,main['oracle_b15'])>=.9:verdict='CROSS_SCENE_STATE_ETA_GENERALIZATION_SUPPORTED'
    dump_at(folder,'final_decision.json',{'models':summaries,'paired':paired,'source_selected_eta_baseline':pred['source_selected_eta_baseline'],'classification':verdict,'new_rollout':0,'predictions_sha256':sha(folder/'target_predictions.json')})
    print(fold,json.dumps({'classification':verdict,'models':{k:(v['b15'],v['unresolved']) for k,v in summaries.items()},'paired':paired}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['inputs','predict','evaluate']);ap.add_argument('--fold',default='toy');a=ap.parse_args()
    {'inputs':inputs,'predict':predict,'evaluate':evaluate}[a.action](a.fold)
