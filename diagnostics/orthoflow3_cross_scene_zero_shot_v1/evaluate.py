"""Frozen source models -> frozen target proposals -> cached outcome evaluation."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import json,hashlib,sys,csv
from collections import defaultdict,Counter
from datetime import datetime,timezone
import numpy as np
import jax,jax.numpy as jnp
from flax import serialization
from scipy.special import expit
from scipy.spatial.distance import cdist
from scipy.stats import binomtest
from .build_source import ROOT,OUT,SOURCES,load,dump,sha,con
from .train import model_for,data,gather,metrics
from diagnostics.orthoflow3_unified_representation_v1 import representation as rep

CONF=ROOT/'diagnostics/orthoflow3_k16_critic_canonicalization_v1/phase_a/confirmation'
def write_csv(name,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def physical_inputs():
    # Target base-policy inference is permitted; no target eta/Q labels loaded.
    from new_benchmark_common.macflow import load_checkpoint,sample_bounded_actions
    from ring_exchange.environment import LocalFrameConfig,RingExchangeEnv
    from ring_exchange.local_frame import local_observation,local_actions_to_world
    ds=ROOT/'diagnostics/ring_exchange_stage1/base_u_v10_local_dataset/manifest.json'
    man=load(ds);cfg=LocalFrameConfig(**man['scenario_config'])
    policy,_=load_checkpoint(ROOT/'diagnostics/ring_exchange_stage1/base_u_v10_local_macflow/best.pkl',expected_environment_fingerprint=man['environment_fingerprint'])
    ss={r['state_uid']:r for r in load(CONF/'state_manifest.json')['states'] if r['scenario']=='ring_exchange'}
    ps=[r for r in load(CONF/'proposals.json')['states'] if r['scenario']=='ring_exchange']
    items=[];flats=[]
    for p in ps:
        state=ss[p['state_uid']];phy=state['physical'];env=RingExchangeEnv(cfg)
        env.reset(np.asarray(phy['positions']),velocities=np.asarray(phy['velocities']),goals=np.asarray(phy['goals']))
        obs=local_observation(env.positions,env.velocities,env.goals,cfg)
        token=int(hashlib.sha256(p['state_uid'].encode()).hexdigest()[:8],16)
        key=jax.random.fold_in(jax.random.PRNGKey(2026100106),token)
        act=np.asarray(sample_bounded_actions(policy,obs[None],key)[0],float)
        flow=local_actions_to_world(act,env.positions);flow*=np.minimum(1.,cfg.max_speed/np.maximum(np.linalg.norm(flow,axis=-1,keepdims=True),1e-30))
        flat=np.r_[obs.ravel(),flow.ravel()];flats.append(flat)
        native={'scenario':'ring_exchange','structured_state':{**phy,'timestep':0},'conditioning':{'flat':flat.tolist()},'environment_descriptor':{'obstacle_center':[0.,0.]}}
        items.append(rep.entities(rep.parse(native)))
    return ps,ss,rep.batch(items),np.asarray(flats,np.float32)

def references(flats,ps):
    """Target-supervised comparisons, never used for source model selection."""
    source=ROOT/'diagnostics/orthoflow3_c1_generalization_v1';sys.path.insert(0,str(source))
    from train_interaction import InteractionCritic,SPLIT,parse2
    import pyarrow.parquet as pq
    summaries=[load(source/f'interaction_models/seed{s}/ring_full/summary.json') for s in (17,23,41)]
    selected=min(summaries,key=lambda r:(r['dev_nll'],r['seed']))
    rows=pq.read_table(ROOT/'datasets/orthoflow3_basin_dataset_v2_audited/states.parquet',filters=[('scenario','=','ring_exchange')]).to_pylist()
    trainids=set(SPLIT['state_uids']['ring_exchange']['train'])
    hh=np.array([parse2(r['conditioning'])['flat'] for r in rows if r['state_uid'] in trainids],np.float32)
    hm,hs=hh.mean(0),hh.std(0);hs[hs<1e-6]=1
    assert not trainids & {p['state_uid'] for p in ps}
    model=InteractionCritic();p=model.init(jax.random.PRNGKey(0),jnp.zeros((1,80)),jnp.zeros((1,3)),jnp.zeros((1,100)),jnp.zeros((1,3)))
    p=serialization.from_bytes(p,open(selected['checkpoint'],'rb').read())
    eta=np.array([r['etas'][1:] for r in ps],np.float32).reshape(-1,3)
    h=np.repeat((flats-hm)/hs,16,axis=0)
    score=expit(np.asarray(model.apply(p,jnp.asarray(h),jnp.asarray((eta-np.asarray(SPLIT['eta_coord_center'],np.float32))/np.asarray(SPLIT['eta_coord_radius'],np.float32)),method=model.ring))).reshape(-1,16)
    return score,{'checkpoint':selected['checkpoint'],'seed':selected['seed'],'training':'Ring-only corrected-safety B15 classifier; scores are not single-continuation Q probabilities','reference_only':True}

def predict():
    if (OUT/'target_predictions.json').exists():return load(OUT/'target_predictions.json')
    frozen=load(OUT/'models_frozen.json');base=load(OUT/'eta_baselines_frozen.json')
    for kind in ('shared','eta_only'):
        for r in frozen[kind]['runs']:assert sha(r['checkpoint'])==r['sha256']
    ps,ss,x,flats=physical_inputs();norm=load(OUT/'normalization.json')
    ee=np.array([r['etas'][1:] for r in ps],np.float32).reshape(-1,3)
    e=(ee-np.array(norm['eta_center'],np.float32))/np.array(norm['eta_radius'],np.float32)
    idx=np.repeat(np.arange(len(ps)),16);scores={}
    for kind in ('shared','eta_only'):
        model=model_for(kind);init=model.init(jax.random.PRNGKey(0),gather(x,[0]),jnp.zeros((1,3)))
        for r in frozen[kind]['runs']:
            p=serialization.from_bytes(init,open(r['checkpoint'],'rb').read())
            pred=jax.jit(lambda bx,be:jax.nn.sigmoid(model.apply(p,bx,be)))
            chunks=[np.asarray(pred(gather(x,idx[i:i+128]),jnp.asarray(e[i:i+128]))) for i in range(0,len(idx),128)]
            scores[f'{kind}_seed{r["seed"]}']=np.concatenate(chunks).reshape(-1,16)
        scores[kind]=scores[f'{kind}_seed{frozen[kind]["selected"]["seed"]}']
        scores[kind+'_ensemble']=np.mean([scores[f'{kind}_seed{s}'] for s in (17,23,41)],axis=0)
    kern=np.load(OUT/'eta_kernel.npz');linear=np.load(OUT/'eta_linear.npz')
    scores['eta_kernel']=np.clip(float(kern['prior'])+np.exp(-cdist(e,kern['z'],'sqeuclidean')/(2*float(kern['bandwidth'])**2))@kern['alpha'],0,1).reshape(-1,16)
    scores['global_linear']=expit(np.column_stack([e,np.ones(len(e))])@linear['beta']).reshape(-1,16)
    # All choices below precede target outcome access.
    eta_baseline=min([('eta_only',frozen['eta_only']['selected']['validation_nll']),('eta_kernel',base['kernel']['validation_nll'])],key=lambda v:v[1])[0]
    scores['ring_supervised_joint_reference']=np.array([r['scores'][1:] for r in ps])
    scores['ring_only_reference'],reference=references(flats,ps)
    trainrows,_,traine,_,_=data();support=np.asarray([traine[i] for i,r in enumerate(trainrows) if r['split']=='train'])
    nearest=cdist(e,support).min(1).reshape(-1,16)
    result={'frozen_at':datetime.now(timezone.utc).isoformat(),'target_labels_opened':False,
      'models_sha256':sha(OUT/'models_frozen.json'),'target_proposals_sha256':sha(CONF/'proposals.json'),
      'source_selected_eta_baseline':eta_baseline,'ring_only_reference':reference,'K':16,'mean_excluded':True,
      'rows':[{'state_uid':p['state_uid'],'controller_uid':p['controller_uid'],'eta':p['etas'][1:],
       'nearest_source_eta_distance':nearest[i].tolist(),'scores':{k:v[i].astype(float).tolist() for k,v in scores.items()}} for i,p in enumerate(ps)]}
    dump('target_predictions.json',result);np.savez_compressed(OUT/'target_entities.npz',**x)
    return result

def evaluate():
    pred=predict();prediction_sha=sha(OUT/'target_predictions.json')
    from shared_rollout_db.src.rollout_db import eta_identity
    from shared_rollout_db.src.planner import preflight
    requests=[]
    for p in pred['rows']:
        for e in p['eta']:requests.append({'state_uid':p['state_uid'],'eta_uid':eta_identity(e)[0],'controller_uid':p['controller_uid'],'seed_keys':[json.dumps({'future_index':i},separators=(',',':')) for i in range(16)]})
    dump('planned_rollouts.json',{'requests':requests,'execute_missing':False})
    # Cache audit only; this experiment never invokes a rollout executor.
    import subprocess
    subprocess.run([sys.executable,'-m','shared_rollout_db.plan','--manifest',str(OUT/'planned_rollouts.json'),'--output',str(OUT/'cache_preflight.json')],cwd=ROOT,check=True,capture_output=True)
    c=con();details=[];truth=[]
    for p in pred['rows']:
        qq=[];robust=[];lo=[];hi=[];num=[];coll=[]
        for e in p['eta']:
            rr=c.execute("SELECT seed_key,success,numerical_failure,collision,conflict_quarantined,compatibility_quality FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=?",(p['state_uid'],eta_identity(e)[0],p['controller_uid'])).fetchall()
            by={json.loads(r['seed_key']).get('future_index'):r for r in rr if r['compatibility_quality']=='EXACT_REUSE' and not r['conflict_quarantined']}
            good=[by[i] for i in range(16) if i in by and not by[i]['numerical_failure']]
            k=sum(r['success'] for r in good);u=16-len(good);low=k/16;high=(k+u)/16
            qq.append(low if not u else None);lo.append(low);hi.append(high);num.append(u);coll.append(sum(r['collision'] for r in good))
            robust.append(True if k>=15 else False if k+u<15 else None)
        truth.append({'state_uid':p['state_uid'],'q':qq,'lower':lo,'upper':hi,'robust':robust,'unknown_seeds':num,'collisions':coll})
        for name,s in p['scores'].items():
            j=int(np.argmax(s));details.append({'method':name,'state_uid':p['state_uid'],'selected_index':j+1,'p':s[j],'q':qq[j],'Q_lower':lo[j],'Q_upper':hi[j],
             'b15':robust[j],'oracle_b15':any(v is True for v in robust),'oracle_lower':max(lo),'oracle_upper':max(hi),
             'regret_lower':max(0,max(lo)-hi[j]),'regret_upper':max(hi)-lo[j],'numerical_seeds':num[j],'collision':coll[j],
             'severe_false_positive':s[j]>=.95 and hi[j]<=.5})
    c.close();write_csv('target_selection.csv',details);dump('target_truth.json',truth)
    summaries={}
    for name in pred['rows'][0]['scores']:
        rr=[r for r in details if r['method']==name];valid=[r for r in rr if r['q'] is not None]
        scores=np.array([r['scores'][name] for r in pred['rows']]).ravel();q=np.array([np.nan if v is None else v for t in truth for v in t['q']]);ok=np.isfinite(q)
        summaries[name]={'states':len(rr),'b15':sum(r['b15'] is True for r in rr),'unresolved':sum(r['b15'] is None for r in rr),'oracle_b15':sum(r['oracle_b15'] for r in rr),
            'mean_selected_Q_exact':float(np.mean([r['q'] for r in valid])) if valid else None,'selected_exact_states':len(valid),
            'mean_selected_Q_lower':float(np.mean([r['Q_lower'] for r in rr])),'mean_selected_Q_upper':float(np.mean([r['Q_upper'] for r in rr])),
            'mean_regret_lower':float(np.mean([r['regret_lower'] for r in rr])),'mean_regret_upper':float(np.mean([r['regret_upper'] for r in rr])),
            'severe_false_positive':sum(r['severe_false_positive'] for r in rr),'collisions':sum(r['collision'] for r in rr),'probability_metrics':metrics(scores[ok],q[ok])}
    paired={};lookup={(r['method'],r['state_uid']):r for r in details};rng=np.random.default_rng(20261002)
    for other in (pred['source_selected_eta_baseline'],'global_linear','ring_only_reference','ring_supervised_joint_reference'):
        a=[lookup[('shared',p['state_uid'])] for p in pred['rows']];b=[lookup[(other,p['state_uid'])] for p in pred['rows']]
        valid=[(x,y) for x,y in zip(a,b) if x['b15'] is not None and y['b15'] is not None]
        diff=np.array([int(x['b15'])-int(y['b15']) for x,y in valid]);res=int(sum(diff>0));br=int(sum(diff<0))
        boot=diff[rng.integers(0,len(diff),(10000,len(diff)))].mean(1)
        paired[other]={'resolved_pairs':len(diff),'excluded_numerical_pairs':len(a)-len(diff),'rescue':res,'break':br,'net':res-br,'rate_difference':float(diff.mean()),'paired_95CI':np.quantile(boot,[.025,.975]).tolist(),'exact_two_sided_p':float(binomtest(res,res+br,.5).pvalue) if res+br else 1.}
    # Exact repeated eta is required for a logically valid state reversal.
    shared_eta=defaultdict(list)
    for p,t in zip(pred['rows'],truth):
        for j,e in enumerate(p['eta']):shared_eta[tuple(float(v).hex() for v in e)].append((p['state_uid'],j,t['q'][j]))
    repeats={k:v for k,v in shared_eta.items() if len({z[0] for z in v})>=2}
    interactions={'rule':load(OUT/'protocol.json')['interaction_subset'],'unique_eta':len(shared_eta),'eta_repeated_across_states':len(repeats),'status':'DATA_COVERAGE_INSUFFICIENT' if not repeats else 'REQUIRES_EXACT_PAIR_ENUMERATION','ranking_reversal_claim':False}
    dump('interaction_subset.json',interactions)
    eta=summaries[pred['source_selected_eta_baseline']];main=summaries['shared'];n=main['states'];ci=paired[pred['source_selected_eta_baseline']]['paired_95CI']
    supported=ci[0]>0 and main['b15']/max(1,main['oracle_b15'])>=.9
    decision='CROSS_SCENE_STATE_ETA_GENERALIZATION_SUPPORTED' if supported else 'CROSS_SCENE_STATE_CONDITIONING_NOT_YET_SUPPORTED'
    if min(main['b15'],eta['b15'])/n>=.95:decision='TARGET_SCENE_UNDERDISCRIMINATIVE'
    result={'classification':decision,'models':summaries,'paired':paired,'source_selected_eta_baseline':pred['source_selected_eta_baseline'],'target_predictions_sha256':prediction_sha,'new_rollouts':0,'generator_gate_passed':supported,
      'strictness':'target-label-free model training/selection under historically target-informed inherited schema; not target-naive design or end-to-end zero-shot proposals',
      'target_generator':'frozen historical generator had Ring training; used solely as fixed candidate provider',
      'source_support_limitations':['Four full-count evidence has no clear-failure TRAIN pair','Four source snapshots are mid-trajectory; target is true-t0','source shapes are segments; target circular geometry has unseen curvature','source joint-generator-aligned proposals excluded for Ring-label ancestry'],
      'interaction':interactions,'no_post_test_model_changes':True}
    dump('final_decision.json',result);dump('working_state.json',{'stage':'critic_fold_complete','new_rollouts':0,'target_labels_opened':True,'source_models_frozen':True,'generator_gate_passed':supported})
    print(json.dumps({'classification':decision,'models':{k:{f:v[f] for f in ('b15','unresolved','mean_selected_Q_exact')} for k,v in summaries.items()},'paired':paired},indent=2))

if __name__=='__main__':evaluate()
