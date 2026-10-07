"""Freeze every source-selected score, then evaluate identical cached K16 pools."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import json,argparse,sys,subprocess
import numpy as np
import jax,jax.numpy as jnp
from flax import serialization
from scipy.stats import binomtest
from .data import ROOT,OUT,OLD,FIRST,FOLDS,VARIANTS,baseline,load,sha,dump,csvout,con
from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import model_for,gather
from shared_rollout_db.src.rollout_db import eta_identity
jax.config.update('jax_default_matmul_precision','highest')

def predict():
    if (OUT/'target_predictions.json').exists():return
    # Gate before any scoring: ALL three variants and all four folds are frozen.
    for fold in FOLDS:
        for variant in VARIANTS:assert (OUT/fold/variant/'models_frozen.json').exists()
    result={};model_hashes={}
    for fold in FOLDS:
        target=load(OLD/'targets'/fold/'manifest.json');x=dict(np.load(OLD/'targets'/fold/'entities.npz'));norm=load(OUT/fold/'normalization.json')
        eta=np.array([p['eta'] for p in target],np.float32).reshape(-1,3);e=(eta-np.array(norm['eta_center'],np.float32))/np.array(norm['eta_radius'],np.float32);ix=np.repeat(np.arange(len(target)),16);scores={}
        prior=load(baseline(fold)/'target_predictions.json');by={p['state_uid']:p for p in prior['rows']}
        for p in target:assert p['eta']==by[p['state_uid']]['eta']
        olddecision=load(baseline(fold)/'final_decision.json')
        for kind in ('shared','eta_only'):
            scores['old_'+kind]=np.array([by[p['state_uid']]['scores'][kind] for p in target])
        scores['old_source_selected_eta']=np.array([by[p['state_uid']]['scores'][olddecision['source_selected_eta_baseline']] for p in target])
        for variant in VARIANTS:
            fr=load(OUT/fold/variant/'models_frozen.json');model_hashes[fold+'/'+variant]=sha(OUT/fold/variant/'models_frozen.json')
            for kind in ('shared','eta_only'):
                m=model_for(kind);init=m.init(jax.random.PRNGKey(0),gather(x,[0]),jnp.zeros((1,3)))
                for run in fr[kind]['runs']:
                    assert run['target_labels_used'] is False and sha(run['checkpoint'])==run['sha256']
                    p=serialization.from_bytes(init,open(run['checkpoint'],'rb').read());fn=jax.jit(lambda bx,be:jax.nn.sigmoid(m.apply(p,bx,be)))
                    s=np.concatenate([np.asarray(fn(gather(x,ix[j:j+128]),jnp.asarray(e[j:j+128]))) for j in range(0,len(ix),128)]).reshape(-1,16)
                    scores[f'{variant}_{kind}_seed{run["seed"]}']=s
                scores[variant+'_'+kind]=scores[f'{variant}_{kind}_seed{fr[kind]["selected"]["seed"]}'];scores[variant+'_'+kind+'_ensemble']=np.mean([scores[f'{variant}_{kind}_seed{s}'] for s in (17,23,41)],axis=0)
        result[fold]={'state_uids':[p['state_uid'] for p in target],'scores':{k:v.tolist() for k,v in scores.items()},'normalization_sha256':sha(OUT/fold/'normalization.json'),'proposals_sha256':sha(OLD/'targets'/fold/'manifest.json')}
    dump('target_predictions.json',{'target_outcomes_opened_in_this_experiment':False,'model_freezes':model_hashes,'folds':result})
    dump('working_state.json',{'stage':'all_target_predictions_frozen','new_rollout':0})

def truth():
    c=con();out={};updates=[]
    for fold in FOLDS:
        manifest=load(baseline(fold)/'planned_rollouts.json');dump(f'{fold}/planned_rollouts.json',manifest)
        subprocess.run([sys.executable,'-m','shared_rollout_db.plan','--manifest',str(OUT/fold/'planned_rollouts.json'),'--output',str(OUT/fold/'cache_preflight.json')],cwd=ROOT,check=True,capture_output=True)
        target=load(OLD/'targets'/fold/'manifest.json');previous={r['state_uid']:r for r in load(baseline(fold)/'target_truth.json')};rr=[]
        for p in target:
            t={'state_uid':p['state_uid'],'lower':[],'upper':[],'robust':[],'q':[],'unknown':[],'collision':[]}
            for j,e in enumerate(p['eta']):
                found=c.execute("SELECT seed_key,success,numerical_failure,collision FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=? AND compatibility_quality='EXACT_REUSE' AND conflict_quarantined=0",(p['state_uid'],eta_identity(e)[0],p['controller_uid'])).fetchall();by={json.loads(r['seed_key']).get('future_index'):r for r in found};good=[by[i] for i in range(16) if i in by and not by[i]['numerical_failure']];s=sum(r['success'] for r in good);u=16-len(good);lo=s/16;hi=(s+u)/16;b=True if s>=15 else False if s+u<15 else None
                t['lower'].append(lo);t['upper'].append(hi);t['robust'].append(b);t['q'].append(lo if u==0 else None);t['unknown'].append(u);t['collision'].append(sum(r['collision'] for r in good))
                prior=previous[p['state_uid']]
                if prior['lower'][j]!=lo or prior['upper'][j]!=hi:updates.append({'fold':fold,'state_uid':p['state_uid'],'index':j,'old_lower':prior['lower'][j],'old_upper':prior['upper'][j],'new_lower':lo,'new_upper':hi})
            rr.append(t)
        out[fold]=rr
    c.close();dump('cached_truth.json',out);dump('outcome_version_audit.json',{'updates':updates,'old_results_not_overwritten':True,'new_rollouts':0});return out

def evaluate():
    predict();pred=load(OUT/'target_predictions.json');tt=truth();summaries=[];selected=[];paired=[];tails=[]
    for fold,p in pred['folds'].items():
        t=tt[fold];n=len(t);oracle=np.array([any(v is True for v in r['robust']) for r in t]);bymethod={}
        for method,values in p['scores'].items():
            z=np.array(values);pick=z.argmax(1);b=[r['robust'][j] for r,j in zip(t,pick)];lo=np.array([r['lower'][j] for r,j in zip(t,pick)]);hi=np.array([r['upper'][j] for r,j in zip(t,pick)]);pr=z[np.arange(n),pick]
            bymethod[method]=b
            r={'fold':fold,'method':method,'states':n,'B15':sum(v is True for v in b),'unresolved':sum(v is None for v in b),'oracle_B15':int(oracle.sum()),'gap_lower':int(oracle.sum())-sum(v is True or v is None for v in b),'gap_upper':int(oracle.sum())-sum(v is True for v in b),'selection_on_available_B15':sum(v is True and o for v,o in zip(b,oracle))/int(oracle.sum()),'selected_prediction_mean':float(pr.mean()),'selected_Q_lower':float(lo.mean()),'selected_Q_upper':float(hi.mean()),'overestimation_lower':float((pr-hi).mean()),'overestimation_upper':float((pr-lo).mean()),'regret_lower':float(np.mean([max(0,max(q['lower'])-h) for q,h in zip(t,hi)])),'regret_upper':float(np.mean([max(q['upper'])-l for q,l in zip(t,lo)])),'false_high_p90_Q_le_half':int(sum((pr>.9)&(hi<=.5))),'false_high_p95_Q_le_half':int(sum((pr>.95)&(hi<=.5))),'p95_selected_count':int(sum(pr>.95)),'p95_B15_precision_lower':sum(v is True and prob>.95 for v,prob in zip(b,pr))/max(1,sum(pr>.95)) if sum(pr>.95) else None,'p95_B15_precision_upper':sum((v is True or v is None) and prob>.95 for v,prob in zip(b,pr))/max(1,sum(pr>.95)) if sum(pr>.95) else None,'collisions':sum(q['collision'][j] for q,j in zip(t,pick))}
            summaries.append(r)
            for i,j in enumerate(pick):selected.append({'fold':fold,'method':method,'state_uid':t[i]['state_uid'],'index':int(j),'p':pr[i],'B15':b[i],'Q_lower':lo[i],'Q_upper':hi[i]})
            # Upper tail of ALL proposals separately, no max-selection conflation.
            for a,bb in ((0,.5),(.5,.7),(.7,.8),(.8,.9),(.9,.95),(.95,1.000001)):
                mask=(z>=a)&(z<bb);count=int(mask.sum());ql=np.array([q['lower'] for q in t]);qh=np.array([q['upper'] for q in t]);yes=np.array([[v is True for v in q['robust']] for q in t]);unknown=np.array([[v is None for v in q['robust']] for q in t])
                tails.append({'fold':fold,'method':method,'p_min':a,'p_max':min(bb,1.),'count':count,'predicted_mean':float(z[mask].mean()) if count else None,'Q_lower':float(ql[mask].mean()) if count else None,'Q_upper':float(qh[mask].mean()) if count else None,'B15_precision_lower':float(yes[mask].mean()) if count else None,'B15_precision_upper':float((yes|unknown)[mask].mean()) if count else None})
        for variant in VARIANTS:
            main=variant+'_shared'
            for control in (variant+'_eta_only','old_shared','old_source_selected_eta','full_continuation_shared'):
                if main==control:continue
                a=bymethod[main];b=bymethod[control];d=np.array([int(x)-int(y) for x,y in zip(a,b) if x is not None and y is not None]);res=int(sum(d>0));br=int(sum(d<0));rng=np.random.default_rng(2026100301);boot=d[rng.integers(0,len(d),(10000,len(d)))].mean(1) if len(d) else np.array([np.nan])
                paired.append({'fold':fold,'main':main,'control':control,'resolved_states':len(d),'rescue':res,'break':br,'net_resolved':res-br,'net_lower_all_states':sum(int(x is True)-int(y is True or y is None) for x,y in zip(a,b)),'net_upper_all_states':sum(int(x is True or x is None)-int(y is True) for x,y in zip(a,b)),'rate_diff_CI_low':float(np.quantile(boot,.025)),'rate_diff_CI_high':float(np.quantile(boot,.975)),'exact_p':float(binomtest(res,res+br,.5).pvalue) if res+br else 1.})
    csvout('loso_results.csv',summaries);csvout('selected_proposals.csv',selected);csvout('paired_comparisons.csv',paired);csvout('upper_tail_calibration.csv',tails)
    dump('working_state.json',{'stage':'frozen_evaluation_complete','new_rollouts':0})
    show=[r for r in summaries if r['method'] in ['old_shared','full_continuation_shared','partial_count_shared','partial_count_eta_only','negative_binary_shared','negative_binary_eta_only']]
    print(json.dumps(show,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['predict','evaluate']);a=ap.parse_args();globals()[a.action]()
