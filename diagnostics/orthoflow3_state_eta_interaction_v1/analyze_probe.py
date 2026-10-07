"""Shared-eta counterfactual analysis. Does not tune any model."""
import numpy as np
from scipy.stats import beta
from .audit import *
from .baselines import features

def run():
    records=load(OUT/'probe/evidence.json');pp=load(OUT/'probe/predictions_frozen.json')
    etas=np.asarray(load(OUT/'probe/protocol.json')['shared_eta']);metrics=[];groups=[];witness=[];anova=[];support=[]
    for block in pp:
        fold=block['fold'];sids=block['state_uids'];by={(r['state_uid'],r['eta_index']):r for r in records}
        q=np.array([[np.nan if by[s,i]['q'] is None else by[s,i]['q'] for i in range(len(etas))] for s in sids])
        ps,allx=ev.inputs(fold);ix=[next(i for i,p in enumerate(ps) if p['state_uid']==s) for s in sids];x={k:v[ix] for k,v in allx.items()}
        for name,folder in folders(fold).items():
            m=Frozen(folder);h=m.encode(x);ee=np.tile(etas,(len(sids),1,1));score=np.asarray(block['scores'][name]);np.testing.assert_allclose(m.score(h,ee),score,atol=2e-6)
            for which,z in [('true_Q',q),('predicted_Q',score),('predicted_logit',m.score(h,ee,logits=True))]:
                valid=np.isfinite(z);aa,bb=np.where(valid);design=np.column_stack((np.ones(len(aa)),np.eye(len(z))[aa,1:],np.eye(z.shape[1])[bb,1:]))
                y=z[valid];fit=design@np.linalg.lstsq(design,y,rcond=None)[0];var=float(np.mean((y-y.mean())**2))
                anova.append({'fold':fold,'model':name,'field':which,'evaluated_cells':len(y),
                    'residual_RMSE_after_best_additive_table':float(np.sqrt(np.mean((y-fit)**2))),
                    'nonadditive_variance_fraction':float(np.mean((y-fit)**2)/var) if var>1e-12 else None,
                    'interpretation':'descriptive table decomposition, not a trained baseline or proof of population interaction'})
            rr=pq.read_table(folder/'source_pairs.parquet').to_pylist();te=np.asarray([r['eta'] for r in rr if r['split']=='train'])
            dist=np.linalg.norm((etas[:,None]-te[None])/m.radius,axis=-1).min(1)
            support.append({'fold':fold,'model':name,'probe_eta':etas.tolist(),'normalized_nearest_train_eta_distance':dist.tolist(),
                'unseen_eta_indices':[i for i,d in enumerate(dist) if d>1e-10],
                'state_train_overlap':len(set(sids)&{r['state_uid'] for r in rr if r['split']=='train'})})
            met,rows=contrasts(q,score);metrics.append({'fold':fold,'model':name,'variant':'full',**met})
            mt,_=contrasts(q[:,:8],score[:,:8]);metrics.append({'fold':fold,'model':name,'variant':'full_domain_only',**mt})
            # A direct eta-only function ablation: marginalize predictions over
            # state; this is diagnostic, not ground-truth training or a new model.
            pooled=np.tile(score.mean(0),(len(sids),1));mt,_=contrasts(q,pooled)
            metrics.append({'fold':fold,'model':name,'variant':'state_marginalized_frozen_function',**mt})
            b=np.load(OUT/f'baseline_{fold}_{name}.npz');ef=features((etas-m.center)/m.radius,b['omega'],b['phase']);ef=np.tile(ef,(len(sids),1));hh=np.repeat((h-b['hmean'])/b['hstd'],len(etas),axis=0)
            for kind in ('eta_only_RFF','additive_RFF'):
                z=((ef if kind=='eta_only_RFF' else np.column_stack((ef,hh)))@b['beta_'+kind]).reshape(q.shape)
                mt,_=contrasts(q,z);metrics.append({'fold':fold,'model':name,'variant':kind,**mt})
            ref=reference_entities(m,fold,x)
            for group in GROUPS:
                for mode in ('permutation','train_reference'):
                    if mode=='train_reference' and ref is None:continue
                    for seed in (7300,7301,7302,7303,7304) if mode=='permutation' else (None,):
                        donor={k:v[derange(len(sids),seed)] for k,v in x.items()} if mode=='permutation' else ref
                        xx=change_group(x,donor,group);z=m.score(m.encode(xx),ee);mt,_=contrasts(q,z)
                        dm,_=contrasts(q[:,:8],z[:,:8])
                        groups.append({'fold':fold,'model':name,'group':group,'mode':mode,'seed':seed,
                            'score_change':float(np.mean(abs(z-score))),**mt,
                            'domain_only_strong_sign_accuracy':dm['strong_sign_accuracy'],
                            'domain_only_delta_spearman':dm['delta_spearman']})
            # Probability nonadditivity versus logit-only nonadditivity matters:
            # sigmoid(A(h)+B(eta)) can yield delta_Q !=0 but cannot reverse ranks.
            logit=m.score(h,ee,logits=True);mt,_=contrasts(logit,logit)
            metrics.append({'fold':fold,'model':name,'variant':'functional_logit_nonadditivity',
                'delta_absolute_mean':float(np.mean([abs(r[4]) for r in contrasts(logit,logit)[1]]))})
            for row in rows:
                a,b0,i,j,dt,dp,da,db,dpa,dpb=row
                if da*db>=0 or min(abs(da),abs(db))<.25:continue
                ai,bi,ii,ji=map(int,(a,b0,i,j));qs=np.array([q[ai,ii],q[ai,ji],q[bi,ii],q[bi,ji]])
                # Conservative simultaneous finite-seed binomial intervals. Not
                # a guarantee of population success; no correction across all
                # searched quadruples, so exploratory certificates only.
                k=np.rint(qs*16).astype(int);alpha=.05/4
                low=np.array([0. if z==0 else beta.ppf(alpha/2,z,17-z) for z in k]);high=np.array([1. if z==16 else beta.ppf(1-alpha/2,z+1,16-z) for z in k])
                positive=(low[0]>high[1] and high[2]<low[3]);negative=(high[0]<low[1] and low[2]>high[3])
                witness.append({'fold':fold,'model':name,'states':[sids[ai],sids[bi]],'eta_indices':[ii,ji],
                    'eta':[etas[ii].tolist(),etas[ji].tolist()],'Q':qs.tolist(),'delta_true':float(dt),'delta_pred':float(dp),
                    'predicted_reversal_direction_correct':bool(np.sign(da)==np.sign(dpa) and np.sign(db)==np.sign(dpb)),
                    'four_cell_95_intervals_support_reversal':bool(positive or negative),
                    'intervals':list(zip(low.tolist(),high.tolist()))})
            np.savez_compressed(OUT/f'probe_{fold}_{name}_contrasts.npz',q=q,p=score,contrasts=np.asarray(rows))
    dump('probe_interaction_metrics.json',metrics);dump('feature_interaction_metrics.json',groups);dump('ranking_reversal_witnesses.json',witness)
    dump('functional_additivity_decomposition.json',anova);dump('probe_training_support.json',support)
    return {'witnesses_including_models':len(witness),'metrics':len(metrics)}

if __name__=='__main__':print(run())
