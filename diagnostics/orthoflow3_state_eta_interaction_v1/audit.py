"""Frozen critic functional audit. No generator or production-model updates."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('OMP_NUM_THREADS','2')
import json, hashlib, argparse, itertools
from pathlib import Path
from collections import defaultdict
import numpy as np
import pyarrow.parquet as pq
import jax,jax.numpy as jnp
from flax import serialization
from scipy.special import expit
from scipy.stats import spearmanr
from diagnostics.orthoflow3_loso_root_cause_v1 import eval as ev
from diagnostics.orthoflow3_cross_scene_zero_shot_v1 import train as tr
from diagnostics.orthoflow3_unified_representation_v1 import representation as rep
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1];PRE=ev.OUT
SCENES={'ring':'ring_exchange','four':'four_way_intersection','db':'double_bottleneck','toy':'toy_giveway'}
load=ev.load
def dump(n,x):
    p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def corr(a,b):
    a=np.asarray(a).ravel();b=np.asarray(b).ravel();ok=np.isfinite(a)&np.isfinite(b);a=a[ok];b=b[ok]
    return float(spearmanr(a,b).statistic) if len(a)>2 and np.std(a)>1e-12 and np.std(b)>1e-12 else None
def folders(fold):return {'scene_specific':PRE/('single_'+fold),'joint_supervised':PRE/'joint_enriched',
                          'joint_generic':PRE/'joint','LOSO':ev.FIRST if fold=='ring' else PRE/fold}

class Frozen:
    def __init__(self,folder,kind='shared'):
        self.folder=folder;self.kind=kind;self.norm=load(folder/'normalization.json')
        frozen=load(folder/'models_frozen.json')[kind]['selected'];self.checkpoint=frozen
        assert ev.sha(frozen['checkpoint'])==frozen['sha256']
        sample=dict(np.load(folder/'source_entities.npz'));m=tr.model_for(kind)
        init=m.init(jax.random.PRNGKey(0),tr.gather(sample,[0]),jnp.zeros((1,3)))
        self.params=serialization.from_bytes(init,Path(frozen['checkpoint']).read_bytes());p=self.params['params']
        if kind=='shared':self.enc=jax.jit(lambda x:rep.Encoder().apply({'params':p['physical_encoder']},x))
        def head(h,e):
            def layer(x,name):return x@p[name]['kernel']+p[name]['bias']
            z=jax.nn.silu(layer(e,'eta_encoder'))
            if kind=='shared':z=jnp.concatenate((h,z),axis=-1)
            z=jax.nn.silu(layer(z,'shared1'));z=jax.nn.silu(layer(z,'shared2'))
            return layer(z,'out')[...,0]
        self.head=jax.jit(head)
        self.center=np.asarray(self.norm['eta_center'],np.float32);self.radius=np.asarray(self.norm['eta_radius'],np.float32)
        self.sample=sample
        # Independent full executable-model equivalence, not just parameter assumptions.
        x=tr.gather(sample,[0]);e=jnp.array([[.1,-.2,.3]])
        h=self.encode(x);np.testing.assert_allclose(self.head(h,e),m.apply(self.params,x,e),atol=2e-6,rtol=2e-6)
    def encode(self,x):
        if self.kind!='shared':return np.zeros((len(x['globals']),0),np.float32)
        return np.concatenate([np.asarray(self.enc(tr.gather(x,np.arange(i,min(i+128,len(x['globals'])))))) for i in range(0,len(x['globals']),128)])
    def score(self,h,etas,logits=False):
        e=np.asarray(etas,np.float32);shape=e.shape[:-1];e=(e.reshape(-1,3)-self.center)/self.radius
        hh=np.repeat(h,shape[-1],axis=0) if len(shape)==2 else h
        z=np.concatenate([np.asarray(self.head(jnp.asarray(hh[i:i+256]),jnp.asarray(e[i:i+256]))) for i in range(0,len(e),256)]).reshape(shape)
        return z if logits else expit(z)

def derange(n,seed):
    rng=np.random.default_rng(seed);p=rng.permutation(n)
    while np.any(p==np.arange(n)):p=rng.permutation(n)
    return p
def select_metrics(score,truth,reference):
    pick=score.argmax(1);ref=reference.argmax(1);lo=np.array([x['lower'] for x in truth]);hi=np.array([x['upper'] for x in truth]);r=[x['robust'][j] for x,j in zip(truth,pick)];ix=np.arange(len(pick))
    return {'states':len(pick),'b15':sum(v is True for v in r),'unresolved':sum(v is None for v in r),
        'selected_Q_lower':float(lo[ix,pick].mean()),'selected_Q_upper':float(hi[ix,pick].mean()),
        'oracle_B15_gap':sum(any(v is True for v in x['robust']) for x in truth)-sum(v is True for v in r),
        'Q_regret_lower':float(np.maximum(0,lo.max(1)-hi[ix,pick]).mean()),'Q_regret_upper':float((hi.max(1)-lo[ix,pick]).mean()),
        'top1_agreement':float(np.mean(pick==ref)),'score_spearman':corr(score,reference),
        'score_MAE_change':float(np.mean(abs(score-reference))),
        'severe_false_positive':int(np.sum((score[ix,pick]>=.95)&(hi[ix,pick]<=.5)))}

# Groups overlap only where a physical quantity has two roles; no ranking implied.
GROUPS={
 'agent_relative':{'agents':[1,2],'pairs':list(range(9))},
 'obstacle_geometry':{'obstacles':list(range(13))},
 'Flow_reference':{'agents':[3,4,5,6],'globals':[5]},
 'safety_quantities':{'agents':[7,8],'pairs':[7,8],'obstacles':[2],'globals':[6,7]},
 'goal_relative':{'agents':[0,9],'pairs':[5,6]},
 'remaining_time':{'globals':[0,1,13]},
 'policy_frame':{'agents':[10,11,12,13,14]},
 'entity_masks_counts':{'agent_mask':None,'obstacle_mask':None}}
def change_group(x,donor,group):
    out={k:v.copy() for k,v in x.items()}
    for key,channels in GROUPS[group].items():
        if channels is None:out[key]=donor[key].copy()
        else:out[key][...,channels]=donor[key][...,channels]
    return out
def reference_entities(model,fold,x):
    ss=load(model.folder/'source_states.json');ids=[i for i,s in enumerate(ss) if s['split']=='train' and s['scenario']==SCENES[fold]]
    if not ids:return None
    # Conditional mean uses only TRAIN, same scene and counts; never zero-filled OOD.
    trainx=model.sample;ref={}
    for k,v in x.items():
        val=trainx[k][ids].mean(0)
        # Crop padded entity axes to target shapes; no actual entity discarded.
        if k=='agents':val=val[:v.shape[1]]
        elif k=='pairs':val=val[:v.shape[1],:v.shape[2]]
        elif k=='obstacles':val=val[:v.shape[1],:v.shape[2]]
        elif k in ('agent_mask','obstacle_mask'):val=val[:v.shape[1]]
        assert val.shape==v.shape[1:],(k,val.shape,v.shape)
        ref[k]=np.broadcast_to(val,v.shape).copy()
    return ref

def preregister():
    if (OUT/'protocol.json').exists():return
    dump('protocol.json',{'phase1':'all frozen models, no production retraining or generator change',
      'models':'scene_specific, joint_enriched, joint_generic, LOSO source-VAL-selected; all four frozen target cohorts',
      'K':'exact archived 16 stochastic only, no mean added; do not regenerate proposals',
      'shuffle_seeds':list(range(7300,7320)),'shuffle':'independent within-scene state/eta-set derangements',
      'interaction':'(Qa_i-Qa_j)-(Qb_i-Qb_j); same EXACT eta IDs, independent states, full fixed-count evidence only',
      'strong_delta':.5,'reversal_margins':.25,'weak_delta_exclusion':.125,
      'group_tests':GROUPS,'replacement':'TRAIN same-scene conditional feature mean; no zero replacements; correlations may break',
      'severe_FP':'predicted>=.95 and Q upper<=.5',
      'baseline_phase2':'fixed ridge 0.01, random Fourier eta basis 128 seed8301 bandwidth1 normalized eta; additive includes train-standardized frozen h; TRAIN only, no production weights changed',
      'uncertainty':'quadruples share states/eta and are not independent trials; B15 is empirical; no threshold flip treated as proof',
      'new_rollouts_initial':0})
    (OUT/'ALREADY_TESTED.md').write_text('# Reused evidence\n\nRead orthoflow3_loso_root_cause_v1/final_report.txt and orthoflow3_cross_scene_zero_shot_v1/final_report.md. Existing source-only folds, scene-specific and enriched-joint checkpoints, K16 outcomes, input replay, ten prior state shuffles and absence of exact cross-scene validation reversals are reused. No retraining of these models, K sweep, safety audit, or generator sampling. This audit adds independent state/eta/joint controls, within-scene shared-eta contrasts, additive function controls and grouped input interventions.\n')

def controls():
    preregister();results=[];feature=[]
    for fold in SCENES:
        ps,x=ev.inputs(fold);eta=np.asarray([p['eta'] for p in ps]);truth=load((ev.FIRST if fold=='ring' else PRE/fold)/'target_truth.json')
        assert [p['state_uid'] for p in ps]==[t['state_uid'] for t in truth]
        for name,folder in folders(fold).items():
            m=Frozen(folder);h=m.encode(x);base=m.score(h,eta)
            scores={'original':base};results.append({'fold':fold,'model':name,'control':'original','seed':None,**select_metrics(base,truth,base)})
            for seed in range(7300,7320):
                p=derange(len(ps),seed);ep=derange(len(ps),seed+10000)
                for control,hh,ee in [('state_shuffle',h[p],eta),('eta_shuffle',h,eta[ep]),('joint_shuffle',h[p],eta[ep])]:
                    z=m.score(hh,ee);scores[f'{control}_{seed}']=z
                    metrics=select_metrics(z,truth,base) if control=='state_shuffle' else {
                        'top1_agreement':float(np.mean(z.argmax(1)==base.argmax(1))),
                        'score_spearman':corr(z,base),'score_MAE_change':float(np.mean(abs(z-base))),
                        'true_Q_status':'N/A: donor eta must be evaluated on recipient state, never borrow donor Q'}
                    results.append({'fold':fold,'model':name,'control':control,'seed':seed,**metrics})
            ref=reference_entities(m,fold,x)
            for group in GROUPS:
                for mode in ('permutation','train_reference'):
                    if mode=='train_reference' and ref is None:
                        feature.append({'fold':fold,'model':name,'group':group,'mode':mode,'status':'N/A: target-scene TRAIN reference prohibited in LOSO'});continue
                    for seed in (7300,7301,7302,7303,7304) if mode=='permutation' else (None,):
                        donor={k:v[derange(len(ps),seed)] for k,v in x.items()} if mode=='permutation' else ref
                        xx=change_group(x,donor,group);z=m.score(m.encode(xx),eta)
                        feature.append({'fold':fold,'model':name,'group':group,'mode':mode,'seed':seed,
                            'max_input_change':max(float(np.max(abs(xx[k]-x[k]))) for k in x),**select_metrics(z,truth,base)})
            np.savez_compressed(OUT/f'{fold}_{name}_scores.npz',**scores)
            dump('shuffle_metrics.json',results);dump('feature_metrics.json',feature)
            print(fold,name,'done',flush=True)

def contrasts(q,p,ids=None):
    # Dense common eta panel. Missing Q stays NaN; not inferred from neighbors.
    rows=[]
    for a,b in itertools.combinations(range(len(q)),2):
        for i,j in itertools.combinations(range(q.shape[1]),2):
            if not np.isfinite(q[[a,a,b,b],[i,j,i,j]]).all():continue
            d1=q[a,i]-q[a,j];d2=q[b,i]-q[b,j];pd1=p[a,i]-p[a,j];pd2=p[b,i]-p[b,j]
            rows.append([a,b,i,j,d1-d2,pd1-pd2,d1,d2,pd1,pd2])
    if not rows:return {'quadruples':0},[]
    z=np.asarray(rows);z[:,5][abs(z[:,5])<1e-6]=0;z[:,8:10][abs(z[:,8:10])<1e-6]=0
    true=z[:,4];pred=z[:,5];strong=abs(true)>=.5;clear=abs(true)>=.125
    reversal=(z[:,6]*z[:,7]<0)&(np.minimum(abs(z[:,6]),abs(z[:,7]))>=.25)
    predicted=(z[:,8]*z[:,9]<0)
    resolved=np.minimum(abs(z[:,6]),abs(z[:,7]))>=.25
    met={'quadruples':len(z),'delta_true_quantiles':np.quantile(true,[0,.1,.5,.9,1]).tolist(),
         'delta_pred_quantiles':np.quantile(pred,[0,.1,.5,.9,1]).tolist(),
         'delta_spearman':corr(true,pred),'delta_MAE':float(np.mean(abs(true-pred))),
         'sign_accuracy_clear':float(np.mean(np.sign(true[clear])==np.sign(pred[clear]))) if clear.any() else None,
         'strong_count':int(strong.sum()),'strong_sign_accuracy':float(np.mean(np.sign(true[strong])==np.sign(pred[strong]))) if strong.any() else None,
         'strong_delta_spearman':corr(true[strong],pred[strong]),'reversal_count':int(reversal.sum()),
         'reversal_recall':float(predicted[reversal].mean()) if reversal.any() else None,
         'reversal_detection_accuracy_clear':float(np.mean(predicted[resolved]==reversal[resolved])) if resolved.any() else None,
         'reversal_precision_clear':float(np.mean(reversal[predicted&resolved])) if (predicted&resolved).any() else None,
         'reversal_direction_accuracy':float(np.mean((np.sign(z[reversal,6])==np.sign(z[reversal,8]))&(np.sign(z[reversal,7])==np.sign(z[reversal,9])))) if reversal.any() else None,
         'predicted_reversal_count':int(predicted.sum()),'descriptive_correlated_quadruples':True}
    return met,rows

def existing_interactions():
    output=[];coverage=[]
    for fold in SCENES:
        for name,folder in folders(fold).items():
            # Evaluation uses the same frozen enriched-joint VAL physical states
            # for all models; train membership is recorded, not hidden.
            source=PRE/'joint_enriched';ss=load(source/'source_states.json');rr=pq.read_table(source/'source_pairs.parquet').to_pylist()
            selected=[s for s in ss if s['scenario']==SCENES[fold] and s['split']=='validation']
            sid=[s['state_uid'] for s in selected];by=defaultdict(dict);etas={}
            for r in rr:
                if r['state_uid'] not in sid:continue
                etas[r['eta_uid']]=r['eta'];by[r['state_uid']][r['eta_uid']]=r['q']
            keys=sorted(k for k in etas if sum(k in by[s] for s in sid)>=2)
            q=np.array([[by[s].get(k,np.nan) for k in keys] for s in sid])
            if not keys:continue
            m=Frozen(folder);x=rep.batch([rep.entities(s['physical']) for s in selected]);eta=np.tile([etas[k] for k in keys],(len(sid),1,1));p=m.score(m.encode(x),eta)
            met,rows=contrasts(q,p);output.append({'fold':fold,'model':name,**met})
            np.savez_compressed(OUT/f'existing_{fold}_{name}_interactions.npz',q=q,p=p,contrasts=np.asarray(rows))
            if name=='joint_supervised':coverage.append({'fold':fold,'states':sid,'eta_uids':keys,'shared_etas':len(keys),'quadruples':met['quadruples'],
                'source':'joint_enriched VAL only; exact eta_uid matching','train_state_overlap':len(set(sid)&{s['state_uid'] for s in load(folder/'source_states.json') if s['split']=='train'})})
    dump('existing_interaction_metrics.json',output);dump('shared_eta_coverage.json',coverage)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['protocol','controls','interactions']);arg=ap.parse_args()
    {'protocol':preregister,'controls':controls,'interactions':existing_interactions}[arg.stage]()
