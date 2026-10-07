"""Post-hoc capacity/support adjudication, isolated from strict LOSO runs."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import argparse,json,copy
from collections import Counter,defaultdict
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from .run import ROOT,OUT,FIRST,SCENES,FOLDS,rep,old,load,sha,dump,dump_at
from .eval import inputs,score_model,write_csv

def save_dataset(name,states,rows,normalization=None):
    d=OUT/name;d.mkdir(exist_ok=True);used={r['state_uid'] for r in rows};states=[s for s in states if s['state_uid'] in used];ix={s['state_uid']:i for i,s in enumerate(states)}
    rows=[dict(r,state_index=ix[r['state_uid']]) for r in rows]
    scenarios=sorted({r['scenario'] for r in rows})
    counts={}
    for sc in scenarios:
        tr={s['family'] for s in states if s['scenario']==sc and s['split']=='train'};va={s['family'] for s in states if s['scenario']==sc and s['split']=='validation'}
        assert tr and va and not tr&va,(sc,tr&va)
        counts[sc]={sp:{'states':sum(s['scenario']==sc and s['split']==sp for s in states),'pairs':sum(r['scenario']==sc and r['split']==sp for r in rows),'failure':sum(r['scenario']==sc and r['split']==sp and r['q']<=.5 for r in rows)} for sp in ('train','validation')}
    targets=[]
    for f in ('toy','db','four','ring'):
        pp,_=inputs(f);targets+=pp
    assert not {p['state_uid'] for p in targets}&used
    targetfamilies={str(p.get('family')) for p in targets if p.get('family') is not None}
    assert not targetfamilies&{str(s['family']) for s in states}
    np.savez_compressed(d/'source_entities.npz',**rep.batch([rep.entities(s['physical']) for s in states]));pq.write_table(pa.Table.from_pylist(rows),d/'source_pairs.parquet')
    dump_at(d,'source_states.json',states)
    if normalization is None:
        e=np.array([r['eta'] for r in rows if r['split']=='train']);lo=e.min(0);hi=e.max(0);normalization={'eta_center':((lo+hi)/2).tolist(),'eta_radius':np.maximum((hi-lo)/2,1e-6).tolist(),'fit_split':'train','fit_scenes':scenarios,'physical':'fixed units'}
    dump_at(d,'normalization.json',normalization)
    dump_at(d,'source_data_manifest.json',{'counts':counts,'new_rollouts':0,'target_confirmation_state_overlap':0,'post_hoc_supervised_not_zero_shot':True,'pairs_sha256':sha(d/'source_pairs.parquet')})
    dump_at(d,'protocol.json',{**load(OUT/'protocol.json'),'sources':scenarios,'post_hoc_supervised_not_zero_shot':True})
    return counts

def prepare():
    assert not (OUT/'diagnostic_protocol.json').exists()
    dump('diagnostic_protocol.json',{'comparison':'joint generic full-count versus joint enriched full-Q16; identical architecture, W1, seeds, budgets and normalization',
      'claim':'capacity/data-support diagnostic only, not a zero-shot repair',
      'references':'matched same-input scene-only generic controls plus historical supervised references',
      'model_selection':'TRAIN/VAL only; no independent confirmation labels',
      'new_rollouts':0,'generator_modified':False})
    states=load(OUT/'all_source_states.json');rows=pq.read_table(OUT/'all_source_pairs.parquet').to_pylist()
    # Matched scene-only references: prevent architecture and evidence differences
    # from masquerading as a joint-capacity limitation.
    for short,sc in {**FOLDS,'ring':'ring_exchange'}.items():
        save_dataset('single_'+short,[s for s in states if s['scenario']==sc],[r for r in rows if r['scenario']==sc],load(OUT/'joint/normalization.json'))
    base=ROOT/'diagnostics/orthoflow3_k16_critic_canonicalization_v1'
    native=pq.read_table(ROOT/'datasets/orthoflow3_basin_dataset_v2_audited/states.parquet').to_pylist()+load(base/'initial_expansion/states.json')
    by={r['state_uid']:r for r in native};existing={s['state_uid']:s for s in states};keyed={(r['state_uid'],r['eta_uid'],r['controller_uid']):r for r in rows}
    from shared_rollout_db.src.rollout_db import eta_identity
    c=old.con();audit=Counter()
    for path in (base/'phase_a/proposal_Q_evidence.json',base/'initial_expansion/phase_a/proposal_Q_evidence.json'):
        bundle=load(path);assert bundle['complete']
        for r in bundle['rows']:
            if r['proposal_index'] is None or r['Q16'] is None:continue
            s=by[r['state_uid']]
            if s['split'] not in ('train','validation'):continue
            key=(r['state_uid'],eta_identity(r['eta'])[0],r['controller_uid'])
            if key in keyed:audit['duplicate']+=1;continue
            rr=c.execute("SELECT rollout_uid,seed_key,success FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=? AND compatibility_quality='EXACT_REUSE' AND conflict_quarantined=0 AND numerical_failure=0",key).fetchall()
            seedmap={json.loads(x['seed_key']).get('future_index'):x for x in rr}
            valid=[seedmap[i] for i in range(16) if i in seedmap]
            if len(valid)!=16:audit['missing_exact_16']+=1;continue
            q=sum(x['success'] for x in valid)/16
            assert abs(q-r['Q16'])<1e-12
            if key[0] not in existing:existing[key[0]]={'state_uid':key[0],'scenario':s['scenario'],'split':s['split'],'family':s['parent_episode_id'],'physical':rep.parse(s)}
            keyed[key]={'state_uid':key[0],'eta_uid':key[1],'controller_uid':key[2],'eta':r['eta'],'scenario':s['scenario'],'split':s['split'],'origin':'joint_generator_aligned_cached_Q16','requested_n':16,'n':16,'k':int(q*16),'q':q,'rollout_uids':[x['rollout_uid'] for x in valid],'seed_keys':[x['seed_key'] for x in valid]};audit['added_'+s['scenario']]+=1
    c.close()
    counts=save_dataset('joint_enriched',list(existing.values()),list(keyed.values()),load(OUT/'joint/normalization.json'))
    dump('enrichment_audit.json',{'counts':counts,'audit':dict(audit),'source_proposals_may_have_all_scene_generator_ancestry':True,'eligible_for_strict_LOSO':False,'frozen_confirmation_used':False})
    print(json.dumps({'counts':counts,'audit':dict(audit)},indent=2))

def evaluate():
    rows=[];allpred={}
    for fold in ('toy','db','four','ring'):
        pp,x=inputs(fold);eta=np.array([p['eta'] for p in pp],np.float32);scores={}
        for name in ('joint','joint_enriched','single_'+fold):
            scores.update({name+'_'+k:v for k,v in score_model(OUT/name,x,eta).items()})
        allpred[fold]={k:v.tolist() for k,v in scores.items()}
    dump('diagnostic_predictions_frozen.json',allpred)
    # Labels are opened only after all diagnostics have fixed their scores.
    for fold in allpred:
        folder=FIRST if fold=='ring' else OUT/fold;truth=load(folder/'target_truth.json')
        for name,values in allpred[fold].items():
            scores=np.array(values);pick=scores.argmax(1);b=[t['robust'][j] for t,j in zip(truth,pick)]
            rows.append({'fold':fold,'method':name,'states':len(b),'b15':sum(v is True for v in b),'unresolved':sum(v is None for v in b),'oracle':sum(any(v is True for v in t['robust']) for t in truth),'Q_lower':float(np.mean([t['lower'][j] for t,j in zip(truth,pick)])),'Q_upper':float(np.mean([t['upper'][j] for t,j in zip(truth,pick)])),'severe_false_positive':sum(scores[i,j]>=.95 and t['upper'][j]<=.5 for i,(t,j) in enumerate(zip(truth,pick)))})
    write_csv(OUT/'joint_capacity_diagnostic.csv',rows)
    print(json.dumps([r for r in rows if r['method'] in ('joint_shared','joint_enriched_shared') or r['method']=='single_'+r['fold']+'_shared'],indent=2,default=lambda x:x.item()))

def state_dependence():
    import jax,jax.numpy as jnp
    from flax import serialization
    from diagnostics.orthoflow3_cross_scene_zero_shot_v1 import train as tr
    decisions={};records=[]
    for fold in ('toy','db','four','ring'):
        ps,x=inputs(fold);eta=np.array([p['eta'] for p in ps],np.float32);scores=score_model(OUT/('single_'+fold),x,eta,'eta_only');cases={'target_supervised_eta_only':scores['eta_only']}
        # A diagnostic of reliance on within-scene h, not a new trained model.
        # Ten fixed derangements; no outcome-based choice among permutations.
        folder=OUT/'joint_enriched';norm=load(folder/'normalization.json');e=(eta.reshape(-1,3)-np.array(norm['eta_center'],np.float32))/np.array(norm['eta_radius'],np.float32)
        ck=load(folder/'models_frozen.json')['shared']['selected'];m=tr.model_for('shared');p=m.init(jax.random.PRNGKey(0),tr.gather(x,[0]),jnp.zeros((1,3)));p=serialization.from_bytes(p,open(ck['checkpoint'],'rb').read());fn=jax.jit(lambda bx,be:jax.nn.sigmoid(m.apply(p,bx,be)))
        rng=np.random.default_rng(2026100209)
        for repeat in range(10):
            ix=rng.permutation(len(ps))
            while np.any(ix==np.arange(len(ps))):ix=rng.permutation(len(ps))
            ii=np.repeat(ix,16);values=np.concatenate([np.asarray(fn(tr.gather(x,ii[j:j+128]),jnp.asarray(e[j:j+128]))) for j in range(0,len(ii),128)]).reshape(-1,16)
            cases['shuffled_h_'+str(repeat)]=values
        decisions[fold]={k:v.tolist() for k,v in cases.items()}
    dump('state_dependence_predictions.json',{'posthoc_diagnostic':True,'not_new_zero_shot_model':True,'scores':decisions})
    for fold,cases in decisions.items():
        truth=load((FIRST if fold=='ring' else OUT/fold)/'target_truth.json')
        for name,values in cases.items():
            pick=np.argmax(values,axis=1);b=[t['robust'][j] for t,j in zip(truth,pick)]
            records.append({'fold':fold,'method':name,'b15':sum(v is True for v in b),'unresolved':sum(v is None for v in b),'Q_lower':float(np.mean([t['lower'][j] for t,j in zip(truth,pick)])),'Q_upper':float(np.mean([t['upper'][j] for t,j in zip(truth,pick)]))})
    write_csv(OUT/'state_dependence_diagnostic.csv',records)
    print(json.dumps(records,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','evaluate','state_dependence']);a=p.parse_args();globals()[a.action]()
