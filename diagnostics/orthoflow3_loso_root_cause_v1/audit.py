"""Evidence audit: label ascertainment, physical support and source fit."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import argparse,json
from collections import defaultdict,Counter
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from .run import ROOT,OUT,FIRST,SCENES,FOLDS,load,sha,dump,old,rep,configure
from .eval import inputs,write_csv

def summary_vector(item):
    x=rep.entities(item);v=[]
    for key in ('agents','pairs','obstacles'):
        a=x[key]
        if key=='pairs':a=a[~np.eye(len(a),dtype=bool)]
        else:a=a.reshape(-1,a.shape[-1])
        v.extend(np.r_[a.mean(0),a.std(0),a.min(0),a.max(0)])
    return np.r_[v,x['globals'],len(x['agents']),len(x['obstacle_mask'])]

def support():
    allstates=load(OUT/'all_source_states.json');out=[];neighbors=[]
    for fold,sc in {**FOLDS,'ring':'ring_exchange'}.items():
        folder=FIRST if fold=='ring' else OUT/fold;rows=pq.read_table(folder/'source_pairs.parquet').to_pylist();states=load(folder/'source_states.json');st=[s for s in states if s['split']=='train']
        pred=load(folder/'target_predictions.json')['rows'];truth=load(folder/'target_truth.json')
        if fold=='ring':
            native=load(ROOT/'diagnostics/orthoflow3_k16_critic_canonicalization_v1/phase_a/confirmation/state_manifest.json')['states'];by={r['state_uid']:r for r in native}
            tx=dict(np.load(FIRST/'target_entities.npz'))
            # Ring physical entities can be reduced directly without reconstructing Flow.
            def frombatch(x,i):
                n=int(x['agent_mask'][i].sum());m=int(x['obstacle_mask'][i].sum());v=[]
                for key in ('agents','pairs','obstacles'):
                    a=x[key][i,:n]
                    if key=='pairs':a=a[:,:n][~np.eye(n,dtype=bool)]
                    elif key=='obstacles':a=a[:,:m].reshape(-1,a.shape[-1])
                    v.extend(np.r_[a.mean(0),a.std(0),a.min(0),a.max(0)])
                return np.r_[v,x['globals'][i],n,m]
            targets=np.array([frombatch(tx,i) for i in range(len(pred))])
        else:targets=np.array([summary_vector(s) for s in load(OUT/'targets'/fold/'physical.json')])
        z=np.array([summary_vector(s['physical']) for s in st]);center=z.mean(0);scale=z.std(0);scale=np.maximum(scale,.05)
        zz=(z-center)/scale;tt=(targets-center)/scale;dist=cdist(tt,zz)/np.sqrt(zz.shape[1])
        internal=cdist(zz,zz)/np.sqrt(zz.shape[1]);fam=[str(s['scenario'])+':'+str(s['family']) for s in st]
        for i in range(len(st)):
            for j in range(len(st)):
                if fam[i]==fam[j]:internal[i,j]=np.inf
        nn=internal.min(1);d=dist.min(1)
        norm=load(folder/'normalization.json');er=np.asarray(norm['eta_radius']);ec=np.asarray(norm['eta_center']);tr=[r for r in rows if r['split']=='train'];traineta=(np.array([r['eta'] for r in tr])-ec)/er;targeteta=(np.array([p['eta'] for p in pred]).reshape(-1,3)-ec)/er
        ed=cdist(targeteta,traineta);eta_nn=ed.min(1)
        si={s['state_uid']:i for i,s in enumerate(st)};statepair=np.array([si[r['state_uid']] for r in tr]);joint=np.sqrt(np.repeat(dist,16,axis=0)[:,statepair]**2+ed**2)
        closest=joint.argmin(1);jdist=joint.min(1)
        for i,p in enumerate(pred):
            j=int(np.argmax(p['scores']['shared']));r=tr[closest[16*i+j]]
            neighbors.append({'fold':fold,'state_uid':p['state_uid'],'selected_q_lower':truth[i]['lower'][j],'selected_q_upper':truth[i]['upper'][j],'joint_distance':float(jdist[16*i+j]),'nearest_scene':r['scenario'],'nearest_state':r['state_uid'],'nearest_q':r['q'],'eta_distance':float(ed[16*i+j,closest[16*i+j]]),'physical_distance':float(dist[i,statepair[closest[16*i+j]]])})
        sourcecurves=sorted({o['curvature'] for s in st for o in s['physical']['obstacles']});constant=(np.ptp(z,axis=0)<1e-8);novel=constant&(np.any(abs(targets-z[0])>1e-6,axis=0))
        out.append({'fold':fold,'source_states':len(st),'target_states':len(pred),'state_nn_median':float(np.median(d)),'source_leave_family_nn_median':float(np.median(nn)),'source_leave_family_nn_p95':float(np.quantile(nn,.95)),'target_above_source_p95_fraction':float(np.mean(d>np.quantile(nn,.95))),'eta_nn_min':float(eta_nn.min()),'eta_nn_median':float(np.median(eta_nn)),'joint_nn_median':float(np.median(jdist)),'source_constant_channels_novel_in_target':int(novel.sum()),'source_curvatures':json.dumps(sourcecurves),'source_t0_fraction':float(np.mean([s['physical']['remaining_fraction']==1 for s in st])),'source_positive_fraction':float(np.mean([r['q']>=15/16 for r in tr])),'source_failure_fraction':float(np.mean([r['q']<=.5 for r in tr]))})
    write_csv(OUT/'support_by_fold.csv',out);write_csv(OUT/'selected_joint_neighbors.csv',neighbors)
    dump('support_metric_definition.json',{'physical_summary':'per physical-channel mean/std/min/max over agent/pair/obstacle sets +globals +entity counts','normalization':'sourceTRAIN only, std floor0.05 in fixed physical units','distance':'RMS standardized summary L2; joint sqrt(physical_RMS^2 +eta_normalized_L2^2)','limitation':'summary distances are descriptive, not proof of representation aliasing/injectivity; floor prespecified before outcomes analyzed here','cross_scene_exact_alias_test':'distinct constant geometry/controller context channels rule out exact input identity on observed states; approximate conflict cannot establish nonidentifiability'})
    print(json.dumps(out,indent=2))

def ascertainment():
    labels=pq.read_table(ROOT/'datasets/orthoflow3_basin_dataset_v2_audited/eta_labels.parquet').to_pylist();audit=[]
    for sc in SCENES[1:]:
        for split in ('train','validation'):
            rr=[r for r in labels if r['scenario']==sc and r['split']==split];full=[r for r in rr if r['seed_count']>=16 and not r['numerical_failure_count']]
            audit.append({'scenario':sc,'split':split,'all_pairs':len(rr),'known_B15_false':sum(r['robust_15of16'] is False for r in rr),'known_B15_true':sum(r['robust_15of16'] is True for r in rr),'full_count_pairs':len(full),'full_known_B15_false':sum(r['robust_15of16'] is False for r in full),'full_Q_le_half':sum(r['success_count']/r['seed_count']<=.5 for r in full),'partial_pairs':sum(r['seed_count']<16 for r in rr),'early_stop_reasons':json.dumps(dict(Counter(str(r['early_stop_reason']) for r in rr if r['seed_count']<16)))})
    write_csv(OUT/'label_ascertainment.csv',audit)
    # All admission and outcome rules inherited, not silently relaxed to improve models.
    dump('ascertainment_interpretation.json',{'mechanism':'Outcome-dependent full-count availability creates survivorship bias in probability-supervision subset','not_claimed':'partial k/n is not certified Q16; cannot fill missing trials as failures or successes','next_possible_protocol':'retain censored sequences with appropriate stopping-rule likelihood, or collect predetermined full budgets on source-only design; separate from current frozen LOSO'})
    print(json.dumps(audit,indent=2))

def local_control():
    from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import metrics
    results=[];choices={}
    for fold in ('toy','db','four','ring'):
        folder=FIRST if fold=='ring' else OUT/fold;rows=pq.read_table(folder/'source_pairs.parquet').to_pylist();states=load(folder/'source_states.json')
        tr=[r for r in rows if r['split']=='train'];va=[r for r in rows if r['split']=='validation'];trainids=sorted({r['state_index'] for r in tr});summ=np.array([summary_vector(s['physical']) for s in states]);scale=np.maximum(summ[trainids].std(0),.05);center=summ[trainids].mean(0);z=(summ-center)/scale
        norm=load(folder/'normalization.json');er=np.array(norm['eta_radius']);ec=np.array(norm['eta_center'])
        et=np.array([r['eta'] for r in tr]);ev=np.array([r['eta'] for r in va]);yti=np.array([r['q'] for r in tr]);yv=np.array([r['q'] for r in va]);ti=[r['state_index'] for r in tr];vi=[r['state_index'] for r in va]
        ds=cdist(z[vi],z[ti],'sqeuclidean')/z.shape[1];de=cdist((ev-ec)/er,(et-ec)/er,'sqeuclidean');best=None;curve=[]
        def predict(dist,k):
            ix=np.argpartition(dist,k-1,axis=1)[:,:k];d=np.take_along_axis(dist,ix,axis=1);w=1/(np.sqrt(d)+.01);return (w*yti[ix]).sum(1)/w.sum(1)
        for alpha in (.25,1.,4.):
            for k in (1,5,15):
                p=predict(ds+alpha*de,k);loss=np.mean([metrics(p[[r['scenario']==sc for r in va]],yv[[r['scenario']==sc for r in va]])['nll'] for sc in sorted({r['scenario'] for r in tr})]);curve.append({'eta_distance_weight':alpha,'k':k,'source_val_nll':float(loss)})
                if best is None or loss<best[0]:best=(loss,alpha,k)
        pp,xx=inputs(fold)
        if fold=='ring':
            def reduce(x,i):
                n=int(x['agent_mask'][i].sum());m=int(x['obstacle_mask'][i].sum());v=[]
                for key in ('agents','pairs','obstacles'):
                    a=x[key][i,:n]
                    if key=='pairs':a=a[:,:n][~np.eye(n,dtype=bool)]
                    elif key=='obstacles':a=a[:,:m].reshape(-1,a.shape[-1])
                    v.extend(np.r_[a.mean(0),a.std(0),a.min(0),a.max(0)])
                return np.r_[v,x['globals'][i],n,m]
            target=np.array([reduce(xx,i) for i in range(len(pp))])
        else:target=np.array([summary_vector(s) for s in load(OUT/'targets'/fold/'physical.json')])
        targeteta=np.array([r['eta'] for r in pp]).reshape(-1,3)
        ds=np.repeat(cdist((target-center)/scale,z[ti],'sqeuclidean')/z.shape[1],16,axis=0);de=cdist((targeteta-ec)/er,(et-ec)/er,'sqeuclidean')
        p=predict(ds+best[1]*de,best[2]).reshape(-1,16);choices[fold]={'selected':{'eta_distance_weight':best[1],'k':best[2],'source_val_nll':float(best[0])},'curve':curve,'predictions':p.tolist(),'post_hoc_diagnostic':True}
    dump('local_model_frozen.json',choices)
    for fold,choice in choices.items():
        truth=load((FIRST if fold=='ring' else OUT/fold)/'target_truth.json');scores=np.array(choice['predictions']);pick=scores.argmax(1);b=[t['robust'][j] for t,j in zip(truth,pick)]
        results.append({'fold':fold,**choice['selected'],'b15':sum(v is True for v in b),'unresolved':sum(v is None for v in b),'Q_lower':float(np.mean([t['lower'][j] for t,j in zip(truth,pick)])),'Q_upper':float(np.mean([t['upper'][j] for t,j in zip(truth,pick)]))})
    write_csv(OUT/'local_control.csv',results);print(json.dumps(results,indent=2))

def source_fit():
    import jax,jax.numpy as jnp
    from flax import serialization
    results=[];reversals=[]
    for fold in ('toy','db','four','joint','joint_enriched'):
        tr,_=configure(fold);rows,x,e,y,si=tr.data();fr=load(OUT/fold/'models_frozen.json');pred={}
        for kind in ('shared','eta_only'):
            model=tr.model_for(kind);p=model.init(jax.random.PRNGKey(0),tr.gather(x,[0]),jnp.zeros((1,3)));p=serialization.from_bytes(p,open(fr[kind]['selected']['checkpoint'],'rb').read());fn=jax.jit(lambda bx,be:jax.nn.sigmoid(model.apply(p,bx,be)))
            pred[kind]=np.concatenate([np.asarray(fn(tr.gather(x,si[i:i+128]),jnp.asarray(e[i:i+128]))) for i in range(0,len(rows),128)])
        for sc in load(OUT/fold/'protocol.json')['sources']:
            for split in ('train','validation'):
                ix=np.array([i for i,r in enumerate(rows) if r['scenario']==sc and r['split']==split]);by=defaultdict(list)
                for i in ix:by[rows[i]['state_uid']].append(i)
                for kind in pred:
                    metrics=tr.metrics(pred[kind][ix],y[ix]);correct=total=0;selected=[];oracle=[]
                    for ii in by.values():
                        ii=np.array(ii);order=np.argmax(pred[kind][ii]);selected.append(y[ii[order]]>=15/16);oracle.append(np.max(y[ii])>=15/16)
                        dy=y[ii,None]-y[ii][None];dp=pred[kind][ii,None]-pred[kind][ii][None];mask=np.triu(abs(dy)>=.25,1);total+=int(mask.sum());correct+=int(((dp*dy>0)&mask).sum())
                    results.append({'fold':fold,'scenario':sc,'split':split,'method':kind,'pairs':len(ix),**metrics,'pairwise_accuracy':correct/total if total else None,'ordering_pairs':total,'selected_B15':sum(selected),'oracle_B15':sum(oracle),'states':len(by)})
        # Exact eta ranking reversal among validation states, model-independent threshold.
        statepairs=defaultdict(dict)
        for i,r in enumerate(rows):
            if r['split']=='validation':statepairs[(r['scenario'],r['state_uid'])][r['eta_uid']]=(i,r['q'])
        ss=list(statepairs)
        for ia,a in enumerate(ss):
            for b in ss[ia+1:]:
                if a[0]==b[0]:continue
                common=sorted(statepairs[a].keys()&statepairs[b].keys())
                for ji,e1 in enumerate(common):
                    for e2 in common[ji+1:]:
                        i1,q1=statepairs[a][e1];i2,q2=statepairs[a][e2];j1,v1=statepairs[b][e1];j2,v2=statepairs[b][e2]
                        if abs(q1-q2)<.25 or abs(v1-v2)<.25 or (q1-q2)*(v1-v2)>=0:continue
                        reversals.append({'fold':fold,'scenario_a':a[0],'scenario_b':b[0],'state_a':a[1],'state_b':b[1],'eta_a':e1,'eta_b':e2,'shared_both_correct':bool((pred['shared'][i1]-pred['shared'][i2])*(q1-q2)>0 and (pred['shared'][j1]-pred['shared'][j2])*(v1-v2)>0),'eta_only_both_correct':bool((pred['eta_only'][i1]-pred['eta_only'][i2])*(q1-q2)>0 and (pred['eta_only'][j1]-pred['eta_only'][j2])*(v1-v2)>0)})
    write_csv(OUT/'source_fit_metrics.csv',results)
    if reversals:write_csv(OUT/'source_cross_scene_reversals.csv',reversals)
    dump('source_reversal_summary.json',{'cases':len(reversals),'by_fold':dict(Counter(r['fold'] for r in reversals)),'rule':'validation source states from different scenes, exact same two eta, empirical Q margin>=.25 in both with reversed sign','limitation':'empirical finite-seed contrasts and correlated pairs, descriptive not independent statistical trials'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['support','ascertainment','source_fit','local_control']);a=p.parse_args();globals()[a.action]()
