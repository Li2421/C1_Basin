"""Read-only mechanism reanalysis. Never calls an environment or writes a source cache."""
from __future__ import annotations
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import collections
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
PARENT = Path('/home/zhihan/research/Basin_C1_hard_field_20261006')
CACHE = PARENT / 'transfer_mechanism_v2'
FIELD = Path('/home/zhihan/research/Basin_C1_flow_field_poc_20261004')
CHAINS = ('TT', 'FT', 'TF', 'FF')
SCENES = ('toy_give_way', 'ring_exchange')

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)): return [clean(v) for v in x]
    if isinstance(x, (np.bool_, bool)): return bool(x)
    if isinstance(x, (np.integer, int)): return int(x)
    if isinstance(x, (np.floating, float)): return float(x) if np.isfinite(x) else None
    return x
def save(name, obj): (HERE/name).write_text(json.dumps(clean(obj), indent=2, allow_nan=False)+'\n')
def stats(x):
    a=np.asarray(x, float); a=a[np.isfinite(a)]
    return dict(n=len(a), quantiles=np.quantile(a,[0,.1,.5,.9,1]) if len(a) else [], mean=a.mean() if len(a) else None)
def angle(a,b):
    a=np.asarray(a); b=np.asarray(b)
    n=np.linalg.norm(a)*np.linalg.norm(b)
    return np.degrees(np.arccos(np.clip(np.dot(a,b)/n,-1,1))) if n>1e-12 else np.nan
def rel(a,b): return np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-12)
def label(q): return True if q[0]>=15/16 else False if q[1]<15/16 else None
def group(a,b): return 'unresolved' if a is None or b is None else ('both_success' if a and b else 'loss' if a else 'rescue' if b else 'both_failure')

def verify():
    man=read(CACHE/'manifest.json'); checks={}
    for name,h in man['source_hashes'].items(): checks[str(FIELD/name)]=sha(FIELD/name)==h
    for name,h in man['input_hashes'].items(): checks[str(PARENT/name)]=sha(PARENT/name)==h
    for v in man['checkpoint_registration'].values(): checks[v['checkpoint']]=sha(v['checkpoint'])==v['checkpoint_sha256']
    checks['probe_amendment']=sha(CACHE/'probe.py')==read(CACHE/'measurement_amendment.json')['measurement_code_sha256_after']
    assert all(checks.values()),checks
    journals={}
    for p in sorted((CACHE/'journals').glob('*.jsonl')):
        for line in p.read_text().splitlines():
            r=json.loads(line); journals[r['id']]=r
    return checks,journals

def partial_rank(frame, feature, target, within=False):
    a=frame[[feature,target,'input_norm','eta_norm','state_uid']].dropna()
    if len(a)<10:return np.nan
    xx=np.stack([rankdata(a[k]) for k in [feature,target,'input_norm','eta_norm']],axis=1)
    control=np.c_[np.ones(len(a)),xx[:,2:]]
    if within:
        control=np.c_[control,pd.get_dummies(a.state_uid).to_numpy()[:,1:]]
    rr=xx[:,:2]-control@np.linalg.lstsq(control,xx[:,:2],rcond=None)[0]
    return np.corrcoef(rr.T)[0,1] if (rr.std(0)>1e-10).all() else np.nan

def main():
    checks,journals=verify()
    cfg=read(PARENT/'screening_manifest.json'); etas=np.array(cfg['eta'])
    labels={r['state_uid']:r for r in read(PARENT/'paired_states.json')}
    tasks=read(CACHE/'tasks.json')['tasks']
    allrows=[]; initial=[]; errors=[]; spectra=[]; vectors=[]; examples=[]
    pair_errors=collections.defaultdict(list); operators=collections.defaultdict(list)
    raw_checked=0
    for ti,t in enumerate(tasks):
        p=CACHE/'results'/t['id']; journal=journals[t['id']]
        assert sha(p.with_suffix('.json'))==journal['json_sha256']
        data=read(p.with_suffix('.json')); assert data['replay_max_abs_error']==0
        z=None
        if t['kind']=='initial':
            assert sha(p.with_suffix('.npz'))==journal['npz_sha256'];raw_checked+=1
            z=np.load(p.with_suffix('.npz'))
            b=z['context0_basis']; d=b.shape[0]
            ids=np.arange(16)*(1+6*d+18)+1
            assert np.allclose(z['context0_deltas'][ids],etas@b.T,rtol=0,atol=2e-16)
            output=z['context0_exec_all'][:,ids]
            near=np.linalg.norm(etas[:,None]-etas[None,:],axis=-1)+np.eye(16)*1e9
            near=near.argmin(1)
        for row in data['rows']:
            uid=row['state_uid']; sc=row['scene']; j=row['eta_index']; phase=row['phase']; delta=np.array(row['delta'])
            labs=labels[uid]['cells']
            gg=group(label(labs['TT']['Q_distribution'][j]),label(labs['FF']['Q_distribution'][j]))
            meta=dict(scene=sc,state_uid=uid,seed=row['seed'],eta_index=j,phase=phase,group=gg,
                      task_id=t['id'],input_norm=np.linalg.norm(delta),eta_norm=np.linalg.norm(row['eta']))
            for ci,ch in enumerate(CHAINS):
                q=row['chains'][ch]
                a={**meta,'chain':ch,'stable_old':q['fd_stable']}
                for k in ['epsilon_relative_change','one_sided_mismatch','eta_epsilon_relative_change','chain_rule_relative_error',
                          'field_activity_switch_fraction','terminal_activity_switch_fraction','projection_active_fraction',
                          'total_secant_gain','rotation_degrees','effective_norm','exec_norm','terminal_projection_deformation']:
                    a[k]=q[k]
                a['reliable']=bool(q['fd_stable'] and q['chain_rule_relative_error'] is not None and q['chain_rule_relative_error']<=.05)
                allrows.append(a)
                if z is None:continue
                mm=z[f'context0_center{j}_M_exec']; m=mm[ci,2]
                jj=z[f'context0_center{j}_J_eta_exec'][ci,2]
                mp=z[f'context0_center{j}_M_pre'][ci,2]
                mf=z[f'context0_center{j}_M_flow'][2]
                a['coarse_medium_change']=rel(mm[ci,0],mm[ci,1])
                a['predicted_norm']=np.linalg.norm(m@delta)
                a['predicted_angle']=angle(delta,m@delta)
                a['baseline_offset']=row['zero_baseline_native_difference'][ch]
                a['final_projection_ratio']=np.linalg.norm(m@delta)/max(np.linalg.norm(mp@delta),1e-12)
                a['flow_gain']=np.linalg.norm(mf@delta)/max(np.linalg.norm(delta),1e-12)
                if np.isfinite(m).all():
                    u,s,vh=np.linalg.svd(m)
                    a.update(sigma_max=s[0],sigma_min=s[-1],rank_1e3=int((s>1e-3*s[0]).sum()),
                             condition=s[0]/s[-1] if s[-1]>1e-3*s[0] else np.nan,
                             singular_alignment=abs(vh[0]@delta)/max(np.linalg.norm(delta),1e-12),
                             determinant=np.linalg.det(m),negative_determinant=np.linalg.det(m)<0,
                             determinant_identified=s[-1]>1e-3*s[0] and a['reliable'] and
                              len(set(np.sign(np.linalg.det(v)) for v in mm[ci]))==1,
                             scalar_residual=np.linalg.norm(m-np.trace(m)/d*np.eye(d))/max(np.linalg.norm(m),1e-12))
                    for k in range(3):
                        a[f'basis{k}_angle']=angle(b[:,k],m@b[:,k])
                        a[f'basis{k}_gain']=np.linalg.norm(m@b[:,k])/max(np.linalg.norm(b[:,k]),1e-12)
                    if row['seed']==0 and j==0 and uid in [x['state']['uid'] for x in tasks if x['kind']=='initial' and x['seed']==0][:2]:
                        vectors.append(dict(**meta,chain=ch,M=m,B=b,J_eta=jj,U=u,singular_values=s,Vh=vh))
                de=etas-etas[j]; actual=output[ci]-output[ci,j]; predicted=de@jj.T
                ee=np.linalg.norm(actual-predicted,axis=1); nn=np.linalg.norm(actual,axis=1)
                ratios=ee/np.maximum(nn,1e-8)
                zz=z['context0_exec_all'][ci,0]-output[ci,j]; zpred=-jj@etas[j]
                a.update(nearest_eta_distance=np.linalg.norm(de[near[j]]),nearest_error=ee[near[j]],
                         nearest_relative_error=ratios[near[j]],back_to_zero_error=np.linalg.norm(zz-zpred),
                         back_to_zero_relative_error=np.linalg.norm(zz-zpred)/max(np.linalg.norm(zz),1e-8))
                if a['reliable']:
                    ix=np.arange(16)!=j
                    pair_errors[sc,ch].append(np.c_[np.linalg.norm(de[ix],axis=1),ee[ix],ratios[ix],nn[ix]])
                    if ch=='FF' and row['seed']==0:
                        examples.append({**meta,'error':a['back_to_zero_error'],'relative_error':a['back_to_zero_relative_error'],
                                         'epsilon_change':a['epsilon_relative_change'],'eta':etas[j],'delta':delta,
                                         'actual_to_zero':zz,'linear_prediction':zpred,'M':m,'J_eta':jj})
                initial.append(a.copy())
                if ch=='FF' and np.isfinite(m).all():operators[sc,uid,j].append(m)
        if z is not None:z.close()
        if ti%400==0: print('cached tasks read',ti,flush=True)
    df=pd.DataFrame(allrows); ini=pd.DataFrame(initial)
    df.to_csv(HERE/'all_context_quality.csv.gz',index=False);ini.to_csv(HERE/'initial_context_metrics.csv.gz',index=False)
    summary={}
    for keys,g in df.groupby(['scene','phase','chain']):
        summary[':'.join(keys)]=dict(n=len(g),reliable=int(g.reliable.sum()),stable_old=int(g.stable_old.sum()),
            **{k:stats(g[k]) for k in ['epsilon_relative_change','one_sided_mismatch','chain_rule_relative_error',
                                     'terminal_activity_switch_fraction','field_activity_switch_fraction']})
    save('stability.json',summary)
    geom={};lin={};parr={}
    for (sc,ch),g in ini.groupby(['scene','chain']):
        reliable=g[g.reliable];k=f'{sc}:{ch}'
        geom[k]=dict(n=len(g),reliable=len(reliable),**{v:stats(reliable[v]) for v in ['sigma_max','sigma_min','rank_1e3','condition',
                   'scalar_residual','predicted_angle','basis0_angle','basis1_angle','basis2_angle','basis0_gain','basis1_gain','basis2_gain',
                   'total_secant_gain','rotation_degrees','baseline_offset','coarse_medium_change','final_projection_ratio']},
                   amplified_fraction=float((reliable.sigma_max>1.01).mean()),near_rank_collapse_fraction=float((reliable.rank_1e3<(4 if sc=='toy_give_way' else 8)).mean()),
                   identified_determinants=int(reliable.determinant_identified.sum()),negative_identified=int((reliable.determinant_identified & reliable.negative_determinant).sum()))
        ar=np.concatenate(pair_errors[sc,ch]);parr[k]=ar
        lin[k]=dict(nearest={v:stats(reliable[v]) for v in ['nearest_eta_distance','nearest_error','nearest_relative_error']},
                    to_zero={v:stats(reliable[v]) for v in ['back_to_zero_error','back_to_zero_relative_error']},
                    all_candidate_pairs=dict(n=len(ar),eta_distance=stats(ar[:,0]),absolute_error=stats(ar[:,1]),
                        relative_error=stats(ar[:,2]),fraction_error_over_25pct=float((ar[:,2]>.25).mean()),
                        fraction_error_over_100pct=float((ar[:,2]>1).mean())))
    save('geometry.json',geom);save('candidate_linearization.json',lin)
    np.savez_compressed(HERE/'candidate_pair_errors.npz',**parr)
    save('representative_vectors.json',vectors)
    save('linearization_counterexamples.json',sorted(examples,key=lambda v:v['error'],reverse=True)[:16])
    means=ini.groupby(['scene','state_uid','eta_index','chain']).mean(numeric_only=True).reset_index()
    paired=[]
    for (sc,uid,j),g in means.groupby(['scene','state_uid','eta_index']):
        gg=g.set_index('chain');q=labels[uid]['cells'];tt=q['TT']['Q_distribution'][j];ff=q['FF']['Q_distribution'][j]
        rr=dict(scene=sc,state_uid=uid,eta_index=j,group=group(label(tt),label(ff)),Qdiff_low=ff[0]-tt[1],Qdiff_high=ff[1]-tt[0],
                input_norm=gg.loc['FF','input_norm'],eta_norm=gg.loc['FF','eta_norm'])
        for ch in ['TT','FF']:
            for k in ['predicted_norm','predicted_angle','singular_alignment','final_projection_ratio','flow_gain','sigma_max','sigma_min',
                      'rotation_degrees','total_secant_gain','effective_norm','back_to_zero_relative_error','field_activity_switch_fraction','reliable']:
                rr[ch+'_'+k]=gg.loc[ch,k]
        rr['secant_log_ratio']=np.log(max(rr['FF_effective_norm'],1e-8)/max(rr['TT_effective_norm'],1e-8))
        rr['angle_difference']=rr['FF_rotation_degrees']-rr['TT_rotation_degrees']
        paired.append(rr)
    paired=pd.DataFrame(paired);paired.to_csv(HERE/'basin_transfer_pairs.csv',index=False)
    table={}
    for (sc,gr),g in paired.groupby(['scene','group']):
        table[sc+':'+gr]=dict(cells=len(g),families=g.state_uid.nunique(),**{k:stats(g[k]) for k in [
            'FF_predicted_norm','FF_predicted_angle','FF_singular_alignment','FF_final_projection_ratio','FF_flow_gain',
            'FF_sigma_max','FF_total_secant_gain','TT_total_secant_gain','secant_log_ratio','angle_difference','FF_back_to_zero_relative_error']})
    save('basin_groups.json',table)
    associations={};rng=np.random.default_rng(202610071)
    for sc,gg in paired.groupby('scene'):
        families=gg.state_uid.unique();by={s:gg[gg.state_uid==s] for s in families}
        features=['FF_predicted_norm','FF_predicted_angle','FF_singular_alignment','FF_final_projection_ratio','FF_flow_gain',
                  'FF_sigma_max','secant_log_ratio','angle_difference']
        for feat in features:
            estimates={}
            for target in ['Qdiff_low','Qdiff_high']:
                vals=[]
                for k in range(1000):
                    sample=pd.concat([by[s].assign(state_uid=str(i)) for i,s in enumerate(rng.choice(families,len(families),replace=True))])
                    vals.append(partial_rank(sample,feat,target,True))
                estimates[target]=dict(partial_rank=partial_rank(gg,feat,target,True),cluster_bootstrap95=np.nanquantile(vals,[.025,.975]))
            associations[sc+':'+feat]=estimates
    save('associations.json',dict(method='Within-family partial Spearman controlling input norm and eta norm; 1000 family bootstraps. Unknown Q endpoints are sensitivity scenarios, not sharp rank bounds. Exploratory, no multiplicity correction.',results=associations))
    sensitivity={}
    for sc,g in paired.groupby('scene'):
        complete=g[(g.FF_reliable==1)&(g.TT_reliable==1)]
        sensitivity[sc]=dict(total_cells=len(g),all_contexts_reliable_cells=len(complete),
            correlations={feature:{target:partial_rank(complete,feature,target,True) for target in ['Qdiff_low','Qdiff_high']}
                          for feature in features})
    save('association_reliability_sensitivity.json',sensitivity)
    state_controls(labels,etas)
    save('provenance.json',dict(checks=checks,cached_tasks=len(tasks),json_hashes_checked=len(tasks),npz_hashes_checked=raw_checked,
                              action_contexts=len(df)//4,new_rollouts=0,source_cache=str(CACHE),script_sha256=sha(__file__)))
    print('cache analysis complete',flush=True)

def state_controls(labels,etas):
    out={};counter=[];records=[]
    for sc in SCENES:
        states=[u for u,l in labels.items() if l['scene']==sc];ms={k:[] for k in ['flow','TF','FF','TT']};bs=[]
        for uid in states:
            z=np.load(CACHE/'state_dependence_control'/f'{uid}.npz')
            for ch in ['TT','TF','FF']:ms[ch].append(z['center0_M_exec'][CHAINS.index(ch),2])
            ms['flow'].append(z['center0_M_flow'][2]);bs.append(z['basis'])
            records.append(dict(scene=sc,state_uid=uid,M_FF=ms['FF'][-1],M_flow=ms['flow'][-1],B=bs[-1]))
        for ch,arr in ms.items():
            m=np.stack(arr);av=m.mean(0);norm=np.linalg.norm(av)
            distances=[]
            for i in range(len(m)):
                for j in range(i):
                    dist=2*np.linalg.norm(m[i]-m[j])/max(np.linalg.norm(m[i])+np.linalg.norm(m[j]),1e-12)
                    distances.append(dist)
            out[sc+':'+ch]=dict(relative_Frobenius_dispersion=np.sqrt(np.mean(np.sum((m-av)**2,axis=(1,2))))/norm,
                pairwise_distance=stats(distances),sigma_max=stats(np.linalg.svd(m,compute_uv=False)[:,0]),
                sigma_min=stats(np.linalg.svd(m,compute_uv=False)[:,-1]))
        for i in range(len(states)):
            for j in range(i):
                dist=2*np.linalg.norm(ms['FF'][i]-ms['FF'][j])/(np.linalg.norm(ms['FF'][i])+np.linalg.norm(ms['FF'][j]))
                groups=[]
                for k in range(16):
                    a,b=[labels[u]['cells'] for u in [states[i],states[j]]]
                    groups.append((group(label(a['TT']['Q_distribution'][k]),label(a['FF']['Q_distribution'][k])),
                                   group(label(b['TT']['Q_distribution'][k]),label(b['FF']['Q_distribution'][k]))))
                counter.append(dict(scene=sc,state1=states[i],state2=states[j],distance=dist,groups=groups,
                    same_group_count=sum(a==b for a,b in groups),different_flip_status_count=sum((a in ['loss','rescue'])!=(b in ['loss','rescue']) for a,b in groups)))
    chosen={}
    for sc in SCENES:
        aa=[a for a in counter if a['scene']==sc]
        chosen[sc]=dict(nearest_with_different_flip_status=min([a for a in aa if a['different_flip_status_count']],key=lambda a:a['distance']),
                        distant_with_unchanged_memberships=max([a for a in aa if a['same_group_count']>=8],key=lambda a:a['distance']))
    save('fixed_noise_state_dependence.json',out);save('state_counterexamples.json',chosen);save('fixed_noise_operators.json',records)

if __name__=='__main__':main()
