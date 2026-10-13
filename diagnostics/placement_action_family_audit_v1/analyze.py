"""Read-only four-placement action-family audit; no controller or rollout calls."""
from __future__ import annotations
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import sys, json, itertools, collections
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
HERE=Path(__file__).resolve().parent
PREV=HERE.parent/'field_transfer_operator_audit_v3'
sys.path.insert(0,str(PREV))
from cache_analysis import read, sha, clean, stats, angle, rel, label, group, verify, CACHE, PARENT, CHAINS
DISPLAY=('TT','TF','FT','FF')
PAIRS=list(itertools.combinations(DISPLAY,2))
FEATURES=['log_effective_ratio','finite_angle','action_shift_scaled','baseline_shift_scaled',
          'subspace_distance','lateral_gain_difference']
def save(name,obj):
    (HERE/name).write_text(json.dumps(clean(obj),indent=2,allow_nan=False)+'\n')
def subspace(j,error):
    if not np.isfinite(j).all() or not np.isfinite(error):
        return None,np.full(3,np.nan),np.full((3,3),np.nan),-1,np.nan
    u,s,v=np.linalg.svd(j,full_matrices=False)
    tau=max(.001*s[0],3*error,1e-7)
    rank=int((s>tau).sum())
    q=u[:,:rank]
    return q,s,v,rank,tau
def pair_space(q1,q2):
    if q1 is None or q2 is None:
        return {k:np.nan for k in ['principal_min','principal_max','principal_rms','principal_angles',
                                  'rank_difference','subspace_distance','subspace_gap','same_rank']}
    a=np.degrees(np.arccos(np.clip(np.linalg.svd(q1.T@q2,compute_uv=False),0,1)))
    p=q1@q1.T-q2@q2.T
    return dict(principal_min=a.min() if len(a) else np.nan,
                principal_max=a.max() if len(a) else np.nan,
                principal_rms=np.sqrt(np.mean(a*a)) if len(a) else np.nan,
                principal_angles=json.dumps(a.tolist()),rank_difference=q2.shape[1]-q1.shape[1],
                subspace_distance=np.linalg.norm(p)/np.sqrt(max(q1.shape[1],q2.shape[1],1)),
                subspace_gap=np.linalg.norm(p,2),
                same_rank=q1.shape[1]==q2.shape[1])
def summarize(frame,cols):
    return {k:stats(frame[k]) for k in cols}

def extract():
    checks,journals=verify()
    tasks=read(CACHE/'tasks.json')['tasks']
    etas=np.array(read(PARENT/'screening_manifest.json')['eta'])
    records=[]; pairs=[]; clouds=[]; cloudpairs=[]; representatives=[];interactions=[]
    hashes=[]; context_count=0; chain_residuals=[]
    for ti,t in enumerate(tasks):
        base=CACHE/'results'/t['id']; journal=journals[t['id']]
        for ext,key in [('.json','json_sha256'),('.npz','npz_sha256')]:
            h=sha(base.with_suffix(ext)); assert h==journal[key]
            hashes.append(dict(path=str(base.with_suffix(ext)),sha256=h))
        data=read(base.with_suffix('.json'));assert data['replay_max_abs_error']==0
        with np.load(base.with_suffix('.npz')) as z:
            memo={}
            for row in data['rows']:
                cp=row['array_prefix'].split('_center')[0]
                center=int(row['array_prefix'].split('_center')[1]);j=row['eta_index']
                if cp not in memo:
                    b=z[cp+'_basis'];d=b.shape[0]
                    ex=z[cp+'_exec_all'];pre=z[cp+'_pre_all'];ds=z[cp+'_deltas']
                    memo[cp]=(b,ex,pre,ds)
                b,ex,pre,ds=memo[cp];d=b.shape[0]
                idx=1+center*(1+6*d+18)
                delta=np.array(row['delta']);eta=np.array(row['eta'])
                assert np.max(np.abs(ds[idx]-b@eta))<3e-15
                for ci,ch in enumerate(CHAINS):
                    assert np.max(np.abs(ex[ci,idx]-np.ravel(row['chains'][ch]['exec_action'])))<1e-12
                meta={k:row[k] for k in ['scene','state_uid','seed','eta_index','phase','step','anchor_chain','fallback']}
                meta.update(task_id=t['id'],array_prefix=row['array_prefix'],input_norm=np.linalg.norm(delta),eta_norm=np.linalg.norm(eta))
                js=z[row['array_prefix']+'_J_eta_exec'];ms=z[row['array_prefix']+'_M_exec']
                jp=z[row['array_prefix']+'_J_eta_pre']
                basis_s=np.linalg.svd(b,compute_uv=False)
                brank=int((basis_s>max(1e-10*basis_s[0],1e-8)).sum())
                bq=np.linalg.svd(b,full_matrices=False)[0][:,:brank]
                cr={};qs={}
                for ci,ch in enumerate(CHAINS):
                    jmat=js[ci,2]; err=np.linalg.norm(jmat-js[ci,1],2) if np.isfinite(js[ci,1:]).all() else np.nan
                    q,s,v,rank,tau=subspace(jmat,err);qs[ch]=q
                    old=row['chains'][ch]
                    reliable=bool(np.isfinite(js[ci]).all() and np.isfinite(ms[ci]).all() and old['fd_stable']
                        and old['eta_epsilon_relative_change'] is not None and old['eta_epsilon_relative_change']<=.05
                        and old['chain_rule_relative_error'] is not None and old['chain_rule_relative_error']<=.05)
                    eff=ex[ci,idx]-ex[ci,0];gain=np.linalg.norm(eff)/max(np.linalg.norm(delta),1e-12)
                    a=dict(**meta,chain=ch,reliable=reliable,
                        sigma1=s[0],sigma2=s[1],sigma3=s[2],rank_raw=int((s>.001*s[0]).sum()),
                        rank_resolved=rank,rank_tau=tau,rank_error_estimate=err,
                        condition=s[0]/s[-1] if rank==3 else np.nan,
                        j_relative_step_change=rel(jmat,js[ci,1]),j_coarse_medium_change=rel(js[ci,0],js[ci,1]),
                        chain_rule_error=old['chain_rule_relative_error'],
                        basis_sigma1=basis_s[0],basis_sigma3=basis_s[-1],basis_condition=basis_s[0]/max(basis_s[-1],1e-12),basis_rank=brank,
                        effective_norm=np.linalg.norm(eff),finite_gain=gain,finite_rotation=angle(delta,eff),
                        action_norm=np.linalg.norm(ex[ci,idx]),baseline_norm=np.linalg.norm(ex[ci,0]),
                        projection_local_ratio=np.linalg.norm(jmat)/max(np.linalg.norm(jp[ci,2]),1e-12),
                        projection_finite_pre_norm=np.linalg.norm(pre[ci,idx]-pre[ci,0]),
                        field_switch_fraction=old['field_activity_switch_fraction'],terminal_switch_fraction=old['terminal_activity_switch_fraction'],
                        projection_active_fraction=old['projection_active_fraction'],
                        backzero_error=np.linalg.norm(eff-jmat@eta),
                        backzero_relative_error=np.linalg.norm(eff-jmat@eta)/max(np.linalg.norm(eff),1e-8))
                    span=np.linalg.svd(ms[ci,2]@bq,compute_uv=False) if np.isfinite(ms[ci,2]).all() else np.full(3,np.nan)
                    a.update(physical_span_sigma1=span[0],physical_span_sigma3=span[-1] if brank==3 else np.nan,
                             physical_span_sigma_min=span[-1])
                    for bi in range(3):
                        bn=np.linalg.norm(b[:,bi]);nonzero=bn>1e-8
                        gg=np.linalg.norm(jmat[:,bi])/bn if nonzero else np.nan
                        aa=angle(b[:,bi],jmat[:,bi]) if gg>.01 else np.nan
                        a.update({f'basis{bi}_gain':gg,f'basis{bi}_angle':aa,f'basis{bi}_input_norm':bn,f'basis{bi}_nonzero':nonzero,
                                  f'basis{bi}_annihilated':gg<=.01 if nonzero else np.nan,
                                  f'basis{bi}_attenuated':gg<.95 if nonzero else np.nan,f'basis{bi}_amplified':gg>1.05 if nonzero else np.nan,
                                  f'basis{bi}_preserved':(.95<=gg<=1.05 and aa<=5) if nonzero else np.nan})
                    cr[ch]=a;records.append(a)
                    if row['phase']=='initial' and row['seed']==0 and j==4:
                        representatives.append(dict(**meta,chain=ch,B=b,J_eta=jmat,U=np.linalg.svd(jmat,full_matrices=False)[0],
                            singular_values=s,Vh=v,rank_resolved=rank,finite_effect=eff,baseline=ex[ci,0]))
                for ch1,ch2 in PAIRS:
                    c1,c2=CHAINS.index(ch1),CHAINS.index(ch2)
                    e1=ex[c1,idx]-ex[c1,0];e2=ex[c2,idx]-ex[c2,0]
                    shift=ex[c2,idx]-ex[c1,idx];base_shift=ex[c2,0]-ex[c1,0]
                    residual=np.linalg.norm(shift-base_shift-(e2-e1));chain_residuals.append(residual)
                    p=dict(**meta,contrast=ch1+'-'+ch2,
                        reliable=cr[ch1]['reliable'] and cr[ch2]['reliable'],
                        **pair_space(qs[ch1],qs[ch2]),
                        rank1=cr[ch1]['rank_resolved'],rank2=cr[ch2]['rank_resolved'],
                        finite_angle=angle(e1,e2),
                        log_effective_ratio=np.log(max(np.linalg.norm(e2),1e-8)/max(np.linalg.norm(e1),1e-8)),
                        effective1=np.linalg.norm(e1),effective2=np.linalg.norm(e2),
                        action_shift=np.linalg.norm(shift),action_shift_scaled=np.linalg.norm(shift)/max(np.linalg.norm(delta),1e-12),
                        baseline_shift=np.linalg.norm(base_shift),baseline_shift_scaled=np.linalg.norm(base_shift)/max(np.linalg.norm(delta),1e-12),
                        correction_family_shift=np.linalg.norm(e2-e1),
                        correction_family_shift_scaled=np.linalg.norm(e2-e1)/max(np.linalg.norm(delta),1e-12),
                        lateral_gain_difference=cr[ch2]['basis1_gain']-cr[ch1]['basis1_gain'],
                        sigma1_difference=cr[ch2]['sigma1']-cr[ch1]['sigma1'],
                        j_relative_difference=np.linalg.norm(js[c2,2]-js[c1,2])/max(np.linalg.norm(js[c1,2]),1e-12),
                        j_absolute_difference=np.linalg.norm(js[c2,2]-js[c1,2]))
                    pairs.append(p)
                interaction=ex[3,idx]-ex[1,idx]-ex[2,idx]+ex[0,idx]
                base_interaction=ex[3,0]-ex[1,0]-ex[2,0]+ex[0,0]
                interactions.append(dict(**meta,action_interaction_norm=np.linalg.norm(interaction),
                    action_interaction_scaled=np.linalg.norm(interaction)/max(np.linalg.norm(delta),1e-12),
                    correction_interaction_norm=np.linalg.norm(interaction-base_interaction),
                    baseline_interaction_norm=np.linalg.norm(base_interaction)))
                context_count+=1
            if t['kind']=='initial':
                b,ex,pre,ds=memo['context0'];d=b.shape[0];idxs=1+np.arange(16)*(1+6*d+18)
                outputs=ex[:,idxs];bs=np.linalg.svd(b,compute_uv=False)
                meta=dict(scene=t['state']['scenario'],state_uid=t['state']['uid'],seed=t['seed'],task_id=t['id'])
                for ci,ch in enumerate(CHAINS):
                    out=outputs[ci];cs=np.linalg.svd(out-out.mean(0),compute_uv=False)
                    clouds.append(dict(**meta,chain=ch,diameter=cdist(out,out).max(),
                        centered_rms=np.sqrt(np.mean(np.sum((out-out.mean(0))**2,1))),
                        cloud_sigma1=cs[0],cloud_sigma3=cs[2],
                        empirical_affine_rank=int((cs>.001*cs[0]).sum()),
                        baseline_displacement_mean=np.linalg.norm(out-ex[ci,0],axis=1).mean()))
                for a,bb in PAIRS:
                    aa=outputs[CHAINS.index(a)];ab=outputs[CHAINS.index(bb)]
                    dist=cdist(aa,ab);same=np.diag(dist)
                    cloudpairs.append(dict(**meta,contrast=a+'-'+bb,hausdorff=max(dist.min(0).max(),dist.min(1).max()),
                        mean_nearest=.5*(dist.min(0).mean()+dist.min(1).mean()),
                        matched_rms=np.sqrt(np.mean(same**2)),matched_max=same.max(),
                        nearest_eta_diff_fraction=float((dist.argmin(1)!=np.arange(16)).mean())))
        if ti%400==0: print('read',ti,'/',len(tasks),'contexts',context_count,flush=True)
    df=pd.DataFrame(records);pf=pd.DataFrame(pairs)
    inf=pd.DataFrame(interactions);inf.to_csv(HERE/'placement_interactions.csv.gz',index=False)
    save('placement_interactions.json',{':'.join(k):summarize(g,['action_interaction_norm','action_interaction_scaled',
        'correction_interaction_norm','baseline_interaction_norm']) for k,g in inf.groupby(['scene','phase'])})
    df.to_csv(HERE/'context_metrics.csv.gz',index=False);pf.to_csv(HERE/'paired_context_metrics.csv.gz',index=False)
    cf=pd.DataFrame(clouds);cpf=pd.DataFrame(cloudpairs)
    cf.to_csv(HERE/'sampled_action_families.csv',index=False);cpf.to_csv(HERE/'sampled_family_distances.csv',index=False)
    save('representative_jacobians.json',representatives)
    geom={};pair_summary={}
    cols=['sigma1','sigma2','sigma3','rank_raw','rank_resolved','condition','rank_error_estimate',
          'j_relative_step_change','j_coarse_medium_change','basis_condition','physical_span_sigma1','physical_span_sigma3',
          'effective_norm','finite_gain','finite_rotation','projection_local_ratio','backzero_relative_error']
    cols += [f'basis{i}_{k}' for i in range(3) for k in ['gain','angle','input_norm','nonzero','annihilated','attenuated','amplified','preserved']]
    for key,g in df.groupby(['scene','phase','chain']):
        v=g[g.reliable]
        geom[':'.join(key)]=dict(total=len(g),reliable=len(v),**summarize(v,cols),
            rank_raw_counts=v.rank_raw.value_counts().sort_index().to_dict(),
            rank_resolved_counts=v.rank_resolved.value_counts().sort_index().to_dict())
    for key,g in pf.groupby(['scene','phase','contrast']):
        v=g[g.reliable]
        pair_summary[':'.join(key)]=dict(total=len(g),reliable=len(v),**summarize(v,[
            'principal_min','principal_max','subspace_distance','subspace_gap','rank_difference','same_rank',
            'finite_angle','log_effective_ratio','action_shift','baseline_shift','correction_family_shift',
            'j_relative_difference']),
            rank_pairs=v.groupby(['rank1','rank2']).size().rename_axis(['rank1','rank2']).reset_index(name='n').to_dict('records'))
    save('geometry.json',geom);save('paired_geometry.json',pair_summary)
    save('sampled_families.json',{
        'individual':{':'.join(k):summarize(g,['diameter','centered_rms','cloud_sigma1','cloud_sigma3','empirical_affine_rank','baseline_displacement_mean']) for k,g in cf.groupby(['scene','chain'])},
        'paired':{':'.join(k):summarize(g,['hausdorff','mean_nearest','matched_rms','matched_max','nearest_eta_diff_fraction']) for k,g in cpf.groupby(['scene','contrast'])}})
    save('provenance.json',dict(checks=checks,hashes_checked=len(hashes),contexts=context_count,new_rollouts=0,
                              new_action_evaluations=0,decomposition_residual_max=max(chain_residuals),
                              script_sha256=sha(__file__),cache=str(CACHE)))
    save('input_hashes.json',hashes)
    print('extraction complete',flush=True)

def residual_rank(g,feature,target,eta_fe=False):
    a=g[[feature,target,'input_norm','eta_norm','state_uid','eta_index']].dropna()
    if len(a)<30:return np.nan
    ranked=np.stack([rankdata(a[k]) for k in [feature,target,'input_norm','eta_norm']],axis=1)
    ctl=np.c_[np.ones(len(a)),ranked[:,2:],pd.get_dummies(a.state_uid).to_numpy()[:,1:]]
    if eta_fe:ctl=np.c_[ctl,pd.get_dummies(a.eta_index).to_numpy()[:,1:]]
    rr=ranked[:,:2]-ctl@np.linalg.lstsq(ctl,ranked[:,:2],rcond=None)[0]
    return np.corrcoef(rr.T)[0,1] if (rr.std(0)>1e-10).all() else np.nan

def bootstrap_rank(values,codes,eta_codes=None):
    mask=np.isfinite(values).all(1);a=values[mask];codes=codes[mask]
    if len(a)<30:return np.nan
    rr=rankdata(a,axis=0)
    counts=np.bincount(codes,minlength=24)
    means=np.stack([np.bincount(codes,weights=rr[:,k],minlength=24)/np.maximum(counts,1) for k in range(4)],axis=1)
    rr=rr-means[codes]
    ctl=rr[:,2:]
    if eta_codes is not None:
        e=np.eye(16)[eta_codes[mask]][:,1:]
        em=np.stack([np.bincount(codes,weights=e[:,k],minlength=24)/np.maximum(counts,1) for k in range(15)],axis=1)
        ctl=np.c_[ctl,e-em[codes]]
    res=rr[:,:2]-ctl@np.linalg.lstsq(ctl,rr[:,:2],rcond=None)[0]
    return np.corrcoef(res.T)[0,1] if (res.std(0)>1e-10).all() else np.nan

def associate():
    labels={v['state_uid']:v for v in read(PARENT/'paired_states.json')}
    pf=pd.read_csv(HERE/'paired_context_metrics.csv.gz',low_memory=False);pf=pf[pf.phase=='initial']
    if pf.same_rank.dtype==object:pf['same_rank']=pf.same_rank.map({'True':1.,'False':0.,True:1.,False:0.})
    # Finite values do not require local differentiability. Local metrics use only reliable paired estimates.
    local=['principal_min','principal_max','subspace_distance','subspace_gap','rank_difference','same_rank',
           'lateral_gain_difference','sigma1_difference','j_relative_difference','j_absolute_difference']
    pf.loc[~pf.reliable,local]=np.nan
    cell=pf.groupby(['scene','state_uid','eta_index','contrast']).mean(numeric_only=True).reset_index()
    classes=[]
    for _,v in cell.iterrows():
        a,b=v.contrast.split('-');q=labels[v.state_uid]['cells'];qa=q[a]['Q_distribution'][v.eta_index];qb=q[b]['Q_distribution'][v.eta_index]
        classes.append(dict(group=group(label(qa),label(qb)),q1low=qa[0],q1high=qa[1],q2low=qb[0],q2high=qb[1],
                            qdiff_low=qb[0]-qa[1],qdiff_high=qb[1]-qa[0]))
    cell=pd.concat([cell.reset_index(drop=True),pd.DataFrame(classes)],axis=1)
    cell.to_csv(HERE/'basin_action_family_pairs.csv',index=False)
    groups={}
    for k,g in cell.groupby(['scene','contrast']):
        groups[':'.join(k)]=dict(counts=g.group.value_counts().to_dict(),
            by_group={gr:dict(n=len(v),states=v.state_uid.nunique(),**summarize(v,FEATURES+['correction_family_shift','j_relative_difference','principal_max','effective1','effective2'])) for gr,v in g.groupby('group')})
    save('basin_groups.json',groups)
    # All predeclared contrasts and features; 500 state-cluster bootstrap resamples.
    assoc={};rng=np.random.default_rng(2026100719)
    for k,g in cell.groupby(['scene','contrast']):
        g=g.reset_index(drop=True)
        families=g.state_uid.unique();by={s:np.flatnonzero(g.state_uid.to_numpy()==s) for s in families}
        draws=rng.choice(families,(500,len(families)),replace=True)
        index_draws=[np.concatenate([by[s] for s in draw]) for draw in draws]
        codes=np.repeat(np.arange(len(families)),16)
        out={}
        for feat in FEATURES:
            result={}
            for target in ['qdiff_low','qdiff_high']:
                values=g[[feat,target,'input_norm','eta_norm']].to_numpy(float)
                vals=[bootstrap_rank(values[ix],codes) for ix in index_draws]
                eta_codes=g.eta_index.to_numpy(int)
                vals_eta=[bootstrap_rank(values[ix],codes,eta_codes[ix]) for ix in index_draws]
                result[target]=dict(rho=residual_rank(g,feat,target),cluster95=np.nanquantile(vals,[.025,.975]) if np.isfinite(vals).any() else [np.nan,np.nan],
                                   rho_state_eta_fe=residual_rank(g,feat,target,True),
                                   cluster95_state_eta_fe=np.nanquantile(vals_eta,[.025,.975]) if np.isfinite(vals_eta).any() else [np.nan,np.nan],
                                   rho_all_seeds_reliable=residual_rank(g[g.reliable==1],feat,target))
            out[feat]=result
        assoc[':'.join(k)]=out
        print('association',k,flush=True)
    save('associations.json',dict(method='Partial Spearman: correction norm, eta norm, state FE; 500 state bootstraps. eta FE and all-seeds-reliable sensitivities. Q endpoints are scenarios, not sharp correlation bounds. Exploratory, no multiplicity correction.',results=assoc))
    # Counterexample selection is deterministic and explicitly outcome conditioned.
    counter=[]
    for (sc,contrast),g in cell.groupby(['scene','contrast']):
        if contrast not in ['TT-TF','TT-FT','TT-FF','TF-FF','FT-FF']:continue
        flipped=g[g.group.isin(['loss','rescue'])]
        same=g[g.group.isin(['both_success','both_failure'])]
        for kind,subset,col,ascending in [('small_initial_action_flip',flipped,'action_shift',True),
                ('small_initial_jacobian_flip',flipped,'j_absolute_difference',True),
                ('large_initial_action_same_label',same,'action_shift',False)]:
            for _,v in subset.sort_values(col,ascending=ascending).head(2).iterrows():
                counter.append(dict(kind=kind,**v.to_dict()))
    save('counterexample_cells.json',counter)
    save('basin_counts.json',{sc:{ch:dict(lower=sum(v['cells'][ch]['B15_count'][0] for v in labels.values() if v['scene']==sc),
                                          upper=sum(v['cells'][ch]['B15_count'][1] for v in labels.values() if v['scene']==sc)) for ch in DISPLAY} for sc in sorted(cell.scene.unique())})
    print('associations complete',flush=True)

if __name__=='__main__':
    mode=sys.argv[1] if len(sys.argv)>1 else 'all'
    if mode in ['all','extract']:extract()
    if mode in ['all','associate']:associate()
