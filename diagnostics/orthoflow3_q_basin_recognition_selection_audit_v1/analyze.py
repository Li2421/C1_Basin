"""Basin recognition and selection from frozen sufficient statistics only."""
import csv, gzip, hashlib, json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.stats import binomtest, rankdata

HERE=Path(__file__).resolve().parent
SEED=20261006

def read(p):return json.loads(Path(p).read_text())
def clean(v):
    if isinstance(v,dict):return {k:clean(x)for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x)for x in v]
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,float) and not np.isfinite(v):return None
    return v
def write(p,d):Path(p).write_text(json.dumps(clean(d),indent=2,allow_nan=False)+'\n')
def csvout(name,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    op=gzip.open if name.endswith('.gz') else open
    with op(HERE/name,'wt',newline='')as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows([clean(r)for r in rows])
def membership(s,f):return np.where(s>=15,1,np.where(f>=2,0,-1))
def auc(y,z):
    y=np.asarray(y);z=np.asarray(z);ok=y>=0;y=y[ok];z=z[ok];np_=sum(y==1);nn=sum(y==0)
    if not np_ or not nn:return np.nan
    return float((rankdata(z)[y==1].sum()-np_*(np_+1)/2)/(np_*nn))
def ap(y,z):
    ok=y>=0;y=y[ok];z=z[ok];n=sum(y==1)
    if not n:return np.nan
    order=np.argsort(-z,kind='stable');y=y[order];z=z[order]
    ends=np.r_[np.flatnonzero(np.diff(z)!=0),len(y)-1]
    tp=np.cumsum(y==1)[ends];return float(np.sum(np.diff(np.r_[0,tp])*tp/(ends+1))/n)
def ci_mean(a):
    a=np.asarray(a,float);a=a[np.isfinite(a)]
    if not len(a):return [np.nan,np.nan]
    ids=np.random.default_rng(SEED).integers(len(a),size=(10000,len(a)))
    return np.quantile(a[ids].mean(1),[.025,.975]).tolist()
def wilson(s,n):
    if not n:return [np.nan,np.nan]
    z=1.959963984540054;p=s/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0,c-h),min(1,c+h)]
def sign_p(w,l):return float(binomtest(w,w+l,.5).pvalue) if w+l else 1.
def holm(vals):
    vals=np.array(vals);ix=np.argsort(vals);a=np.minimum(1,np.maximum.accumulate(vals[ix]*(len(vals)-np.arange(len(vals)))));r=np.empty(len(vals));r[ix]=a;return r
def evaluate(s,f,z,states,cohort,model):
    b=membership(s,f);valid=b>=0;p=expit(z);n,k=s.shape;h=np.arange(n);order=np.argsort(-z,axis=1,kind='stable');j=order[:,0]
    y=b[h,j];eligible=(b==1).any(1);per=[]
    for i in range(n):
        au=auc(b[i],z[i]);nb=int(sum(b[i]==1));rank=1+int(np.flatnonzero(b[i,order[i]]==1)[0]) if nb else np.nan
        positive=np.flatnonzero(b[i]==1);prank=rankdata(-z[i],method='average')[positive]
        row=dict(cohort=cohort,model=model,state_uid=states[i],state_index=i,K=k,robust_candidates=nb,unknown_candidates=int(sum(b[i]<0)),
                 oracle_eligible=bool(eligible[i]),within_state_auc=au,best_robust_rank=rank,
                 mean_robust_rank_percentile=float(np.mean((k-prank)/max(1,k-1))) if nb else np.nan,
                 selected_index=int(j[i]),selected_B15=int(y[i]),selected_Q_lower=s[i,j[i]]/16,selected_Q_upper=(16-f[i,j[i]])/16,
                 oracle_Q_lower=float(s[i].max()/16),oracle_Q_upper=float((16-f[i]).max()/16),
                 top2_contains_robust=bool((b[i,order[i,:min(k,2)]]==1).any()),top4_contains_robust=bool((b[i,order[i,:min(k,4)]]==1).any()),
                 top4_enrichment=float((b[i,order[i,:min(k,4)]]==1).mean()/(nb/k)) if nb else np.nan)
        # Interval regret; chosen itself is in the oracle set, so lower bound is >=0.
        row['Q_regret_lower']=max(0,row['oracle_Q_lower']-row['selected_Q_upper'])
        row['Q_regret_upper']=max(0,row['oracle_Q_upper']-row['selected_Q_lower'])
        per.append(row)
    aus=np.array([r['within_state_auc']for r in per]);mixed=np.isfinite(aus);known=b>=0;total=s+f
    nll=float((s*np.logaddexp(0,-z)+f*np.logaddexp(0,z)).sum()/total.sum())
    # These are per-continuation proper scores, not a calibrated membership probability.
    brier=float((s*(1-p)**2+f*p**2).sum()/total.sum())
    wci=wilson(sum(y==1),sum(eligible));aci=ci_mean(aus)
    row=dict(cohort=cohort,model=model,N=n,K=k,oracle_eligible=int(eligible.sum()),mixed_states=int(mixed.sum()),
             robust_candidates=int((b==1).sum()),unknown_candidates=int((b<0).sum()),candidate_AUROC=auc(b.ravel(),z.ravel()),
             candidate_AP=ap(b.ravel(),z.ravel()),candidate_prevalence=float((b[known]==1).mean()),
             trial_Brier=brier,trial_NLL=nll,positive_mean_score=float(p[b==1].mean()) if (b==1).any() else np.nan,
             negative_mean_score=float(p[b==0].mean()) if (b==0).any() else np.nan,
             state_AUC_mean=float(aus[mixed].mean()) if mixed.any()else np.nan,state_AUC_ci_low=aci[0],state_AUC_ci_high=aci[1],
             state_AUC_q25=float(np.nanquantile(aus,.25))if mixed.any()else np.nan,state_AUC_median=float(np.nanmedian(aus))if mixed.any()else np.nan,
             state_AUC_q75=float(np.nanquantile(aus,.75))if mixed.any()else np.nan,state_AUC_above_half=int(sum(aus>.5)),
             selected_B15=int(sum(y==1)),selected_unknown=int(sum(y<0)),top1_eligible_rate=float(np.mean(y[eligible]==1))if eligible.any()else np.nan,
             top1_eligible_rate_upper=float(np.mean(y[eligible]!=0))if eligible.any()else np.nan,
             top1_eligible_Wilson_low=wci[0],top1_eligible_Wilson_high=wci[1],oracle_at_K=float(eligible.mean()),
             oracle_gap_all_state_fraction=float(eligible.mean()-(y==1).mean()),
             proposal_failures=int(((b==0).all(1)).sum()),oracle_unknown=int((~eligible&~(b==0).all(1)).sum()),
             ranking_failures=int((eligible&(y==0)).sum()),ranking_unknown=int((eligible&(y<0)).sum()),
             top2_eligible=int(sum(r['top2_contains_robust'] for r in per)),top4_eligible=int(sum(r['top4_contains_robust'] for r in per)),
             unique_top1_eta=int(len(set(j))),
             selected_Q_lower=float(np.mean([r['selected_Q_lower']for r in per])),selected_Q_upper=float(np.mean([r['selected_Q_upper']for r in per])),
             oracle_Q_lower=float(np.mean([r['oracle_Q_lower']for r in per])),oracle_Q_upper=float(np.mean([r['oracle_Q_upper']for r in per])),
             regret_lower=float(np.mean([r['Q_regret_lower']for r in per])),regret_upper=float(np.mean([r['Q_regret_upper']for r in per])))
    return row,per

def state_cases(s,f):
    b=membership(s,f);lo=s/16;hi=(16-f)/16;n,k=s.shape;sets={x:[]for x in ['B15_swap','Q16_gap_025']}
    for i,l in combinations(range(n),2):
        for a,d in combinations(range(k),2):
            if b[i,a]>=0 and b[i,d]>=0 and b[l,a]>=0 and b[l,d]>=0 and (b[i,a]-b[i,d])*(b[l,a]-b[l,d])==-1:
                sets['B15_swap'].append([i,l,a,d,int(b[i,a]-b[i,d])])
            signs=[]
            for h in [i,l]:signs.append(1 if lo[h,a]-hi[h,d]>=.25 else -1 if lo[h,d]-hi[h,a]>=.25 else 0)
            if signs[0]*signs[1]==-1:sets['Q16_gap_025'].append([i,l,a,d,signs[0]])
    return {k:np.array(v,int).reshape(-1,5)for k,v in sets.items()}
def score_cases(cases,z):
    if not len(cases):return np.array([],bool)
    i,l,a,b,truth=cases.T
    return (np.sign(z[i,a]-z[i,b])==truth)&(np.sign(z[l,a]-z[l,b])==-truth)
def same_eta(s,f,z):
    b=membership(s,f);wins=ties=total=0;state=set()
    for i,j in combinations(range(len(b)),2):
        ok=(b[i]>=0)&(b[j]>=0)&(b[i]!=b[j]);d=(z[i]-z[j])*(b[i]-b[j]);total+=sum(ok);wins+=sum(ok&(d>0));ties+=sum(ok&(d==0))
        if ok.any():state.update([i,j])
    return dict(same_eta_membership_changes=int(total),same_eta_correct_direction=int(wins),same_eta_ties=int(ties),
                same_eta_direction_accuracy_half_ties=(wins+.5*ties)/total if total else np.nan,same_eta_states=len(state))
def paired(s,f,za,zb,pa,pb,cohort,a,b):
    B=membership(s,f);i=np.arange(len(s));aa=B[i,np.argmax(za,1)];bb=B[i,np.argmax(zb,1)]
    known=(aa>=0)&(bb>=0);d=aa[known]-bb[known];w=int(sum(d>0));l=int(sum(d<0));ci=ci_mean(d)
    ar=np.array([r['within_state_auc']for r in pa]);br=np.array([r['within_state_auc']for r in pb]);ad=ar-br;mask=np.isfinite(ad);ac=ci_mean(ad)
    lower=(aa==1).astype(int)-(bb!=0).astype(int);upper=(aa!=0).astype(int)-(bb==1).astype(int)
    identical=np.argmax(za,1)==np.argmax(zb,1);lower[identical]=0;upper[identical]=0
    return dict(cohort=cohort,model=a,reference=b,paired_states=int(known.sum()),unresolved_states=int((~known).sum()),rescue=w,break_count=l,
                all_state_delta_lower=float(lower.mean()),all_state_delta_upper=float(upper.mean()),
                selected_rate_delta=float(d.mean()),delta_ci_low=ci[0],delta_ci_high=ci[1],McNemar_p=sign_p(w,l),
                AUC_paired_states=int(mask.sum()),AUC_delta=float(ad[mask].mean())if mask.any()else np.nan,AUC_delta_ci_low=ac[0],AUC_delta_ci_high=ac[1],
                AUC_state_sign_p=sign_p(int(sum(ad[mask]>1e-12)),int(sum(ad[mask]<-1e-12))))

def main_panels():
    metrics=[];perstate=[];reversals=[];comparisons=[];random=[];figselect={};atlas={};candidate_counts=[]
    for meta in read(HERE/'cohorts.json'):
        name=meta['name'];d=dict(np.load(HERE/'inputs'/f'{name}.npz'));s=d['success'];f=d['failure'];b=membership(s,f)
        for i in range(meta['N']):
            for j in range(meta['K']):candidate_counts.append(dict(cohort=name,state_uid=meta['states'][i],candidate_index=j,success=int(s[i,j]),failure=int(f[i,j]),unobserved_or_numerical=int(16-s[i,j]-f[i,j]),B15=int(b[i,j]),eta1=d['eta'][i,j,0],eta2=d['eta'][i,j,1],eta3=d['eta'][i,j,2]))
        scores={k[3:]:v.astype(float)for k,v in d.items()if k.startswith('z::')};states=meta['states'];details={}
        figselect[name]=min(range(len(states)),key=lambda i:hashlib.sha256(('q-basin-audit-v1|'+states[i]).encode()).hexdigest())
        sets=state_cases(s,f)if meta['shared_eta']else {}
        rnpz={};rmodels=[]
        for model,z in scores.items():
            row,ps=evaluate(s,f,z,states,name,model);row['tier']=meta['tier']
            if model.startswith('fixed_source_common_eta__'):
                # The one-hot preference encodes a single choice, not predictions for the other eta.
                row['score_semantics']='choice_only; no calibrated probability or complete ordering'
                for k in list(row):
                    if k.startswith(('candidate_','state_AUC_','trial_','positive_mean','negative_mean','top2_','top4_')) or k=='mixed_states':row[k]=np.nan
                for pp in ps:
                    for k in ['within_state_auc','best_robust_rank','mean_robust_rank_percentile','top2_contains_robust','top4_contains_robust','top4_enrichment']:pp[k]=np.nan
            else:row['score_semantics']='predicted per-continuation success logit'
            metrics.append(row);details[model]=ps;perstate.extend(ps)
            if meta['shared_eta'] and not model.startswith('fixed_source_common_eta__') and model.split('__')[-1] in ['correct','none','wrong_base','wrong_controller','wrong_controller_alt','joint_state_context_shuffle','state_context_shuffle']:
                extra=same_eta(s,f,z)
                for kind,cs in sets.items():
                    good=score_cases(cs,z);rr=dict(cohort=name,model=model,reversal=kind,cases=len(cs),both_correct=int(good.sum()),joint_accuracy=float(good.mean())if len(good)else np.nan,
                                                  unique_states=len(set(cs[:,:2].ravel())),**extra)
                    reversals.append(rr);rnpz[f'{kind}::{model}']=good
                rmodels.append(model)
        if sets:
            for k,v in sets.items():rnpz['cases::'+k]=v
            np.savez_compressed(HERE/'reversals'/f'{name}.npz',**rnpz)
        # Paired controls preserve both states and candidate bank. No model seed pooling.
        for model,z in scores.items():
            bits=model.split('__');kind,seed,cond=bits
            if cond!='correct' and not(kind=='physical_context' and cond=='correct'):continue
            refs=[]
            if kind in ['full_context','physical_context','critic','raw_bypass','trunk_only','full_update','full_update_matched400','expanded_reference','old_trunk']:
                refs += [f'eta_only__{seed}__correct',f'eta_only__{seed}__none','global_train_eta__0__correct','eta_mle__0__correct',
                         'eta_only_kernel__frozen__correct','eta_only_mlp__frozen__correct']
            if kind.startswith('controller_cv_full'):refs += [f'controller_cv_eta_only__{seed}__correct',f'family_cv_eta_only__{seed}__correct','fixed_source_common_eta__0__correct']
            if kind.startswith('family_cv_') and kind!='family_cv_eta_only':refs += [f'family_cv_eta_only__{seed}__correct','fixed_source_common_eta__0__correct']
            refs += [f'{kind}__{seed}__{c}'for c in ['wrong_base','wrong_controller','wrong_controller_alt','state_shuffle','joint_state_context_shuffle','state_context_shuffle']]
            for ref in dict.fromkeys(refs):
                if ref in scores and ref!=model:comparisons.append(paired(s,f,z,scores[ref],details[model],details[ref],name,model,ref))
        eligible=(b==1).any(1);lo=(b==1).mean(1);hi=(b!=0).mean(1)
        random.append(dict(cohort=name,N=len(s),K=s.shape[1],oracle_eligible=int(eligible.sum()),random_expected_B15_count=float(lo.sum()),random_expected_B15_upper=float(hi.sum()),
                           random_eligible_rate=float(lo[eligible].mean())if eligible.any()else None,random_all_state_rate=float(lo.mean())))
    # Multiplicity families: cohort x reference-kind, retaining each training seed as a model, not independent N.
    groups=defaultdict(list)
    for i,r in enumerate(comparisons):groups[r['cohort'],r['reference'].split('__')[0],r['reference'].split('__')[-1]].append(i)
    for ids in groups.values():
        for i,p in zip(ids,holm([comparisons[j]['McNemar_p']for j in ids])):comparisons[i]['McNemar_p_Holm']=float(p)
    csvout('recognition_selection.csv',metrics);csvout('per_state.csv.gz',perstate);csvout('state_reversals.csv',reversals);csvout('candidate_counts.csv.gz',candidate_counts)
    csvout('paired_comparisons.csv',comparisons);csvout('random_baselines.csv',random);write(HERE/'figure_selection.json',figselect)
    write(HERE/'summary.json',dict(models=metrics,random=random,paired=comparisons,state_reversals=reversals))
    return metrics

def field_panels():
    out=[];casesall=[];recognition=[];per=[]
    for meta in read(HERE/'inputs/source_fields.json'):
        scene=meta['scene'];d=dict(np.load(HERE/'inputs'/f'source_field_{scene}.npz'));cs=['base','alt','second'][:meta['controllers']]
        s=np.stack([d[c+'_s']for c in cs]);f=np.stack([d[c+'_f']for c in cs]);b=membership(s,f);lo=s/16;hi=(16-f)/16
        groups=defaultdict(list)
        for i,r in enumerate(meta['rows']):groups[r['state_uid']].append(i)
        cases={x:[]for x in ['B15_swap','Q16_interval']};changes=[]
        for c0,c1 in combinations(range(len(cs)),2):
            for i in range(s.shape[1]):
                if b[c0,i]>=0 and b[c1,i]>=0 and b[c0,i]!=b[c1,i]:changes.append((c0,c1,i,int(b[c0,i]-b[c1,i])))
            for uid,ids in groups.items():
                for a,e in combinations(ids,2):
                    if min(b[c0,a],b[c0,e],b[c1,a],b[c1,e])>=0 and (b[c0,a]-b[c0,e])*(b[c1,a]-b[c1,e])==-1:
                        cases['B15_swap'].append((c0,c1,a,e,int(b[c0,a]-b[c0,e])))
                    signs=[1 if lo[c,a]>hi[c,e] else -1 if lo[c,e]>hi[c,a]else 0 for c in [c0,c1]]
                    if signs[0]*signs[1]==-1:cases['Q16_interval'].append((c0,c1,a,e,signs[0]))
        for kind,arr in cases.items():
            for c0,c1,a,e,sg in arr:casesall.append(dict(scene=scene,reversal=kind,state_uid=meta['rows'][a]['state_uid'],controller_a=cs[c0],controller_b=cs[c1],eta_a=a,eta_b=e,true_sign_a=sg))
        for info in meta['models']:
            key=info['model'];z=d['z::'+key][:len(cs)];p=expit(z);wins=sum((z[c0,i]-z[c1,i])*sg>0 for c0,c1,i,sg in changes);ties=sum(z[c0,i]==z[c1,i]for c0,c1,i,sg in changes)
            row=dict(scene=scene,model=key,source_VAL=True,heldout_controller=info['heldout_controller'],membership_changes=len(changes),
                     direction_correct=int(wins),direction_ties=int(ties),direction_accuracy_half_ties=(wins+.5*ties)/len(changes)if changes else np.nan)
            for kind,arr in cases.items():
                good=sum(np.sign(z[c0,a]-z[c0,e])==sg and np.sign(z[c1,a]-z[c1,e])==-sg for c0,c1,a,e,sg in arr)
                row[kind+'_cases']=len(arr);row[kind+'_both_correct']=int(good)
            out.append(row)
            # One physical state per row; controller groups kept separate. Pad only nonexistent candidates with -1e9.
            for c in range(len(cs)):
                for condition in ['correct','wrong_context']:
                    zz=z[c if condition=='correct'else(c+1)%len(cs)];n=len(groups);k=max(map(len,groups.values()))
                    ss=np.zeros((n,k));ff=np.zeros((n,k));zs=np.full((n,k),-1e9)
                    for j,ids in enumerate(groups.values()):ss[j,:len(ids)]=s[c,ids];ff[j,:len(ids)]=f[c,ids];zs[j,:len(ids)]=zz[ids]
                    rr,pp=evaluate(ss,ff,zs,list(groups),'source_VAL_'+scene+'_'+cs[c],key+'__'+condition)
                    # Padded cells must not be interpreted as empirical unresolved trials.
                    padded=n*k-len(meta['rows']);rr['padding_cells']=padded;rr['unknown_candidates']-=padded;rr['not_independent_TEST']=True
                    recognition.append(rr);per.extend(pp)
    csvout('field_response_reversals.csv',out);csvout('field_reversal_cases.csv',casesall);csvout('source_VAL_recognition_selection.csv',recognition);csvout('source_VAL_per_state.csv.gz',per)

def current_field_panels():
    rows=read(HERE/'inputs/current_source_field_rows.json');pred=dict(np.load(HERE/'inputs/current_source_fields.npz'))
    s=np.array([r['s16']for r in rows]);f=np.array([r['f16']for r in rows]);assert np.all(s+f<=16)
    b=membership(s,f);lo=s/16;hi=(16-f)/16;group=defaultdict(dict);out=[];truth=[]
    for i,r in enumerate(rows):
        group[r['scenario'],r['state_uid']].setdefault(r['controller_uid'],{})[r['eta_uid']]=i
    for scene in sorted({r['scenario']for r in rows}):
        changes=[];rev={k:[]for k in ['B15_swap','Q16_interval']};states=set()
        for (sc,uid),controllers in group.items():
            if sc!=scene:continue
            for ca,cb in combinations(controllers,2):
                aa=controllers[ca];bb=controllers[cb];etas=sorted(set(aa)&set(bb))
                for eta in etas:
                    a,c=aa[eta],bb[eta]
                    if b[a]>=0 and b[c]>=0 and b[a]!=b[c]:changes.append((a,c,b[a]-b[c]));states.add(uid)
                for ea,eb in combinations(etas,2):
                    a,d,c,e=aa[ea],aa[eb],bb[ea],bb[eb]
                    if min(b[a],b[d],b[c],b[e])>=0 and (b[a]-b[d])*(b[c]-b[e])==-1:rev['B15_swap'].append((a,d,c,e,b[a]-b[d]))
                    ta=1 if lo[a]>hi[d]else -1 if lo[d]>hi[a]else 0
                    tb=1 if lo[c]>hi[e]else -1 if lo[e]>hi[c]else 0
                    if ta*tb==-1:rev['Q16_interval'].append((a,d,c,e,ta))
        for model,z in pred.items():
            if not model.endswith('correct'):continue
            win=sum((z[a]-z[c])*sg>0 for a,c,sg in changes);tie=sum(z[a]==z[c]for a,c,sg in changes)
            row=dict(scene=scene,model=model,VAL_selected=True,membership_changes=len(changes),changed_states=len(states),direction_correct=int(win),direction_ties=int(tie),
                     direction_accuracy_half_ties=(win+.5*tie)/len(changes)if changes else np.nan)
            for kind,arr in rev.items():
                row[kind+'_cases']=len(arr);row[kind+'_both_correct']=int(sum(np.sign(z[a]-z[d])==sg and np.sign(z[c]-z[e])==-sg for a,d,c,e,sg in arr))
            out.append(row)
        truth.append(dict(scene=scene,changes=changes,reversals=rev))
    csvout('current_full_field_response_reversals.csv',out);write(HERE/'current_full_field_cases.json',truth)

def main():
    (HERE/'reversals').mkdir(exist_ok=True)
    rows=main_panels();field_panels();current_field_panels()
    print(json.dumps(dict(model_conditions=len(rows),new_rollouts=0,analysis='complete')))

if __name__=='__main__':main()
