"""Frozen round10 phase analysis. No rollout imports and no database access."""
import os,sys,json,hashlib,csv,datetime
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent.parent
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
for k,v in [('MPLCONFIGDIR','mplconfig'),('XDG_CACHE_HOME','cache'),('TMPDIR','tmp')]:
    p=HERE/v;p.mkdir(exist_ok=True);os.environ[k]=str(p)
def guard(event,args):
    if event=='open':
        path,mode,flags=args
        if isinstance(path,(str,bytes,os.PathLike)):
            writing=(isinstance(mode,str) and any(x in mode for x in 'wax+')) or (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
            if writing and not Path(path).resolve().is_relative_to(HERE):raise PermissionError(f'External write blocked: {path}')
    if event in ('os.remove','os.mkdir','os.rmdir','os.rename'):
        for path in args[:2] if event=='os.rename' else args[:1]:
            if isinstance(path,(str,bytes,os.PathLike)) and not Path(path).resolve().is_relative_to(HERE):raise PermissionError(f'External mutation blocked: {path}')
sys.addaudithook(guard)
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0,str(ROOT/'diagnostics/controller_eta_switch_time_v1/vendor'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R10=ROOT/'diagnostics/controller_universal_eta_schedule_v1'; R7=ROOT/'diagnostics/controller_behavior_repr_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def write(name,x):(HERE/name).write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def ci(x):return np.quantile(x,[.025,.975],axis=0)
def csvwrite(name,rows):
    with (HERE/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    reg=read(HERE/'PREREGISTRATION.receipt.json');assert sha(HERE/'PREREGISTRATION.md')==reg['sha256'];assert (HERE/'START_GATE_AUTHORIZATION.receipt.json').exists()
    inventory=read(HERE/'HISTORICAL_ASSET_INDEX.json')
    for row in inventory['assets']:assert sha(ROOT/row['path'])==row['sha256'],row['path']
    assert read(R10/'main_POSTFLIGHT.json')['status']=='PASS'
    z=np.load(R10/'LABELS.npz');Y=z['success'];U=z['unknown'];av=z['available'].astype(bool);names=z['controllers'].tolist();arms=z['arms'].tolist()
    assert Y.shape==U.shape==(14,6,64,4);assert av.sum()==73;assert U.sum()==131
    assert np.all((Y==0)|(Y==1)) and np.all((U==0)|(U==1)) and np.all(Y+U<=1)
    assert arms==['F10','F13','Freg','S300','S400','S500'];assert av[:,[0,1,3,4,5]].all()
    cases=read(R10/'main_CASES.json');assert len(cases)==int(av.sum()*256)==18688
    slots=set()
    for c in cases:
        key=(names.index(c['base_controller_name']),arms.index(c['arm']),c['state_index'],c['seed']);assert key not in slots;slots.add(key);assert av[key[:2]]
    y=Y.mean(-1);u=U.mean(-1);lo=y.mean(-1);hi=(y+u).mean(-1)
    rng=np.random.default_rng(reg['bootstrap_seed']);draw=rng.integers(0,64,(5000,64));weights=np.array([np.bincount(a,minlength=64)/64 for a in draw]);blo=np.einsum('bs,cas->bca',weights,y);bhi=np.einsum('bs,cas->bca',weights,y+u)
    effects={};table=[];rate_rows=[]
    for i,n in enumerate(names):
        for a,arm in enumerate(arms):
            if av[i,a]:rate_rows.append(dict(controller=n,arm=arm,success_lower=lo[i,a],success_upper=hi[i,a],success_count=int(Y[i,a].sum()),unknown_count=int(U[i,a].sum())))
        effects[n]={}
        for s in [5,3,4]:
            for label,a,b in [('E',s,0),('L',s,1)]+([('N',1,0)] if s==5 else []):
                point=lo[i,a]-lo[i,b];bounds=[lo[i,a]-hi[i,b],hi[i,a]-lo[i,b]];bc=ci(blo[:,i,a]-blo[:,i,b]);env=[ci(blo[:,i,a]-bhi[:,i,b])[0],ci(bhi[:,i,a]-blo[:,i,b])[1]]
                key=label if label=='N' else f'{label}_{arms[s]}'
                effects[n][key]=dict(point=point,unknown_bounds=bounds,CI95_point=bc,CI95_unknown_envelope=env)
                table.append(dict(controller=n,effect=label,schedule=arms[s] if label!='N' else 'fixed',point_pp=point*100,unknown_lower_pp=bounds[0]*100,unknown_upper_pp=bounds[1]*100,CI95_lower_pp=bc[0]*100,CI95_upper_pp=bc[1]*100,CI95_envelope_lower_pp=env[0]*100,CI95_envelope_upper_pp=env[1]*100))
    E=lo[:,5]-lo[:,0];L=lo[:,5]-lo[:,1];N=lo[:,1]-lo[:,0];truth=np.c_[E,L,N];assert np.max(abs(E-L-N))<1e-14
    p1a=int(((N<0)&(E>=.1)).sum());neg=int((N<0).sum());both=int(((E>=.1)&(L>=.1)).sum());p1=bool(p1a>=neg/2 and both>=5);p2=bool((E>0).sum()>=11 and (L>0).sum()>=11)
    P1=dict(pass_prediction=p1,N_negative=neg,negative_with_E_ge10pp=p1a,both_E_L_ge10pp=both,thresholds=dict(fraction=.5,both_count=5));P2=dict(pass_prediction=p2,E_positive=int((E>0).sum()),L_positive=int((L>0).sum()),N_signs=dict(negative=neg,zero=int((N==0).sum()),positive=int((N>0).sum())))
    # Feasible lower/upper count ranges provide unknown sensitivity without changing point rules.
    P2['unknown_compatible_positive_count_ranges']={label:[sum(effects[n][f'{label}_S500']['unknown_bounds'][0]>0 for n in names),sum(effects[n][f'{label}_S500']['unknown_bounds'][1]>0 for n in names)] for label in ['E','L']}
    oracle=np.max(np.where(av,lo,-np.inf),axis=1)
    def evaluate(choices):
        rows=[]
        for i,a in enumerate(choices):
            other=[j for j in range(6) if av[i,j] and j!=a];bounds=[max(0.,max([lo[i,j]-hi[i,a] for j in other],default=0)),max(0.,max([hi[i,j]-lo[i,a] for j in other],default=0))]
            rows.append(dict(controller=names[i],selected_arm=arms[a],success_lower=lo[i,a],success_upper=hi[i,a],oracle_arm=arms[int(np.argmax(np.where(av[i],lo[i],-np.inf)))],oracle_success_lower=oracle[i],regret=oracle[i]-lo[i,a],regret_unknown_bounds=bounds))
        r=np.array([x['regret'] for x in rows]);return dict(mean_regret=float(r.mean()),max_regret=float(r.max()),mean_unknown_bounds=np.mean([x['regret_unknown_bounds'] for x in rows],axis=0),max_unknown_bounds=np.max([x['regret_unknown_bounds'] for x in rows],axis=0),folds=rows)
    selections={};menus={'fixed':[0,1],'expanded':[0,1,3,4,5]}
    for label,menu in menus.items():
        choices=[];means=[]
        for i in range(14):
            source=[j for j in range(14) if j!=i];mean=lo[source][:,menu].mean(0);choices.append(menu[int(mean.argmax())]);means.append(dict(zip([arms[a] for a in menu],mean)))
        selections[label]=evaluate(choices)
        for row,m in zip(selections[label]['folds'],means):row['source_mean_lower']=m
    P3=dict(**selections,pass_prediction=bool(selections['expanded']['mean_regret']<=selections['fixed']['mean_regret']/3 and selections['expanded']['max_regret']<=.05))
    f=np.load(R7/'FEATURES.npz');fnames=f['controller_names'].tolist();ids=[fnames.index(n) for n in names];keep=np.ones(160,bool);keep[105:110]=False
    features={'R3-PCA2':f['R3_96'][ids].astype(float),'R5-PCA2':f['R5'][ids][:,keep].astype(float)};assert features['R5-PCA2'].shape==(14,155)
    P4={};prediction_rows=[];fold_params=[]
    for rep,raw in features.items():
        assert np.isfinite(raw).all();pred=np.zeros((14,3));baseline=np.zeros((14,3));choices=[];near_names=[]
        for i in range(14):
            source=[j for j in range(14) if j!=i];mean=raw[source].mean(0);sd=raw[source].std(0);scale=np.where(sd<1e-6,1.,sd);standard=(raw-mean)/scale;_,sing,vt=np.linalg.svd(standard[source],full_matrices=False)
            for v in vt:
                if v[np.argmax(abs(v))]<0:v*=-1
            pc=vt[:2];coords=standard@pc.T;x=coords[source];h=coords[i];mu=x.mean(0);ym=truth[source].mean(0);xc=x-mu;coef=np.linalg.solve(xc.T@xc+np.eye(2),xc.T@(truth[source]-ym));pred[i]=ym+(h-mu)@coef;baseline[i]=ym
            near=source[int(np.linalg.norm(x-h,axis=1).argmin())];near_names.append(names[near]);menu=menus['expanded'];choices.append(menu[int(lo[near,menu].argmax())])
            fold_params.append(dict(representation=rep,held=names[i],sources=[names[j] for j in source],mean=mean,scale=scale,components=pc,ridge_coefficients=coef,ridge_intercept=ym-mu@coef,nearest_controller=names[near],explained_variance_ratio=(sing[:2]**2/(sing**2).sum())))
        assert np.max(abs(pred[:,0]-pred[:,1]-pred[:,2]))<1e-12
        metrics={}
        for k,label in enumerate(['E','L','N']):
            metrics[label]=dict(Spearman=float(spearmanr(pred[:,k],truth[:,k]).statistic),sign_accuracy=float(np.mean(np.sign(pred[:,k])==np.sign(truth[:,k]))),source_mean_Spearman=float(spearmanr(baseline[:,k],truth[:,k]).statistic),source_mean_sign_accuracy=float(np.mean(np.sign(baseline[:,k])==np.sign(truth[:,k]))),rmse=float(np.sqrt(np.mean((pred[:,k]-truth[:,k])**2))),source_mean_rmse=float(np.sqrt(np.mean((baseline[:,k]-truth[:,k])**2))))
            for i,n in enumerate(names):prediction_rows.append(dict(representation=rep,controller=n,effect=label,truth_pp=truth[i,k]*100,prediction_pp=pred[i,k]*100,source_mean_prediction_pp=baseline[i,k]*100))
        nn=evaluate(choices)
        for row,near in zip(nn['folds'],near_names):row['nearest_controller']=near
        nn['mean_improvement_vs_source_average']=selections['expanded']['mean_regret']-nn['mean_regret'];nn['lower_mean_regret_than_source_average']=nn['mean_regret']<selections['expanded']['mean_regret']
        P4[rep]=dict(ridge_metrics=metrics,nearest_neighbor=nn)
    exceptions=[names[i] for i in range(14) if not(E[i]>=.1 and L[i]>=.1)];historical=read(R10/'NUMERICAL_RESULTS.json')['outcome_summaries'];failure_summaries={n:{arm:historical[f'{n}/{arm}'] for a,arm in enumerate(arms) if av[names.index(n),a]} for n in exceptions}
    result=dict(status='COMPLETE',evidence_level='DEVELOPMENT',H_compress='SUPPORTED' if p1 and p2 else 'PARTIAL' if p1 or p2 else 'REJECTED',P1=P1,P2=P2,P3=P3,P4=P4,effects=effects,exception_failure_summaries=failure_summaries,bootstrap=dict(seed=reg['bootstrap_seed'],draws=5000,cluster='64 states, retaining four seeds, shared draws across controllers and arms'),data=dict(controllers=14,states=64,seeds=[0,1,2,3],observations=18688,unknown=131,available_arms=73),learning_interpretation='P4 reports descriptive metrics without an invented predictive significance threshold; sign accuracy must be compared with source-mean reference. No general claim of unlearnability follows from fourteen controllers.',scope='Frozen Ring/FF/H700 outcomes, fourteen existing controllers; no rollout, DB write or modification of historical assets. Numerical-completion start gate authorized by user; round10 video status remains unchanged.')
    write('results.json',result);write('FOLD_PREPROCESSING.json',fold_params);csvwrite('E_L_N.csv',table);csvwrite('ARM_RATES.csv',rate_rows);csvwrite('RIDGE_PREDICTIONS.csv',prediction_rows)
    for key in menus:csvwrite(f'P3_{key}_folds.csv',[{k:v for k,v in r.items() if k not in ['source_mean_lower','regret_unknown_bounds']} for r in selections[key]['folds']])
    fig,ax=plt.subplots(figsize=(10,8));colors=np.where(N<0,'#2477b3',np.where(N>0,'#d45128','#555555'))
    for i,n in enumerate(names):
        ec=effects[n]['E_S500']['CI95_point'];lc=effects[n]['L_S500']['CI95_point'];ax.errorbar(E[i]*100,L[i]*100,xerr=[[max(0,E[i]-ec[0])*100],[max(0,ec[1]-E[i])*100]],yerr=[[max(0,L[i]-lc[0])*100],[max(0,lc[1]-L[i])*100]],fmt='o',color=colors[i],alpha=.8,capsize=2);ax.annotate(str(i+1),(E[i]*100,L[i]*100),xytext=(5,5),textcoords='offset points')
    lim=[min(-10,float(min(E.min(),L.min())*100)-5),max(float(max(E.max(),L.max())*100)+8,75)];ax.plot(lim,lim,'k--',lw=1,label='E=L (N=0)');ax.axvline(0,color='gray',lw=.6);ax.axhline(0,color='gray',lw=.6);ax.axvline(10,color='gray',ls=':',lw=.6);ax.axhline(10,color='gray',ls=':',lw=.6)
    for color,label in [('#2477b3','N<0: fixed eta10 preference'),('#d45128','N>0: fixed eta13 preference')]:ax.scatter([],[],color=color,label=label)
    ax.set(xlabel='E = S500 - F10 (pp)',ylabel='L = S500 - F13 (pp)',title='Phase decomposition (DEVELOPMENT); 95% state-cluster point CI',xlim=lim,ylim=lim);ax.legend(loc='upper left');fig.tight_layout();fig.savefig(HERE/'E_L_scatter.png',dpi=180);fig.savefig(HERE/'E_L_scatter.pdf');plt.close(fig)
    csvwrite('SCATTER_CONTROLLER_INDEX.csv',[dict(index=i+1,controller=n) for i,n in enumerate(names)])
    # Verify frozen inputs after the full run; write guard also prevents external output.
    verified=[]
    for row in inventory['assets']:
        assert sha(ROOT/row['path'])==row['sha256'];verified.append(row['path'])
    write('ANALYSIS_AUDIT.json',dict(status='PASS',preregistration_verified=True,input_hashes_unchanged=verified,paired_effect_identity_max_error=float(np.max(abs(E-L-N))),case_slots_unique=len(slots),unknowns_preserved=int(U.sum()),source_only_PCA=True,no_target_labels_in_action_selection=True,no_rollout_or_DB_imports=True,write_scope=str(HERE)))
    print(json.dumps(clean(dict(H_compress=result['H_compress'],P1=P1,P2=P2,P3={k:v for k,v in P3.items() if k=='pass_prediction' or k in []},regrets={k:[selections[k]['mean_regret'],selections[k]['max_regret']] for k in menus},P4={r:{'ridge':v['ridge_metrics'],'nn_mean':v['nearest_neighbor']['mean_regret'],'nn_max':v['nearest_neighbor']['max_regret']} for r,v in P4.items()})),indent=2))
if __name__=='__main__':main()
