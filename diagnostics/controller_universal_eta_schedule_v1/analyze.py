"""Prospective fixed schedule/oracle endpoint; no interim outcome access."""
from core import *
from collections import Counter,defaultdict
sys.path.insert(0,str(R9/'vendor'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import rankdata

def ci(x):return np.quantile(x,[.025,.975],axis=0)
def corr_rows(x,y):
    x=x-x.mean(-1,keepdims=True);y=y-y.mean(-1,keepdims=True);den=np.sqrt((x*x).sum(-1)*(y*y).sum(-1));return np.divide((x*y).sum(-1),den,out=np.full_like(den,np.nan),where=den>0)

def statistics(Y,U,available,destination=HERE):
    y=Y.mean(-1);u=U.mean(-1);lo=y.mean(-1);hi=(y+u).mean(-1);fixed=[0,1,2];f=np.where(available[:,fixed],lo[:,fixed],np.nan);fh=np.where(available[:,fixed],hi[:,fixed],np.nan);oracle=np.nanmax(f,axis=1);oraclehi=np.nanmax(fh,axis=1)
    rng=np.random.default_rng(202610130100);draw=rng.integers(0,64,(10000,64));weights=np.array([np.bincount(z,minlength=64)/64 for z in draw]);blo=np.einsum('bs,cas->bca',weights,y);bhi=np.einsum('bs,cas->bca',weights,y+u);bfix=np.where(available[None,:,fixed],blo[:,:,fixed],np.nan);bfixhi=np.where(available[None,:,fixed],bhi[:,:,fixed],np.nan);boracle=np.nanmax(bfix,axis=2);boraclehi=np.nanmax(bfixhi,axis=2)
    groups=dict(all=list(range(14)),dev=[CONTROLLERS.index(n) for n in DEV],held_out=[i for i,n in enumerate(CONTROLLERS) if n not in DEV]);resamples={key:np.asarray(group)[rng.integers(0,len(group),(10000,len(group)))] for key,group in groups.items()};drawids=np.arange(10000)[:,None];out={};percontroller={n:dict(group='dev' if n in DEV else 'held_out',arms={},oracle_point=float(oracle[i]),oracle_unknown_bounds=[float(oracle[i]),float(oraclehi[i])],oracle_point_arm=ARMS[fixed[int(np.nanargmax(f[i]))]],comparisons={}) for i,n in enumerate(CONTROLLERS)}
    for i,name in enumerate(CONTROLLERS):
        for ai,arm in enumerate(ARMS):
            if not available[i,ai]:continue
            percontroller[name]['arms'][arm]=dict(success_lower=float(lo[i,ai]),success_upper=float(hi[i,ai]),success_count=int(Y[i,ai].sum()),unknown_count=int(U[i,ai].sum()),CI_point=ci(blo[:,i,ai]).tolist(),CI_unknown_envelope=[float(ci(blo[:,i,ai])[0]),float(ci(bhi[:,i,ai])[1])])
    for arm in ('S500','S300','S400'):
        a=ARMS.index(arm);point=lo[:,a]-oracle;gaplo=lo[:,a]-oraclehi;gaphi=hi[:,a]-oracle;source=lo[:,a]-lo[:,0];srclo=lo[:,a]-hi[:,0];srchi=hi[:,a]-lo[:,0];bd=blo[:,:,a]-boracle;bdlo=blo[:,:,a]-boraclehi;bdhi=bhi[:,:,a]-boracle
        for i,name in enumerate(CONTROLLERS):percontroller[name]['comparisons'][arm]=dict(gain_vs_oracle_point=float(point[i]),gain_unknown_bounds=[float(gaplo[i]),float(gaphi[i])],CI_point=ci(bd[:,i]).tolist(),CI_unknown_envelope=[float(ci(bdlo[:,i])[0]),float(ci(bdhi[:,i])[1])],gain_vs_F10_point=float(source[i]),gain_vs_F10_bounds=[float(srclo[i]),float(srchi[i])])
        out[arm]={}
        for group,indices in groups.items():
            idx=np.array(indices);bs=bd[drawids,resamples[group]].mean(1);bsl=bdlo[drawids,resamples[group]].mean(1);bsh=bdhi[drawids,resamples[group]].mean(1);pointCI=ci(bs);lCI=ci(bsl);hCI=ci(bsh);e3=int((source[idx]<-.10).sum());worst=[CONTROLLERS[i] for i in idx if point[i]<-.05];catastrophic=[CONTROLLERS[i] for i in idx if source[i]<-.10];need_e3=group!='held_out';pointpass=bool(pointCI[0]>0 and not worst and (not need_e3 or e3==0));certainpass=bool(lCI[0]>0 and np.all(gaplo[idx]>=-.05) and (not need_e3 or np.all(srclo[idx]>=-.10)));possiblepass=bool(hCI[0]>0 and np.all(gaphi[idx]>=-.05) and (not need_e3 or np.all(srchi[idx]>=-.10)));status='PASS' if certainpass else 'FAIL' if not possiblepass else 'INCONCLUSIVE'
            out[arm][group]=dict(controller_count=len(idx),controllers=[CONTROLLERS[i] for i in idx],mean_gain=float(point[idx].mean()),mean_gain_unknown_bounds=[float(gaplo[idx].mean()),float(gaphi[idx].mean())],hierarchical_CI_point=pointCI.tolist(),hierarchical_CI_unknown_envelope=[float(lCI[0]),float(hCI[1])],lower95_feasible_range=[float(lCI[0]),float(hCI[0])],point_positive_lower95=bool(pointCI[0]>0),point_no_controller_deficit_gt5pp=not bool(worst),point_E3=e3,E3_required=need_e3,point_gate_pass=pointpass,status=status,certain_pass_unknown=certainpass,possible_pass_unknown=possiblepass,deficit_controllers_gt5pp=worst,catastrophic_vs_F10=catastrophic,robust_deficit_controllers_gt5pp=[CONTROLLERS[i] for i in idx if gaphi[i]<-.05],uncertain_deficit_controllers=[CONTROLLERS[i] for i in idx if gaplo[i]<-.05<=gaphi[i]])
    main=out['S500'];univers=main['all']['status']=='PASS' and main['held_out']['status']=='PASS'
    if univers:hypothesis='SUPPORTED';mapping='G_universal and independent held-out gate passed'
    elif main['dev']['status']=='FAIL' and main['held_out']['status']=='FAIL':hypothesis='REJECTED';mapping='Both dev and held-out groups fail the registered gate'
    elif main['dev']['status']=='PASS' and main['held_out']['status']=='FAIL':
        hypothesis='REJECTED' if main['held_out']['robust_deficit_controllers_gt5pp'] else 'INCONCLUSIVE';mapping='Dev passes, held-out does not; partial-overfitting gate pattern (wide-CI failure alone is not proof of mechanism)'
    else:hypothesis='INCONCLUSIVE';mapping='Unmapped mixed group outcome or unknown-flippable gate; no forced conclusion'
    result=dict(H_schedule=hypothesis,mapping=mapping,primary='S500',G_universal=univers,group_metrics=out,controllers=percontroller,bootstrap=dict(draws=10000,seed=202610130100,state_cluster64=True,controller_resampled=True,oracle_reselected_each_draw=True),scope='Prospective preregistered fresh-state endpoint; explanatory conclusions DEVELOPMENT; Ring/FF/H700/14Flow, Q4 not B15',DEVELOPMENT=True)
    # Secondary gain-regime association, fixed independent bootstrap stream.
    rng2=np.random.default_rng(202610130101);wd=np.array([np.bincount(z,minlength=64)/64 for z in rng2.integers(0,64,(10000,64))]);yy=np.einsum('bs,cas->bca',wd,y);oo=np.nanmax(np.where(available[None,:,fixed],yy[:,:,fixed],np.nan),axis=2);ix=rng2.integers(0,14,(10000,14));rows=np.arange(10000)[:,None];xx=(yy[:,:,0]-yy[:,:,1])[rows,ix];gg=(yy[:,:,5]-oo)[rows,ix];pb=corr_rows(xx,gg);sb=corr_rows(rankdata(xx,axis=1),rankdata(gg,axis=1));x=lo[:,0]-lo[:,1];g=lo[:,5]-oracle
    result['secondary_regime_association']=dict(Pearson=float(corr_rows(x[None,:],g[None,:])[0]),Spearman=float(corr_rows(rankdata(x)[None,:],rankdata(g)[None,:])[0]),Pearson_CI=np.nanquantile(pb,[.025,.975]).tolist(),Spearman_CI=np.nanquantile(sb,[.025,.975]).tolist(),undefined_Pearson_draws=int(np.isnan(pb).sum()),undefined_Spearman_draws=int(np.isnan(sb).sum()),bootstrap_seed=202610130101,DEVELOPMENT=True,unknown_note='Association uses registered lower-bound point rates; unknown counts/bounds retained in primary gates')
    destination.mkdir(parents=True,exist_ok=True)
    for group,indices in [('all',list(range(14))),('held_out',groups['held_out'])]:
        fig,ax=plt.subplots(figsize=(17,6));xx=np.arange(len(indices));width=.13;colors=['C0','C1','C2','C3','C4','C5']
        for ai,arm in enumerate(ARMS):
            values=np.array([lo[i,ai] if available[i,ai] else np.nan for i in indices]);lc=np.array([np.quantile(blo[:,i,ai],.025) for i in indices]);hc=np.array([np.quantile(bhi[:,i,ai],.975) for i in indices]);ax.bar(xx+(ai-2.5)*width,values,width,label=arm,color=colors[ai]);ax.errorbar(xx+(ai-2.5)*width,values,yerr=np.maximum(np.r_[values[None,:]-lc[None,:],hc[None,:]-values[None,:]],0),fmt='none',ecolor='black',lw=.7,capsize=1)
        ax.set_xticks(xx,[CONTROLLERS[i].replace('base_u_','').replace('_macflow','') for i in indices],rotation=35,ha='right');ax.set(ylabel='Success lower bound (unknown shown in CI envelope)',ylim=(0,1.05),title=f'Ring universal eta schedule: {group}; prospective endpoint, interpretations DEVELOPMENT');ax.legend(ncol=6);fig.tight_layout();fig.savefig(destination/f'ARM_COMPARISON_{group}.png',dpi=170);plt.close(fig)
    if destination!=HERE:
        result['software_verification_only']=True;result['scope']='Synthetic software fixture, not rollout evidence or prospective endpoint'
    write(destination/'STATISTICAL_RESULTS.json',result);return result

def main():
    prereg();gate();assert (HERE/'ENDPOINT_OPENING.receipt.json').exists();assert read(HERE/'main_POSTFLIGHT.json')['status']=='PASS'
    Y=np.zeros((14,6,64,4));U=Y.copy();available=np.zeros((14,6),bool);records=[];groups=defaultdict(list);lookup={}
    for c in read(HERE/'main_CASES.json'):
        r=read(HERE/'main/raw'/f"{c['rollout_uid']}.json");assert r['case']==c;i=CONTROLLERS.index(c['base_controller_name']);a=ARMS.index(c['arm']);s=c['state_index'];seed=c['seed'];Y[i,a,s,seed]=r['success'];U[i,a,s,seed]=r['numerical_failure'];available[i,a]=True;records.append(r);groups[c['base_controller_name'],c['arm']].append(r);lookup[c['base_controller_name'],c['arm'],s,seed]=r
    assert len(records)==18688 and available.sum()==73;result=statistics(Y,U,available)
    failures={}
    for (name,arm),rows in groups.items():
        counts=Counter(r['outcome'] for r in rows);failuresteps={o:[(r['error']['step'] if r['error'] else r['summary'].get('first_success_step') or r['episode_length']) for r in rows if r['outcome']==o] for o in counts}
        transitions={}
        if arm.startswith('S'):
            for fixed in ('F10','F13'):
                transitions[fixed]=dict(Counter(f"{lookup[name,fixed,r['case']['state_index'],r['case']['seed']]['outcome']}->{r['outcome']}" for r in rows))
        failures[f'{name}/{arm}']=dict(outcomes=dict(counts),occurrence_steps=failuresteps,paired_outcome_transitions=transitions)
    result.update(outcome_summaries=failures,total_main=18688,known_valid=int((1-U[available]).sum()),unknown=int(U.sum()),endpoint_opening=read(HERE/'ENDPOINT_OPENING.receipt.json'),active_preregistration=read(HERE/'PREREGISTRATION_V2.receipt.json'),prospective_endpoint=True)
    np.savez_compressed(HERE/'LABELS.npz',success=Y,unknown=U,available=available,controllers=np.array(CONTROLLERS),arms=np.array(ARMS));write(HERE/'NUMERICAL_RESULTS.json',result);print(json.dumps(dict(H_schedule=result['H_schedule'],group_status={g:o['status'] for g,o in result['group_metrics']['S500'].items()},unknown=result['unknown'])))
if __name__=='__main__':main()
