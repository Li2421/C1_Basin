"""Frozen-preregistration state-cluster statistics and trajectory descriptions."""
from core import *
from collections import Counter
sys.path.insert(0,str(HERE/'vendor'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
GRID=[0]+TIMES+[700]

def load():
    prereg();gate();assert read(HERE/'main_POSTFLIGHT.json')['status']=='PASS'
    old={};historical=read(HERE/'HISTORICAL_RAW_INDEX.json')
    for z in historical:
        assert sha(z['path'])==z['sha256'],'Historical evidence changed'
        c=z['case']
        if c['controller_name'] in CONTROLLERS and c['seed']<4:
            old[c['controller_name'],c['state_index'],c['eta_index'],c['seed']]=read(z['path'])
    Y=np.zeros((4,2,7,64,4));U=Y.copy();records=[]
    for ci,name in enumerate(CONTROLLERS):
        for di,(a,b) in enumerate(((10,13),(13,10))):
            for ti,t in enumerate(GRID):
                if t not in (0,700):continue
                for s in range(64):
                    for seed in range(4):
                        r=old[name,s,b if t==0 else a,seed];Y[ci,di,ti,s,seed]=r['success'];U[ci,di,ti,s,seed]=r['numerical_failure']
    for c in read(HERE/'main_CASES.json'):
        r=read(HERE/'main/raw'/f"{c['rollout_uid']}.json");assert r['case']==c
        ci=CONTROLLERS.index(c['base_controller_name']);di=0 if c['eta_a_index']==10 else 1;ti=GRID.index(c['switch_step']);s=c['state_index'];seed=c['seed']
        Y[ci,di,ti,s,seed]=r['success'];U[ci,di,ti,s,seed]=r['numerical_failure'];records.append(r)
    return Y,U,records

def interval(x):return np.quantile(x,[.025,.975],axis=0).tolist()
def ratio_bounds(nlo,nhi,dlo,dhi):
    if dlo<=0:return [None,None]
    x=[nlo/dlo,nlo/dhi,nhi/dlo,nhi/dhi];return [min(x),max(x)]

def curves(Y,U,destination=HERE):
    rng=np.random.default_rng(202610130901);draw=rng.integers(0,64,(5000,64));weights=np.array([np.bincount(x,minlength=64)/64 for x in draw]);y=Y.mean(-1);u=U.mean(-1);lo=y.mean(-1);hi=(y+u).mean(-1)
    blo=np.einsum('bs,cdts->bcdt',weights,y);bhi=np.einsum('bs,cdts->bcdt',weights,y+u)
    out={};effects=[];boot_effects=[];half_draws=[];retention_draws=[];rescue_draws=[];valid_draws=[];fig,axs=plt.subplots(2,2,figsize=(11,8),sharex=True)
    for ci,name in enumerate(CONTROLLERS):
        lines=[];ax=axs.flat[ci];base10=y[ci,0,-1];base13=y[ci,0,0];unk10=u[ci,0,-1];unk13=u[ci,0,0];D=float((base13-base10).mean());Dlo=float((base13-base10-unk10).mean());Dhi=float((base13+unk13-base10).mean());sign=1 if D>0 else -1
        for di,direction in enumerate(('10to13','13to10')):
            for ti,t in enumerate(GRID):
                row=dict(direction=direction,t=t,success_bounds=[lo[ci,di,ti],hi[ci,di,ti]],unknown_count=int(U[ci,di,ti].sum()),success_CI_envelope=[np.quantile(blo[:,ci,di,ti],.025),np.quantile(bhi[:,ci,di,ti],.975)],paired={})
                for eta,baseline,unk in ((10,base10,unk10),(13,base13,unk13)):
                    a=y[ci,di,ti]-baseline;al=a-unk;ah=a+u[ci,di,ti]
                    same=(t==0 and eta==(13 if di==0 else 10)) or (t==700 and eta==(10 if di==0 else 13))
                    if same:al=np.zeros(64);ah=np.zeros(64);a=np.zeros(64)
                    bl=weights@al;bh=weights@ah
                    row['paired'][str(eta)]=dict(point=float(a.mean()),unknown_bounds=[float(al.mean()),float(ah.mean())],CI_envelope=[float(np.quantile(bl,.025)),float(np.quantile(bh,.975))])
                    if t in TIMES:effects.append((row,str(eta),float(al.mean()),float(ah.mean())));boot_effects.append((bl,bh))
                lines.append(row)
            ax.plot(GRID,lo[ci,di],marker='o',label=direction)
            ax.fill_between(GRID,np.quantile(blo[:,ci,di],.025,axis=0),np.quantile(bhi[:,ci,di],.975,axis=0),alpha=.17)
        ax.axhline(base10.mean(),color='gray',ls=':',label='pure eta10');ax.axhline(base13.mean(),color='gray',ls='--',label='pure eta13');ax.set(title=name.replace('_macflow',''),xlabel='Switch action index',ylabel='Success probability',ylim=(0,1));ax.legend(fontsize=8)
        nd=(y[ci,0]-base10)*sign;nl=nd-(unk10 if sign>0 else u[ci,0]);nh=nd+(u[ci,0] if sign>0 else unk10)
        denom=abs(D);dl=Dlo if sign>0 else -Dhi;dh=Dhi if sign>0 else -Dlo
        ratios=nd.mean(-1)/denom if denom else np.full(7,np.nan);bounds=[ratio_bounds(nl[i].mean(),nh[i].mean(),dl,dh) for i in range(7)]
        # Identical pure eta10 endpoint cancels, eta13 endpoint ratio is exactly one.
        bounds[0]=[1.,1.] if dl>0 else [None,None];bounds[-1]=[0.,0.] if dl>0 else [None,None]
        passing=[t for t,r in zip(GRID,ratios) if r>=.5];half=max(passing) if passing else None
        certain=[t for t,b in zip(GRID,bounds) if b[0] is not None and b[0]>=.5];possible=[t for t,b in zip(GRID,bounds) if b[1] is None or b[1]>=.5]
        bd=weights@((base13-base10)*sign);bn=np.einsum('bs,ts->bt',weights,nd);good=bd>0;br=np.divide(bn,bd[:,None],out=np.full_like(bn,np.nan),where=good[:,None]);halfboot=np.max(np.where(br>=.5,np.array(GRID)[None,:],-1),axis=1);halfboot[~good]=-1
        rescue=(y[ci,1,GRID.index(200)]-base13)*(-1 if sign>0 else 1);rl=rescue-(u[ci,1,GRID.index(200)] if sign>0 else unk13);rh=rescue+(unk13 if sign>0 else u[ci,1,GRID.index(200)])
        half_draws.append(halfboot);retention_draws.append(br[:,GRID.index(300)]);rescue_draws.append(np.divide(weights@rescue,bd,out=np.full(5000,np.nan),where=good));valid_draws.append(good)
        out[name]=dict(curves=lines,D_point=D,D_unknown_bounds=[Dlo,Dhi],effect_sign=sign,retention_ratios=ratios.tolist(),retention_unknown_bounds=bounds,t_half=half,t_half_unknown_range=[max(certain) if certain else None,max(possible) if possible else None],passing_times=passing,t_half_bootstrap_counts=dict(Counter(map(str,halfboot))),bootstrap_nonpositive_denominator=int((~good).sum()),reverse_t200_harm_rescue_ratio=float(rescue.mean()/denom) if denom else None,reverse_t200_harm_rescue_bounds=ratio_bounds(rl.mean(),rh.mean(),dl,dh))
    # Simultaneous max-T intervals for the full preregistered family of 80 paired comparisons.
    bl=np.array([b[0] for b in boot_effects]).T;bh=np.array([b[1] for b in boot_effects]).T;ml=np.array([x[2] for x in effects]);mh=np.array([x[3] for x in effects]);sl=bl.std(0);sh=bh.std(0)
    z=np.maximum(np.abs((bl-ml)/np.maximum(sl,1e-12)),np.abs((bh-mh)/np.maximum(sh,1e-12)));q=float(np.quantile(z.max(1),.95))
    for i,(row,eta,l,h) in enumerate(effects):row['paired'][eta]['simultaneous_CI_envelope']=[float(l-q*sl[i]),float(h+q*sh[i])]
    destination.mkdir(parents=True,exist_ok=True);fig.suptitle('DEVELOPMENT: state-bootstrap 95% bands include unknown envelopes');fig.tight_layout();fig.savefig(destination/'SUCCESS_CURVES.png',dpi=180);plt.close(fig)
    def check(lo,hi,threshold,ge=True):
        if lo is None or hi is None:return None
        if ge:return True if lo>=threshold else False if hi<threshold else None
        return True if hi<=threshold else False if lo>threshold else None
    late=[];early=[]
    for name in CONTROLLERS[:2]:
        o=out[name];l,h=o['t_half_unknown_range'];late.append(check(l,h,300));early.append(check(l,h,100,False))
    for name in CONTROLLERS[2:]:
        o=out[name];late.append(check(*o['retention_unknown_bounds'][GRID.index(300)],.5));late.append(check(*o['reverse_t200_harm_rescue_bounds'],.5))
    # Preregistration explicitly calls mixed/intermediate timing INCONCLUSIVE.
    classify=lambda xs:'SUPPORTED' if all(x is True for x in xs) else 'INCONCLUSIVE'
    valid=np.logical_and.reduce(valid_draws);late_boot=(half_draws[0]>=300)&(half_draws[1]>=300)&(retention_draws[2]>=.5)&(retention_draws[3]>=.5)&(rescue_draws[2]>=.5)&(rescue_draws[3]>=.5)
    early_valid=valid_draws[0]&valid_draws[1];early_boot=(half_draws[0]<=100)&(half_draws[1]<=100)&early_valid
    result=dict(controllers=out,H_late=classify(late),H_early=classify(early),late_component_checks=late,early_component_checks=early,timing_status_note='Joint preregistered gate; intermediate/mixed or unknown-flippable timing is INCONCLUSIVE, not forced binary',late_bootstrap_gate_pass_fraction=float(late_boot[valid].mean()) if valid.any() else None,late_bootstrap_valid_draws=int(valid.sum()),early_bootstrap_gate_pass_fraction=float(early_boot[early_valid].mean()) if early_valid.any() else None,early_bootstrap_valid_draws=int(early_valid.sum()),bootstrap_draws=5000,simultaneous_maxT_q95=q,DEVELOPMENT=True)
    write(destination/'CURVE_RESULTS.json',result);return result

def mechanisms(records,destination=HERE):
    p=read(HERE/'protocol.json');goal_tol=.09;rows=[];feature=[];target=[];state=[];ctl=[];excluded=Counter();prefix=[]
    old_outcomes={}
    for z in read(HERE/'HISTORICAL_RAW_INDEX.json'):
        c=z['case']
        if c['controller_name'] in CONTROLLERS and c['seed']<4:
            r=read(z['path']);old_outcomes[c['controller_name'],c['state_index'],c['seed'],c['eta_index']]=r['outcome']
    baselines={}
    for c in read(HERE/'gate_CASES.json'):
        r=read(HERE/'gate/raw'/f"{c['rollout_uid']}.json");baselines[c['base_controller_name'],c['state_index'],c['seed'],c['expected_eta_index']]=r
    for r in records:
        c=r['case'];ci=CONTROLLERS.index(c['base_controller_name'])
        with np.load(r['trace_path'],allow_pickle=False) as z:
            pos=z['positions'];vel=z['velocities'];goals=z['goals'];meta=json.loads(z['metadata_json'].item());cfg=meta['config'];gt=cfg.get('goal_tolerance',goal_tol);radius=cfg.get('robot_radius',cfg.get('agent_radius',0.));n=pos.shape[1];dist=np.linalg.norm(pos-goals,axis=-1);speed=np.linalg.norm(vel,axis=-1)
            samples=z['sample_steps'];stall=[];surface=[]
            for k in samples:
                displacement=np.linalg.norm(pos[max(0,k-40):k+1]-pos[k],axis=-1).max(0);stall.append(int(np.sum((dist[k]>gt)&(displacement<.02))) if k>=40 else None)
                dd=np.linalg.norm(pos[k,:,None,:]-pos[k,None,:,:],axis=-1);dd[np.eye(n,dtype=bool)]=np.inf;surface.append(float(dd.min()-2*radius))
            gd=goals-pos[0];opposing=[]
            for i in range(n):
                for j in range(i+1,n):
                    if gd[i]@gd[j]<0:opposing.append((i,j))
            encounter=None
            if opposing:
                candidates=[(float(np.linalg.norm(pos[:,i]-pos[:,j],axis=-1).min()),i,j,int(np.linalg.norm(pos[:,i]-pos[:,j],axis=-1).argmin())) for i,j in opposing];d,i,j,k=min(candidates);encounter=dict(pair=[i,j],step=k,location=((pos[k,i]+pos[k,j])/2).tolist(),center_distance=d)
            angles=np.unwrap(np.arctan2(pos[:,:,1],pos[:,:,0]),axis=0)
            rows.append(dict(case=c,outcome=r['outcome'],unknown=bool(r['numerical_failure']),episode_length=r['episode_length'],switch_reached=r['switch_reached'],sample_steps=samples.tolist(),angular_progress_per_robot=(angles[samples]-angles[0]).tolist(),mean_goal_distance=dist[samples].mean(1).tolist(),mean_speed=speed[samples].mean(1).tolist(),stall_candidates=stall,min_pair_surface_separation=surface,projection_active_fraction=float(z['projection_active'].mean()) if len(z['projection_active']) else None,min_simulator_swept_clearance=r['minimum_swept_clearance'],encounter=encounter))
            key=(c['base_controller_name'],c['state_index'],c['seed'],c['eta_a_index'])
            if key in baselines:
                b=baselines[key]
                with np.load(b['trace_path'],allow_pickle=False) as bz:
                    length=min(c['switch_step']+1,len(pos),len(bz['positions']));same=np.array_equal(pos[:length],bz['positions'][:length]) and np.array_equal(vel[:length],bz['velocities'][:length]);preterminal_same=None if r['numerical_failure'] else (b['physical_steps']>c['switch_step'] or (r['outcome']==b['outcome'] and r['episode_length']==b['episode_length'] and not r['switch_reached']));prefix.append(dict(rollout_uid=c['rollout_uid'],steps_compared=length-1,prefix_identical=same,preterminal_outcome_identical=preterminal_same))
            if c['eta_a_index']!=10:continue
            if r['numerical_failure']:excluded['unknown']+=1;continue
            if not r['switch_reached']:excluded['preterminal']+=1;continue
            s=r['switch_snapshot'];feature.append(np.r_[np.ravel(s['positions']),np.ravel(s['velocities']),np.ravel(s['goals']),c['switch_step']/700]);target.append(r['success']);state.append(c['state_index']);ctl.append(ci)
    assert all(x['prefix_identical'] and x['preterminal_outcome_identical'] is not False for x in prefix),'Intervention changed pre-switch dynamics or known early termination'
    write(destination/'PREFIX_AUDIT.json',dict(checks=prefix,passed=True,scope='All main cases with available exact gate baseline trajectory; no new baseline rollout'))
    write(destination/'MECHANISM_ROWS.json',rows)
    fig,axs=plt.subplots(2,2,figsize=(11,8));summary={}
    for ci,name in enumerate(CONTROLLERS):
        summary[name]={};ax=axs.flat[ci]
        for direction in (10,13):
            for t in TIMES:
                selected=[r for r in rows if r['case']['base_controller_name']==name and r['case']['eta_a_index']==direction and r['case']['switch_step']==t];known=[r for r in selected if not r['unknown']];out=Counter(r['outcome'] for r in selected);finalstall=[r['stall_candidates'][-1] for r in known if r['stall_candidates'][-1] is not None]
                if not selected:continue
                summary[name][f'{direction}to{23-direction}_t{t}']=dict(outcomes=dict(out),episode_length_by_outcome={o:[r['episode_length'] for r in selected if r['outcome']==o] for o in out},preterminal_fraction=sum(not r['switch_reached'] for r in selected)/len(selected),mean_final_stall_candidates=float(np.mean(finalstall)) if finalstall else None,mean_projection_fraction=float(np.mean([r['projection_active_fraction'] for r in known if r['projection_active_fraction'] is not None])),minimum_swept_clearance=min([r['min_simulator_swept_clearance'] for r in known if r['min_simulator_swept_clearance'] is not None],default=None))
                summary[name][f'{direction}to{23-direction}_t{t}']['paired_outcome_transitions_from_pure_source']=dict(Counter(f"{old_outcomes[name,r['case']['state_index'],r['case']['seed'],direction]}->{r['outcome']}" for r in selected))
                if direction==10:
                    xs=np.arange(0,701,10);means=[]
                    for k in xs:
                        vals=[r['mean_goal_distance'][r['sample_steps'].index(int(k))] for r in known if int(k) in r['sample_steps']];means.append(np.mean(vals) if vals else np.nan)
                    ax.plot(xs,means,label=f't={t}')
        ax.set(title=name.replace('_macflow',''),xlabel='Step (only still-observed trajectories)',ylabel='Mean goal distance');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(destination/'MECHANISM_GOAL_DISTANCE.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(2,4,figsize=(14,6))
    for ci,name in enumerate(CONTROLLERS):
        for direction in (10,13):
            arms=[summary[name].get(f'{direction}to{23-direction}_t{t}') for t in TIMES]
            if not all(arms):continue
            label=f'{direction}to{23-direction}'
            axs[0,ci].plot(TIMES,[a['mean_final_stall_candidates'] for a in arms],marker='o',label=label)
            axs[1,ci].plot(TIMES,[a['mean_projection_fraction'] for a in arms],marker='o',label=label)
        axs[0,ci].set(title=name.replace('_macflow',''),ylabel='Final local stall candidates (not deadlock)');axs[1,ci].set(xlabel='Switch action index',ylabel='Episode projection activation fraction')
        if axs[0,ci].lines:axs[0,ci].legend(fontsize=8);axs[1,ci].legend(fontsize=8)
    fig.suptitle('DEVELOPMENT: descriptive summaries of valid observed trajectories');fig.tight_layout();fig.savefig(destination/'MECHANISM_STALL_PROJECTION.png',dpi=170);plt.close(fig)
    # Fixed state0/seed0 examples: selection independent of observed outcomes.
    selected_records={(r['case']['base_controller_name'],r['case']['eta_a_index'],r['case']['switch_step'],r['case']['state_index'],r['case']['seed']):r for r in records}
    fig,axs=plt.subplots(4,4,figsize=(12,12));example_manifest=[]
    for ci,name in enumerate(CONTROLLERS):
        examples=[('pure eta10',baselines.get((name,0,0,10))),('pure eta13',baselines.get((name,0,0,13))),('10to13 t300',selected_records.get((name,10,300,0,0))),('13to10 t300',selected_records.get((name,13,300,0,0)))]
        for j,(label,r) in enumerate(examples):
            ax=axs[ci,j]
            if r is None:ax.set_axis_off();continue
            with np.load(r['trace_path'],allow_pickle=False) as z:
                cfg=json.loads(z['metadata_json'].item())['config'];positions=z['positions'];goals=z['goals'];ax.add_patch(plt.Circle((0,0),cfg['obstacle_radius'],facecolor='gray',alpha=.4));ax.add_patch(plt.Circle((0,0),cfg['outer_radius'],fill=False,color='black'))
                for robot in range(positions.shape[1]):
                    color=f'C{robot}';ax.plot(positions[:,robot,0],positions[:,robot,1],color=color,lw=1);ax.scatter(*positions[0,robot],color=color,s=12);ax.scatter(*goals[robot],color=color,s=25,marker='*')
                    t=r['case']['switch_step']
                    if label.startswith(('10to','13to')) and t<len(positions):ax.scatter(*positions[t,robot],color=color,s=25,marker='x')
            ax.set_aspect('equal');ax.set(title=f"{name.replace('_macflow','')}\n{label}: {r['outcome']}",xlim=(-3.5,3.5),ylim=(-3.5,3.5));ax.tick_params(labelsize=7)
            example_manifest.append(dict(label=label,case=r['case'],outcome=r['outcome'],trace_path=r['trace_path'],trace_sha256=r['trace_sha256'],complete=not bool(r['numerical_failure'])))
    fig.suptitle('DEVELOPMENT: fixed state0/seed0; full observed paths, x marks switch; unknown is partial');fig.tight_layout();fig.savefig(destination/'MECHANISM_FIXED_PATHS.png',dpi=150);plt.close(fig);write(destination/'MECHANISM_EXAMPLES.json',example_manifest)
    write(destination/'MECHANISM_RESULTS.json',dict(summary=summary,excluded_classifier=dict(excluded),stall_definition='unfinished and max displacement <0.02m in last40steps, diagnostic only; no runtime deadlock event',survivor_note='Time curves use observed trajectories, not terminal imputation; composition changes with termination',DEVELOPMENT=True))
    np.savez_compressed(destination/'CLASSIFIER_INPUT.npz',X=np.array(feature),y=np.array(target),state=np.array(state),controller=np.array(ctl))
    return summary

if __name__=='__main__':
    Y,U,records=load();curves(Y,U);mechanisms(records)
