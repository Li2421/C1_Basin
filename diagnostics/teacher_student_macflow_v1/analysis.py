"""Paired controller-then-state clustered bootstrap; preserves unknowns."""
from core import *
from scipy.stats import spearmanr
ARMS=['nominal only','safety baseline','F10','F13','S300','S500','B-RHR']
MENU=['F10','F13','S300','S500']
def ci(x):return np.quantile(x,[.025,.975]).tolist()
def summary(point,boot):return dict(mean=float(point),CI95=ci(boot))
def analyze():
    from teacher_choices import main as teacher_choice_predict
    teacher_choice_predict()
    teacherchoices=read(HERE/'DEVELOPMENT_TEACHER_CHOICES.json')['controllers']
    p=read(HERE/'main_protocol.json');names=p['controller_order'];C=len(names);S=32;A=len(ARMS);rows=read(HERE/'main_CASES.json');db={r['rollout_uid']:r for r in read(HERE/'main_DB_ROWS.json')};y=np.zeros((C,S,2,A));u=y.copy();collision=y.copy();timeout=y.copy();stall=y.copy();raw_index={}
    for c in rows:
        r=db[c['rollout_uid']];ix=(names.index(c['base_controller_name']),c['state_index'],c['seed'],ARMS.index(c['arm']));y[ix]=r['success'] if not r['numerical_failure'] else 0;u[ix]=r['numerical_failure'];collision[ix]=r['collision'];timeout[ix]=r['timeout'];path=HERE/'main/raw'/f"{c['rollout_uid']}.json"
        if path.exists():
            raw=read(path);raw_index[(c['base_controller_name'],c['state_index'],c['seed'],c['arm'])]=raw
            z=np.load(raw['trace_path']);slow=z['slow_mask'];counts=np.zeros(4,int);has=False
            for mask in slow:counts=np.where(mask,counts+1,0);has=has or bool((counts>=40).any())
            stall[ix]=has
    choices=read(HERE/'FROZEN_CHOICES.json')['controllers'];rng=np.random.default_rng(2026103032);controllers=rng.integers(0,C,(5000,C));states=rng.integers(0,S,(5000,C,S));means=y.mean((1,2));uppermeans=(y+u).mean((1,2));boot=y.mean(2)[controllers[:,:,None],states,:].mean(2);boot_upper=(y+u).mean(2)[controllers[:,:,None],states,:].mean(2);menu=[ARMS.index(a) for a in MENU];fixed=menu[:2];schedules=menu[2:];idx=np.arange(C)
    def selected(matrix,label):
        sel=np.array([ARMS.index(choices[n][label]) for n in names]);return matrix[idx,sel],sel
    table={}
    for ai,arm in enumerate(ARMS):
        table[arm]=dict(success=summary(means[:,ai].mean(),boot[:,:,ai].mean(1)),success_upper=float(uppermeans[:,ai].mean()),collision=summary(collision[:,:,:,ai].mean(),collision.mean(2)[controllers[:,:,None],states,ai].mean((1,2))),timeout=summary(timeout[:,:,:,ai].mean(),timeout.mean(2)[controllers[:,:,None],states,ai].mean((1,2))),unknown=float(u[:,:,:,ai].mean()),stall_diagnostic=float(stall[:,:,:,ai].mean()))
    selectors={}
    bc=collision.mean(2)[controllers[:,:,None],states,:].mean(2);bt=timeout.mean(2)[controllers[:,:,None],states,:].mean(2)
    for label in ['A3_full','A0_full','G_E','S500','F10','A3_fixed','A0_fixed']:
        point,sel=selected(means,label);bs=boot[np.arange(5000)[:,None],np.arange(C)[None,:],sel[controllers]];bsu=boot_upper[np.arange(5000)[:,None],np.arange(C)[None,:],sel[controllers]]
        selectors[label]=dict(success=summary(point.mean(),bs.mean(1)),success_upper=float(uppermeans[idx,sel].mean()),collision=summary(np.mean([collision[i,:,:,sel[i]].mean() for i in idx]),bc[np.arange(5000)[:,None],np.arange(C)[None,:],sel[controllers]].mean(1)),timeout=summary(np.mean([timeout[i,:,:,sel[i]].mean() for i in idx]),bt[np.arange(5000)[:,None],np.arange(C)[None,:],sel[controllers]].mean(1)),_point=point,_boot=bs)
    oracle=means[:,menu].max(1);boracle=boot[:,:,menu].max(2);oracle_sel=np.asarray(menu)[means[:,menu].argmax(1)];oracle_bsel=np.asarray(menu)[boot[:,:,menu].argmax(2)];table['oracle']=dict(success=summary(oracle.mean(),boracle.mean(1)),success_upper=float(uppermeans[:,menu].max(1).mean()),collision=summary(np.mean([collision[i,:,:,oracle_sel[i]].mean() for i in idx]),bc[np.arange(5000)[:,None],np.arange(C)[None,:],oracle_bsel].mean(1)),timeout=summary(np.mean([timeout[i,:,:,oracle_sel[i]].mean() for i in idx]),bt[np.arange(5000)[:,None],np.arange(C)[None,:],oracle_bsel].mean(1)))
    base=ARMS.index('safety baseline');s500=ARMS.index('S500');ts1=summary((means[:,s500]-means[:,base]).mean(),(boot[:,:,s500]-boot[:,:,base]).mean(1));ts1['identification_interval']=[float((means[:,s500]-uppermeans[:,base]).mean()),float((uppermeans[:,s500]-means[:,base]).mean())];ts1['CI_unknown_envelope']=[float(np.quantile((boot[:,:,s500]-boot_upper[:,:,base]).mean(1),.025)),float(np.quantile((boot_upper[:,:,s500]-boot[:,:,base]).mean(1),.975))];ts1['passed_robust_unknown']=ts1['CI_unknown_envelope'][0]>0;ts1['passed']=ts1['CI95'][0]>0
    schedule=means[:,schedules].max(1);fix=means[:,fixed].max(1);ts2=summary((schedule-fix).mean(),(boot[:,:,schedules].max(2)-boot[:,:,fixed].max(2)).mean(1));ts2['coverage']=float((schedule>=fix-.01).mean());ts2['coverage_definite']=float((schedule>=uppermeans[:,fixed].max(1)-.01).mean());ts2['coverage_possible']=float((uppermeans[:,schedules].max(1)>=fix-.01).mean());ts2['identification_interval']=[float((schedule-uppermeans[:,fixed].max(1)).mean()),float((uppermeans[:,schedules].max(1)-fix).mean())];ts2['CI_unknown_envelope']=[float(np.quantile((boot[:,:,schedules].max(2)-boot_upper[:,:,fixed].max(2)).mean(1),.025)),float(np.quantile((boot_upper[:,:,schedules].max(2)-boot[:,:,fixed].max(2)).mean(1),.975))];ts2['passed_robust_unknown']=ts2['CI_unknown_envelope'][0]>0 and ts2['coverage_definite']>=.7;ts2['passed']=ts2['CI95'][0]>0 and ts2['coverage']>=.7
    a3=selectors['A3_full'];regret=summary((oracle-a3['_point']).mean(),(boracle-a3['_boot']).mean(1));gain=summary((a3['_point']-means[:,base]).mean(),(a3['_boot']-boot[:,:,base]).mean(1));ts3a=dict(regret=regret,gain_over_safety=gain,passed=regret['mean']<=.03 and gain['CI95'][0]>0,comparisons={})
    a3sel=np.array([ARMS.index(choices[n]['A3_full']) for n in names]);other_lo=means[:,menu].copy();other_hi=uppermeans[:,menu].copy()
    for i in range(C):other_lo[i,menu.index(a3sel[i])]=-np.inf;other_hi[i,menu.index(a3sel[i])]=-np.inf
    ts3a['regret_identification_interval']=[float(np.maximum(0,other_lo.max(1)-uppermeans[idx,a3sel]).mean()),float(np.maximum(0,other_hi.max(1)-means[idx,a3sel]).mean())]
    a3upper=boot_upper[np.arange(5000)[:,None],np.arange(C)[None,:],a3sel[controllers]];ts3a['gain_CI_unknown_envelope']=[float(np.quantile((a3['_boot']-boot_upper[:,:,base]).mean(1),.025)),float(np.quantile((a3upper-boot[:,:,base]).mean(1),.975))];ts3a['passed_robust_unknown']=ts3a['regret_identification_interval'][1]<=.03 and ts3a['gain_CI_unknown_envelope'][0]>0;ts3a['passed']=regret['mean']<=.03 and gain['CI95'][0]>0
    for label in ['S500','A0_full','G_E','F10']:ts3a['comparisons'][label]=summary((a3['_point']-selectors[label]['_point']).mean(),(a3['_boot']-selectors[label]['_boot']).mean(1))
    prefer10=int((means[:,fixed[0]]-means[:,fixed[1]]>=.1).sum());prefer13=int((means[:,fixed[1]]-means[:,fixed[0]]>=.1).sum());a3f=selectors['A3_fixed'];a0f=selectors['A0_fixed'];delta=summary((a0f['_point']-a3f['_point']).mean(),(a0f['_boot']-a3f['_boot']).mean(1));ts3b=dict(preference_F10_count=prefer10,preference_F13_count=prefer13,regret_A3_minus_A0=delta,status='INCONCLUSIVE' if min(prefer10,prefer13)<3 else 'PASS' if delta['CI95'][1]<0 else 'REJECTED',agreement=float(np.mean([choices[n]['A3_fixed']==MENU[int(means[i,fixed].argmax())] for i,n in enumerate(names)])))
    transitions=dict(total_nominal_collision=0,safety_success=0,safety_timeout=0,safety_unknown=0,safety_collision=0,safety_other=0)
    for i in range(C):
        for s in range(S):
            for seed in range(2):
                if collision[i,s,seed,0]:
                    transitions['total_nominal_collision']+=1
                    k='safety_unknown' if u[i,s,seed,1] else 'safety_success' if y[i,s,seed,1] else 'safety_timeout' if timeout[i,s,seed,1] else 'safety_collision' if collision[i,s,seed,1] else 'safety_other';transitions[k]+=1
    transitions['proportions']={k:v/max(1,transitions['total_nominal_collision']) for k,v in transitions.items() if k!='total_nominal_collision'}
    # Pair teacher DEVELOPMENT references by original state index and seed.
    teachers={mode:np.full((C,S,2,4),np.nan) for mode in ['local','central','shared']};teacher_u={mode:np.zeros((C,S,2,4)) for mode in teachers};teacher_collision={mode:np.zeros((C,S,2,4)) for mode in teachers};teacher_timeout={mode:np.zeros((C,S,2,4)) for mode in teachers}
    old=D/'decentralized_pipeline_v1'
    for c in read(old/'main_CASES.json'):
        if c['base_controller_name'] not in names or c['pipeline'] not in teachers or c['arm'] not in MENU:continue
        r=read(old/'main/raw'/f"{c['rollout_uid']}.json");ix=(names.index(c['base_controller_name']),c['state_index'],c['seed'],MENU.index(c['arm']));teachers[c['pipeline']][ix]=0 if r['numerical_failure'] else r['success'];teacher_u[c['pipeline']][ix]=r['numerical_failure'];teacher_collision[c['pipeline']][ix]=r['collision'];teacher_timeout[c['pipeline']][ix]=r['timeout']
    ts5={};teacher_tables={}
    for mode,t in teachers.items():
        assert np.isfinite(t).all();tm=t.mean((1,2));tb=t.mean(2)[controllers[:,:,None],states,:].mean(2);teacher_tables[mode]={};ts5[mode]={}
        for ai,arm in enumerate(MENU):
            si=ARMS.index(arm);teacher_tables[mode][arm]=dict(success=summary(tm[:,ai].mean(),tb[:,:,ai].mean(1)),success_upper=float((t+teacher_u[mode])[:,:,:,ai].mean()),collision=float(teacher_collision[mode][:,:,:,ai].mean()),timeout=float(teacher_timeout[mode][:,:,:,ai].mean()))
            ts5[mode][arm]=summary((means[:,si]-tm[:,ai]).mean(),(boot[:,:,si]-tb[:,:,ai]).mean(1));ts5[mode][arm]['identification_interval']=[float((means[:,si]-(t+teacher_u[mode]).mean((1,2))[:,ai]).mean()),float((uppermeans[:,si]-tm[:,ai]).mean())]
        for model in ['A3','A0']:
            sel=np.array([MENU.index(teacherchoices[n][model]) for n in names]);teacher_tables[mode][model+'_historical_R5']=dict(success=summary(tm[idx,sel].mean(),tb[np.arange(5000)[:,None],np.arange(C)[None,:],sel[controllers]].mean(1)),success_upper=float((t+teacher_u[mode]).mean((1,2))[idx,sel].mean()),collision=float(np.mean([teacher_collision[mode][i,:,:,sel[i]].mean() for i in idx])),timeout=float(np.mean([teacher_timeout[mode][i,:,:,sel[i]].mean() for i in idx])))
        teacher_tables[mode]['oracle']=dict(success=summary(tm.max(1).mean(),tb.max(2).mean(1)),success_upper=float((t+teacher_u[mode]).mean((1,2)).max(1).mean()))
        teacher_tables[mode]['always-S500']=teacher_tables[mode]['S500']
    # Reuse round28 D-shared RHR; never relabel this as D-local.
    rh=np.zeros((C,S,2));rhc=rh.copy();rht=rh.copy()
    for c in read(old/'main_CASES.json'):
        if c['pipeline']=='rhr' and c['base_controller_name'] in names:
            r=read(old/'main/raw'/f"{c['rollout_uid']}.json");ix=(names.index(c['base_controller_name']),c['state_index'],c['seed']);rh[ix]=r['success'] if not r['numerical_failure'] else 0;rhc[ix]=r['collision'];rht[ix]=r['timeout']
    teacher_tables['shared']['B-RHR (radius infinity)']=dict(success=summary(rh.mean(),rh.mean(2)[controllers[:,:,None],states].mean((1,2))),success_upper=float(rh.mean()),collision=float(rhc.mean()),timeout=float(rht.mean()))
    # Teacher nominal/safety additions (same evaluation states; all14 teachers may be reported separately).
    if (HERE/'teacher_DB_ROWS.json').exists():
        cases=read(HERE/'teacher_CASES.json');by={r['rollout_uid']:r for r in read(HERE/'teacher_DB_ROWS.json')};extra={arm:np.zeros((C,S,2)) for arm in ['nominal only','safety baseline']};ec={a:np.zeros((C,S,2)) for a in extra};et={a:np.zeros((C,S,2)) for a in extra};eu={a:np.zeros((C,S,2)) for a in extra}
        for c in cases:
            if c['base_controller_name'] in names:
                r=by[c['rollout_uid']];ix=(names.index(c['base_controller_name']),c['state_index'],c['seed']);extra[c['arm']][ix]=r['success'] if not r['numerical_failure'] else 0;ec[c['arm']][ix]=r['collision'];et[c['arm']][ix]=r['timeout'];eu[c['arm']][ix]=r['numerical_failure']
        for arm,t in extra.items():
            teacher_tables['local'][arm]=dict(success=summary(t.mean(),t.mean(2)[controllers[:,:,None],states].mean((1,2))),success_upper=float((t+eu[arm]).mean()),collision=float(ec[arm].mean()),timeout=float(et[arm].mean()),unknown=float(eu[arm].mean()))
            si=ARMS.index(arm);ts5['local'][arm]=summary((means[:,si]-t.mean((1,2))).mean(),(boot[:,:,si]-t.mean(2)[controllers[:,:,None],states].mean(2)).mean(1))
    inherit=[];tm=teachers['local'].mean((1,2))
    for i,n in enumerate(names):
        corr=spearmanr(means[i,menu],tm[i]);inherit.append(dict(controller=n,student_best=MENU[int(means[i,menu].argmax())],teacher_best=MENU[int(tm[i].argmax())],student_fixed=MENU[int(means[i,fixed].argmax())],teacher_fixed=MENU[int(tm[i,:2].argmax())],spearman=float(corr.statistic) if np.isfinite(corr.statistic) else None))
    for d in selectors.values():d.pop('_point');d.pop('_boot')
    study=read(HERE/'CAPABILITY_GATE.json')['study_status'];result=dict(utc=utc(),status='INCONCLUSIVE' if study=='INCONCLUSIVE' else 'COMPLETE',eligible_controllers=names,n_controllers=C,states=S,seeds=2,bootstrap_replicates=5000,unknown_total=int(u.sum()),unknown_treatment='lower success=0, upper success=1; not known failures, no reruns',safety_layer_collisions=int(collision[:,:,:,1:].sum()),TS1=ts1,TS2=ts2,TS3a=ts3a,TS3b=ts3b,TS4=dict(inheritance=inherit,best_arm_agreement=float(np.mean([x['student_best']==x['teacher_best'] for x in inherit])),fixed_preference_agreement=float(np.mean([x['student_fixed']==x['teacher_fixed'] for x in inherit])),student_majority_fixed_baseline=max(sum(x['student_fixed']=='F10' for x in inherit),sum(x['student_fixed']=='F13' for x in inherit))/C,fixed_preference_contingency={t:{a:sum(x['teacher_fixed']==t and x['student_fixed']==a for x in inherit) for a in ['F10','F13']} for t in ['F10','F13']},mean_spearman=float(np.mean([x['spearman'] for x in inherit if x['spearman'] is not None])) if any(x['spearman'] is not None for x in inherit) else None),TS5=ts5,student_table=table,selectors=selectors,teacher_tables=teacher_tables,transitions=transitions,controller_rates={n:{arm:float(means[i,a]) for a,arm in enumerate(ARMS)} for i,n in enumerate(names)},confirmatory_interpretation='No confirmatory claim if fewer than8 passed capability gate; teacher labels DEVELOPMENT',evidence='student labels new and preregistered; teacher comparisons DEVELOPMENT')
    result['protocol_deviations']=['PILOT_RUNNER_PROVENANCE.json','JOURNAL_PATH_DEVIATION.json'];result['source_file_hash_audit']='see INTEGRITY_COMPARISON.json (separate from failed write-scope audit)';result['write_scope_audit']='FAIL:448 pilot journals created in PoC copy, preserved and recovered through official merger; source files unchanged'
    write(HERE/'results.json',result);np.savez_compressed(HERE/'analysis_arrays.npz',success_lower=y,unknown=u,collision=collision,timeout=timeout,controller_names=names,arms=ARMS)
    (HERE/'transition_table.md').write_text('# Nominal collision → safety baseline\n\n'+json.dumps(transitions,indent=2)+'\n')
    (HERE/'inheritance_table.md').write_text('| Controller | Student best | Teacher best | Spearman |\n|---|---|---|---|\n'+''.join(f"| {r['controller']} | {r['student_best']} | {r['teacher_best']} | {r['spearman']} |\n" for r in inherit))
    lines=['| Family/arm | Success lower–upper | 95% CI | Collision | Timeout |','|---|---:|---:|---:|---:|']
    def fmt(v):return f'{100*v:.2f}%'
    for arm,t in table.items():
        s=t['success'];lines.append(f"| Student/{arm} | {fmt(s['mean'])}–{fmt(t['success_upper'])} | {fmt(s['CI95'][0])}, {fmt(s['CI95'][1])} | {fmt(t.get('collision',{}).get('mean',0)) if arm!='oracle' else fmt(t['collision']['mean'])} | {fmt(t.get('timeout',{}).get('mean',0)) if arm!='oracle' else fmt(t['timeout']['mean'])} |")
    for label,t in selectors.items():
        s=t['success'];lines.append(f"| Student/{label} | {fmt(s['mean'])}–{fmt(t['success_upper'])} | {fmt(s['CI95'][0])}, {fmt(s['CI95'][1])} | {fmt(t['collision']['mean'])} | {fmt(t['timeout']['mean'])} |")
    for mode,tab in teacher_tables.items():
        for arm,t in tab.items():
            s=t['success'];lines.append(f"| Teacher/{mode}/{arm} | {fmt(s['mean'])}–{fmt(t.get('success_upper',s['mean']))} | {fmt(s['CI95'][0])}, {fmt(s['CI95'][1])} | {fmt(t['collision']) if 'collision' in t else 'see raw'} | {fmt(t['timeout']) if 'timeout' in t else 'see raw'} |")
    lines+=['','Teacher D-local B-RHR: not evaluated; round28 B-RHR uses D-shared and is shown with radius infinity. Teacher A3/A0 use historical R5 descriptors with frozen round16 models and are DEVELOPMENT references, not newly probed D-local descriptors; student selector rows retain pre-evaluation frozen choices.']
    (HERE/'paper_table.md').write_text('\n'.join(lines)+'\n');print('Analysis',result['status'],ts1,ts2,ts3a['passed'],ts3b['status'],flush=True)
if __name__=='__main__':
    analyze()
    from metrics import main as metrics
    metrics()
    seal('ANALYSIS_COMPLETE.receipt.json',[HERE/'results.json',HERE/'paper_table.md',HERE/'rollout_metrics.csv',HERE/'FAILURE_DIAGNOSTICS.json',HERE/'analysis.py',HERE/'metrics.py'],rollouts_rerun=0)
