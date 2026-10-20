"""Paired controller/state clustered analysis of immutable new and reused labels."""
from core import *
from audit_utils import source_inventory
import collections,csv
from scipy.stats import spearmanr
ARMS=['nominal only','safety baseline','F10','F13','S300','S500','B-RHR','B-RHR-tuned'];MENU=[2,3,4,5]
OLD=D/'teacher_student_macflow_v1'
def interval(point,boot):return dict(mean=float(point),CI95=np.quantile(boot,[.025,.975]).tolist())
def collect(path,names):
    cases=read(path/'main_CASES.json');by={r['rollout_uid']:r for r in read(path/'main_DB_ROWS.json')};arrays={k:np.full((len(names),32,2,8),np.nan) for k in ['success','unknown','collision','timeout']}
    for c in cases:
        n=c['base_controller_name']
        if n not in names:continue
        r=by[c['rollout_uid']];ix=(names.index(n),c['state_index'],c['seed'],ARMS.index(c['arm']))
        arrays['success'][ix]=r['success'] if not r['numerical_failure'] else 0;arrays['unknown'][ix]=r['numerical_failure'];arrays['collision'][ix]=r['collision'];arrays['timeout'][ix]=r['timeout']
    assert all(np.isfinite(x).all() for x in arrays.values())
    return arrays

def analyze_set(label,names,a,choices,confirmatory_allowed):
    C=len(names)
    if C==0:return dict(status='INCONCLUSIVE',reason='No eligible frame-consistent student',n_controllers=0,controllers=[])
    y=a['success'];u=a['unknown'];means=y.mean((1,2));upper=(y+u).mean((1,2));rng=np.random.default_rng(2026103032);controllers=rng.integers(0,C,(5000,C));states=rng.integers(0,32,(5000,C,32));br=np.arange(5000)[:,None];cr=np.arange(C)[None,:];idx=np.arange(C)
    def draw(x):return x.mean(2)[controllers[:,:,None],states,:].mean(2)
    boot=draw(y);bu=draw(y+u);bc=draw(a['collision']);bt=draw(a['timeout']);table={}
    for j,arm in enumerate(ARMS):table[arm]=dict(success=interval(means[:,j].mean(),boot[:,:,j].mean(1)),success_upper=interval(upper[:,j].mean(),bu[:,:,j].mean(1)),collision=interval(a['collision'][:,:,:,j].mean(),bc[:,:,j].mean(1)),timeout=interval(a['timeout'][:,:,:,j].mean(),bt[:,:,j].mean(1)),unknown=float(u[:,:,:,j].mean()))
    selectors={}
    for label2 in ['A3_full','A0_full','A3_fixed','A0_fixed','G_E','S500','F10']:
        sel=np.array([ARMS.index(choices[n][label2]) for n in names]);point=means[idx,sel];bs=boot[br,cr,sel[controllers]];bsu=bu[br,cr,sel[controllers]];selectors[label2]=dict(point=point,boot=bs,upper_boot=bsu,indices=sel)
        table[label2]=dict(success=interval(point.mean(),bs.mean(1)),success_upper=interval(upper[idx,sel].mean(),bsu.mean(1)),collision=interval(a['collision'].mean((1,2))[idx,sel].mean(),bc[br,cr,sel[controllers]].mean(1)),timeout=interval(a['timeout'].mean((1,2))[idx,sel].mean(),bt[br,cr,sel[controllers]].mean(1)))
    oracle=means[:,MENU].max(1);bo=boot[:,:,MENU].max(2);osel=np.array(MENU)[means[:,MENU].argmax(1)];bosel=np.array(MENU)[boot[:,:,MENU].argmax(2)];table['oracle']=dict(success=interval(oracle.mean(),bo.mean(1)),success_upper=interval(upper[:,MENU].max(1).mean(),bu[:,:,MENU].max(2).mean(1)),collision=interval(a['collision'].mean((1,2))[idx,osel].mean(),bc[br,cr,bosel].mean(1)),timeout=interval(a['timeout'].mean((1,2))[idx,osel].mean(),bt[br,cr,bosel].mean(1)))
    def contrast(j,k):
        r=interval((means[:,j]-means[:,k]).mean(),(boot[:,:,j]-boot[:,:,k]).mean(1));r.update(identification_interval=[float((means[:,j]-upper[:,k]).mean()),float((upper[:,j]-means[:,k]).mean())],CI_unknown_envelope=[float(np.quantile((boot[:,:,j]-bu[:,:,k]).mean(1),.025)),float(np.quantile((bu[:,:,j]-boot[:,:,k]).mean(1),.975))]);r['passed']=r['CI95'][0]>0;return r
    ts1=contrast(5,1);ts6=contrast(5,6);ts6b=contrast(5,7);sf=means[:,4:6].max(1)-means[:,2:4].max(1);sfb=boot[:,:,4:6].max(2)-boot[:,:,2:4].max(2);ts2=interval(sf.mean(),sfb.mean(1));ts2['coverage']=float((sf>=-.01).mean());ts2['passed']=ts2['CI95'][0]>0 and ts2['coverage']>=.7;ts2['CI_unknown_envelope']=[float(np.quantile((boot[:,:,4:6].max(2)-bu[:,:,2:4].max(2)).mean(1),.025)),float(np.quantile((bu[:,:,4:6].max(2)-boot[:,:,2:4].max(2)).mean(1),.975))]
    a3=selectors['A3_full'];gain=interval((a3['point']-means[:,1]).mean(),(a3['boot']-boot[:,:,1]).mean(1));regret=interval((oracle-a3['point']).mean(),(bo-a3['boot']).mean(1));ts3a=dict(regret=regret,gain_over_safety=gain,passed=regret['mean']<=.03 and gain['CI95'][0]>0,comparisons={})
    for other in ['A0_full','G_E','S500','F10']:ts3a['comparisons'][other]=interval((a3['point']-selectors[other]['point']).mean(),(a3['boot']-selectors[other]['boot']).mean(1))
    ts3a['gain_CI_unknown_envelope']=[float(np.quantile((a3['boot']-bu[:,:,1]).mean(1),.025)),float(np.quantile((a3['upper_boot']-boot[:,:,1]).mean(1),.975))]
    ts3a['regret_identification_interval']=[float(np.maximum(0,oracle-upper[idx,a3['indices']]).mean()),float(np.maximum(0,upper[:,MENU].max(1)-a3['point']).mean())]
    prefer10=int((means[:,2]-means[:,3]>=.1).sum());prefer13=int((means[:,3]-means[:,2]>=.1).sum());delta=interval((selectors['A0_fixed']['point']-selectors['A3_fixed']['point']).mean(),(selectors['A0_fixed']['boot']-selectors['A3_fixed']['boot']).mean(1));ts3b=dict(preference_F10_count=prefer10,preference_F13_count=prefer13,regret_A3_minus_A0=delta,status='INCONCLUSIVE' if min(prefer10,prefer13)<3 else 'PASS' if delta['CI95'][1]<0 else 'NOT_PASS')
    for r in [ts1,ts2,ts3a,ts6,ts6b]:r['status']=('PASS' if r['passed'] else 'NOT_PASS') if confirmatory_allowed else 'INCONCLUSIVE'
    if not confirmatory_allowed:ts3b['numerical_status']=ts3b['status'];ts3b['status']='INCONCLUSIVE'
    # Frozen joint-teacher comparisons, read-only, matched state/seed.
    teachers={mode:np.full((C,32,2,4),np.nan) for mode in ['local','central','shared']}
    teacher_unknown={mode:np.zeros((C,32,2,4)) for mode in teachers}
    for c in read(D/'decentralized_pipeline_v1/main_CASES.json'):
        if c['base_controller_name'] in names and c['pipeline'] in teachers and c['arm'] in ARMS[2:6]:
            r=read(D/'decentralized_pipeline_v1/main/raw'/f"{c['rollout_uid']}.json");ix=(names.index(c['base_controller_name']),c['state_index'],c['seed'],ARMS[2:6].index(c['arm']));teachers[c['pipeline']][ix]=r['success'] if not r['numerical_failure'] else 0;teacher_unknown[c['pipeline']][ix]=r['numerical_failure']
    assert all(np.isfinite(t).all() for t in teachers.values());teacher=teachers['local'];tm=teacher.mean((1,2));tb=teacher.mean(2)[controllers[:,:,None],states,:].mean(2);inherit=[]
    for i,n in enumerate(names):
        corr=spearmanr(means[i,2:6],tm[i]).statistic
        inherit.append(dict(controller=n,student_best=ARMS[2+int(means[i,2:6].argmax())],teacher_best=ARMS[2+int(tm[i].argmax())],student_fixed=ARMS[2+int(means[i,2:4].argmax())],teacher_fixed=ARMS[2+int(tm[i,:2].argmax())],spearman=float(corr) if np.isfinite(corr) else None))
    ts4=dict(inheritance=inherit,best_arm_agreement=float(np.mean([x['student_best']==x['teacher_best'] for x in inherit])))
    ts5={}
    for mode,t in teachers.items():
        m=t.mean((1,2));b=t.mean(2)[controllers[:,:,None],states,:].mean(2);tu=(t+teacher_unknown[mode]).mean((1,2));ts5[mode]={}
        for j,arm in enumerate(ARMS[2:6]):
            ts5[mode][arm]=interval((means[:,j+2]-m[:,j]).mean(),(boot[:,:,j+2]-b[:,:,j]).mean(1));ts5[mode][arm]['identification_interval']=[float((means[:,j+2]-tu[:,j]).mean()),float((upper[:,j+2]-m[:,j]).mean())]
    ts4['fixed_preference_agreement']=float(np.mean([x['student_fixed']==x['teacher_fixed'] for x in inherit]))
    result=dict(analysis_set=label,status='COMPLETE' if confirmatory_allowed else 'INCONCLUSIVE',n_controllers=C,controllers=names,states=32,seeds=2,bootstrap_replicates=5000,bootstrap='paired; controllers then state clusters, both seeds kept together; reselect maxima within each bootstrap',unknown_total=int(u.sum()),safety_layer_collisions=int(a['collision'][:,:,:,1:].sum()),table=table,TS1=ts1,TS2=ts2,TS3a=ts3a,TS3b=ts3b,TS6=ts6,TS6b=ts6b,always_S500_minus_A3=interval((means[:,5]-a3['point']).mean(),(boot[:,:,5]-a3['boot']).mean(1)),TS4=ts4,TS5=ts5,controller_rates={n:{arm:float(means[i,j]) for j,arm in enumerate(ARMS)} for i,n in enumerate(names)})
    lines=['| Arm/selector | Success lower | 95% CI | Success upper | Collision | Timeout |','|---|---:|---|---:|---:|---:|']
    for arm in ARMS+['A3_full','S500','oracle']:
        r=table[arm];v=r['success'];lo,hi=v['CI95'];display='always S500' if arm=='S500' and len(lines)>9 else 'A3 zero-shot' if arm=='A3_full' else arm;lines.append(f"| {display} | {v['mean']:.2%} | [{lo:.2%}, {hi:.2%}] | {r['success_upper']['mean']:.2%} | {r['collision']['mean']:.2%} | {r['timeout']['mean']:.2%} |")
    (HERE/f'paper_table_{label.lower()}.md').write_text('\n'.join(lines)+'\n');return result

def failures(names,a):
    rows=[];transfers=[];by={};totals=collections.Counter()
    def category(i,s,k,arm):
        if a['unknown'][i,s,k,arm]:return 'unknown'
        if a['success'][i,s,k,arm]:return 'success'
        if a['collision'][i,s,k,arm]:return 'collision'
        if a['timeout'][i,s,k,arm]:return 'timeout'
        return 'other'
    for i,n in enumerate(names):
        outcomes=collections.Counter();transition=collections.Counter();groups={g:dict(count=0,rescued_F13=0,rescued_S500=0,rescued_BRHR=0,rescued_BRHR_tuned=0) for g in ['own_stagnation','safety_induced_stagnation']}
        for s in range(32):
            for k in range(2):
                nominal=category(i,s,k,0);safe=category(i,s,k,1);outcomes[nominal]+=1;transition[(nominal,safe)]+=1;transfers.append(dict(controller=n,state=s,seed=k,nominal=nominal,safety=safe))
                group='own_stagnation' if nominal=='timeout' and not a['collision'][i,s,k,0] else 'safety_induced_stagnation' if nominal=='collision' and safe=='timeout' else None
                if group:
                    groups[group]['count']+=1
                    for arm,key in [(3,'rescued_F13'),(5,'rescued_S500'),(6,'rescued_BRHR'),(7,'rescued_BRHR_tuned')]:groups[group][key]+=int(a['success'][i,s,k,arm])
        for g,v in groups.items():
            totals[g]+=v['count'];rows.append(dict(controller=n,source=g,**v))
        by[n]=dict(nominal_outcomes=dict(outcomes),nominal_to_safety=[dict(nominal=x,safety=y,count=v) for (x,y),v in transition.items()],failure_sources=groups)
    with (HERE/'failure_sources.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['controller','source','count','rescued_F13','rescued_S500','rescued_BRHR','rescued_BRHR_tuned']);w.writeheader();w.writerows(rows)
    write(HERE/'TRANSFERS.json',transfers);write(HERE/'FAILURE_SOURCES.json',dict(controllers=by,totals=dict(totals),timeout_is_not_deadlock=True,unknown_is_not_known_failure=True));return dict(totals)

def reuse_audit():
    from shared_rollout_db.src.rollout_db import connect
    from shared_rollout_db.src.planner import preflight
    local_names=[read(HERE/'TEACHERS.json')[i]['name'] for i in [8,10,12,13]]
    local_distill={r['teacher']:r for r in read(OLD/'DISTILLATION_REPORT.json')['controllers']}
    local_checkpoints={}
    for n in local_names:
        ck=OLD/'students'/n/'best.pkl';complete=read(ck.parent/'COMPLETE.json');d=local_distill[n];assert complete['checkpoint_sha256']==sha(ck) and d['best_step']!=0 and d['mean_cosine']>=.90 and d['within_05_mps']>=.80;local_checkpoints[n]=sha(ck)
    oldreceipt=read(OLD/'CHOICES_FROZEN.receipt.json')
    for path,h in oldreceipt.get('files',oldreceipt.get('sha256',{})).items():assert sha(path)==h
    report={'local_checkpoints':local_checkpoints,'local_choices_sha256':sha(OLD/'FROZEN_CHOICES.json'),'local_choices_receipt_sha256':sha(OLD/'CHOICES_FROZEN.receipt.json')}
    for phase in ['main','teacher']:
        cases=[c for c in read(OLD/f'{phase}_CASES.json') if phase=='teacher' or c['base_controller_name'] in local_names];snapshot={r['rollout_uid']:r for r in read(OLD/f'{phase}_DB_ROWS.json')}
        manifest=HERE/f'REUSED_{phase.upper()}_PLANNED_ROLLOUTS.json';write(manifest,dict(requests=[{k:c[k] for k in ['state_uid','eta_uid','controller_uid','seed_key']} for c in cases]));pf=preflight(manifest);write(HERE/f'REUSED_{phase.upper()}_PREFLIGHT.json',pf)
        with connect(True) as db:
            for c in cases:
                row=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone();assert row and not row['conflict_quarantined'];assert all(row[k]==snapshot[c['rollout_uid']][k] for k in ['state_uid','eta_uid','controller_uid','seed_key','success','collision','timeout','numerical_failure','episode_length'])
        report[phase]=dict(records=len(cases),summary=pf['summary'],snapshot_sha256=sha(OLD/f'{phase}_DB_ROWS.json'),unknown_preserved=sum(snapshot[c['rollout_uid']]['numerical_failure'] for c in cases),new_rollouts=0)
    write(HERE/'REUSED_EVIDENCE_AUDIT.json',report)

def main():
    assert (HERE/'RHR_FROZEN.receipt.json').exists() and (HERE/'CHOICES_FROZEN.receipt.json').exists()
    reuse_audit()
    ts=read(HERE/'TEACHERS.json');new=read(HERE/'CAPABILITY_GATE.json')['eligible'];prior=[ts[i]['name'] for i in [1,7]];local=[ts[i]['name'] for i in [8,10,12,13]];primary=prior+new;combined=primary+local
    choices={**read(D/'teacher_student_macflow_v1/FROZEN_CHOICES.json')['controllers'],**read(D/'teacher_student_macflow_v1b/FROZEN_CHOICES.json')['controllers'],**read(HERE/'FROZEN_CHOICES.json')['controllers']};data=collect(HERE,combined);allowed=len(primary)>=8;sets={}
    for label,names in [('CONFIRMATORY',primary),('SENSITIVITY',new),('DEVELOPMENT',combined)]:
        ix=[combined.index(n) for n in names];subset={k:v[ix] for k,v in data.items()};sets[label]=analyze_set(label,names,subset,choices,allowed)
    sources=failures(combined,data)
    result=dict(utc=utc(),**sets,regate=read(HERE/'REGATE.json'),capability_gate=read(HERE/'CAPABILITY_GATE.json'),rhr_selected=read(HERE/'RHR_SELECTED.json'),failure_sources=sources,primary_prior_labels_opened=True,primary_provenance='round30b2 previously evaluated + newly regated and capability-eligible students; revised gate preregistered before new dev metrics and before eight candidate closed loops',development_provenance='primary plus4 round30 local students, previously opened labels',sensitivity_provenance='new round30c regated/capability-eligible students only',confirmatory_minimum=8,unknown_policy='lower0 upper1, retained and never rerun, not known failures',tuning_asymmetry='RHR tuned on all valid students; eta fixed on joint teachers and never tuned for students')
    write(HERE/'results.json',result);np.savez_compressed(HERE/'analysis_arrays.npz',**data,controller_names=combined,arms=ARMS);source_inventory('imports_analysis.json');seal('ANALYSIS_COMPLETE.receipt.json',[HERE/'results.json',HERE/'analysis_arrays.npz',HERE/'FAILURE_SOURCES.json',HERE/'TRANSFERS.json',HERE/'analysis.py'],rollouts_rerun=0)
if __name__=='__main__':main()
