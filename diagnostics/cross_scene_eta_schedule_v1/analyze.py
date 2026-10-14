from core import *
from collections import Counter,defaultdict
import csv
from shared_rollout_db.src.rollout_db import connect

def csvwrite(name,rows):
    with (HERE/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(clean(rows))
def ci(x):return np.quantile(x,[.025,.975],axis=0)
def decision(certain,possible):return 'SUPPORTED' if certain else 'REJECTED' if not possible else 'INCONCLUSIVE'
def regret_bounds(lo,hi,selected):
    other=[a for a in range(7) if a!=selected];return max(0.,float(np.max(lo[other]-hi[selected]))),max(0.,float(np.max(hi[other]-lo[selected])))
def possible_source(lo,hi,candidates):
    best=max(lo[candidates]);return [a for i,a in enumerate(candidates) if hi[a]>=best and all(hi[a]>lo[j] for j in candidates[:i])]
def statistics(Y,U):
    lc=Y.sum((-1,-2)).astype(np.int64);hc=(Y+U).sum((-1,-2)).astype(np.int64);lo=lc/160;hi=hc/160;slc=lc.sum(1);shc=hc.sum(1);sl=slc/480;sh=shc/480;rng=np.random.default_rng(202610131301);weights=np.array([[np.bincount(x,minlength=40)/40 for x in rng.integers(0,40,(5000,40))] for s in SCENES]);mult=np.rint(weights*40).astype(np.int64);blc=np.einsum('sbk,scak->bsca',mult,Y.sum(-1).astype(np.int64));bhc=np.einsum('sbk,scak->bsca',mult,(Y+U).sum(-1).astype(np.int64));bl=blc/160;bh=bhc/160;scb=blc.sum(2);ucb=bhc.sum(2);bs=scb/480;bu=ucb/480;fixed=[0,1,2];x1rows=[];x1controller=[];x1boot=(scb[:,:,4]-scb[:,:,:3].max(-1))/480;point_counts=slc[:,4]-slc[:,:3].max(-1);bound_counts=np.c_[slc[:,4]-shc[:,:3].max(-1),shc[:,4]-slc[:,:3].max(-1)];point=point_counts/480;bounds=bound_counts/480
    for s,sc in enumerate(SCENES):
        x1rows.append(dict(scene=sc,S13_t2=sl[s,4],best_fixed_arm=ARMS[int(sl[s,:3].argmax())],best_fixed=sl[s,:3].max(),difference=point[s],unknown_lower=bounds[s,0],unknown_upper=bounds[s,1],CI95=ci(x1boot[:,s]),CI95_unknown_envelope=[ci(bs[:,s,4]-bu[:,s,:3].max(-1))[0],ci(bu[:,s,4]-bs[:,s,:3].max(-1))[1]],point_pass=bool(point_counts[s]>=-24)))
        for c,n in enumerate(NAMES):
            delta=lo[s,c,4]-lo[s,c,:3].max();bd=bl[:,s,c,4]-bl[:,s,c,:3].max(-1);x1controller.append(dict(scene=sc,controller=n,S13_t2=lo[s,c,4],best_fixed=lo[s,c,:3].max(),difference=delta,CI95=ci(bd),unknown_lower=lo[s,c,4]-hi[s,c,:3].max(),unknown_upper=hi[s,c,4]-lo[s,c,:3].max()))
    X1=dict(status=decision(bool((bound_counts[:,0]>=-24).all()),bool((bound_counts[:,1]>=-24).all())),point_pass=bool((point_counts>=-24).all()),margin=.05,scenes=x1rows,controllers=x1controller,CI_is_descriptive_not_extra_gate=True)
    methods={};x2rows=[];bootreg={};unknown_boot={}
    for method,menu in [('fixed',fixed),('expanded',list(range(7)))]:
        regrets=[];lower=[];upper=[];regret_counts=[];lower_counts=[];upper_counts=[];choices=[];br=np.zeros((5000,4));bmin=br.copy();bmax=br.copy()
        for s,sc in enumerate(SCENES):
            sources=[i for i in range(4) if i!=s];source=slc[sources].sum(0);slo=source;shi=shc[sources].sum(0);a=menu[int(source[menu].argmax())];possible=possible_source(slo,shi,menu);rc=int(slc[s].max()-slc[s,a]);lbc=int(min(regret_bounds(slc[s],shc[s],a0)[0] for a0 in possible));ubc=int(max(regret_bounds(slc[s],shc[s],a0)[1] for a0 in possible));r=rc/480;lb=lbc/480;ub=ubc/480;regret_counts.append(rc);lower_counts.append(lbc);upper_counts.append(ubc);regrets.append(r);lower.append(lb);upper.append(ub);choices.append(ARMS[a]);source_b=scb[:,sources].sum(1);source_u=ucb[:,sources].sum(1);ids=np.array(menu)[source_b[:,menu].argmax(1)];br[:,s]=(scb[:,s].max(-1)-scb[np.arange(5000),s,ids])/480
            for b in range(5000):
                pos=possible_source(source_b[b],source_u[b],menu);bmin[b,s]=min(regret_bounds(scb[b,s],ucb[b,s],a0)[0] for a0 in pos)/480;bmax[b,s]=max(regret_bounds(scb[b,s],ucb[b,s],a0)[1] for a0 in pos)/480
            x2rows.append(dict(scene=sc,method=method,selected=ARMS[a],possible_source_choices=[ARMS[q] for q in possible],held_success_lower=sl[s,a],held_success_upper=sh[s,a],oracle_arm=ARMS[int(sl[s].argmax())],oracle=sl[s].max(),regret=r,regret_lower=lb,regret_upper=ub,CI95=ci(br[:,s]),CI95_unknown_envelope=[ci(bmin[:,s])[0],ci(bmax[:,s])[1]],source_scene_only=True))
        methods[method]=dict(integer_regret_counts=regret_counts,integer_unknown_lower_counts=lower_counts,integer_unknown_upper_counts=upper_counts,scene_denominator=480,choices=choices,regrets=regrets,mean_regret=float(np.mean(regrets)),max_regret=float(np.max(regrets)),mean_unknown_bounds=[float(np.mean(lower)),float(np.mean(upper))],max_unknown_bounds=[float(np.max(lower)),float(np.max(upper))],mean_CI95=ci(br.mean(1)),max_CI95=ci(br.max(1)));bootreg[method]=br;unknown_boot[method]=(bmin,bmax)
    f=methods['fixed'];e=methods['expanded'];pointpass=max(e['integer_regret_counts'])<=24 and 3*sum(e['integer_regret_counts'])<=sum(f['integer_regret_counts']);certain=max(e['integer_unknown_upper_counts'])<=24 and 3*sum(e['integer_unknown_upper_counts'])<=sum(f['integer_unknown_lower_counts']);possible=max(e['integer_unknown_lower_counts'])<=24 and 3*sum(e['integer_unknown_lower_counts'])<=sum(f['integer_unknown_upper_counts'])
    X2=dict(status=decision(certain,possible),point_pass=bool(pointpass),methods=methods,scenes=x2rows,mean_regret_ratio=e['mean_regret']/f['mean_regret'] if f['mean_regret'] else None,paired_mean_improvement_CI95=ci((bootreg['fixed']-bootreg['expanded']).mean(1)),bootstrap_reselects_source_argmax=True,tie_rule='earliest menu arm')
    effects=[]
    for s,sc in enumerate(SCENES):
        for c,n in enumerate(NAMES):
            for a,fa,sw in [(13,1,4),(15,2,6)]:
                Ec=int(lc[s,c,sw]-lc[s,c,0]);Lc=int(lc[s,c,sw]-lc[s,c,fa]);Nc=int(lc[s,c,fa]-lc[s,c,0]);E=Ec/160;L=Lc/160;N=Nc/160;assert abs(E-L-N)<1e-14
                effects.append(dict(scene=sc,controller=n,early_eta=a,E=E,L=L,N=N,both_ge10pp=bool(Ec>=16 and Lc>=16),E_CI95=ci(bl[:,s,c,sw]-bl[:,s,c,0]),L_CI95=ci(bl[:,s,c,sw]-bl[:,s,c,fa]),N_CI95=ci(bl[:,s,c,fa]-bl[:,s,c,0]),E_unknown_bounds=[lo[s,c,sw]-hi[s,c,0],hi[s,c,sw]-lo[s,c,0]],L_unknown_bounds=[lo[s,c,sw]-hi[s,c,fa],hi[s,c,sw]-lo[s,c,fa]],N_unknown_bounds=[lo[s,c,fa]-hi[s,c,0],hi[s,c,fa]-lo[s,c,0]]))
    X3=dict(status='INCONCLUSIVE',kind='DESCRIPTIVE_NO_BINARY_THRESHOLD',effects=effects,both_ge10pp=sum(x['both_ge10pp'] for x in effects),both_ge10pp_per_scene={sc:sum(x['both_ge10pp'] for x in effects if x['scene']==sc) for sc in SCENES},total_pairs=24,identity_E_minus_L_equals_N=True)
    return dict(integer_count_arithmetic_for_exact_gates_and_ties=True,X1=X1,X2=X2,X3=X3,scene_rate_CI95=np.moveaxis(ci(bs),0,-1),scene_rate_CI95_unknown_envelope=np.stack([ci(bs)[0],ci(bu)[1]],axis=-1),controller_rate_CI95=np.moveaxis(ci(bl),0,-1),scene_rates_lower=sl,scene_rates_upper=sh,controller_rates_lower=lo,controller_rates_upper=hi,bootstrap=dict(seed=202610131301,draws=5000,clusters_per_scene=40,controllers_retained=3,seeds_retained=4,independent_draws_across_scenes=True,shared_draws_across_arms=True)),x1controller,x2rows,effects

def main():
    prereg();gate();p=read(HERE/'protocol.json');cases=read(HERE/'main_CASES.json');assert len(cases)==13440;paths=[HERE/'main/raw'/f"{c['rollout_uid']}.json" for c in cases];assert all(q.exists() for q in paths);inventory=[dict(uid=c['rollout_uid'],sha256=sha(q)) for c,q in zip(cases,paths)];assert not (HERE/'ENDPOINT_OPENING.receipt.json').exists() or read(HERE/'ENDPOINT_OPENING.receipt.json')['raw_inventory']==inventory;freeze(HERE/'ENDPOINT_OPENING.receipt.json',dict(utc=read(HERE/'ENDPOINT_OPENING.receipt.json')['utc'] if (HERE/'ENDPOINT_OPENING.receipt.json').exists() else utc(),all13440_raw_present=True,raw_inventory=inventory,prereg_receipt_sha256=sha(HERE/'PREREGISTRATION.receipt.json'),no_interim_outcome_analysis=True));Y=np.zeros((4,3,7,40,4));U=Y.copy();seen=set();outcomes={sc:Counter() for sc in SCENES};failures=[];transitions=defaultdict(Counter);raw_by={};rates=[]
    with connect(True) as db:
        for c,path in zip(cases,paths):
            r=read(path);assert r['case']==c and r['protocol_sha256']==sha(HERE/'protocol.json');assert sha(r['trace_path'])==r['trace_sha256'];d=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone();assert d and not d['conflict_quarantined'];assert all(d[k]==r[k] for k in ['success','deadlock','timeout','collision','numerical_failure']);slot=(SCENES.index(c['scene']),NAMES.index(c['base_controller_name']),ARMS.index(c['arm']),c['state_index'],c['seed']);assert slot not in seen;seen.add(slot);Y[slot]=r['success'];U[slot]=r['numerical_failure'];outcomes[c['scene']][r['outcome']]+=1;raw_by[slot]=dict(outcome=r['outcome'])
            if r['outcome']!='success':failures.append(dict(scene=c['scene'],controller=c['base_controller_name'],arm=c['arm'],state_index=c['state_index'],seed=c['seed'],rollout_uid=c['rollout_uid'],type=r['outcome'],occurrence_step=r['error']['step'] if r['error'] else r['episode_length'],error=r['error'],native_first_deadlock_step=r['summary'].get('first_deadlock_step'),candidate_ever=r['summary'].get('ever_candidate_deadlock')))
    result,controllers,x2rows,effects=statistics(Y,U)
    for sc_i,sc in enumerate(SCENES):
        for c_i,n in enumerate(NAMES):
            for a_i,arm in enumerate(ARMS):
                rate=result['controller_rates_lower'][sc_i,c_i,a_i];upper=result['controller_rates_upper'][sc_i,c_i,a_i];rates.append(dict(scene=sc,controller=n,arm=arm,success=int(Y[sc_i,c_i,a_i].sum()),unknown=int(U[sc_i,c_i,a_i].sum()),total=160,lower=rate,upper=upper))
            for si in range(40):
                for seed in range(4):
                    for fixed,sw in [(0,4),(1,4),(0,6),(2,6)]:
                        a=raw_by[sc_i,c_i,fixed,si,seed];b=raw_by[sc_i,c_i,sw,si,seed];transitions[(sc,n,ARMS[fixed],ARMS[sw])][a['outcome']+' -> '+b['outcome']]+=1
    write(HERE/'FAILURE_RECORDS.json',failures);write(HERE/'FAILURE_TRANSITIONS.json',[dict(scene=k[0],controller=k[1],baseline=k[2],schedule=k[3],counts=v) for k,v in transitions.items()]);csvwrite('ARM_RATES.csv',rates);write(HERE/'X1_CONTROLLER.json',controllers);write(HERE/'X2_SCENES.json',x2rows);write(HERE/'X3_EFFECTS.json',effects);np.savez_compressed(HERE/'LABELS.npz',success=Y,unknown=U,scenes=np.array(SCENES),controllers=np.array(NAMES),arms=np.array(ARMS))
    X1=result['X1']['status'];X2=result['X2']['status']
    if X1=='SUPPORTED' or X2=='SUPPORTED':mapping='SUPPORTED shared action-compression/H_compress explanation on this prospective E8 four-scene population; not proof of a unique root cause.'
    else:mapping='Cross-scene regime elimination not prospectively confirmed; inspect descriptive within-scene phase effects and remaining LOSO regret. No post-hoc schedule/threshold tuning.'
    sl=result['scene_rates_lower'];descriptive=dict(best_schedule_minus_best_fixed={sc:float(sl[i,3:].max()-sl[i,:3].max()) for i,sc in enumerate(SCENES)},each_scene_has_both_effects_ge10pp=all(result['X3']['both_ge10pp_per_scene'][sc]>0 for sc in SCENES),no_schedule_point_superiority_in_Toy_DB_Four=bool(all(sl[i,3:].max()<=sl[i,:3].max() for i in range(3))),source_selected_schedules=result['X2']['methods']['expanded']['choices'])
    result.update(descriptive_mechanism_mapping=descriptive,status='NUMERICAL_COMPLETE_PENDING_VIDEO_GATE',conclusion_mapping=mapping,scenes=SCENES,controllers=NAMES,arms=ARMS,unknown=int(U.sum()),postflight=dict(status='PASS',planned=13440,actual=13440,unique_slots=len(seen),trace_hashes_verified=13440,missing=0,quarantined=0,unknown=int(U.sum()),outcomes=outcomes,unknown_never_rerun=True),scope='Prospective new40states/scene; original E8 affine Flow-reference controllers; no critic training; scene/Flow nested; old evidence DEVELOPMENT')
    write(HERE/'main_POSTFLIGHT.json',result['postflight']);write(HERE/'results.json',result);print(json.dumps(clean(dict(X1=X1,X2=X2,X3='DESCRIPTIVE',unknown=int(U.sum()),X1_scenes=result['X1']['scenes'],X2_methods=result['X2']['methods']))))
if __name__=='__main__':main()
