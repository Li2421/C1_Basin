from core import *
from models import csvwrite,MENU
from collections import Counter
from shared_rollout_db.src.rollout_db import connect,canonical
from shared_rollout_db.src.planner import preflight

def summarize(Y,U,available,arms,names,choices,methods,seed):
    lo=Y.mean(axis=(2,3));hi=(Y+U).mean(axis=(2,3));oracle=np.max(np.where(available,lo,-np.inf),axis=1);nstate=Y.shape[2];rng=np.random.default_rng(seed);weights=np.array([np.bincount(d,minlength=nstate)/nstate for d in rng.integers(0,nstate,(5000,nstate))]);bl=np.einsum('bs,cas->bca',weights,Y.mean(-1));bh=np.einsum('bs,cas->bca',weights,(Y+U).mean(-1));bo=np.max(np.where(available[None],bl,-np.inf),axis=2);out={};rows=[]
    for method in methods:
        selected=[arms.index(c[method]) for c in choices];regret=oracle-lo[np.arange(len(names)),selected];bounds=[];gains=[];gainbounds=[]
        for i,a in enumerate(selected):
            other=[j for j in range(len(arms)) if available[i,j] and j!=a];bounds.append([max([0.]+[lo[i,j]-hi[i,a] for j in other]),max([0.]+[hi[i,j]-lo[i,a] for j in other])]);b=arms.index(choices[i]['source_average']);gain=lo[i,a]-lo[i,b];gains.append(gain);gainbounds.append([0.,0.] if a==b else [lo[i,a]-hi[i,b],hi[i,a]-lo[i,b]])
            rows.append(dict(method=method,controller=names[i],selected_arm=arms[a],success_lower=lo[i,a],success_upper=hi[i,a],oracle_arm=arms[int(np.argmax(np.where(available[i],lo[i],-np.inf)))],oracle_lower=oracle[i],regret=regret[i],regret_lower=bounds[-1][0],regret_upper=bounds[-1][1],gain_vs_source_average=gain))
        breg=bo-bl[:,np.arange(len(names)),selected]
        baseline_selected=[arms.index(c['source_average']) for c in choices]
        bgain=(bl[:,np.arange(len(names)),selected]-bl[:,np.arange(len(names)),baseline_selected]).mean(1)
        point_pass=bool(np.mean(gains)>0 and regret.max()<=.05);glo,ghi=np.mean(gainbounds,axis=0);maxlo,maxhi=np.max(bounds,axis=0);certain=bool(glo>0 and maxhi<=.05);possible=bool(ghi>0 and maxlo<=.05);status='SUPPORTED' if certain else 'REJECTED' if not possible else 'INCONCLUSIVE'
        out[method]=dict(mean_regret=float(regret.mean()),max_regret=float(regret.max()),mean_regret_unknown_bounds=np.mean(bounds,axis=0),max_regret_unknown_bounds=[maxlo,maxhi],mean_gain_vs_source_average=float(np.mean(gains)),mean_gain_unknown_bounds=[glo,ghi],mean_gain_CI95=np.quantile(bgain,[.025,.975]),point_gate_pass=point_pass,transfer_status=status,mean_regret_CI95=np.quantile(breg.mean(1),[.025,.975]),max_regret_CI95=np.quantile(breg.max(1),[.025,.975]),choices=[r for r in rows if r['method']==method])
    arm_rows=[dict(controller=n,arm=arm,success_count=int(Y[i,a].sum()),unknown_count=int(U[i,a].sum()),success_lower=lo[i,a],success_upper=hi[i,a]) for i,n in enumerate(names) for a,arm in enumerate(arms) if available[i,a]]
    return out,rows,arm_rows

def main():
    prereg();labels_allowed();gate();selector=read(HERE/'SELECTOR_MODELS.json');assert sha(HERE/'SELECTOR_MODELS.json')==read(HERE/'SELECTOR_FROZEN.receipt.json')['models_sha256'];p=read(HERE/'protocol.json');names=p['controller_order'];states=p['test_states'];old=read(R5/'stage_2/protocol.json');cases=read(HERE/'main_CASES.json');assert len(cases)==3072
    rawpaths=[HERE/'main/raw'/f"{c['rollout_uid']}.json" for c in cases];assert all(q.exists() for q in rawpaths)
    raw_inventory=[dict(rollout_uid=c['rollout_uid'],sha256=sha(q)) for c,q in zip(cases,rawpaths)]
    opening=HERE/'BLEND_ENDPOINT_OPENING.receipt.json'
    if opening.exists():assert read(opening)['raw_inventory']==raw_inventory
    else:write(opening,dict(utc=utc(),all_new_raw_present=len(rawpaths),raw_inventory=raw_inventory,selector_receipt_sha256=sha(HERE/'SELECTOR_FROZEN.receipt.json'),R6_receipt_sha256=sha(HERE/'R6_FROZEN.receipt.json'),no_interim_new_endpoint_analysis=True))
    Y=np.zeros((12,4,32,4));U=Y.copy();lookup={};oldindex=[];manifest=[]
    from diagnostics.controller_q_diversity_v1.experiment import case
    for i,n in enumerate(names):
        for a,eta in enumerate([10,13]):
            for si,s in enumerate(states):
                seeds=[]
                for seed in range(4):
                    c=case(old,s,n,eta,seed);rawpath=R5/'stage_2/raw'/f"{c['rollout_uid']}.json";r=read(rawpath);assert r['case']==c;Y[i,a,si,seed]=r['success'];U[i,a,si,seed]=r['numerical_failure'];seeds.append(c['seed_key']);oldindex.append(dict(rollout_uid=c['rollout_uid'],source_path=str(rawpath),sha256=sha(rawpath),unknown=bool(r['numerical_failure'])));lookup[n,MENU[a],si,seed]=r
                manifest.append(dict(state_uid=c['state_uid'],eta_uid=c['eta_uid'],controller_uid=c['controller_uid'],seed_keys=seeds))
    write(HERE/'fixed_PLANNED_ROLLOUTS.json',dict(requests=manifest));write(HERE/'fixed_CACHE_PREFLIGHT.json',preflight(HERE/'fixed_PLANNED_ROLLOUTS.json'));write(HERE/'FIXED_REUSED_INDEX.json',oldindex)
    unknown=0;counts=Counter();traces=0
    with connect(True) as db:
        for c,q in zip(cases,rawpaths):
            r=read(q);assert r['case']==c and r['protocol_sha256']==sha(HERE/'protocol.json');assert sha(Path(r['trace_path']))==r['trace_sha256'];traces+=1;d=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone();assert d and not d['conflict_quarantined'];assert all(d[k]==r[k] for k in ['success','deadlock','timeout','collision','numerical_failure']);i=names.index(c['base_controller_name']);a=MENU.index(c['arm']);s=c['state_index'];seed=c['seed'];Y[i,a,s,seed]=r['success'];U[i,a,s,seed]=r['numerical_failure'];unknown+=r['numerical_failure'];counts[r['outcome']]+=1;lookup[c['base_controller_name'],c['arm'],s,seed]=r
        probe_rows=db.execute('SELECT count(*) FROM rollout WHERE experiment_uid=?',(EXP+'_unlabeled',)).fetchone()[0];assert probe_rows==0
    assert np.all(Y+U<=1);av=np.ones((12,4),bool);np.savez_compressed(HERE/'BLEND_LABELS.npz',success=Y,unknown=U,available=av,controller_names=np.array(names),arms=np.array(MENU))
    blend,rows,armrows=summarize(Y,U,av,MENU,names,selector['blend_choices'],['G','G_E','source_average'],202610131102);csvwrite('BLEND_ACTIONS_REGRETS.csv',rows);csvwrite('BLEND_ARM_RATES.csv',armrows)
    z=np.load(R10/'LABELS.npz');realnames=[c['controller'] for c in selector['real_choices']];ids=[z['controllers'].tolist().index(n) for n in realnames];real,rr,ar=summarize(z['success'][ids],z['unknown'][ids],z['available'][ids],z['arms'].tolist(),realnames,selector['real_choices'],['G','G_E','source_average','nearest_neighbor'],202610131101);csvwrite('REAL_ACTIONS_REGRETS.csv',rr)
    b=read(HERE/'STAGE_B.json');HL=b['H_L'];HG=blend['G']['transfer_status'];HGE=blend['G_E']['transfer_status']
    if HL=='SUPPORTED' and HG=='SUPPORTED':mapping='Decomposed controller-Q learned and transferred within tested family; prospective real-controller/state confirmation remains necessary.'
    elif HL=='REJECTED' and HGE=='SUPPORTED':mapping='Transferable component is early E; late component either covered by schedules or requires another representation.'
    elif HL=='REJECTED' and HG=='REJECTED' and HGE=='REJECTED':mapping='Controller descriptors explain E but remaining regret cannot be eliminated by these frozen zero-shot rules; known limitation.'
    else:mapping='INCONCLUSIVE/unmapped mixed pattern under frozen conclusion table; no forced favorable conclusion.'
    post=dict(status='PASS',main_requests=3072,new_trace_verified=traces,unknown=unknown,quarantined=0,missing=0,raw_counts=counts,existing_fixed_records_reused=3072,existing_fixed_unknown=int(U[:,:2].sum()),probe_success_rows=probe_rows,genuinely_missing_executed_only=True)
    write(HERE/'main_POSTFLIGHT.json',post);result=dict(status='NUMERICAL_COMPLETE',H_L=HL,H_transfer=HG,H_transfer_E_only=HGE,conclusion_mapping=mapping,stage_B=b,real_LOCO=real,blend_endpoint=blend,probe_audit=read(HERE/'R6_AUDIT.json'),postflight=post,evidence_level='DEVELOPMENT; blends NEW_CONTROLLER_TEST_FROM_OPENED_ANCHORS, not independent confirmation',videos='AGENTS pure Q evaluation exemption preregistered; no video entry required',label_gates=dict(R6=read(HERE/'R6_FROZEN.receipt.json'),selectors=read(HERE/'SELECTOR_FROZEN.receipt.json'),endpoint_utc=read(HERE/'BLEND_ENDPOINT_OPENING.receipt.json')['utc']))
    write(HERE/'results.json',result);print(json.dumps(clean(dict(H_L=HL,H_transfer=HG,H_transfer_E_only=HGE,blend={k:{a:v[a] for a in ['mean_regret','max_regret','mean_gain_vs_source_average','point_gate_pass','transfer_status']} for k,v in blend.items()},unknown=unknown))))
if __name__=='__main__':main()
