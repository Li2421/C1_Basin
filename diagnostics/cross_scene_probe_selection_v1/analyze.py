"""Read evaluation labels only after the immutable probe selection receipt."""
from core import *
from collections import defaultdict,Counter

def main():
    receipt=read(HERE/'PROBE_SELECTION_FREEZE.receipt.json');assert sha(HERE/'PROBE_SELECTION_FREEZE.json')==receipt['sha256'];freeze=read(HERE/'PROBE_SELECTION_FREEZE.json');active=freeze['active_variants'];choice=freeze['selections'];cases=[c for c in read(HERE/'main_CASES.json') if c['variant'] in active];assert all((HERE/'main/raw'/f"{c['rollout_uid']}.json").exists() for c in cases)
    arr={v:np.zeros((40,3,2,7),np.int8) for v in active};unk={v:np.zeros((40,3,2,7),np.int8) for v in active};fails=defaultdict(lambda:defaultdict(list));name={n:i for i,n in enumerate(NAMES)};arm={a:i for i,a in enumerate(ARMS)}
    for c in cases:
        r=read(HERE/'main/raw'/f"{c['rollout_uid']}.json");assert r['case']==c;v=c['variant'];ix=(c['state_index'],name[c['base_controller_name']],c['seed'],arm[c['arm']]);arr[v][ix]=r['success'];unk[v][ix]=r['numerical_failure'];
        if not r['success']:fails[v][c['arm']].append(dict(outcome=r['outcome'],step=r['episode_length'],uid=c['rollout_uid']))
    rates={v:arr[v].mean(axis=(0,1,2)) for v in active};upper={v:(arr[v]+unk[v]).mean(axis=(0,1,2)) for v in active};oracle={v:float(np.max(rates[v])) for v in active};selected={v:float(rates[v][arm[choice[v]]]) for v in active};base={v:float(rates[v][arm['S15_t2']]) for v in active};regret={v:oracle[v]-selected[v] for v in active};menu={v:float(oracle[v]-max(rates[v][arm['S13_t1']],rates[v][arm['S15_t2']])) for v in active}
    rng=np.random.default_rng(202610142020);B=5000;boot_d2=np.empty(B);boot_d1=np.empty(B);boot_max=np.empty(B)
    for b in range(B):
        rv=[];dv=[]
        for v in active:
            ids=rng.integers(0,40,40);rr=arr[v][ids].mean(axis=(0,1,2));rv.append(float(np.max(rr)-rr[arm[choice[v]]]));dv.append(float(rr[arm['S15_t2']]-rr[arm[choice[v]]]))
        boot_d1[b]=float(np.mean(rv));boot_max[b]=float(np.max(rv));boot_d2[b]=float(np.mean(dv))
    d1_mean=float(np.mean(list(regret.values())));d1_max=float(max(regret.values()));d2=float(np.mean([base[v]-selected[v] for v in active]));d2ci=np.quantile(boot_d2,[.025,.975]).tolist()
    regret_bounds={v:[max(0.,float(np.max(rates[v])-upper[v][arm[choice[v]]])),float(np.max(upper[v])-rates[v][arm[choice[v]]])] for v in active};d1_mean_bounds=[float(np.mean([regret_bounds[v][i] for v in active])) for i in (0,1)];d1_max_bounds=[float(max(regret_bounds[v][i] for v in active)) for i in (0,1)];d2_bounds=[float(np.mean([rates[v][arm['S15_t2']]-upper[v][arm[choice[v]]] for v in active])),float(np.mean([upper[v][arm['S15_t2']]-rates[v][arm[choice[v]]] for v in active]))]
    d1_status='SUPPORTED' if d1_mean_bounds[1]<=.03 and d1_max_bounds[1]<=.10 else 'REJECTED' if d1_mean_bounds[0]>.03 or d1_max_bounds[0]>.10 else 'INCONCLUSIVE';d2_status='SUPPORTED' if d2_bounds[1]<0 and d2ci[1]<0 else 'REJECTED' if d2_bounds[0]>=0 or d2ci[0]>=0 else 'INCONCLUSIVE'
    result=dict(schema='round20_results_v1',utc=utc(),active_variants=active,probe_selection_receipt_sha256=sha(HERE/'PROBE_SELECTION_FREEZE.receipt.json'),rows=len(cases),unknown=int(sum(unk[v].sum() for v in active)),rates={v:{a:float(rates[v][i]) for i,a in enumerate(ARMS)} for v in active},upper_rates={v:{a:float(upper[v][i]) for i,a in enumerate(ARMS)} for v in active},selected=choice,oracle={v:dict(arm=ARMS[int(np.argmax(rates[v]))],rate=oracle[v]) for v in active},regret=regret,regret_bounds=regret_bounds,menu_regret=menu,D1=dict(status=d1_status,mean=d1_mean,max=d1_max,mean_ci=np.quantile(boot_d1,[.025,.975]).tolist(),max_ci=np.quantile(boot_max,[.025,.975]).tolist(),unknown_mean_bounds=d1_mean_bounds,unknown_max_bounds=d1_max_bounds,thresholds=[.03,.10]),D2=dict(status=d2_status,mean_regret_difference=d2,ci=d2ci,unknown_bounds=d2_bounds,baseline='S15_t2'),D3=dict(status='DESCRIPTIVE',within_3pp=sum(x<=.03 for x in menu.values()),total=len(active),fraction=float(np.mean([x<=.03 for x in menu.values()]))),failure_summary={v:{a:dict(counts=dict(Counter(x['outcome'] for x in rows)),earliest=min([x['step'] for x in rows],default=None),latest=max([x['step'] for x in rows],default=None)) for a,rows in fs.items()} for v,fs in fails.items()},failure_records=fails)
    result['D3']['status']='SUPPORTED' if all(x<=.03 for x in menu.values()) else 'REJECTED'
    if len(active)<4:result['D1']['status']=result['D2']['status']=result['D3']['status']='INCONCLUSIVE'
    write(HERE/'results.json',result)
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(15,5));x=np.arange(len(active));w=.11
    for i,a in enumerate(ARMS):ax.bar(x+(i-3)*w,[rates[v][i] for v in active],w,label=a)
    ax.set_xticks(x,[v.replace('_',' ') for v in active],rotation=30,ha='right');ax.set_ylim(0,1.08);ax.set_ylabel('Success rate');ax.legend(ncol=7,loc='upper center');fig.tight_layout();fig.savefig(HERE/'seven_arms_by_variant.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,6));
    for v,f in freeze['features'].items():ax.scatter(f['13']['stall'],f['15']['stall'],label=v)
    ax.set_xlabel('stall(η13)');ax.set_ylabel('stall(η15)');ax.plot([0,1],[.05,1.05],ls='--',c='gray');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(HERE/'probe_stall_scatter.png',dpi=180);plt.close(fig)
    print(result['D1'],result['D2'],result['D3'])
if __name__=='__main__':main()
