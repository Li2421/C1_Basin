"""Paired final-checkpoint estimates and training curves; no model selection."""
import csv,json,hashlib,os
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/tmp/c1-four-mpl')
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt=None

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/c1_four_objectives_multiseed';ARMS=['P','PS','Pg','full']

def describe(rows):
    return dict(n=len(rows),success=sum(r['success'] for r in rows),deadlock=sum(r['deadlock'] and r['controller_error'] is None and not r['collision'] for r in rows),timeout=sum(r['timeout'] for r in rows),
        collision=sum(r['collision'] for r in rows),controller_errors=sum(r['controller_error'] is not None for r in rows),
        any_deadlock_events=sum(r['any_deadlock'] for r in rows),
        success_rate=float(np.mean([r['success'] for r in rows])),mean_stagnation=float(np.mean([r['stagnation'] for r in rows])),
        median_stagnation=float(np.median([r['stagnation'] for r in rows])),total_stagnation=float(sum(r['stagnation'] for r in rows)))

def main():
    p=json.loads((OUT/'protocol.json').read_text());assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in p['source_hashes'].items())
    rows=[];run_stats=[];histories={}
    for arm in ARMS:
        for seed in p['training_seeds']:
            name=f'{arm}_seed{seed}';dest=OUT/'runs'/name
            done=json.loads((dest/'complete.json').read_text());assert done['scheduled']==80 and done['source_hashes_verified']
            histories[(arm,seed)]=json.loads((dest/'history.json').read_text())
            rr=[json.loads(f.read_text()) for f in (OUT/'evaluation'/name).glob('4*.json')];assert len(rr)==192
            rows.extend(rr)
            for cond in ['matched','independent']:run_stats.append(dict(arm=arm,training_seed=seed,condition=cond,**describe([r for r in rr if r['condition']==cond])))
    baseline=[json.loads(f.read_text()) for f in (OUT/'evaluation/baseline').glob('4*.json')];assert len(baseline)==192
    for seed in p['training_seeds']:
        hs=[json.loads((OUT/'runs'/f'{arm}_seed{seed}'/'complete.json').read_text())['initial_parameter_hash'] for arm in ARMS];assert len(set(hs))==1
        schedules=[[r['rid'] for r in histories[(arm,seed)]] for arm in ARMS];assert all(s==schedules[0] for s in schedules)
    pooled={arm:{cond:describe([r for r in rows if r['arm']==arm and r['condition']==cond]) for cond in ['matched','independent']} for arm in ARMS}
    rng=np.random.default_rng(20261101);contrasts={};lookup={(r['arm'],r['training_seed'],r['rid'],r['execution_seed']):r for r in rows}
    for before,after in [('P','Pg'),('PS','full'),('P','PS'),('Pg','full'),('P','full')]:
        for cond in ['matched','independent']:
            exseeds=[p['matched_execution_seed']] if cond=='matched' else p['independent_execution_seeds']
            d_success=np.zeros((3,64));d_stag=np.zeros((3,64))
            for j,s in enumerate(p['training_seeds']):
                for i,rid in enumerate(p['test_ids']):
                    aa=[lookup[(before,s,rid,k)] for k in exseeds];bb=[lookup[(after,s,rid,k)] for k in exseeds]
                    d_success[j,i]=np.mean([float(b['success'])-float(a['success']) for a,b in zip(aa,bb)])
                    d_stag[j,i]=np.mean([b['stagnation']-a['stagnation'] for a,b in zip(aa,bb)])
            si=rng.integers(0,3,(10000,3));ii=rng.integers(0,64,(10000,64))
            bs=d_success[si[:,:,None],ii[:,None,:]].mean((1,2));bt=d_stag[si[:,:,None],ii[:,None,:]].mean((1,2))
            contrasts[before+' -> '+after+' / '+cond]=dict(success_difference=float(d_success.mean()),success_difference_by_seed=d_success.mean(1).tolist(),
                success_difference95=np.quantile(bs,[.025,.975]).tolist(),stagnation_difference=float(d_stag.mean()),stagnation_difference_by_seed=d_stag.mean(1).tolist(),
                stagnation_difference95=np.quantile(bt,[.025,.975]).tolist())
    allrows=rows+baseline
    safety=dict(counts={k:sum(r['safety'][k] for r in allrows) for k in ['steps','agent_collision_steps','wall_collision_steps','outside_endpoints','cbf_violations','speed_violations']},
        minima={k:min(r['safety'][k] for r in allrows) for k in ['min_center_separation','min_agent_surface_clearance','min_wall_surface_clearance','min_pair_cbf_residual','min_wall_cbf_residual']})
    training={}
    for arm in ARMS:
        training[arm]={}
        for seed in p['training_seeds']:
            h=histories[(arm,seed)];norms=[r['gradient_norm'] for r in h if 'gradient_norm' in r]
            training[arm][str(seed)]=dict(scheduled=len(h),accepted=sum(r['accepted'] for r in h),base_errors=sum(r['base_error'] is not None for r in h),
                proposal_errors=sum(len(r['proposal_errors']) for r in h),backtrack_halvings=sum(r.get('backtracks',0) for r in h),max_gradient_norm=max(norms,default=None))
    result=dict(protocol=p,pooled=pooled,baseline={c:describe([r for r in baseline if r['condition']==c]) for c in ['matched','independent']},
        per_training_seed=run_stats,contrasts=contrasts,safety=safety,training=training,total_executions=len(allrows),
        verification=dict(frozen_hashes=True,identical_initialization_within_seed=True,identical_sample_schedule_within_seed=True),
        uncertainty='Paired bootstrap resamples3 training seeds and64 initial states; preserves pairing and averages independent execution replicas within state. Only3 training seeds, not a universal guarantee.')
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with (OUT/'per_seed_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(run_stats[0]));writer.writeheader();writer.writerows(run_stats)
    curve_rows=[]
    for arm in ARMS:
        for step in range(80):
            per_seed=[histories[(arm,s)][step] for s in p['training_seeds']]
            terms=np.asarray([r.get('terms',[np.nan]*4) for r in per_seed])
            curve_rows.append(dict(arm=arm,step=step+1,accepted_rate=float(np.mean([r['accepted'] for r in per_seed])),
                P_mean=float(np.nanmean(terms[:,0])),g_mean=float(np.nanmean(terms[:,1])),S_mean=float(np.nanmean(terms[:,2])),
                full_mean=float(np.nanmean(terms[:,3])),gradient_norm_median=float(np.nanmedian([r.get('gradient_norm',np.nan) for r in per_seed]))))
    with (OUT/'training_curves.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(curve_rows[0]));writer.writeheader();writer.writerows(curve_rows)
    result['training_curve_data']='training_curves.csv; matplotlib unavailable in this locked experiment environment, so no raster plot was created.'
    if plt is not None:
        fig,axes=plt.subplots(2,3,figsize=(14,8));colors=dict(P='#222222',PS='#d08b00',Pg='#2673bf',full='#bd3860')
        for ax,index,title in zip(axes.flat[:4],range(4),['P','g','S (conditioned)','Frozen full risk']):
            for arm in ARMS:
                yy=np.asarray([[r.get('terms',[np.nan]*4)[index] for r in histories[(arm,s)]] for s in p['training_seeds']])
                ax.plot(np.arange(1,81),np.nanmean(yy,axis=0),label=arm,color=colors[arm])
            ax.set_title(title);ax.set_xlabel('Scheduled update');ax.grid(alpha=.2)
        axes[0,0].legend();fig.tight_layout();fig.savefig(OUT/'training_curves.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(pooled=pooled,baseline=result['baseline'],contrasts=contrasts,training=training,safety=safety),indent=2))

if __name__=='__main__':main()
