"""State-local equal-count sampling; paired hierarchical bootstrap, never fixed-index oracle."""
from common import *
from frozen_networks import stable_int
from catalog_names import full_type
import math,csv

def probability_best(successes,count,total=56):return 1.-(math.comb(total-successes,count)/math.comb(total,count) if total-successes>=count else 0.)

def per_state(lower,upper,names,state_uids,draws=None,rng=None):
    # Exact distribution of the mean of 200 independent subset maxima:
    # each subset is uniform without replacement, so its binary maximum is
    # Bernoulli(1-C(56-s,k)/C(56,k)). Lower/upper share the same draw via a
    # three-category multinomial. This integrates out irrelevant subset IDs;
    # the 200-trial randomness is redrawn for EVERY bootstrap state occurrence.
    n=len(lower);metrics=np.zeros((n,2,12));subsets=[]
    for i,(lo,hi) in enumerate(zip(lower,upper)):
        fixed=[]
        for count in (9,8):
            if draws is None:
                ix=np.stack([np.random.default_rng(stable_int('crosstype5-equal-count',202610195104,str(names[i]),str(state_uids[i]),count,repeat)).choice(56,count,replace=False) for repeat in range(200)])
                values=np.asarray([lo[ix].max(-1).mean(),hi[ix].max(-1).mean()]);subsets.append(dict(row=i,count=count,indices=ix))
            else:
                pl=probability_best(int(lo[:56].sum()),count);pu=probability_best(int(hi[:56].sum()),count);counts=rng.multinomial(200,[pl,max(0.,pu-pl),max(0.,1-pu)])
                values=np.asarray([counts[0],counts[0]+counts[1]])/200
            fixed.append(values)
        for b,v in enumerate((lo,hi)):
            schedule=v[57:66].max();query=v[66:74].max();human=v[56]
            metrics[i,b]=[fixed[0][b],schedule,fixed[1][b],query,v[:56].mean(),v[57:66].mean(),v[66:74].mean(),v[57],v[74],v[66],schedule-human,query-human]
    return metrics,subsets

METRICS=('冻结提议对·随机9对样本内最佳','时间表生成器提议对·9对样本内最佳','冻结提议对·随机8对样本内最佳','切换时重查·8个样本内最佳','冻结提议对·平均成功率','时间表生成器提议对·平均成功率','切换时重查·平均成功率','时间表生成器提议对·均值提议','冻结提议对·均值对角元','切换时重查·均值为前期η','时间表生成器提议对·样本内最佳相对人定双 η·按进度切换','切换时重查·样本内最佳相对人定双 η·按进度切换')

def draw_rows(data,types,rng,point=False):
    out=[]
    for t in types if point else rng.choice(types,len(types),replace=True):
        names=sorted(set(data['names'][data['types']==t].tolist()))
        for name in names if point else rng.choice(names,len(names),replace=True):
            pool=np.flatnonzero((data['types']==t)&(data['names']==name));assert len(pool)==8
            out.extend(pool if point else rng.choice(pool,len(pool),replace=True))
    return np.asarray(out,int)

def aggregate(values,data,rows,types,point=False):
    # draw_rows has contiguous equal-sized controller blocks inside each type;
    # retain bootstrap type multiplicities, with each type occurrence equal weight.
    result=[];start=0
    for t in types:
        count=len(set(data['names'][data['types']==t]))*8
        result.append(values[start:start+count].mean(0));start+=count
    assert start==len(rows);return np.mean(result,0)

def main():
    with np.load(HERE/'COVERAGE_BUNDLE.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    assert data['lower'].shape==(240,75);rng=np.random.default_rng(202610195105);results={};subsets=[]
    point_values,subsets=per_state(data['lower'],data['upper'],data['names'],data['state_uids'])
    groups={t:[t] for t in sorted(set(data['types'].tolist()))};groups['类型等权合并']=sorted(set(data['types'].tolist()))
    for group,types in groups.items():
        boot=[]
        for rep in range(5001):
            tt=types if rep==0 else rng.choice(types,len(types),replace=True).tolist();rows=[];type_values=[]
            for t in tt:
                cn=sorted(set(data['names'][data['types']==t].tolist()));cv=[]
                for name in cn if rep==0 else rng.choice(cn,len(cn),replace=True):
                    pool=np.flatnonzero((data['types']==t)&(data['names']==name));ix=pool if rep==0 else rng.choice(pool,len(pool),replace=True)
                    if rep==0:v=point_values[ix]
                    else:v,_=per_state(data['lower'][ix],data['upper'][ix],data['names'][ix],data['state_uids'][ix],draws=True,rng=rng)
                    cv.append(v.mean(0))
                type_values.append(np.mean(cv,0))
            boot.append(np.mean(type_values,0))
        boot=np.asarray(boot)*100
        metrics={m:dict(point=float(boot[0,0,k]),CI95=np.quantile(boot[1:,0,k],[.025,.975]),unknown_upper=dict(point=float(boot[0,1,k]),CI95=np.quantile(boot[1:,1,k],[.025,.975]))) for k,m in enumerate(METRICS)}
        differences={}
        for label,a,b in [('时间表生成器提议对相对同数量冻结提议对',1,0),('切换时重查相对同数量冻结提议对',3,2)]:
            diff=boot[:,:,a]-boot[:,:,b];differences[label]=dict(point=float(diff[0,0]),CI95=np.quantile(diff[1:,0],[.025,.975]),unknown_upper_difference=float(diff[0,1]),unknown_interval_conservative=[float(boot[0,0,a]-boot[0,1,b]),float(boot[0,1,a]-boot[0,0,b])])
        results[group]=dict(metrics=metrics,comparisons=differences)
        print('Coverage bootstrap complete',group,flush=True)
    write(HERE/'COVERAGE_RESULTS.json',dict(utc=utc(),evidence='DEVELOPMENT',bootstrap=5000,seed=202610195105,groups=results,equal_count_resampled_inside_each_bootstrap=True,subset_maximum_distribution='exact 200-trial multinomial hypergeometric integration',no_cross_state_index_oracle=True,unknown_bounds_conservative=True))
    # Point subsamples remain reproducible exactly, separately for k=9 and k=8.
    np.savez_compressed(HERE/'COVERAGE_POINT_SUBSETS.npz',random9=np.stack([r['indices'] for r in subsets if r['count']==9]),random8=np.stack([r['indices'] for r in subsets if r['count']==8]))
    lines=['# 后期η提议覆盖（开发证据）','','每个state独立比较；冻结提议对按数量9/8不放回抽样200次；5000次类型→controller→state配对bootstrap，每次重新抽样。未使用跨state固定提议序号oracle。','','| 类型 | 时间表9对最佳 % | 冻结随机9对最佳 % | 配对差 pp [95% CI] | 重查8个最佳 % | 冻结随机8对最佳 % | 配对差 pp [95% CI] |','|---|---|---|---|---|---|---|']
    for t,r in results.items():
        m=r['metrics'];d=list(r['comparisons'].values());lines.append('| '+full_type(t)+' | '+' | '.join([f"{m[METRICS[1]]['point']:.2f}",f"{m[METRICS[0]]['point']:.2f}",f"{d[0]['point']:.2f} [{d[0]['CI95'][0]:.2f}, {d[0]['CI95'][1]:.2f}]",f"{m[METRICS[3]]['point']:.2f}",f"{m[METRICS[2]]['point']:.2f}",f"{d[1]['point']:.2f} [{d[1]['CI95'][0]:.2f}, {d[1]['CI95'][1]:.2f}]"] )+' |')
    lines+=['','全部平均成功率、三种均值提议、相对人定双 η·按进度切换的差及unknown上界见 COVERAGE_RESULTS.json。','', '探索成功动作的后期η分布见 figures/late_eta_distribution.svg；仅从真实仿真标签读取。']
    (HERE/'COVERAGE.md').write_text('\n'.join(lines)+'\n')
    exploration_plot()
    seal('COVERAGE_STATISTICS.receipt.json',[HERE/n for n in ('COVERAGE_RESULTS.json','COVERAGE_POINT_SUBSETS.npz','COVERAGE.md','figures/late_eta_distribution.svg','coverage_statistics.py','COVERAGE_BUNDLE.receipt.json','COVERAGE_BOOTSTRAP_IMPLEMENTATION.md')])

def exploration_plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with np.load(HERE/'TRAINING_BUNDLE.npz',allow_pickle=False) as z:labels=z['labels'][:,:,34:40,0];eta=z['action'][:,:,34:40,3:6];frozen=z['proposals']
    base=read(HERE/'BASE_PROTOCOL.json');fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    for j,ax in enumerate(axes):
        for outcome,color,label in [(0,'#b87d77','Known failure'),(1,'#218575','Success')]:
            values=eta[...,j][labels==outcome]
            if len(values):ax.hist(values,bins=35,density=True,alpha=.45,color=color,label=f'{label} (n={len(values)})')
        ax.axvline(base['eta'][13][j],color='#7b3fa0',linestyle='--',label='eta13 (reference only)');ax.axvline(base['eta'][10][j],color='#c08016',linestyle=':',label='eta10 (reference only)');ax.axvspan(float(frozen[...,j].min()),float(frozen[...,j].max()),color='#4566ad',alpha=.12,label='Frozen proposals range');ax.set_xlabel(f'Late eta coordinate {j+1}');ax.set_ylabel('Density')
    axes[0].legend(fontsize=7);fig.suptitle('Exploration labels only; all 19 types are development evidence');fig.tight_layout();p=HERE/'figures/late_eta_distribution.svg';p.parent.mkdir(exist_ok=True);fig.savefig(p);plt.close(fig)
if __name__=='__main__':main()
