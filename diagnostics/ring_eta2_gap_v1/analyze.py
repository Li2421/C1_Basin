"""Prespecified Round18 TEST analysis; cannot execute before TEST freeze and postflight."""
from core import *
import html
from scipy.stats import pearsonr,spearmanr

ARMS=['F10','F13','F2','S200','S300','S400','S500','S600']
MODELS=['C0','C1','C3','C2','A0']
POLICIES=[m+'_fixed' for m in MODELS]+[m+'_state' for m in MODELS]+['F10']

def corr(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float);mask=np.isfinite(x)&np.isfinite(y)
    if mask.sum()<3 or np.std(x[mask])<1e-10 or np.std(y[mask])<1e-10:return dict(Pearson=None,Spearman=None,n=int(mask.sum()))
    return dict(Pearson=float(pearsonr(x[mask],y[mask]).statistic),
                Spearman=float(spearmanr(x[mask],y[mask]).statistic),n=int(mask.sum()))

def evaluate(label,choices,indices,states=None):
    states=np.arange(48) if states is None else np.asarray(states)
    yy=label[np.asarray(indices)][:,states]
    good=yy!=-1;success=yy==1
    count=good.sum((1,3));wins=success.sum((1,3))
    if np.any(count==0):return None
    rate=wins/count;oracle=rate.max(1)
    out=dict(oracle=oracle.tolist(),rates=rate.tolist(),known_per_arm=count.tolist())
    for policy in POLICIES:
        if policy=='F10':chosen=np.zeros((len(indices),len(states)),int)
        elif policy.endswith('_state'):
            chosen=np.array([[ARMS.index(choices[i][policy][s]) for s in states] for i in indices],int)
        else:
            chosen=np.array([[ARMS.index(choices[i][policy])]*len(states) for i in indices],int)
        selected=np.take_along_axis(success,chosen[:,:,None,None],axis=2)[:,:,0,:]
        selected_known=np.take_along_axis(good,chosen[:,:,None,None],axis=2)[:,:,0,:]
        denom=selected_known.sum((1,2))
        if np.any(denom==0):return None
        value=selected.sum((1,2))/denom
        regret=(oracle-value)*100
        out[policy]=dict(mean_regret_pp=float(regret.mean()),max_regret_pp=float(regret.max()),
                         per_controller_regret_pp=regret.tolist(),per_controller_success=value.tolist(),
                         per_controller_known_seed_count=denom.tolist())
    return out

def boot_contrasts(label,choices,groups):
    rng=np.random.default_rng(1818)
    pairs=[('B1_C2_minus_C0','C2_fixed','C0_fixed','eta2'),
           ('B4_C2_minus_A0','C2_fixed','A0_fixed','all'),
           ('coverage_C1_minus_C0','C1_fixed','C0_fixed','all'),
           ('representation_C3_minus_C0','C3_fixed','C0_fixed','all'),
           ('representation_controlled_C2_minus_C1','C2_fixed','C1_fixed','all'),
           ('coverage_controlled_C2_minus_C3','C2_fixed','C3_fixed','all')]
    draws={name:[] for name,*_ in pairs}
    for _ in range(10000):
        sample=rng.integers(0,48,size=48)
        values={g:evaluate(label,choices,idx,sample) if idx else None for g,idx in groups.items() if g in ('all','eta2')}
        for name,a,b,g in pairs:
            v=values[g]
            draws[name].append(None if v is None else v[a]['mean_regret_pp']-v[b]['mean_regret_pp'])
    out={}
    for name,a,b,g in pairs:
        vals=draws[name]
        out[name]=dict(group=g,contrast=f'{a} minus {b}',draws=10000,seed=1818,
                       CI95_pp=np.quantile(vals,[.025,.975]).tolist() if all(v is not None for v in vals) else None,
                       bootstrap_mean_pp=float(np.mean(vals)) if all(v is not None for v in vals) else None)
    return out

def figures(results):
    colors=['#2b6aa4','#e07436','#53a082','#8b61a3','#b24a62']
    def start(w,h):return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">','<rect width="100%" height="100%" fill="white"/>']
    def tx(x,y,s,size=12):return f'<text x="{x}" y="{y}" font-family="sans-serif" font-size="{size}">{html.escape(str(s))}</text>'
    parts=start(1200,600);parts.append(tx(20,35,'Round18 TEST mean regret (pp)',18))
    vals=[results['groups'][g]['policies'][m+'_fixed']['mean_regret_pp'] for g in ('eta2','non_eta2') for m in MODELS if results['groups'][g]['status']=='COMPLETE']
    ymax=max(vals+[1])*1.15
    for gi,g in enumerate(('eta2','non_eta2')):
        x0=100+gi*570;parts.append(tx(x0,70,g,16))
        if results['groups'][g]['status']!='COMPLETE':continue
        for j,m in enumerate(MODELS):
            v=results['groups'][g]['policies'][m+'_fixed']['mean_regret_pp'];h=v/ymax*380;x=x0+j*95
            parts.append(f'<rect x="{x}" y="{480-h:.1f}" width="70" height="{h:.1f}" fill="{colors[j]}"/>')
            parts.append(tx(x,505,m,13))
    parts.append('</svg>');(HERE/'MODEL_REGRET.svg').write_text('\n'.join(parts))
    parts=start(1400,780);parts.append(tx(20,30,'Frozen TEST controller choices',18))
    for j,m in enumerate(MODELS):parts.append(tx(570+j*155,60,m,12))
    for i,r in enumerate(results['controllers']):
        y=85+i*36;parts.append(tx(20,y+16,r['controller'][:55],10))
        for j,m in enumerate(MODELS):
            a=r['frozen_choices'][m+'_fixed'];x=570+j*155;parts.append(f'<rect x="{x}" y="{y}" width="135" height="27" fill="{colors[j]}"/>');parts.append(tx(x+6,y+18,a,12))
    parts.append('</svg>');(HERE/'CONTROLLER_CHOICES.svg').write_text('\n'.join(parts))
    parts=start(850,680);parts.append(tx(20,30,'eta2 probe mean progress vs F2 advantage (TEST)',17))
    rows=[r for r in results['controllers'] if r['eta2_probe']['progress'] is not None and r['F2_advantage_pp'] is not None]
    xv=[r['eta2_probe']['progress'] for r in rows];yv=[r['F2_advantage_pp'] for r in rows]
    xmin,xmax=min(xv+[0]),max(xv+[1]);ymin,ymax=min(yv+[-1]),max(yv+[1])
    parts.append('<line x1="90" y1="600" x2="760" y2="600" stroke="black"/><line x1="90" y1="600" x2="90" y2="90" stroke="black"/>')
    for r in rows:
        x=90+(r['eta2_probe']['progress']-xmin)/(xmax-xmin)*670
        y=600-(r['F2_advantage_pp']-ymin)/(ymax-ymin)*510
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{colors[0] if r["family"]=="v3" else colors[1]}"/>')
    parts.append(tx(300,650,'eta2 probe progress',13));parts.append(tx(10,90,'F2 advantage (pp)',13));parts.append('</svg>')
    (HERE/'ETA2_PROBE_SCATTER.svg').write_text('\n'.join(parts))

def main():
    test_labels_allowed()
    assert read(HERE/'test_DELIVERY_AUDIT.json')['status']=='PASS'
    choices=read(HERE/'TEST_CHOICES.json');names=choices['controller_order'];rows=choices['controllers']
    with np.load(HERE/'test_LABELS.npz',allow_pickle=False) as z:
        assert list(z['controller_names'])==names;lab=z['Y'].astype(np.int8)
    assert lab.shape==(18,48,8,4) and np.isin(lab,[-1,0,1]).all()
    count=(lab!=-1).sum((1,3));wins=(lab==1).sum((1,3))
    rate=np.divide(wins,count,out=np.full((18,8),np.nan),where=count>0)
    f2_margin=(rate[:,2]-np.nanmax(np.delete(rate,2,axis=1),axis=1))*100
    eta2_idx=[i for i,x in enumerate(f2_margin) if np.isfinite(x) and x>=3-1e-12]
    other_idx=[i for i in range(18) if i not in eta2_idx]
    groups={'all':list(range(18)),'eta2':eta2_idx,'non_eta2':other_idx}
    group_results={}
    for g,idx in groups.items():
        value=evaluate(lab,rows,idx) if idx else None
        group_results[g]=dict(status='COMPLETE' if value else 'INCONCLUSIVE',controller_count=len(idx),
                              policies={p:value[p] for p in POLICIES} if value else None,
                              unknown=int(np.sum(lab[idx]==-1)) if idx else 0,
                              oracle=value['oracle'] if value else None)
    assert group_results['all']['status']=='COMPLETE'
    boot=boot_contrasts(lab,rows,groups)
    allp=group_results['all']['policies'];etap=group_results['eta2']['policies'] if group_results['eta2']['status']=='COMPLETE' else None
    nonp=group_results['non_eta2']['policies'] if group_results['non_eta2']['status']=='COMPLETE' else None
    b1ci=boot['B1_C2_minus_C0']['CI95_pp'];b4ci=boot['B4_C2_minus_A0']['CI95_pp']
    headroom=len(eta2_idx)>=3
    b1='INCONCLUSIVE_NO_HEADROOM' if not headroom else 'INCONCLUSIVE' if b1ci is None or etap is None else 'SUPPORTED' if b1ci[1]<0 and etap['C2_fixed']['max_regret_pp']<=5 else 'REJECTED'
    catastrophes=[]
    for i,name in enumerate(names):
        drop=(allp['F10']['per_controller_success'][i]-allp['C2_fixed']['per_controller_success'][i])*100
        if drop>=10-1e-12:catastrophes.append(dict(controller=name,F10_minus_C2_pp=float(drop)))
    b2='SUPPORTED' if allp['C2_fixed']['mean_regret_pp']<=2 and allp['C2_fixed']['max_regret_pp']<=5 and not catastrophes else 'REJECTED'
    b3='INCONCLUSIVE' if nonp is None else 'SUPPORTED' if nonp['C2_fixed']['mean_regret_pp']<=nonp['C0_fixed']['mean_regret_pp']+.5 else 'REJECTED'
    b4='INCONCLUSIVE' if b4ci is None else 'SUPPORTED' if b4ci[1]<0 else 'REJECTED'
    with np.load(HERE/'FEATURE_MATRIX.npz',allow_pickle=False) as z:
        all_names=list(z['controller_names']);eta=z['eta2_raw'].astype(float)
    stat_names=['progress','projection_active','minimum_pair_distance','mean_speed','low_speed_steps']
    controller_rows=[]
    for i,name in enumerate(names):
        raw=eta[all_names.index(name)].reshape(16,5)
        with np.errstate(invalid='ignore'):mean=np.nanmean(raw,axis=0)
        controller_rows.append(dict(controller=name,family=rows[i]['family'],architecture=rows[i]['architecture'],
            seed_offset=rows[i]['seed_offset'],recipe=rows[i]['recipe'],eta2_type=i in eta2_idx,
            F2_advantage_pp=float(f2_margin[i]) if np.isfinite(f2_margin[i]) else None,
            eta2_probe={n:float(v) if np.isfinite(v) else None for n,v in zip(stat_names,mean)},
            frozen_choices={m+'_fixed':rows[i][m+'_fixed'] for m in MODELS},
            success_rate_by_arm={a:float(v) if np.isfinite(v) else None for a,v in zip(ARMS,rate[i])},
            known_seed_count_by_arm={a:int(v) for a,v in zip(ARMS,count[i])},
            regret_pp={m:allp[m+'_fixed']['per_controller_regret_pp'][i] for m in MODELS}))
    correlations={n:corr([r['eta2_probe'][n] if r['eta2_probe'][n] is not None else np.nan for r in controller_rows],f2_margin) for n in stat_names}
    # TRAIN-new is developmental; combine with TEST only to describe regime frequency.
    with np.load(HERE/'train_LABELS.npz',allow_pickle=False) as z:
        train_lab=z['Y'].astype(np.int8);train_names=list(z['controller_names'])
    train_rate=np.divide((train_lab==1).sum((1,3)),(train_lab!=-1).sum((1,3)),out=np.full((18,8),np.nan),where=(train_lab!=-1).sum((1,3))>0)
    train_margin=(train_rate[:,2]-np.nanmax(np.delete(train_rate,2,axis=1),axis=1))*100
    base=read(HERE/'FINAL_BASE_PROFILES.json');occurrence=[]
    for split,ns,margins in [('TRAIN-new',train_names,train_margin),('TEST',names,f2_margin)]:
        for n,m in zip(ns,margins):
            v=base['profiles'][n]
            if v['family']=='v3':occurrence.append(dict(split=split,controller=n,recipe=v['recipe'],architecture=v['architecture'],
                 seed_offset=v['seed_offset'],F2_advantage_pp=float(m) if np.isfinite(m) else None,
                 eta2_type=bool(np.isfinite(m) and m>=3-1e-12)))
    contrasts={}
    for key,a,b,g in [('coverage_C1_minus_C0','C1','C0','all'),('representation_C3_minus_C0','C3','C0','all'),
                      ('representation_controlled_C2_minus_C1','C2','C1','all'),('coverage_controlled_C2_minus_C3','C2','C3','all')]:
        contrasts[key]=dict(mean_regret_difference_pp=allp[a+'_fixed']['mean_regret_pp']-allp[b+'_fixed']['mean_regret_pp'],
                            **boot[key])
    results=dict(evidence_level='PROSPECTIVE_NEW_SEEDS_AND_STATES_WITH_SHARED_V3_RECIPES_AND_ARCHITECTURES',
       prereg_sha256=sha(HERE/'PREREGISTRATION.md'),amendment_sha256=sha(HERE/'PREREGISTRATION_AMENDMENT.md'),
       feature_freeze_sha256=sha(HERE/'FEATURES_FROZEN.receipt.json'),test_freeze_sha256=sha(HERE/'TEST_FREEZE.receipt.json'),
       test_labels_sha256=sha(HERE/'test_LABELS.npz'),controllers=controller_rows,groups=group_results,
       eta2_controller_names=[names[i] for i in eta2_idx],headroom_count=len(eta2_idx),headroom_met=headroom,
       hypotheses=dict(B1=b1,B2=b2,B3=b3,B4=b4),catastrophes=catastrophes,
       bootstrap=boot,factorial_contrasts=contrasts,eta2_probe_correlations=correlations,
       v3_eta2_occurrence=occurrence,unknown_count=int(np.sum(lab==-1)),
       thresholds=dict(eta2_margin_pp=3,headroom_min=3,B1_CI_upper_strictly_below_zero=True,
                       B1_max_regret_pp=5,B2_mean_regret_pp=2,B2_max_regret_pp=5,
                       B2_F10_drop_pp=10,B3_margin_pp=.5,B4_CI_upper_strictly_below_zero=True))
    write(HERE/'results.json',results);figures(results)
    rep=contrasts['representation_controlled_C2_minus_C1'];cov=contrasts['coverage_controlled_C2_minus_C3']
    rep_ci=rep['CI95_pp'];cov_ci=cov['CI95_pp']
    rep_help=rep_ci is not None and rep_ci[1]<0
    cov_help=cov_ci is not None and cov_ci[1]<0
    factor_summary=('表示与TRAIN-new覆盖均显示独立收益' if rep_help and cov_help else
                    '仅新增η2表示显示独立收益' if rep_help else
                    '仅TRAIN-new覆盖显示独立收益' if cov_help else
                    '两种因素的独立收益均未由描述性CI确认')
    lead=(f"η2 缺口 B1 {b1}（TEST η2 型 {len(eta2_idx)}/18，门槛≥3；C2−C0 CI {b1ci}）；"
          f"完整标准 B2 {b2}（C2 平均/最大遗憾 {allp['C2_fixed']['mean_regret_pp']:.2f}/{allp['C2_fixed']['max_regret_pp']:.2f}pp）；"
          f"受控描述性对照显示：{factor_summary}（η2表示 C2−C1 {rep['mean_regret_difference_pp']:.2f}pp，"
          f"TRAIN-new覆盖 C2−C3 {cov['mean_regret_difference_pp']:.2f}pp）。")
    lines=['# Round18 Ring η2 gap','',lead,'',
      '## 预注册门槛','',
      f'B1 {b1}；B2 {b2}；B3 {b3}；B4 {b4}。配对 bootstrap 每项10000次、按48个TEST state成簇；详见 `results.json`。',
      f'B2 灾难数 {len(catastrophes)}。TEST unknown {results["unknown_count"]} 条，保留为unknown且未重跑；timeout未称deadlock。','',
      '## 析因与范围','',
      f'C1−C0 {contrasts["coverage_C1_minus_C0"]["mean_regret_difference_pp"]:.2f}pp；C3−C0 {contrasts["representation_C3_minus_C0"]["mean_regret_difference_pp"]:.2f}pp；C2−C1 {rep["mean_regret_difference_pp"]:.2f}pp；C2−C3 {cov["mean_regret_difference_pp"]:.2f}pp。各95%CI见 `results.json`。',
      'C1−C0 同时包含Round17与TRAIN-new覆盖；C3−C0 同时包含Round17覆盖与表示变化。C2−C1隔离新增η2表示，C2−C3隔离TRAIN-new覆盖。这是预标签澄清，详见 `PREREGISTRATION_AMENDMENT.md`。',
      'v3家族TRAIN-new与TEST共享配方及架构，仅seed不同；TEST state全新。TRAIN-new发生率是开发数据，TEST结果是前瞻。',
      '每个controller选择、F2优势、η2探测统计与相关、按架构/seed的v3发生率见 `results.json`。图见 `MODEL_REGRET.svg`、`CONTROLLER_CHOICES.svg`、`ETA2_PROBE_SCATTER.svg`。','',
      '## 证据','',
      '36 checkpoint 新颖性与FP32兼容性、109 controller短探测、两次冻结、32例一致性门、TRAIN-new与TEST完整rollout、DB preflight/merge/postflight均在本目录留有审计。按AGENTS.md纯Q例外不运行视频入口。']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
    hand=['# Round18 handoff','',lead,'',
      'Python: `/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`。先读 `PREREGISTRATION.md`、预标签澄清、`FEATURES_FROZEN.receipt.json`、`TEST_FREEZE.receipt.json`、两阶段DB postflight审计。',
      '训练计划与36 checkpoint见 `TRAINING_PLAN.json`、`checkpoints/`；无标签探测见 `probe.py`、`FEATURE_MATRIX.npz`；Q标签见 `train_LABELS.npz`、`test_LABELS.npz`，-1=unknown。模型与CV见 `critic/`。',
      '模型选择、阈值与TEST动作在TEST标签前冻结。`results.json`是数值主记录；重新训练或改规则需另开实验。',
      'v3 TEST与TRAIN-new共享配方/架构，不能称独立新配方。本轮纯Q实验按视频例外执行。']
    (HERE/'HANDOFF.md').write_text('\n'.join(hand)+'\n')
    print(lead)

if __name__=='__main__':main()
