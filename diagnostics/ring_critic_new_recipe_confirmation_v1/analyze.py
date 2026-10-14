"""Prespecified Round17 analysis. Must run only after frozen selectors and postflight."""
from core import *
import html
from scipy.stats import pearsonr,spearmanr

ARMS=['F10','F13','F2','S200','S300','S400','S500','S600']
POLICIES=['A3_fixed','A0_fixed','G_E','source_mean','A3_state','A0_state','A3_all_fixed','A3_all_state','F10']

def correlation(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if len(a)<3 or not np.isfinite(a).all() or not np.isfinite(b).all() or np.std(a)<1e-10 or np.std(b)<1e-10:
        return dict(Pearson=None,Spearman=None)
    p=float(pearsonr(a,b).statistic);s=float(spearmanr(a,b).statistic)
    return dict(Pearson=p if np.isfinite(p) else None,Spearman=s if np.isfinite(s) else None)

def evaluate(label,choice_rows,indices,state_index=None):
    """Equal-controller mean regret with known-only seed denominators."""
    states=np.arange(label.shape[1]) if state_index is None else np.asarray(state_index)
    subset=label[np.asarray(indices)][:,states]
    good=(subset!=-1);succ=(subset==1)
    counts=good.sum((1,3));wins=succ.sum((1,3))
    if np.any(counts==0):return None
    rates=wins/counts
    oracle=rates.max(1)
    out=dict(oracle=oracle.tolist(),rates=rates.tolist(),known_per_arm=counts.tolist())
    for name in POLICIES:
        if name=='F10':chosen=np.zeros((len(indices),len(states)),int)
        elif name in ('A3_state','A0_state','A3_all_state'):
            chosen=np.array([[ARMS.index(choice_rows[i][name][s]) for s in states] for i in indices],int)
        else:
            chosen=np.array([[ARMS.index(choice_rows[i][name])]*len(states) for i in indices],int)
        values=np.take_along_axis(succ,chosen[:,:,None,None],axis=2)[:,:,0,:]
        known=np.take_along_axis(good,chosen[:,:,None,None],axis=2)[:,:,0,:]
        denom=known.sum((1,2))
        if np.any(denom==0):return None
        success=values.sum((1,2))/denom
        regret=oracle-success
        out[name]=dict(mean_regret_pp=float(regret.mean()*100),max_regret_pp=float(regret.max()*100),
            per_controller_regret_pp=(regret*100).tolist(),per_controller_success=success.tolist(),
            per_controller_known_seed_count=denom.tolist())
    return out

def bootstrap(label,choices,indices):
    rng=np.random.default_rng(1717);d=[]
    for _ in range(10000):
        pick=rng.integers(0,48,size=48)
        result=evaluate(label,choices,indices,pick)
        if result is None:return None
        d.append(result['A3_fixed']['mean_regret_pp']-result['A0_fixed']['mean_regret_pp'])
    return dict(draws=10000,seed=1717,CI95_pp=np.quantile(d,[.025,.975]).tolist(),
                bootstrap_mean_pp=float(np.mean(d)),state_clusters=48)

def svg_figures(results):
    colors=['#2764a2','#f27c38','#409b78','#8a62aa','#b84a67','#7c8e32','#846b53','#d19b23','#999999']
    def svg(w,h):return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">','<rect width="100%" height="100%" fill="white"/>']
    def tx(x,y,s,size=12):return f'<text x="{x}" y="{y}" font-family="sans-serif" font-size="{size}">{html.escape(str(s))}</text>'
    parts=svg(1300,650);parts.append(tx(30,35,'Round17 mean regret by method (pp)',18))
    all_vals=[results['groups'][g]['policies'][p]['mean_regret_pp'] for g in ('N1','N2') for p in POLICIES]
    ymax=max(all_vals+[1])*1.15
    for gi,group in enumerate(('N1','N2')):
        x0=70+gi*630;y0=460;parts.append(tx(x0,65,group,16))
        parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0+570}" y2="{y0}" stroke="black"/>')
        for j,p in enumerate(POLICIES):
            v=results['groups'][group]['policies'][p]['mean_regret_pp'];h=max(0,v/ymax*350);x=x0+j*60
            parts.append(f'<rect x="{x}" y="{y0-h:.1f}" width="40" height="{h:.1f}" fill="{colors[j]}"/>')
            parts.append(tx(x,y0+18,p,9))
    parts.append('</svg>');(HERE/'METHOD_REGRET.svg').write_text('\n'.join(parts))
    rows=results['controllers'];parts=svg(1500,700);parts.append(tx(20,30,'Frozen controller-level choices',18))
    display=['A3_fixed','A0_fixed','G_E','source_mean','A3_all_fixed']
    for j,p in enumerate(display):parts.append(tx(530+j*165,60,p,12))
    for i,row in enumerate(rows):
        y=85+i*33;parts.append(tx(20,y+15,row['controller'][:48],10))
        for j,p in enumerate(display):
            arm=row['frozen_choices'][p];ix=ARMS.index(arm);x=530+j*165
            parts.append(f'<rect x="{x}" y="{y}" width="145" height="25" fill="{colors[ix]}"/>')
            parts.append(tx(x+5,y+17,arm,12))
    parts.append('</svg>');(HERE/'CONTROLLER_CHOICES.svg').write_text('\n'.join(parts))
    parts=svg(1200,700);parts.append(tx(20,30,'A3 inferred effects vs observed effects',18))
    for j,effect in enumerate(('E','L')):
        x0=80+j*560;y0=600;parts.append(tx(x0,60,effect,18))
        parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0+450}" y2="{y0}" stroke="black"/>')
        parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0-450}" stroke="black"/>')
        values=[(r['effects'][effect]['predicted_A3'],r['effects'][effect]['observed'],r['group']) for r in rows if r['effects'][effect]['observed'] is not None]
        lim=max([abs(a) for v in values for a in v[:2]]+[.01])*1.15
        for a,b,g in values:
            x=x0+225+a/lim*210;y=y0-225-b/lim*210
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{colors[0] if g=="N2" else colors[1]}"/>')
        parts.append(tx(x0+120,670,'predicted',12));parts.append(tx(x0+5,80,'observed',12))
    parts.append('</svg>');(HERE/'E_L_SCATTER.svg').write_text('\n'.join(parts))

def main():
    labels_allowed();assert read(HERE/'CONSISTENCY_GATE.json')['passed']
    assert read(HERE/'POSTFLIGHT_AUDIT.json')['status']=='PASS'
    frozen=read(HERE/'FROZEN_CHOICES.json');names=frozen['controller_order'];choices=frozen['controllers']
    assert len(names)==len(choices) and all(r['controller']==n for r,n in zip(choices,names))
    cases=read(HERE/'wave1_CASES.json');lab=np.full((len(names),48,8,4),-2,np.int8)
    index={n:i for i,n in enumerate(names)};unknown=[];outcomes={}
    for c in cases:
        path=HERE/'wave1/raw'/f"{c['rollout_uid']}.json";r=read(path)
        assert r['case']==c and r['protocol_sha256']==sha(HERE/'protocol.json')
        i=index[c['base_controller_name']];s=c['state_index'];a=ARMS.index(c['arm']);q=c['seed']
        assert lab[i,s,a,q]==-2
        if r['numerical_failure']:
            lab[i,s,a,q]=-1;unknown.append(dict(controller=names[i],state_index=s,arm=c['arm'],seed=q,rollout_uid=c['rollout_uid']))
        else:lab[i,s,a,q]=int(r['success'])
        outcomes[r['outcome']]=outcomes.get(r['outcome'],0)+1
    assert not np.any(lab==-2)
    np.savez_compressed(HERE/'LABELS.npz',Y=lab,controller_names=names,state_uids=frozen['state_uids'],arms=ARMS)
    group_ids={g:[i for i,r in enumerate(choices) if r['group']==g] for g in ('N1','N2')}
    group_results={};controller_rows=[]
    for g,ids in group_ids.items():
        evaluated=evaluate(lab,choices,ids)
        if evaluated is None:group_results[g]=dict(status='INCONCLUSIVE_ZERO_KNOWN');continue
        group_results[g]=dict(status='COMPLETE',controller_count=len(ids),policies={p:evaluated[p] for p in POLICIES},
                              unknown_count=int(np.sum(lab[ids]==-1)),oracle=evaluated['oracle'])
    main_group=group_results['N2'];assert main_group['status']=='COMPLETE'
    boot=bootstrap(lab,choices,group_ids['N2'])
    a3=main_group['policies']['A3_fixed'];a0=main_group['policies']['A0_fixed'];ge=main_group['policies']['G_E']
    k1='INCONCLUSIVE' if boot is None else 'SUPPORTED' if boot['CI95_pp'][1]<0 else 'REJECTED'
    k2='SUPPORTED' if a3['mean_regret_pp']<=ge['mean_regret_pp']+.5 and a3['max_regret_pp']<=5 else 'REJECTED'
    full=evaluate(lab,choices,list(range(len(names))))
    k3_bad=[]
    if full is None:k3='INCONCLUSIVE'
    else:
        for i,n in enumerate(names):
            diff=full['F10']['per_controller_success'][i]-full['A3_fixed']['per_controller_success'][i]
            if diff>=.10-1e-12:k3_bad.append(dict(controller=n,group=choices[i]['group'],F10_minus_A3_pp=float(diff*100)))
        k3='SUPPORTED' if not k3_bad else 'REJECTED'
    rate=np.divide((lab==1).sum((1,3)),(lab!=-1).sum((1,3)),out=np.full((len(names),8),np.nan),where=(lab!=-1).sum((1,3))>0)
    for i,row in enumerate(choices):
        effects={}
        for name,a,b in [('E','S500','F10'),('L','S500','F13')]:
            truth=rate[i,ARMS.index(a)]-rate[i,ARMS.index(b)]
            effects[name]=dict(observed=float(truth) if np.isfinite(truth) else None,
                               predicted_A3=row['A3_predicted_'+name],predicted_A3_all=row['A3_all_predicted_'+name])
        controller_rows.append(dict(controller=names[i],group=row['group'],recipe=row['recipe'],
            frozen_choices={p:row[p] for p in ('A3_fixed','A0_fixed','G_E','source_mean','A3_all_fixed')},
            success_rate_by_arm={a:float(v) if np.isfinite(v) else None for a,v in zip(ARMS,rate[i])},
            known_seed_counts_by_arm={a:int(v) for a,v in zip(ARMS,(lab[i]!=-1).sum((0,2)))},effects=effects))
    effects={}
    for g,ids in group_ids.items():
        effects[g]={}
        for effect in ('E','L'):
            truth=[controller_rows[i]['effects'][effect]['observed'] for i in ids]
            pred=[controller_rows[i]['effects'][effect]['predicted_A3'] for i in ids]
            predall=[controller_rows[i]['effects'][effect]['predicted_A3_all'] for i in ids]
            effects[g][effect]=dict(A3=correlation(pred,truth),A3_all=correlation(predall,truth))
    results=dict(evidence_level='PROSPECTIVE_NEW_CONTROLLERS_AND_STATES',N2_scope='new architecture; training datasets shared with source recipes',
        prereg_sha256=sha(HERE/'PREREGISTRATION.md'),prereg_amendment_sha256=sha(HERE/'PREREGISTRATION_AMENDMENT.md'),
        final_freeze_sha256=sha(HERE/'FINAL_FREEZE.receipt.json'),labels_sha256=sha(HERE/'LABELS.npz'),
        controllers=controller_rows,groups=group_results,unknown_count=len(unknown),unknown_cases=unknown,outcome_counts=outcomes,
        effects=effects,bootstrap_K1=boot,hypotheses=dict(K1=k1,K2=k2,K3=k3),K3_catastrophes=k3_bad,
        thresholds=dict(K1_CI_upper_strictly_below_zero=True,K2_margin_pp=.5,K2_max_regret_pp=5,K3_F10_drop_pp=10))
    write(HERE/'results.json',results)
    svg_figures(results)
    lead=(f"冻结的通用 critic A3 在新训练配方 controller 上：K1 {k1}，相对 A0 固定规则的平均遗憾差 "
          f"{a3['mean_regret_pp']-a0['mean_regret_pp']:.2f}pp，95% CI "
          f"{boot['CI95_pp'] if boot else '不可计算'}；K2 {k2}，A3/G_E 平均遗憾 "
          f"{a3['mean_regret_pp']:.2f}/{ge['mean_regret_pp']:.2f}pp、A3 最大遗憾 {a3['max_regret_pp']:.2f}pp；"
          f"K3 {k3}，F10 低至少10pp 的新 controller 数 {len(k3_bad)}。")
    lines=['# Round17 new-recipe critic confirmation','',lead,'','## 事实','',
           '全部新 controller、48个新 state 与 Q4 标签为本轮前瞻采集。N2 共享原训练数据集，仅为新架构配方；N1 为新数据集配方，四个 controller 的均值比较只作描述。','',
           '| 群体 | controller | A3 fixed 遗憾 | A0 fixed 遗憾 | G_E 遗憾 | A3-all fixed 遗憾 | A3 state 遗憾 | unknown |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for g in ('N1','N2'):
        z=group_results[g];p=z['policies'];lines.append(f"| {g} | {z['controller_count']} | {p['A3_fixed']['mean_regret_pp']:.2f}pp | {p['A0_fixed']['mean_regret_pp']:.2f}pp | {p['G_E']['mean_regret_pp']:.2f}pp | {p['A3_all_fixed']['mean_regret_pp']:.2f}pp | {p['A3_state']['mean_regret_pp']:.2f}pp | {z['unknown_count']} |")
    lines += ['','N2 K1 bootstrap、逐 controller 已知 seed 数、各动作成功率、N1 描述、E/L Pearson/Spearman 及所有策略详值见 `results.json`。图见 `METHOD_REGRET.svg`、`CONTROLLER_CHOICES.svg`、`E_L_SCATTER.svg`。','',
              '## 解释','',
              f"K1/K2 的主判定只针对 N2；K3 根据预标签澄清检查所有 {len(names)} 个可用新 controller。unknown 不计成功或失败且未重跑；timeout 是单独终止原因，未当作 deadlock。",
              '若 K1/K2/K3 全通过，可称新配方前瞻确认；否则按预注册结论映射解释，Q 预测质量与 E/L 相关不作为选择收益的替代证据。',
              '','## 证据与执行','',
              '历史训练包字节和 N1 1,792 文件校验、17 checkpoint 新颖性/兼容性、R5 冻结、A3/A0/G_E/A3-all 选择冻结、48 state 去重、32例一致性门、DB preflight/单进程 merge/postflight 都在本目录留有审计与收据。纯 Q 实验按 AGENTS.md 例外不运行视频入口。']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
    hand=['# Round17 handoff','',lead,'',
          '## Reproduce','',
          'Python: `/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`。先阅读 `PREREGISTRATION.md`、预标签澄清、`FINAL_FREEZE.receipt.json` 与 `POSTFLIGHT_AUDIT.json`。`results.json` 为主数值记录；`LABELS.npz` 保留所有 seed 标签，其中 -1=unknown。',
          '训练入口 `train.sbatch`；无标签探测 `probe_r5.py`/`probe.sbatch`；A3-all `fit_a3_all.py`；选择冻结 `freeze_choices.py`；标签设计/执行 `design.py`/`run.py`；DB 合并 `merge.py`；最终分析 `analyze.py`。任何重新训练或标签重算必须另开实验，不覆盖本批冻结证据。','',
          '## Scope','',
          'N2 是共享旧数据集的新架构配方。N1 是新数据集，但只有4个controller，均值比较仅描述。K3 对所有可用新 controller 执行。纯 Q 视频例外已预注册。']
    (HERE/'HANDOFF.md').write_text('\n'.join(hand)+'\n')
    print(lead)

if __name__=='__main__':main()
