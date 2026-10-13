"""Read-only diagnosis of all saved checkpoints; never selects a TEST winner."""
from pathlib import Path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import sys
import csv
import json
import hashlib
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/'checkpoint_overfitting'
SOURCE=HERE.parent/'orthoflow3_tt_ff_matched_learnability_v1'
sys.path.insert(0,str(HERE.parents[1]))
ARMS=('full_replay','constant_h','constant_context','constant_all')
SEEDS=(17,23,41)
STEPS=tuple(range(100,2001,100))
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(path,x):
    path.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def export():
    from diagnostics.state_nuisance_rootcause_v1.experiment import setup
    jax,jnp,serialization,model,d,e,_=setup()
    OUT.mkdir(exist_ok=True)
    protocol=dict(scope='All20 existing snapshots per arm and seed; Ring FF retrospective overfitting diagnosis',
        arms=list(ARMS),seeds=list(SEEDS),steps=list(STEPS),new_training=0,new_rollouts=0,
        checkpoint_selection='None. All steps reported; original source-selected checkpoints remain unchanged.',
        loss='Stable raw-logit count-weighted Bernoulli NLL; avoid probability clipping at late overconfident checkpoints.',
        metrics=['TRAIN/VAL/TEST NLL','TEST B15 selection','Selected Q bounds','Within-state B15-vs-negative AUROC','Trial Brier','Confident-wrong observed trial fraction'],
        temporal_comparisons='Adjacent saved steps, step100-to2000, and original source-selected-to2000; no TEST-selected optimum',
        uncertainty='Average model seeds within physical state, then bootstrap physical states; temporal checkpoints are not independent replicates')
    write(OUT/'protocol.json',protocol)
    tr=np.flatnonzero((d['scene']=='ring_exchange')&(d['split']=='train'))
    ix=np.flatnonzero((d['scene']=='ring_exchange')&(d['split']=='test'))
    means={k:v[tr].mean(0) for k,v in e.items()}
    cm=d['context'][1,tr].mean(0)
    ii,jj=np.repeat(ix,16),np.tile(np.arange(16),len(ix))
    fun=jax.jit(model('full_context').apply)
    original=np.load(HERE/'intervention_logits.npz')
    sel=read(HERE/'selection.json')['arms']
    results={};provenance=[]
    for arm in ARMS:
        xx={k:(v[ii] if arm in ('full_replay','constant_context') else np.broadcast_to(means[k],(len(ii),*means[k].shape))) for k,v in e.items()}
        cc=cm[jj] if arm in ('constant_all','constant_context') else d['context'][1,ii,jj]
        for seed in SEEDS:
            maxprob=0.
            for step in STEPS:
                path=HERE/'fits'/arm/f'seed{seed}'/f'step{step}.msgpack'
                p=serialization.msgpack_restore(path.read_bytes())
                z=[]
                for start in range(0,len(ii),128):
                    sl=slice(start,start+128)
                    z.append(np.asarray(fun(p,{k:jnp.asarray(v[sl]) for k,v in xx.items()},jnp.asarray(d['eta'][ii[sl],jj[sl]]),jnp.asarray(cc[sl]))))
                z=np.concatenate(z).reshape(len(ix),16)
                if step in (sel[arm]['selected_step'],2000):
                    label='final' if step==2000 else 'selected'
                    ref=original[f'{arm}__{seed}__{label}__test']
                    from scipy.special import expit
                    error=float(abs(expit(z)-expit(ref)).max());maxprob=max(maxprob,error)
                    assert error<.002
                results[f'{arm}__{seed}__{step}']=z
                provenance.append(dict(arm=arm,seed=seed,step=step,checkpoint_sha256=sha(path)))
            print('exported',arm,seed,'20 checkpoints; max probability replay error',maxprob,flush=True)
    np.savez_compressed(OUT/'test_logits.npz',**results)
    write(OUT/'export_manifest.json',dict(checkpoints=provenance,backend=jax.default_backend(),
        source_dataset_sha256=sha(SOURCE/'dataset.npz'),protocol_sha256=sha(OUT/'protocol.json'),
        script_sha256=sha(__file__),test_labels_read=False))


def metrics(z,y,split):
    from scipy.special import expit
    cs,cf,s,f,u=[y[k] for k in ('cs','cf','s','f','u')]
    p=expit(np.asarray(z,float))
    loss=cs*np.logaddexp(0,-z)+cf*np.logaddexp(0,z)
    total=(cs+cf).sum()
    trial_brier=(cs*(1-p)**2+cf*p**2).sum()/total
    out=dict(NLL=float(loss.sum()/total),trial_Brier=float(trial_brier),
        confident_wrong_trial_fraction=float(((p>=.99)*cf+(p<=.01)*cs).sum()/total),
        extreme_probability_candidate_fraction=float(((p>=.99)|(p<=.01)).mean()),
        mean_abs_logit=float(abs(z).mean()))
    details=dict(state_NLL=loss.sum(1)/(cs+cf).sum(1))
    if split!='train':
        ix=np.arange(len(z));chosen=z.argmax(1);hit=s[ix,chosen]>=15
        bad=f[ix,chosen]>=2
        auc=[]
        for i in ix:
            pos=z[i,s[i]>=15];neg=z[i,f[i]>=2]
            if len(pos) and len(neg):
                diff=pos[:,None]-neg[None,:]
                auc.append(float(((diff>0)+.5*(diff==0)).mean()))
        out.update(B15=int(hit.sum()),selection_unknown=int((~(hit|bad)).sum()),
            selected_Q16_lower=float((s[ix,chosen]/16).mean()),
            selected_Q16_upper=float(((s[ix,chosen]+u[ix,chosen])/16).mean()),
            within_state_AUROC=float(np.mean(auc)) if auc else None,
            AUROC_states=len(auc),distinct_selected_eta=int(len(np.unique(chosen))))
        details.update(hit=hit,chosen=chosen)
    return out,details


def analyze():
    from diagnostics.state_nuisance_rootcause_v1.analyze import labels,csvout
    d=np.load(SOURCE/'dataset.npz');truth=np.load(SOURCE/'test_truth.npz')
    yl={sp:labels('ring_exchange','FF',sp,d,truth) for sp in ('train','validation','test')}
    zz=np.load(OUT/'test_logits.npz');rows=[];details={}
    for arm in ARMS:
        for seed in SEEDS:
            history=read(HERE/'fits'/arm/f'seed{seed}'/'history.json')
            for step,h in zip(STEPS,history):
                source=np.load(HERE/'fits'/arm/f'seed{seed}'/f'step{step}_source.npz')
                for split in ('train','validation','test'):
                    z=zz[f'{arm}__{seed}__{step}'] if split=='test' else source[split]
                    m,dd=metrics(z,yl[split],split)
                    if split!='test':
                        target=h['TRAIN_NLL' if split=='train' else 'VAL_NLL']
                        assert abs(m['NLL']-target)<3e-6
                    rows.append(dict(arm=arm,seed=seed,step=step,split=split,**m))
                    details[arm,seed,step,split]=dd
    csvout(OUT/'metrics_by_checkpoint.csv',rows)
    numeric=['NLL','trial_Brier','confident_wrong_trial_fraction','extreme_probability_candidate_fraction',
             'mean_abs_logit','B15','selected_Q16_lower','selected_Q16_upper','within_state_AUROC']
    averaged=[];lookup={}
    for arm in ARMS:
        for step in STEPS:
            for split in ('train','validation','test'):
                rr=[r for r in rows if (r['arm'],r['step'],r['split'])==(arm,step,split)]
                m={k:float(np.mean([r[k] for r in rr])) for k in numeric if all(k in r and r[k] is not None for r in rr)}
                averaged.append(dict(arm=arm,step=step,split=split,**m));lookup[arm,step,split]=m
    csvout(OUT/'mean_metrics_by_checkpoint.csv',averaged)
    trends=[]
    for arm in ARMS:
        for before,after in zip(STEPS[:-1],STEPS[1:]):
            tr=[lookup[arm,i,'train']['NLL'] for i in (before,after)]
            va=[lookup[arm,i,'validation']['NLL'] for i in (before,after)]
            te=[lookup[arm,i,'test']['NLL'] for i in (before,after)]
            hit=[lookup[arm,i,'test']['B15'] for i in (before,after)]
            trends.append(dict(arm=arm,before=before,after=after,TRAIN_NLL_change=tr[1]-tr[0],
                VAL_NLL_change=va[1]-va[0],TEST_NLL_change=te[1]-te[0],TEST_B15_change=hit[1]-hit[0]))
    csvout(OUT/'adjacent_step_changes.csv',trends)
    sel=read(HERE/'selection.json')['arms'];summary={}
    for arm in ARMS:
        entry=dict(source_selected_step=sel[arm]['selected_step'],milestones={})
        for step in sorted({100,sel[arm]['selected_step'],500,1000,2000}):
            record={sp:lookup[arm,step,sp] for sp in ('train','validation','test')}
            record['TEST_B15_by_seed']=[next(r['B15'] for r in rows if (r['arm'],r['seed'],r['step'],r['split'])==(arm,s,step,'test')) for s in SEEDS]
            entry['milestones'][str(step)]=record
        tt=[r for r in trends if r['arm']==arm]
        entry['adjacent_transitions']=dict(total=len(tt),
            train_NLL_down_test_NLL_up=sum(r['TRAIN_NLL_change']<-1e-7 and r['TEST_NLL_change']>1e-7 for r in tt),
            train_NLL_down_test_B15_down=sum(r['TRAIN_NLL_change']<-1e-7 and r['TEST_B15_change']<-.01 for r in tt),
            val_NLL_down_test_B15_down=sum(r['VAL_NLL_change']<-1e-7 and r['TEST_B15_change']<-.01 for r in tt))
        differences=np.mean([details[arm,s,2000,'test']['state_NLL']-details[arm,s,100,'test']['state_NLL'] for s in SEEDS],axis=0)
        draw=np.random.default_rng(202610078).integers(len(differences),size=(10000,len(differences)))
        entry['test_NLL_change_2000_minus100']=dict(mean=float(differences.mean()),family_bootstrap95=np.quantile(differences[draw].mean(1),[.025,.975]).tolist())
        summary[arm]=entry
    write(OUT/'summary.json',summary)
    print(json.dumps(summary,indent=2))


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=list(csv.DictReader((OUT/'mean_metrics_by_checkpoint.csv').open()))
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    get=lambda arm,split,key:np.array([float(next(r[key] for r in rows if (r['arm'],r['split'],int(r['step']))==(arm,split,step))) for step in STEPS])
    ax=axes[0,0]
    for split,color in [('train','#289468'),('validation','#d98935'),('test','#b63d36')]:
        ax.plot(STEPS,get('full_replay',split,'NLL'),label=split.upper(),color=color,lw=2)
    if (OUT/'temperature_metrics.csv').exists():
        calibrated=list(csv.DictReader((OUT/'temperature_metrics.csv').open()))
        values=[np.mean([float(r['NLL']) for r in calibrated if (r['arm'],r['split'],int(r['step']))==('full_replay','test',step)]) for step in STEPS]
        ax.plot(STEPS,values,label='TEST after VAL-only calibration',color='#7453a8',ls='--',lw=2)
    ax.set(title='Full critic: fitting improves, generalization worsens',ylabel='Bernoulli NLL (lower is better)')
    ax.legend(fontsize=9)
    colors={'full_replay':'#b63d36','constant_h':'#d98935','constant_context':'#1974a8','constant_all':'#289468'}
    names={'full_replay':'Full','constant_h':'Fixed h, real C','constant_context':'Real h, fixed C','constant_all':'Fixed h and C'}
    for arm in ARMS:
        style='--' if arm=='constant_h' else '-'
        axes[0,1].plot(STEPS,get(arm,'test','NLL'),color=colors[arm],ls=style,label=names[arm],lw=2)
        axes[1,0].plot(STEPS,get(arm,'test','B15'),color=colors[arm],ls=style,lw=2)
    axes[0,1].set(title='Removing state variation limits overfitting',ylabel='TEST NLL')
    axes[0,1].legend(fontsize=9)
    axes[1,0].set(title='Top-1 selection does not track NLL monotonically',ylabel='Mean B15 selections out of16')
    axes[1,0].axhline(14,color='gray',ls=':',lw=1)
    ax=axes[1,1]
    for arm in ['full_replay','constant_context','constant_all']:
        ax.plot(STEPS,100*get(arm,'test','confident_wrong_trial_fraction'),color=colors[arm],lw=2,label=names[arm])
    ax.set(title='Overconfident errors increase',ylabel='Confident-wrong observed trials (%)')
    ax.legend(fontsize=9)
    for ax in axes.flat:ax.set_xlabel('Training step');ax.grid(alpha=.2)
    fig.suptitle('Ring FF: all saved checkpoints, three training seeds averaged',fontsize=14)
    fig.supxlabel('32 TRAIN / 8 VAL / 16 already-opened TEST states. No retraining, new rollout, or TEST-based model selection.\nNLL computed directly from logits; confident-wrong means p>=.99 with failure or p<=.01 with success.',fontsize=9)
    fig.savefig(OUT/'loss_vs_selection.png',dpi=180);fig.savefig(OUT/'loss_vs_selection.pdf')
    plt.close(fig)


def calibration():
    """VAL-only positive temperature, which provably preserves every argmax."""
    from scipy.optimize import minimize_scalar
    from diagnostics.state_nuisance_rootcause_v1.analyze import labels,csvout
    write(OUT/'calibration_protocol.json',dict(
        motivation='Checkpoint audit found growing NLL but improving Full ranking; test whether confidence contributes using rank-preserving temperature scaling.',
        retrospective=True,transform='logit / T',positive_T_bounds=[.25,16.],
        fit='One T per arm and saved step, minimizing mean three-seed VAL observed-count NLL',
        no_test_labels_for_fit=True,no_checkpoint_selection=True,
        interpretation='Any NLL improvement occurs without changing candidate ordering or top1 selections; finite VAL fit is diagnostic, not a deployment guarantee.'))
    d=np.load(SOURCE/'dataset.npz');truth=np.load(SOURCE/'test_truth.npz')
    yl={sp:labels('ring_exchange','FF',sp,d,truth) for sp in ('validation','test')}
    test=np.load(OUT/'test_logits.npz');rows=[];fits=[]
    for arm in ARMS:
        for step in STEPS:
            vz=[np.load(HERE/'fits'/arm/f'seed{s}'/f'step{step}_source.npz')['validation'] for s in SEEDS]
            def objective(logt):
                return np.mean([metrics(z/np.exp(logt),yl['validation'],'validation')[0]['NLL'] for z in vz])
            opt=minimize_scalar(objective,bounds=(np.log(.25),np.log(16.)),method='bounded',options={'xatol':1e-6})
            temperature=float(np.exp(opt.x))
            fits.append(dict(arm=arm,step=step,temperature=temperature,VAL_NLL=float(opt.fun)))
            for seed,z in zip(SEEDS,vz):
                for split,raw in [('validation',z),('test',test[f'{arm}__{seed}__{step}'])]:
                    scaled=raw/temperature
                    np.testing.assert_array_equal(raw.argmax(1),scaled.argmax(1))
                    m,_=metrics(scaled,yl[split],split)
                    rows.append(dict(arm=arm,step=step,seed=seed,split=split,temperature=temperature,**m))
    csvout(OUT/'temperature_fits.csv',fits)
    csvout(OUT/'temperature_metrics.csv',rows)
    for arm in ARMS:
        for step in [100,2000]:
            rr=[r for r in rows if (r['arm'],r['step'],r['split'])==(arm,step,'test')]
            print(arm,step,'T',rr[0]['temperature'],'TEST_NLL',np.mean([r['NLL'] for r in rr]),'B15',[r['B15'] for r in rr])


def report():
    summary=read(OUT/'summary.json')
    rows=list(csv.DictReader((OUT/'temperature_metrics.csv').open()))
    text=[
        'Ring FF 全checkpoint过拟合、概率校准与选择表现诊断',
        '',
        '结论：完整critic有显著的概率预测过拟合，主要表现为置信度过强；其eta排序和top1选择在这段训练中反而改善。',
        '因此不能把完整模型的NLL恶化直接解释成state越来越干扰eta排序。',
        '',
        '读取既有4组×3seed×20checkpoint=240个模型快照；没有新训练、rollout或按TEST重选模型。',
        '每个checkpoint用同一批TRAIN/VAL/TEST state与同一16-eta候选池计算；训练loss采用全TRAIN平均，而非单batch值。',
        'NLL直接由原始logit通过logaddexp计算，以避免晚期极端概率被clip后低估误差。',
        '',
        '完整critic，三个训练seed平均：',
        'step | TRAIN NLL | VAL NLL | TEST NLL | TEST within-state AUROC | TEST B15 by seed | confident-wrong trials',
    ]
    for step,v in summary['full_replay']['milestones'].items():
        text.append(f"{step} | {v['train']['NLL']:.4f} | {v['validation']['NLL']:.4f} | {v['test']['NLL']:.4f} | {v['test']['within_state_AUROC']:.4f} | {v['TEST_B15_by_seed']} | {100*v['test']['confident_wrong_trial_fraction']:.2f}%")
    diff=summary['full_replay']['test_NLL_change_2000_minus100']
    text.extend([
        '',
        f"Step2000相对100的逐family平均TEST NLL变化为{diff['mean']:.4f}，family bootstrap95%={diff['family_bootstrap95']}。",
        '过拟合证据是TRAIN误差降低而VAL/TEST误差升高。Full的19个相邻checkpoint转换中，12次同时出现TRAIN NLL下降、TEST NLL上升。',
        '没有出现TRAIN NLL下降同时三seed平均TEST B15下降的相邻转换；B15总体改善、局部仍有波动，within-state AUROC从约0.959改善到0.982。',
        '',
        '置信度诊断：正温度T只改变概率强弱，严格保留同一state内eta排序和argmax。',
        '每个checkpoint只用8个VAL family选一个T，三个训练seed共用；TEST只用于事后评价。',
    ])
    cal={}
    for step in (100,2000):
        rr=[r for r in rows if (r['arm'],r['split'],int(r['step']))==('full_replay','test',step)]
        cal[str(step)]=dict(T=float(rr[0]['temperature']),NLL=float(np.mean([float(r['NLL']) for r in rr])),B15=[int(r['B15']) for r in rr])
        text.append(f"Full step{step}: T={cal[str(step)]['T']:.4f}, calibrated TEST NLL={cal[str(step)]['NLL']:.4f}, B15={cal[str(step)]['B15']}（排序不变）。")
    text.extend([
        'Step2000的校准NLL回到约0.325，与原eta-only接近。这支持晚期NLL恶化很大程度来自过度自信；不证明所有state泛化误差已经消失。',
        '',
        '确有“训练loss下降而实际选择变差”的对照：',
    ])
    for step in ('200','1000','2000'):
        v=summary['constant_context']['milestones'][step]
        text.append(f"真实h、固定C，step{step}: TRAIN NLL={v['train']['NLL']:.4f}, TEST NLL={v['test']['NLL']:.4f}, B15={v['TEST_B15_by_seed']}。")
    text.extend([
        '该组训练后期确实损失排序/选择质量；正温度校准无法改变这些选错的eta。因此并非所有变差都只靠校准能修复。',
        '',
        '对上轮根因结论的进一步限定：',
        '- 当前早期checkpoint上，固定context恢复选择的配对事实保留。',
        '- 全训练轨迹表明，Full能学到更好的排序，同时概率越来越过于自信。',
        '- 因此“state相关信息本质是噪声”不成立；应区分概率校准过拟合、排序过拟合，以及checkpoint选择指标。',
        '- 当前Full已按VAL NLL选了最早保存的step100，并非没有early stopping；100步之前没有快照，无法据此定位更早的最优点。',
        '- 不应直接按训练loss更低增加训练步数，也不应在已打开的TEST上挑B15最好的checkpoint。',
        '- 若目标是选eta，应在独立验证state上同时检查选择/排序与概率校准；本轮未部署或替换任何critic。',
        '',
        '数值更正：上轮报告step2000 TEST NLL约0.829使用了clip后的概率。稳定原logit算法得到约0.891；早期selected-checkpoint表和B15计数不受影响。',
        '本轮是既有TEST已打开后的回顾性诊断，独立TEST样本只有16个family；训练seed和checkpoint均不是新增独立样本。',
        '',
        '文件：metrics_by_checkpoint.csv、mean_metrics_by_checkpoint.csv、adjacent_step_changes.csv、summary.json、',
        'temperature_fits.csv、temperature_metrics.csv、protocol.json、calibration_protocol.json、export_manifest.json。',
        '图：loss_vs_selection.png / .pdf。',
    ])
    (OUT/'FINAL_REPORT.txt').write_text('\n'.join(text)+'\n')
    write(OUT/'scientific_decision.json',dict(
        full_probability_overfitting=True,full_selection_overall_deteriorates_with_training=False,
        full_overconfidence_supported_by_rank_preserving_calibration=True,
        state_only_fixed_context_late_selection_overfit=True,
        new_training=0,new_rollouts=0,checkpoints_examined=240,retrospective=True,
        no_test_based_checkpoint_selection=True,calibration=cal,
        correction='Earlier endpoint NLL0.829 was clipped; stable-logit NLL is0.891. Initial ablation-selected results unaffected.'))
    print('Wrote checkpoint_overfitting/FINAL_REPORT.txt and scientific_decision.json')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=('export','analyze','plot','calibration','report'))
    globals()[p.parse_args().action]()
