"""Standalone scientific plots, usable dataset exports, and numerical report."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
OUT=HERE/'real_state_expansion'
sys.path.insert(0,str(REPO))


def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=read(OUT/'protocol.json');result=read(OUT/'research_results.json')
    data=np.load(OUT/'test_dataset.npz');x=np.load(OUT/'physical_inputs.npz')
    exports={}
    for name,mask in [('training',data['split']=='train'),('validation',data['split']=='validation'),
                     ('fresh_test',(data['split']=='test')&(data['origin']=='fresh')),
                     ('original_test',(data['split']=='test')&(data['origin']=='original'))]:
        ix=np.flatnonzero(mask);path=OUT/(name+'_dataset.npz')
        trials=4 if name=='training' else 16
        outcomes=data['outcomes'][ix,:,:trials,:]
        valid=outcomes[...,1]==0
        np.testing.assert_array_equal(((outcomes[...,0]==1)&valid).sum(-1),data['success'][ix])
        np.testing.assert_array_equal(((outcomes[...,0]==0)&valid).sum(-1),data['failure'][ix])
        assert (outcomes>=0).all()
        np.savez_compressed(path,state23=x['physical23'][ix],state92=x['local92'][ix],state25=x['physical25'][ix],
            physical_state23=x['raw_physical23'][ix],eta=np.asarray(p['eta']),normalized_eta=x['eta'],
            positions=np.array([p['states'][i]['physical']['positions'] for i in ix]),
            velocities=np.array([p['states'][i]['physical']['velocities'] for i in ix]),
            goals=np.array([p['states'][i]['physical']['goals'] for i in ix]),
            state23_center=x['physical_center'],state23_scale=x['physical_scale'],
            state92_center=x['local_center'],state92_scale=x['local_scale'],
            orientation_center=x['orientation_center'],orientation_scale=x['orientation_scale'],
            success=data['success'][ix],failure=data['failure'][ix],numerical_unknown=data['numerical'][ix],
            rollout_outcomes=outcomes,state_indices=ix,
            state_uids=np.array([p['states'][i]['state_uid'] for i in ix]))
        exports[name]=dict(path=str(path),states=len(ix),eta=16,trials_per_pair=4 if name=='training' else 16,sha256=sha(path))
    write(OUT/'dataset_exports.json',dict(files=exports,input_controller_features=False,
        target='success/(success+failure), with Bernoulli counts retained; numerical_unknown never counted as a failure',
        label_controller=p['controller']['controller_uid'],native_positions_velocities_goals_in='protocol.json',
        outcome_last_axis=['success','numerical_failure','collision'],all_exported_seed_slots_attempted=True,
        input_norm='original32 TRAIN statistics in physical_inputs.npz; identifiers are metadata only'))
    with (OUT/'state_eta_results.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['state_index','state_uid','split','origin','eta_index','eta1','eta2','eta3','success','failure','numerical_unknown','trials','empirical_success_rate','success_lower','success_upper'])
        for i,state in enumerate(p['states']):
            k=4 if state['metadata']['split']=='train' else 16
            for j,eta in enumerate(p['eta']):
                s,fail,u=[int(data[key][i,j]) for key in ('success','failure','numerical')]
                writer.writerow([i,state['state_uid'],state['metadata']['split'],state['expansion_origin'],j,*eta,s,fail,u,k,s/(s+fail) if s+fail else '',s/k,(s+u)/k])
    colors={'local92_eta':'#0072B2','physical23_eta':'#E69F00','physical23_heads':'#009E73'}
    labels={'local92_eta':'92 physical features + eta','physical23_eta':'23 physical features + eta','physical23_heads':'23 physical features, 16 eta heads'}
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for arm,color in colors.items():
        rr=sorted([r for r in result['results'] if r['arm']==arm and r['rule']=='VAL_NLL' and r['split']=='fresh_test'],key=lambda r:r['n'])
        ns=[r['n'] for r in rr]
        for ax,key in zip(axes.ravel(),('state_effect_rho','same_eta_state_change_accuracy','NLL','top1_success_lower')):
            ax.plot(ns,[r[key] for r in rr],'-o',color=color,label=labels[arm],lw=1.8,ms=5)
        ci=sorted([r for r in result['bootstrap'] if r['arm']==arm],key=lambda r:r['n'])
        axes[0,0].fill_between(ns,[r['state_rho_95'][0] for r in ci],[r['state_rho_95'][1] for r in ci],color=color,alpha=.12)
    for ax,title in zip(axes.ravel(),('State effect correlation (eta means removed)','Same-eta state-change sign accuracy','Bernoulli NLL (lower is better)','Chosen eta: empirical success lower bound')):
        ax.set_title(title,fontsize=11);ax.set_xscale('log',base=2);ax.set_xticks([32,64,128,256,512],[32,64,128,256,512])
        ax.set_xlabel('Independent training states');ax.grid(alpha=.2)
    prior=sorted([r for r in result['results'] if r['arm']=='eta_only' and r['split']=='fresh_test'],key=lambda r:r['n'])
    axes[1,0].plot([r['n'] for r in prior],[r['NLL'] for r in prior],'--',color='#666666',label='eta-only TRAIN prior')
    axes[1,1].plot([r['n'] for r in prior],[r['top1_success_lower'] for r in prior],'--',color='#666666')
    axes[0,0].axhline(0,color='gray',ls=':',lw=1);axes[0,1].axhline(.5,color='gray',ls=':',lw=1)
    axes[0,0].legend(fontsize=8);axes[1,0].legend(fontsize=8)
    fig.suptitle('Real Ring rollouts; no controller inputs\n48 fresh test states, 16 eta, 16 trials per pair; VAL-only checkpoint selection',fontsize=13)
    fig.savefig(OUT/'real_learning_curve.png',dpi=180);fig.savefig(OUT/'real_learning_curve.pdf');plt.close(fig)
    # Separate loss curves show overfitting without selecting a checkpoint from TEST.
    frozen=read(OUT/'models_frozen.json');fig,axes=plt.subplots(1,3,figsize=(13,3.7),layout='constrained')
    for ax,arm in zip(axes,colors):
        for n,ls in ((32,'--'),(512,'-')):
            hh=[read(OUT/'fits'/f'{arm}_n{n}'/f'seed{seed}'/'complete.json')['history'] for seed in (17,23,41)]
            steps=[v['step'] for v in hh[0]]
            for key,col in [('train_NLL','#0072B2'),('validation_NLL','#D55E00')]:
                vals=np.mean([[v[key] for v in h] for h in hh],0)
                ax.plot(steps,vals,ls,color=col,label=f'{n} states {key.removesuffix("_NLL")}')
        ax.set_title(labels[arm],fontsize=10);ax.set_xlabel('Optimizer steps');ax.set_ylabel('Bernoulli NLL');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.savefig(OUT/'overfitting_curves.png',dpi=180);fig.savefig(OUT/'overfitting_curves.pdf');plt.close(fig)
    audit=read(OUT/'test_dataset_audit.json')
    lines=['Real controller-input-free Q(s, eta) state-coverage experiment',
        'Q DATA GENERATION AND NUMERICAL EVALUATION COMPLETE. VIDEOS ARE NOT REQUIRED.',
        '',f'512 TRAIN /32 VAL /64 TEST physical states; TEST=16 previously opened +48 fresh.',
        f'16 exact eta per state,4 TRAIN seeds,16 held-out seeds;{audit["requested_present"]} real rollout records.',
        f'8192 exact cached trials +49152 new trials; numerical unknowns={audit["numerical"]}; collisions={audit["collisions"]}.',
        'No controller identifier, weights, flow, response fingerprints, H20 trajectory or state ID enters a predictor.',
        'The physical execution policy remains fixed to define success probabilities. The protocol retains its provenance.',
        'FF-own valid Bernoulli labels are used uniformly, including the retrained32-state baseline.',
        'Normalization is frozen from the original32 TRAIN states. Every size uses the same32-state validation cohort.',
        'Checkpoint selection uses mean3-fit validation NLL. Fresh TEST labels are generated after the model choices freeze.',
        '', 'PRIMARY FRESH-TEST RESULTS (mean of3 training seeds):',
        'arm | states | selected step | state rho | state-eta interaction rho | same-eta sign accuracy | both reversal directions correct | NLL | top1 success lower']
    for r in result['results']:
        if r['split']=='fresh_test' and r['rule'] in ('VAL_NLL','TRAIN_prior'):
            rho='NA' if r['state_effect_rho'] is None else f'{r["state_effect_rho"]:.4f}'
            interaction='NA' if r['interaction_rho'] is None else f'{r["interaction_rho"]:.4f}'
            rev='NA' if r['reversal_both_correct'] is None else f'{r["reversal_both_correct"]:.4f}'
            lines.append(f'{r["arm"]} | {r["n"]} | {r["step"]} | {rho} | {interaction} | {r["same_eta_state_change_accuracy"]:.4f} | {rev} | {r["NLL"]:.4f} | {r["top1_success_lower"]:.4f}')
    lines+=['','Reversals require two states to prefer opposite eta orders with both empirical gaps at least0.25, including numerical-unknown bounds.',
        'Overlapping reversal pairs are descriptive cases, not independent sample units.',
        '95% intervals resample48 physical states jointly across methods, averaging model seeds within each resample.',
        'See physical_state_bootstrap.json for paired changes versus32 states and versus eta-only.',
        'The synthetic256-state fixture from the previous diagnosis is NOT part of this real training dataset.',
        'A finite512-state learning curve tests the coverage hypothesis; it cannot prove all larger data sizes are sufficient or futile.',
        '', 'Exported training/validation/fresh_test NPZs contain physical inputs, exact eta, Bernoulli counts and per-seed outcomes.',
        'state_eta_results.csv is the direct state-to-outcome data table. protocol.json contains original physical states and fingerprints.']
    (OUT/'FINAL_REPORT.txt').write_text('\n'.join(lines)+'\n')
    print('Numerical report and dataset exports written.',flush=True)


if __name__=='__main__':main()
